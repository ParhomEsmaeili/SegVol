from argparse import Namespace
from pathlib import Path
import torch
import os
import sys
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import monai.transforms as transforms
import copy
from monai.data import MetaTensor
import warnings 
import re 
import gc 
#############################################################################################################

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))) 
from segment_anything_volumetric import sam_model_registry
from network.model import SegVol
# from data_process.demo_data_process import process_ct_gt
import monai.transforms as transforms
from utils.monai_inferers_utils import sliding_window_inference, generate_box, select_points, build_binary_cube, build_binary_points, logits2roi_coor

class MinMaxNormalization(transforms.Transform):
    def __call__(self, data):
        d = dict(data)
        k = "image"
        d[k] = d[k] - d[k].min()
        d[k] = d[k] / np.clip(d[k].max(), a_min=1e-8, a_max=None)
        return d


class DimTranspose(transforms.Transform):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = np.swapaxes(d[key], -1, -3)
        return d


class ForegroundNormalization(transforms.Transform):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            d[key] = self.normalize(d[key])
        return d

    def normalize(self, ct_narray):
        ct_voxel_ndarray = copy.deepcopy(ct_narray)
        ct_voxel_ndarray = ct_voxel_ndarray.flatten()
        thred = np.mean(ct_voxel_ndarray)
        voxel_filtered = ct_voxel_ndarray[(ct_voxel_ndarray > thred)]
        upper_bound = np.percentile(voxel_filtered, 99.95)
        lower_bound = np.percentile(voxel_filtered, 00.05)
        mean = np.mean(voxel_filtered)
        std = np.std(voxel_filtered)
        ### transform ###
        ct_narray = np.clip(ct_narray, lower_bound, upper_bound)
        ct_narray = (ct_narray - mean) / max(std, 1e-8)
        return ct_narray


def load_segvol(sv_model_type:str, sv_ckpt_path:str, clip_ckpt_path:str, infer_device:torch.device):
    # script_directory = os.path.dirname(os.path.abspath(__file__))

    # Add this directory to the sys.path to allow relative path to clip checkpoint
    # if script_directory not in sys.path:
    #     sys.path.append(script_directory)

    args = Namespace(
        test_mode=True,
        resume=sv_ckpt_path,
        infer_overlap=0.5,
        spatial_size=(32, 256, 256),
        patch_size=(4, 16, 16),
        clip_ckpt=str(clip_ckpt_path),  # This might not work if not running the .py in the base dir
    )

    # gpu = 0

    sam_model = sam_model_registry[sv_model_type](args=args)

    segvol_model = SegVol(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        clip_ckpt=args.clip_ckpt,
        roi_size=args.spatial_size,
        patch_size=args.patch_size,
        test_mode=args.test_mode,
    ).to(device=infer_device)


    segvol_model = torch.nn.DataParallel(segvol_model, device_ids=[infer_device]) #[gpu])

    if os.path.isfile(args.resume):
        ## Map model to be loaded to specified single GPU
        # loc = "cuda:{}".format(torch.cuda.current_device()) #gpu)
        # print(loc)
        checkpoint = torch.load(args.resume, map_location=infer_device)#loc)
        segvol_model.load_state_dict(checkpoint["model"], strict=False)
    segvol_model.eval()

    return segvol_model


class InferApp: #(Inferer):

    # pass_prev_prompts = True
    # dim = 3
    # supported_prompts = ("box", "point")

    def __init__(self, dataset_info, infer_device):

        self.dataset_info = dataset_info
        self.infer_device = infer_device

        if self.infer_device.type != "cuda":
            raise RuntimeError("segvol can only be run on cuda.")

        self.app_params = {
            'sv_model_type': "vit",
            'sv_checkpoint_path': 'SegVol_v1.pth',
            'clip_checkpoint_path': 'config/clip'
        }

        #Loading inference model.

        self.load_model()
        self.build_inference_apps()

        #SegVol app parameters. 
        self.sigmoid_mask_threshold = 0.5
        self.infer_overlap = 0.5
    
        self.spatial_size = (32, 256, 256) #This is the spatial size of the zoom-out domain inference. 
        self.patch_size=(4, 16, 16) #We don't use it for anything in the app, but we will want to print it in the app configs.
        
        # self.redo_map_zoomout_to_fg = False #This was a flag which determined whether the zoom-out domain prompts should be remapped
        #for the zoom-in sliding window inference. The original repo demo did this, but I think because they were sampling prompts
        #in the zoom-out domain and so needed an array with the points.... 

        #We had initially turned this off, because its extra lossy to keep adding transforms which aren't needed if we provide the prompts
        #in the native image domain. I.e., if we first map into the FG domain, then we can store the prompts there without having to later 
        # perform a resampling. 
        # 
        # In the SegFM branch, they actually took this approach for the submission they made (with bbox)
        #I presume that they used bbox because it was better for them and their algo can only handle on prompt type at a time......

        #Also, their prompt propagation into zoom-out domain was always done with image arrays (they bypassed the point based method)
        # in SegFM. For boxes this was relatively fine but for points this can be very lossy. This is probably why (along with boxes being
        # easier for most models when operating with low N number of interactions) that points were not incorporated ultimately. 
        # 
        # Since it is partially unclear on how they would handle points in SegFM,and since we are always going to try to be additive 
        # within a reasonable amount of effort, not fixing things which require an unreasonable amount of effort we will ADD the mechanism 
        # required for point propagation but using 1-1 point mappings (at least to get into the FG and zoom-out domain, at that point 
        # everything else is essentially following the logic of their implementation, including the deletion of background points for sliding window.


        # We do this since it was fairly straightforward to perform this map given that the cropped region is always going to be fixed. 
        #We will not be doing anything else for the zoom-in/sliding window).

        # Unlike SAMMed3D which 1) provided their own interpretation on how to map the points AND dynamically cropped the region, 
        # which would require an unreasonable amount of work to modify and experiment with.... 


        self.atomic_edit = True #This is a flag which determines whether the inference app can be used in interactive editing mode.
        # If set to True, the app will accumulate the prompts and run a fresh inference (i.e. no memory of the prediction is kept) each time. 
        # If set to False, the app will not accumulate the prompts and will raise an error if the user tries to use the app in interactive editing mode. 

        self.permitted_prompts = ('points', 'bboxes')#, 'scribbles') 
        
        #Although scribble isn't in the original implementation, one fairly simple mechanism to implement it would be to 
        # convert the set to a bunch of points. So we may allow it in the future. But for now, we will raise an implementation error downstream.

        self.prompt_subtypes = {
            'points':'free_prompts',
            'scribbles':'free_prompts', 
            'bboxes': 'partition_prompts'
        }

        #Setting some of the app parameters which will be passed back through for the logfile.


        self.app_params.update({
            'mask_prob_threshold': self.sigmoid_mask_threshold,
            'sw_infer_overlap': self.infer_overlap,
            'zoomout_spatial_size': self.spatial_size,
            'sw_patch_size': self.patch_size,
            'atomic_edit': self.atomic_edit,
            'permitted_prompts': self.permitted_prompts,
            'prompt_subtypes_map': self.prompt_subtypes,
        })

        #Differentiating between free and partition-based prompts. Partition based prompts are those which (at least within the 
        # back-end) have at the bare minimum impose some type of partitioning of the image space. Free_prompts do not, 
        # even with a brush size. We want to distinguish between free-prompts with a brush size and something like a lasso (especially
        #because points can be easily placed in 3D on a 2D interface, although this is not necessarily true for scribbles). 
        
        #We just write a separate transform for each. We will be pragmatic and just insert the points into the fg crop
        #transforms as it should not be lossy if the points fall into the foreground crop region.

        self.fg_crop_transform_point = transforms.Compose(
            [
                transforms.Orientationd(
                    keys=["image"], axcodes="RAS", 
                ),    #Doesn't really do anything because they never used the metadata in their training? Also we have already
                #orientated the data into RAS, for now anyways, in our validation framework. Just leaving it here anyways
                ForegroundNormalization(keys=["image"]),
                DimTranspose(keys=["image"]),
                MinMaxNormalization(),
                transforms.CropForegroundd(keys=["image"], source_key="image"),
                #They discarded their spatial padding transform? Not sure why... presumably because they just want to rescale
                # just the foreground crop to the spatial size of the zoom-out domain or something...? Not going to ask too
                # many questions here and just follow their implementation.
                transforms.ToTensord(keys=["image"]),
                transforms.ToDeviced(keys=["image"], device=self.infer_device)
            ]
        )

        self.fg_crop_transform_bbox = transforms.Compose(
            [
                transforms.CropForegroundd(keys=["image", "cube_boxes"], source_key="image"), 
                #They discarded the spatial padding transform? Not sure why... presumably because they just want to rescale 
                #just the foreground crop to the spatial size of the zoom-out domain or something...? Not going to ask too 
                #many questions here and just follow their implementation.
                transforms.ToTensord(keys=["image", "cube_boxes"]),
                transforms.ToDeviced(keys=["image", "cube_boxes"], device=self.infer_device)

            ]
        )
        # self.fg_crop_transform_bbox = transforms.Compose(
        #     [
        #         transforms.Orientationd(
        #             keys=["image", "seg"], axcodes="RAS"
        #         ),  # Doesn't actually do anything since the meta data is never used by SegVol in their preprocessing.
        #         ForegroundNormalization(keys=["image"]),
        #         DimTranspose(keys=["image", "seg"]),
        #         MinMaxNormalization(),
        #         transforms.SpatialPadd(keys=["image", "seg"], spatial_size=(32, 256, 256), mode="constant"),
        #         transforms.CropForegroundd(keys=["image", "seg"], source_key="image"),
        #         transforms.ToTensord(keys=["image", "seg"]),
        #         transforms.ToDeviced(keys=["image", "seg"], device=self.infer_device)
        #     ]
        # )

        #Just write two separate transforms for the zoom-out because the points will be handled not using an image representation
        #as this is extremely lossy and we don't want to delete the background points until we have no choice for the zoom-in.
        self.zoom_out_transform_point = transforms.Resized(
            keys=["image"], spatial_size=self.spatial_size, mode='nearest-exact' #mode='nearest'
        )

        self.zoom_out_transform_bbox = transforms.Resized(
            keys=["image", "cube_boxes"], spatial_size=self.spatial_size, mode='nearest-exact'#mode='nearest'
        )

        #

        # self.zoom_out_transform = transforms.Resized(
        #     keys=["image", "seg"], spatial_size=self.spatial_size, mode='nearest', #They used nearest.  
        #     #mode="nearest-exact" is old.
        # )
        
    def app_configs(self):

        #STRONGLY Recommended: A method which returns any configuration specific information for printing to the logfile. Expects a dictionary format.
        return self.app_params 
    
    def load_model(self):
        
        #Just in case of any spooky action at a distance since we have not yet containerised this application.
        base_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        sv_model_type = self.app_params['sv_model_type']
        sv_ckpt_path = os.path.join(base_dir, 'ckpt', self.app_params['sv_checkpoint_path']) 
        clip_ckpt_path = os.path.join(base_dir, self.app_params['clip_checkpoint_path'])

        infer_device = self.infer_device 
        
        self.model = load_segvol(sv_model_type=sv_model_type, sv_ckpt_path=sv_ckpt_path, clip_ckpt_path=clip_ckpt_path, infer_device=infer_device)

    def build_inference_apps(self):
        #Building the inference app, needs to have an end to end system in place for each "model" type which can be passed by the request: 
        # 
        # IS_autoseg, IS_interactive_init, IS_interactive_edit. (all are intuitive wrt what they represent.) 
        
        self.infer_apps = {
            'IS_autoseg':{'binary_predict':self.binary_inference}, #I think we just raise an error for autoseg....this is too OOD for 
            #this algorithm to handle.
            'IS_interactive_init': {'binary_predict':self.binary_inference},
            'IS_interactive_edit': {'binary_predict':self.binary_inference}
            }

    def binary_subject_prep(self, request:dict):
        
        #Here we perform some actions for determining the state of the infer call for adjusting some of our info extraction mechanisms.
         
        #Ordering the set of interaction states provided, first check if there is an initialisation: if so, place that first. 
        im_order = [] 
        init_modes  = {'Automatic Init', 'Interactive Init'}
        edit_names_list = list(set(request['im']).difference(init_modes))

        #Sorting this list.
        edit_names_list.sort(key=lambda test_str : list(map(int, re.findall(r'\d+', test_str))))

        #Extending the ordered list. 
        
        im_order.extend(edit_names_list) 
        #Loading the image and prompts in the input-im domain & the zoom-out domain.
        

        if request['model'] == 'IS_interactive_edit':
            #In this case we are working with an interactive edit
            if self.atomic_edit == True: #Just trying to be explicit here, although we don't need to actually check for equality
                #with a bool...
                key = edit_names_list[-1]
                is_state = request['im'][key]
                #By default SegVol is not configured as editing a given segmentation mask (or logits map), and so 
                #by atomic_edit = True we mean that we are enabling interactive editing but by just accumulating the prompts
                #and running a fresh inference each time. 
                if all([i is None for i in is_state['interaction_torch_format']['interactions'].values()]) or all([i is None for i in is_state['interaction_torch_format']['interactions_labels'].values()]):
                    raise Exception('Cannot be an interactive request without interactive inputs.')
                
                init = False

                assert isinstance(self.input_dom_img, torch.Tensor)
                assert isinstance(self.input_dom_affine, torch.Tensor)
                assert isinstance(self.input_dom_shape, torch.Size) 

                assert isinstance(self.fg_start_coord, np.ndarray)#torch.Tensor)
                assert isinstance(self.fg_end_coord, np.ndarray) #torch.Tensor)
                assert isinstance(self.fg_dom_shape, torch.Size)
                assert isinstance(self.zoomout_dom_shape, torch.Size)

                assert isinstance(self.image_fg_dom, torch.Tensor)
                assert isinstance(self.image_zoomout_dom, torch.Tensor)
                assert isinstance(self.stored_coords, torch.Tensor)
                assert isinstance(self.stored_coords_lbs, torch.Tensor)
                

            else:
                raise Exception('SegVol, by default, is not configured to be used with iterative refinement approaches and atomic_edit was switched off')
        

        elif request['model'] == 'IS_interactive_init':
            key = 'Interactive Init' 
            is_state = request['im'][key]
            
            if all([i is None for i in is_state['interaction_torch_format']['interactions'].values()]) or all([i is None for i in is_state['interaction_torch_format']['interactions_labels'].values()]):
                raise Exception('Cannot be an interactive request without interactive inputs.')
            init = True 
            
            #Just handling some stored variables which will need to be discarded with the new case. 
            try: 
                del self.input_dom_img 
                del self.input_dom_affine 
                del self.input_dom_shape

                del self.fg_start_coord
                del self.fg_end_coord
                del self.fg_dom_shape
                del self.zoomout_dom_shape 
                del self.image_fg_dom
                del self.image_zoomout_dom

                del self.stored_coords
                del self.stored_coords_lbs

                gc.collect() #Collecting garbage to free up memory, as we don't need the previous image data anymore.
                torch.cuda.empty_cache() #Freeing up memory as we had put these tensors onto gpu memory.. we don't want to force memory that is not
                #needed anymore (so we can free it up for other stuff).

            except:
                pass #HACK: We just want to be able to wipe the prior image and the information pertaining to it when a new case is 
            #passed through, but we might not always have these variables pre-set for the first image 
            # that gets provided (although we could pre-set it to None later...)



        elif request['model'] == 'IS_autoseg':
            key = 'Automatic Init'
            is_state = request['im'][key]
            if is_state is not None:
                raise Exception('Autoseg should not have any interaction info.')
            
            #We will for now actually just place this exception here, might be subject to change.
            raise Exception('Autoseg is way too OOD for SegVol, and most I.S Foundation Models..')     

        
        mapped_input_dict = self.binary_prop_to_model(request['image'], is_state, init=init)

        return mapped_input_dict 
    

    def binary_prop_to_model(self, im_dict: dict, is_state: dict | None, init: bool):
        
        #This function will do the bulk of mapping the request information in the model's native domain. Given that it primarily
        #performs the zoom-out mapping, it will also be used for mapping to the zoom-out domain. 

        #Given that the image data is actually fixed, and even the foreground crop is fixed, we can just extract the relevant 
        #image data once, and retain that in memory. But, for the background prompts the authors perform all of their transformations
        #using morphological operations on an array representation of the box. 

        #For the points, it is unclear what they really wanted. Their method was extremely lossy and they abandoned it presumably.
        #Given that the fix for the points is somewhat straightforward given a fixed foreground crop, we will implement it ourselves.

        #Nevertheless, the map to model domains (zoom-out and foreground) will be slightly different as we want to append the box array
        #if bounding box is provided. Moreover, their implementation (though not really breaking for zoom-out inference) will break
        #on the zoom-in if both points and bounding boxes are provided, so we can safely split according to the prompt type! 
    
        #For simplicity we just wrote the transforms as being distinct for the points and bounding boxes, as it is easier than 
        #first performing the foreground normalisation, and then doing the cropping and zoom etc.

        #First extracting some relevant image data from the request: 
        if init:
            self.input_dom_img = im_dict['metatensor'].as_tensor() #Lets discard the metadata as 1) img is in RAS domain already and
            # 2) they didn't actually use it in their implementation. 
            self.input_dom_affine = copy.deepcopy(im_dict['meta_dict']['affine'])
            self.input_dom_shape = copy.deepcopy(self.input_dom_img.shape[1:]) #Assuming a channel-first image is being provided.

            #We will be using this for reinserting the foreground crop prediction patch into the original image space, eventually.

        if bool(is_state):
            #Checking that the state dict containing the prompts is not a NoneType (which only corresponds to Autoseg!)

            p_dict = (is_state['interaction_torch_format']['interactions'], is_state['interaction_torch_format']['interactions_labels'])
            
            # coords = labels = input_p_mask = None
            
            #Determine the prompt types from the input prompt dictionaries
            provided_ptypes = list(set([k for k,v in p_dict[0].items() if v is not None]) & set([k[:-7] for k,v in p_dict[1].items() if v is not None]))
            
            #SegVol is only capable of supporting either points or bounding boxes, but not both at the same time when using the zoom-in.
            #(and probably just generally this should be the approach when adapting it).

            #We will now check whether more than 1 prompt subtype was provided! 
            provided_subtypes = set([self.prompt_subtypes[ptype] for ptype in provided_ptypes])
            
            if not len(provided_subtypes) == 1:
                raise Exception(f'Only one prompt-subtype is permitted for SegVol when using zoom-in activated, we received {len(provided_ptypes)}')

            #now convert provided_subtype to a list so we can actually index it... 
            provided_subtypes = list(provided_subtypes)

            # if provided_ptypes[0] == "points":
            if provided_subtypes[0] == 'free_prompts':
                
                if 'scribbles' in provided_ptypes:
                    raise NotImplementedError('We have not yet incorporated checks for scribbles')
                
                if provided_ptypes[0] != 'points':
                    raise Exception(f'Only points are currently supported for free prompts, received {provided_ptypes[0]}')
            
                #Now we will perform a mapping into the model domains. We will not be doing this with array representation as this is 
                #not provided by default, and is very lossy. We are only trying to be additive! 

                #Merging the coordinates and labels into one tensor as they are provided in list format. 
                
                #We can always assume that there will be points provided because no autoseg inference is being used here. So, lets just 
                #start storing the points. 
                if init:
                    coords = torch.cat(p_dict[0]['points'], dim=0)
                    labels = torch.cat(p_dict[1]['points_labels'], dim=0)
                    self.stored_coords = coords
                    self.stored_coords_lbs = labels
                else:
                    #If init is false, then it means that we are in an editing mode, and so we need to append the clicks to the existing
                    #prompts. We could've done this in like 2 lines, but we are going to split between init and else for the sake of clarity.
                    new_coords = torch.cat(p_dict[0]['points'], dim=0)
                    new_labels = torch.cat(p_dict[1]['points_labels'], dim=0) 
                    coords = torch.cat([self.stored_coords, new_coords], dim=0) #We can't mix and match prompt subtypes so no need to worry. it will always
                    #be a set of points if we are editing for the current moment.
                    labels = torch.cat([self.stored_coords_lbs, new_labels], dim=0) 

                    #store again. 
                    self.stored_coords = coords
                    self.stored_coords_lbs = labels 

                #Prompt zoomout_dom is meant to be the coordinates (as this is the first step in the inference stack). The prompt_fg_dom
                #is the array representation for the sliding window. 
                (img_fg_dom, img_zoomout_dom), (prompt_fg_dom, prompt_zoomout_dom, prompt_zoomout_dom_lbs), (start_coord, end_coord), early_exit_bool = self.map_to_model_domain_points(img=self.input_dom_img, prompt=coords, prompt_lb=labels, init=init)
    


            elif provided_subtypes[0] == "partition_prompts":
                raise NotImplementedError('Not yet been re-checked as it was not part of the preliminary experiments, please fix.')
                if provided_ptypes[0] != 'bboxes':
                    raise Exception(f'Only bboxes are supported for partition prompts, received {provided_ptypes[0]}')
                
                if not init:
                    raise Exception('Segvol is not configured to handle interactive editing with bounding boxes, only initialisation is supported.')
                
                if 'lasso' in provided_ptypes:
                    raise Exception('Zero chance that this algorithm can handle lasso.')
                #Probably redundant check, but we will keep it here for now..        
            
                #We check the labels on the bounding boxes. SegVol can only use foreground bounding boxes.
                if 0 in p_dict[1]['bboxes_labels']:
                    raise Exception('SegVol can only handle foreground bounding boxes, received background bounding boxes in the request!')

                if len(p_dict[0]['bboxes']) != len(p_dict[1]['bboxes_labels']):
                    raise Exception('The number of bounding boxes and the number of bounding box labels do not match!')
                
                if len(p_dict[0]['bboxes']) != 1:
                    raise Exception('SegVol can only handle one bounding box at a time, received multiple bounding boxes in the request!')

                #Lets get rid of the "batch" dimension (i.e. the prompt instance dimension). 
                coords = torch.cat(p_dict[0]['bboxes'], dim=0)
                labels = torch.stack(p_dict[1]['bboxes_labels'])

                #Lastly, we check that the bounding box is 3D. It is unclear how they would handle 2D bounding boxes. For 2D bounding boxes
                #we will temporarily use a convention that the any of the coordinates must be matching. E.g. x_min = x_max, etc.

                if any(coords[0, i] == coords[0, i+3] for i in range(3)):
                    raise NotImplementedError('SegVol does not support 2D bounding boxes, received a 2D bounding box in the request!')
                
                #Placing the prompts into a tensor, we will be doing this in the same capacity as the implementation of SegVol 
                # which might lead to some information loss in extreme changes in the image voxel count. But this is not our method, we are
                # just testing what they've done!             

                #Extracting the set of coordinate info by picking only the foreground bbox as segvol does.
                idxs = torch.argwhere(labels == 1)[:,0].tolist()
                input_bbox = coords[idxs, :]
                if input_bbox.shape[0] > 1: #Might be a redundant check, but we will keep it here for now.
                    raise Exception('Cannot handle more than one foreground bounding box at a given time. Should have already been flagged..')
                elif input_bbox.shape[0] == 0: #Might be a redundant check... 
                    raise Exception('There was no foreground bounding box provided for this given class (class=foreground if binary segmentation task.)')
                    # input_p_mask = torch.zeros_like(input_dom_img)
                else:
                    #Creating the image array representation if we have one bbox!
                    #We don't need to alter this, as long as the input bbox is in the same coordinate system as the input image it will be
                    #consistent for the downstream transforms. 
                    box_mask = build_binary_cube(input_bbox, self.input_dom_shape).unsqueeze(0)
                    #Prompt zoomout_dom is meant to be the coordinates (as this is the first step in the inference stack). The prompt_fg_dom
                    #is the array representation for the sliding window. 
                    (img_fg_dom, img_zoomout_dom), (prompt_fg_dom, prompt_zoomout_dom, prompt_zoomout_dom_lbs), (start_coord, end_coord), early_exit_bool = self.map_to_model_domain_bbox(self.input_dom_img, box_mask)
            else:
                raise Exception('No other prompting subtypes are supported in SegVol.')

            # if input_p_mask is None:
            #     raise Exception('BUG: Prompt mask was not generated despite the fact that there was a valid input prompt, even if it was empty due to handling of binary classes..') 

        elif is_state is None:
            raise Exception('The request should not have been made without any interaction state, this is not enabled for SegVol!')        
            # #Handling empty prompt dict and/or Autosegmentation.
            # input_p_mask = torch.zeros_like(input_dom_img)
            # provided_ptypes = [None]
        else: 
            raise Exception('Unknown state of the request, should not have reached here!')        
            
        return {
            'img_fg_dom': img_fg_dom,
            'img_zoomout_dom': img_zoomout_dom,
            'fg_dom_shape': self.fg_dom_shape,
            'prompt_fg_dom': prompt_fg_dom, #this is in an array representation, with same shape as the image. ,
            'prompt_zoomout_dom': prompt_zoomout_dom, #this is in a sparse representation, i.e. a tensor of coordinates.
            'prompt_zoomout_dom_lbs': prompt_zoomout_dom_lbs, 
            'prompt_subtype': provided_subtypes[0], #provided_ptypes[0], 
            'fg_start_coord': start_coord,
            'fg_end_coord': end_coord,
            'input_dom_affine': self.input_dom_affine,
            'input_dom_shape': self.input_dom_shape,
            'early_exit_bool': early_exit_bool
        }

    def map_to_model_domain_points(self, img: torch.Tensor, prompt: torch.Tensor, prompt_lb: torch.Tensor, init: bool) -> tuple[torch.Tensor, torch.Tensor]:
        #This is a function which will map the points prompts to the model's zoom-out domain in the exact manner implemented
        #in their SegFM branch implementation. 

        #Given that with atomic edit we can assume that the image is always the same, and array representation transforms for the sparse points
        #is very lossy we will not be passing the array representation of the points for mapping to model domain. 


        #First we will perform the mappings for the image. Then we will use the foreground roi coordinates to determine how to map
        #the points to the fg domain, and then to the zoom-out domain, if any point remained after mapping to foreground. We will
        #not be mapping any points from outside of the foreground domain into the foreground domain as this could result in points
        #that are not representative of the original points that were provided. E.g., a background point outside of the fg crop
        #would probably not make sense to clamp the coordinate as it might not eve mean the same thing. In the same way, we can't reasonably
        # map a foregrund point outside of the fg crop region.  


        #If init, then we will perform the image transforms, otherwise we will just retain this image data.
        if init:
            
            item = {}
            #Converts to numpy as this is the standard datastructure for SegVol data processing.
            item["image"] = img.numpy()
            item = self.fg_crop_transform_point(item) 
            #First it performs the transforms which will apply a normalisation, and the cropping on the image.

            #We will deepcopy to prevent any unintended side effects as we pass it through the next transform. May need to be removed for
            # memory efficiency, but we will see whether that is necessary when we test. 
            self.image_fg_dom = copy.deepcopy(item["image"].float().unsqueeze(0))  # Add batch dimension.
            
            #Extracting the coords for the fg cropping for later reinserting when making a pred. And also for mapping the points to fg domain. 
            self.fg_start_coord = copy.deepcopy(item["foreground_start_coord"])  # Store coordinates for reinsertion of segmented foreground patch.
            self.fg_end_coord = copy.deepcopy(item["foreground_end_coord"])
            #NOTE: These are both length 3 (spatial dimensions). And they are in the coordinate system that the fg crop is (i.e. 
            # dim transposed)

            #We will also store the shape of the foreground domain spatially, so that we can map the coordinates. 
            fg_dom_shape = self.image_fg_dom.shape[2:] 
            self.fg_dom_shape = fg_dom_shape 
            #Then we will perform the zoom-out mapping on the image. 
            
            item_zoom_out = self.zoom_out_transform_point(item) 
            self.image_zoomout_dom = copy.deepcopy(item_zoom_out["image"].float().unsqueeze(0))  # Add batch dimension.
            
            self.zoomout_dom_shape = self.image_zoomout_dom.shape[2:] #We will store the zoom-out domain shape for later use. This should be
            #matching the self.spatial_size parameter so lets double check that.
            if tuple(self.zoomout_dom_shape) != tuple(self.spatial_size):
                raise Exception(f'Zoom-out domain spatial shape {self.zoomout_dom_shape} does not match the spatial size {self.spatial_size}.')
            
            del item_zoom_out
            del item #any required info for retention has been deepcopied
            gc.collect() #Collecting garbage to free up memory, as we don't need the item_zoom_out anymore. 
            torch.cuda.empty_cache() #Freeing up memory as we had put these tensors onto gpu memory... another reason we used the deepcopy
            #so that we delele the dict without having reference issues



        #Now we will map the coordinates into the fg_domain and check which ones will remain. If any clicks remain then we will 
        # map to the zoom-out domain. 
        
        # OrientationD transform: We do nothing, it already came in RAS convention, and besides this transform isn't functionally doing anythin
        # as there is no metadata provided as per the original implementation also.

        # DimTranspose
        #prompt is in shape (N, 3) where N is the number of points and 3 is the number of coordinates in RAS order. 
        prompt = prompt.as_tensor() 
        prompt = prompt[:, [2, 1, 0]]  # We swap the axes according to the DimTranspose since our prompts are provided in RAS order.
        #We want to swap the order for ALL of the points from XYZ into ZYX. 

        #They removed the spatialpadd transform so we do not apply it to our coordinate anymore. 

        # CropForegroundd
        prompt = prompt - self.fg_start_coord
        #First we can just subtract the start_coordinate, as the prompt and the fg_start_end coords will be in the dim transposed coordinate system. 

        ranges = [(0, self.fg_dom_shape[i] - 1) for i in range(3)] #We are zero-indexed and the shape includes the 0th index.

        #Next, we will filter out any prompts which will fall outside of the shape of the foreground domain patch. 
        #We do this by discarding any points which would have negative coordinates, OR coordinates which fall outside of the shape of the
        # the foreground domain patch size. 

        #Generate a mask for the points within the foreground dom patch:
        mask = (
            (prompt[:, 0] >= ranges[0][0]) & (prompt[:, 0] <= ranges[0][1]) &
            (prompt[:, 1] >= ranges[1][0]) & (prompt[:, 1] <= ranges[1][1]) &
            (prompt[:, 2] >= ranges[2][0]) & (prompt[:, 2] <= ranges[2][1])
        ) #These are inclusive ranges, because we used -1 in the range extraction.

        filtered_prompts = prompt[mask]
        filtered_lbs = prompt_lb[mask] 

        if filtered_prompts.shape[0] != filtered_lbs.shape[0]:
            raise Exception('The number of filtered prompts and the number of filtered labels do not match!')
        if filtered_prompts.shape[0] == 0:
            print('No points remained in the foreground domain after cropping... early exit as there is no prompt to use for inference')
            return (self.image_fg_dom, self.image_zoomout_dom), (None, None, None), (self.fg_start_coord, self.fg_end_coord), True 
        #If there are no points remaining, we will just return the fg_dom and zoomout_dom images, and None for the prompts. But we are going
        #to early exit anyways so its just kinda there for consistency...
        else:    
            #There are some points remaining, and so we will map them to the zoom-out domain. 
            #First computing the ratio between the fg crop shape, and the zoom-out domain shape. 
            if any([i == 0 for i in self.fg_dom_shape]):
                warnings.warn('The foreground domain shape is zero, this means that the foreground crop is empty....')
                return (self.image_fg_dom, self.image_zoomout_dom), (None, None, None), (self.fg_start_coord, self.fg_end_coord), True
                #We will pass through a NoneType to indicate that there is no possible mechanism for performing inference. There was no foreground
                # crop, and so no points or ROI to use for inference. We also pass a bool denoting early exit, this is a failure case and so
                # we just can't run inference on this image....
            if any([i == 0 for i in self.zoomout_dom_shape]):
                #It will almost certainly raise a flag if the fg crop was zero when the resizing is attempted anyways...
                warnings.warn('The zoom-out domain shape is zero. Should not be possible. Likely only occurs if fg crop is empty, should have been flagged.')
                return (self.image_fg_dom, self.image_zoomout_dom), (None, None, None), (self.fg_start_coord, self.fg_end_coord), True 
                #Early exit as there is no possible mechanism for performing inference. There was no zoom-out domain. 

            #We use a quick and dirty method of resizing the coordinates.
            resizing_ratio = (torch.tensor(self.fg_dom_shape)) / (torch.tensor(self.zoomout_dom_shape))
            
            # resizing_ratio = (torch.tensor(self.fg_dom_shape) - 1) / (torch.tensor(self.zoomout_dom_shape) - 1)
            #We previously subtract 1 because the coordinates are zero-indexed currently (not subvoxel).  This means that we want to map the zero-indexed
            #coordinates between the two domains. But lets be a bit more accurate when we can...

            prompt_zoomout_dom = (filtered_prompts  + 0.5 )/ resizing_ratio - 0.5 
            #We will also round again. 
            prompt_zoomout_dom = torch.round(prompt_zoomout_dom).to(dtype=torch.int32, device=self.infer_device) 
            #We will also just clamp the coordinates to the zoom-out domain shape, just in case.
            for i in range(3):
                prompt_zoomout_dom[:, i] = torch.clamp(prompt_zoomout_dom[:, i], 0, self.zoomout_dom_shape[i] - 1)
            
            assert all(prompt_zoomout_dom[:, 0] >= 0) and all(prompt_zoomout_dom[:, 1] >= 0) and all(prompt_zoomout_dom[:, 2] >= 0), 'Some of the zoom-out domain coordinates are negative, this should not be possible!'
            assert all(prompt_zoomout_dom[:, 0] < self.zoomout_dom_shape[0]) and all(prompt_zoomout_dom[:, 1] < self.zoomout_dom_shape[1]) and all(prompt_zoomout_dom[:, 2] < self.zoomout_dom_shape[2]), 'Some of the zoom-out domain coordinates are out of bounds, this should not be possible!'
            
            
            prompt_zoomout_dom_lbs = filtered_lbs.to(dtype=torch.uint8, device=self.infer_device)

            #For the fg cropped region points that remained, we will build the array representation using the logic that they used in all of their code, which
            #will end up deleting the background points due to them having a integer representation of 0.. 

            #This is used for their sliding window mechanism. 
            prompt_fg_dom = build_binary_points(filtered_prompts, filtered_lbs, self.fg_dom_shape).to(device=self.infer_device)
            #No channel dimension is added here. Probably later.
            return (self.image_fg_dom, self.image_zoomout_dom), (prompt_fg_dom, prompt_zoomout_dom, prompt_zoomout_dom_lbs), (self.fg_start_coord, self.fg_end_coord), False

    def map_to_model_domain_bbox(
        self, img: torch.Tensor, prompt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        raise NotImplementedError('Still requires double-checking, put this off till later as it is not part of our primary experiments')

        # #This is a function which will map the image, and the bounding box prompts to the model's zoom-out domain in the exact 
        # # manner implemented in their SegFM branch implementation. It uses an array representation. 

        # #It then performs a foreground crop based on the image data, and then a resizing.
                
        # item = {}
        # #Converts to numpy as this is the standard datastructure for SegVol data processing.
        # item["image"] = img.numpy()
        # item["cube_boxes"] = prompt.numpy()
        # item = self.fg_crop_transform_bbox(item) 
        # #First it performs the transforms which apply a cropping on the image and the box.

        # #We will copy to prevent any unintended side effects as we pass it through the next transform. Also so we can delete
        # #the variable to dump memory. We might need to remove this though.
        # image_fg_dom = copy.deepcopy(item["image"].float().unsqueeze(0))  # Add batch dimension. 
        # prompt_fg_dom = copy.deepcopy(item["cube_boxes"].float().unsqueeze(0))  # Add batch dimension. 

        # #Extracting the coords for the fg cropping for later reinserting when making a pred.. 
        # self.fg_start_coord = torch.from_numpy(copy.deepcopy(item["foreground_start_coord"]))  # Store coordinates for reinsertion of segmented foreground patch.
        # self.fg_end_coord = torch.from_numpy(copy.deepcopy(item["foreground_end_coord"]))


        # #Now we will perform the zoom-out mapping, which is a resizing of the image and the box prompts to the zoom-out domain.
        
        # #They then extract the sparse representation of the box in the zoom-out domain. NOT for the fg_dom 
        # # (as this is going to be going into the zoom-in sliding window mechanism, which will requires an array representation of 
        # # the box, at least with minor amendments to the ROI). 

        # item_zoom_out = self.zoom_out_transform_bbox(item)
        # # item["zoom_out_image"] = copy.deepcopy(item_zoom_out["image"])
        # # item["zoom_out_cube_boxes"] = item_zoom_out["cube_boxes"]
        # image_zoomout_dom = copy.deepcopy(item_zoom_out["image"].float().unsqueeze(0))  # Add batch dimension.
        # prompt_zoomout_dom = self.mask3D_to_bbox(prompt_zoomout_dom)#Here we will actually convert to the sparse representation within the line.
        

        # #I fear the prior unsqueeze might have been unnecessary now that we have deepcopied the item dict, I think before it was
        # #intended for the resizing...? We will just leave it for now..., it will make very little difference in time used to just re-index.


        # # image_fg_dom, image_zoomout_dom = item["image"].float().unsqueeze(0), item["zoom_out_image"].float().unsqueeze(0)
        # # prompt_fg_dom, prompt_zoomout_dom = item["cube_boxes"].float().unsqueeze(0), item["zoom_out_cube_boxes"].float().unsqueeze(0)
        
        # image_fg_dom = image_fg_dom[0, 0]
        # prompt_fg_dom = prompt_fg_dom[0, 0]
   
        # return (image_fg_dom, image_zoomout_dom), (prompt_fg_dom, prompt_zoomout_dom, prompt_zoomout_dom_lbs), (self.fg_start_coord, self.fg_end_coord)

    def mask3D_to_bbox(self, gt3D, bbox_shift=None):
        """将3D mask转换为bbox坐标和binary cube,使用tensor实现"""
        #They're using notation that they inherited from segFM challenge, because of the use of sitk, in that context the 
        # z coordinate refers to the inferior-superior axis, and would be in the first axis of the tensor. Our input image was
        # originally in the RAS orientation, and so that z coordinate would have been in the third axis. HOWEVER:
        # In their SegFM implementation, they had removed the use of dimtranspose, so because we have used dimtranspose, we are
        # technically using the z coordinate/inferior-superior coord if we find the coordinates in the 0th axis.
        # 
        # This is why it was so important to apply the transform to the bounding box also.
        b_dict = {}
        z_indices, _, _ = torch.where(gt3D > 0)
        if len(z_indices) == 0:
            # print('Fail to detect foreground! mask3D_to_bbox')
            return torch.tensor([-1,-1,-1,-1,-1,-1])
            
        z_min, z_max = z_indices.min(), z_indices.max()
        z_middle = z_indices[len(z_indices)//2]
        D, H, W = gt3D.shape
        
        b_dict['z_min'] = z_min.item()
        b_dict['z_max'] = z_max.item()
        b_dict['z_mid'] = z_middle.item()

        gt_mid = gt3D[z_middle]
        box_2d = self.mask2D_to_bbox(gt_mid, bbox_shift)
        x_min, y_min, x_max, y_max = box_2d
        
        b_dict['z_mid_x_min'] = x_min.item()
        b_dict['z_mid_y_min'] = y_min.item() 
        b_dict['z_mid_x_max'] = x_max.item()
        b_dict['z_mid_y_max'] = y_max.item()

        assert z_min == torch.clamp(z_min, min=0)
        assert z_max == torch.clamp(z_max, max=D-1)
        return torch.tensor([b_dict['z_min'], b_dict['z_mid_y_min'], b_dict['z_mid_x_min'],
                            b_dict['z_max'], b_dict['z_mid_y_max'], b_dict['z_mid_x_max']])

    def mask2D_to_bbox(self, gt2D, bbox_shift=None):
        """将2D mask转换为bbox坐标,使用tensor实现"""
        #NOTE: y and x indices are swapped to indeed match the convention being used. sitk / dimtransposed RAS+ convention.
        # The axes are Z, Y, X, so the gt2D is the YX order. Hence why the indices are written like this.
        y_indices, x_indices = torch.where(gt2D > 0)
        if len(x_indices) == 0:
            return torch.tensor([-1, -1, -1, -1])
            
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        
        H, W = gt2D.shape
        if bbox_shift is None:
            bbox_shift = 0
        else:
            bbox_shift = torch.randint(0, bbox_shift, (1,))[0]
        
        scale_y, scale_x = gt2D.shape
        bbox_shift_x = int(bbox_shift * scale_x/256)
        bbox_shift_y = int(bbox_shift * scale_y/256)
        
        x_min = torch.clamp(x_min - bbox_shift_x, min=0)
        x_max = torch.clamp(x_max + bbox_shift_x, max=W-1) 
        y_min = torch.clamp(y_min - bbox_shift_y, min=0)
        y_max = torch.clamp(y_max + bbox_shift_y, max=H-1)
        
        #They returned the bbox in x,y,x,y but as we will see above, they undid that so that it would be in y,x,y,x when they
        #append it to the z coordinates.
        boxes = torch.tensor([x_min, y_min, x_max, y_max])
        return boxes

    def reformat_zoomout_prompt(self, mapped_inputs:dict):
        #Function which reformats the zoom-out domain's prompts into the prompts required for passing directly into zoom-out inference.

        #Checking if there are any actual input prompts to begin with?:
        if mapped_inputs['prompt_subtype'] is None:
            raise Exception('There were no prompts provided in the request, cannot perform inference without any prompts!')

        #Currently not looking to simulate text prompt, hence it will be switched off here.
        text_prompt = None

        #Here we follow the demo's treatment of spatio-visual prompts, and assume that the zoom-in mechanism is being used. Hence the points and bbox prompts cannot be provided at the same time!
        if mapped_inputs['prompt_subtype'] == 'free_prompts':
            box_prompt = None 
            point_prompt_coords = mapped_inputs['prompt_zoomout_dom'] 
            if point_prompt_coords.numel() == 0 or point_prompt_coords is None:
                raise Exception('There were no points available yet we are here, should have been caught earlier')
            else:
                point_prompt_lbs = mapped_inputs['prompt_zoomout_dom_lbs']
                point_prompt = (point_prompt_coords.unsqueeze(0), point_prompt_lbs.unsqueeze(0))
                #Requires the batch dimension to be added for the model. 

            # print(f'\n pre_zoom shape: {mapped_inputs["img_zoomout_dom"].shape}')
            # print(f'pre_zoom point coord: {torch.argwhere(mapped_inputs["prompt_fg_dom"])}')
            # print(f'post_zoom shape: {mapped_inputs["img_zoomout_dom"].shape}')
            # print(f'post zoom-out point locations {nonzero_indices} \n')

        elif mapped_inputs['prompt_type'] == 'bboxes':
            point_prompt = None 
            box_prompt_coords = mapped_inputs['prompt_zoomout_dom']
            box_prompt_coords = box_prompt_coords.unsqueeze(0) 

            if box_prompt_coords.numel() != 6 or box_prompt_coords is None:
                raise Exception('There were no bounding boxes available yet we are here, should have been caught earlier') 
            # else:
            #     min_d, max_d = nonzero_indices[:, 0].min(), nonzero_indices[:, 0].max()
            #     min_h, max_h = nonzero_indices[:, 1].min(), nonzero_indices[:, 1].max()
            #     min_w, max_w = nonzero_indices[:, 2].min(), nonzero_indices[:, 2].max()
                
            #     box_prompt = torch.tensor([min_d, min_h, min_w, max_d, max_h, max_w]).unsqueeze(0)
            box_prompt = box_prompt_coords #These methods don't support non-foreground bounding boxes so they never take any labels for them.
        else:
            raise Exception('There was an unsupported prompt type inputted by the request!')
        
        assert text_prompt is None
        assert point_prompt is None or (point_prompt[0].numel() > 0 and point_prompt[1].numel() > 0)
        assert box_prompt is None or box_prompt.numel() 

        if point_prompt is None and box_prompt is None:
            raise Exception('There were no prompts provided in the request, cannot perform inference without any prompts! Should have been caught!')
    
        return text_prompt, point_prompt, box_prompt 
    
    @torch.no_grad()
    def binary_zoom_out_predict(self, mapped_inputs:dict):
        # Performing zoom-out inference and mapping back to fg.

        #Reformatting the prompts into the format required for passing into the model
        text_zoomout_input, points_zoomout_input, box_zoomout_input = self.reformat_zoomout_prompt(mapped_inputs)
        
        logits_global_zoom_out = self.model(
            mapped_inputs['img_zoomout_dom'], text=text_zoomout_input, boxes=box_zoomout_input, points=points_zoomout_input
        )

        # resize back global logits to the fg domain.
        logits_fg = F.interpolate(logits_global_zoom_out.cpu(), size=mapped_inputs['fg_dom_shape'], mode="nearest")[
            0
        ][0]

        return logits_fg

    @torch.no_grad()
    def binary_inference(self, request, use_zoom=True):
        
        #First we will map the request to the models' foreground and zoom-out domains, which is the first step in the inference process.
        #Also extracts information required for early exit, and for reinserting the logits into the original image domain.
        mapped_inputs = self.binary_subject_prep(request=request)

        #If early exit then we can't do inference....:

        if mapped_inputs['early_exit_bool']:
            warnings.warn('Early exit requested, either no foreground region was detected in the image or no prompts in the foreground domain')
            return self.binary_process_output(mapped_inputs, None, early_exit_bool=True)
        else:
            #Performing inference on the zoom-out image and mapping back to fg domain.
            logits_fg = self.binary_zoom_out_predict(mapped_inputs)

            #They always use zoom-in, and don't have some heuristic to determine whether to use it or not so we will always use it..
            if not use_zoom:
                assert logits_fg.shape == mapped_inputs['img_fg_dom'].shape 
                return self.binary_process_output(mapped_inputs, logits_fg)
                
            #If we are here, then we will be performing the zoom-in inference, which is a sliding window inference on the fg domain.

            #Extracting the region of interest for zoom-in within the fg, also checks for whether anything was predicted in the zoomout:

            #NOTE: Modification was made in the SegFM implementation, they now use the binary map of the foreground prompts to also inform
            #their roi region. (I.e., don't throw out the regions were prompts were provided in the fg domain). 

            #Comparing the shape of the array representation of the prompt in fg domain that will be used to do the sliding window zoom in.

            assert mapped_inputs['prompt_fg_dom'].shape == mapped_inputs['fg_dom_shape'], f"Prompt fg dom shape {mapped_inputs['prompt_fg_dom'].shape} does not match image fg dom shape {mapped_inputs['img_fg_dom'].shape}"
            #This function might be a little slower because its performing on cpu, and we don't want to push a bunch of memory handling
            #operations onto the back-end functions unless we have to... hopefully this will be fast enough.
            min_d, min_h, min_w, max_d, max_h, max_w = logits2roi_coor(spatial_size=mapped_inputs['fg_dom_shape'], logits_global_single=logits_fg.to(device=self.infer_device), prompt_map=mapped_inputs['prompt_fg_dom'])

            if min_d is None:
                warnings.warn('Warning, for one reason or another, no foreground or prompt was detected and skipping zoom-in. Results may be very poor if the target actually exists....')
                return self.binary_process_output(mapped_inputs, logits_fg, early_exit_bool=True) 
            else:
                #Otherwise, there is not much wrong here, just continue as segvol does.. mapping image, prompts, pred to the zoom-in.

                # Crop roi for zoom-in from the foreground region cropped roi.
                #Unlike the prompt, we didn't need to remove the batch and channel dim from the image. 
                #But the zoomin image array will need it, so lets be careful.
                img_zoomin_dom = mapped_inputs['img_fg_dom'][0,0, min_d:max_d+1, min_h:max_h+1, min_w:max_w+1].unsqueeze(0).unsqueeze(0)
                assert img_zoomin_dom.ndim == 5 
                coarse_pred_zoomin_dom = (torch.sigmoid(logits_fg[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1])>self.sigmoid_mask_threshold).long() 
                #So much use of int64 by the authors........... hopefully this will not cause any memory issues, but we will see.

                # prompt_reflection = None
                prompt_zoomin_dom = mapped_inputs['prompt_fg_dom'][min_d:max_d+1, min_h:max_h+1, min_w:max_w+1]
                prompt_reflection = (
                    prompt_zoomin_dom.unsqueeze(0).unsqueeze(0),
                    coarse_pred_zoomin_dom.unsqueeze(0).unsqueeze(0),
                )

                assert img_zoomin_dom.shape[2:] == coarse_pred_zoomin_dom.shape == prompt_zoomin_dom.shape 

                if mapped_inputs['prompt_subtype'] == 'partition_prompts':
                    raise NotImplementedError('We have not yet double-checked the bounding box with the modifications introduced with the SegFM logic')

                ## inference
                logits_zoomin_dom = sliding_window_inference(
                    img_zoomin_dom,
                    prompt_reflection,
                    self.spatial_size,
                    1,
                    self.model,
                    self.infer_overlap,
                    text=None,
                    use_box=(mapped_inputs['prompt_subtype'] == "partition_prompts"),#"bboxes"),
                    use_point=(mapped_inputs['prompt_subtype'] == "free_prompts")#"points"),
                ).cpu().squeeze()
                
                gc.collect() #Collecting garbage to free up memory
                torch.cuda.empty_cache() 

                #Updating the logits fg with the zoom-in roi logits.
                logits_fg[min_d:max_d + 1, min_h:max_h+1, min_w:max_w+1] = logits_zoomin_dom
                
            assert logits_fg.shape == mapped_inputs['img_fg_dom'].shape[2:]

        #Here we will perform the re-insertion back into the original image input domain.
        return self.binary_process_output(mapped_inputs, logits_fg)

        

    def binary_process_output(self, mapped_inputs:dict | None, logits_fg: torch.Tensor | None, early_exit_bool:bool=False):
        #This func will be reversing the order of operations in the input processing, in order to convert our foreground logits to the outputs desired:
        #probabilistic map & a discretised segmentation, both channel first in the input image domain!

        #If early exit bool was true then there were no prompts or foreground so just need to return an empty probabilistic map and 
        # prediction map... 

        if early_exit_bool:
            output_prob_map = torch.cat(
                [   #We set it to zeroes as all background if early-exited...
                    torch.ones(1, *mapped_inputs['input_dom_shape'], dtype=torch.float32), 
                    torch.zeros(1, *mapped_inputs['input_dom_shape'], dtype=torch.float32)
                ], dim=0)
            output_pred_map = torch.zeros(1, *mapped_inputs['input_dom_shape'], dtype=torch.uint8)
                
        else:
            #Convert to probabilistic map here because we can't pad with -inf to represent the background probability.

            prob_fg = torch.sigmoid(logits_fg)

            #Now mapping to the input image domain.

            #Process entails the creation of a zeros array which we map into the model domain, and then undoing operations 
            # which were applied to map the image into the model domain, in order to undo the map from the logits in model domain 
            # back to the input image domain.
    
            # Dimension transposition was first.
            transpose_dom_shape = mapped_inputs['input_dom_shape'][::-1]
            
            #Padding operation was removed in SegFM so we removed it here. 

            #Create an empty array to insert the foreground probability map.
            prob_transpose_dom = torch.zeros(transpose_dom_shape, dtype=torch.float32
            ) 
            prob_transpose_dom[
                mapped_inputs['fg_start_coord'][0] : mapped_inputs['fg_end_coord'][0],
                mapped_inputs['fg_start_coord'][1] : mapped_inputs['fg_end_coord'][1],
                mapped_inputs['fg_start_coord'][2] : mapped_inputs['fg_end_coord'][2],
            ] = prob_fg 

            #Padding was removed in SegFM and we removed it in the forward propagation into model domain, so we do not need to remove
            #it here. 
 
            # Undo the dim_transpose (which swapped from XYZ to ZYX) to get back to the original input image domain)
            prob_input_dom = torch.permute(prob_transpose_dom, (2, 1, 0))

            assert prob_input_dom.shape == mapped_inputs['input_dom_shape']
            
            #Now we must convert this into the format expected by the validation framework. CHWD for the prob and 1HWD for the discrete pred.

            #The config labels are always corresponding to 0,1 with 0 background and 1 fg. Hence we stack these correspondingly.
            output_prob_list = []
            for label in self.configs_labels_dict.keys():
                if label.title() == 'Background':
                    output_prob_list.append(1-prob_input_dom)
                else:
                    output_prob_list.append(prob_input_dom)  
            output_prob_map = torch.stack(output_prob_list)

            output_pred_map = (prob_input_dom > self.sigmoid_mask_threshold).to(dtype=torch.uint8).unsqueeze(0)


        return (output_prob_map, output_pred_map, mapped_inputs['input_dom_affine'])


    def __call__(self, request:dict):

        if len(request['config_labels_dict']) == 2:
            class_type = 'binary'
        elif len(request['config_labels_dict']) > 2:
            class_type = 'multi'
            raise NotImplementedError 
        else:
            raise Exception('Should not have received less than two semantic class labels at minimum')
        
        #We create a duplicate so we can transform the data from metatensor format to the torch tensor format compatible with the inference script.
        modif_request = copy.deepcopy(request) 

        app = self.infer_apps[modif_request['model']][f'{class_type}_predict']

        #Setting the configs label dictionary for this inference request.
        self.configs_labels_dict = modif_request['config_labels_dict']


        probs_tensor, pred, affine = app(request=modif_request)


        pred = pred.to(device='cpu')
        probs_tensor = probs_tensor.to(device='cpu')
        affine = affine.to(device='cpu')
        del modif_request 
        gc.collect() 
        torch.cuda.empty_cache()

        assert probs_tensor.shape[1:] == request['image']['metatensor'].shape[1:]
        assert pred.shape[1:] == request['image']['metatensor'].shape[1:] 
        assert torch.all(affine == request['image']['metatensor'].meta['affine'])
        assert isinstance(probs_tensor, torch.Tensor) 
        assert isinstance(pred, torch.Tensor)
        assert isinstance(affine, torch.Tensor)

        output = {
            'probs':{
                'metatensor':probs_tensor,
                'meta_dict':{'affine': affine}
            },
            'pred':{
                'metatensor':pred,
                'meta_dict':{'affine': affine}
            },
        }
        return output 

if __name__ == '__main__':
   
    infer_app = InferApp(
        {'dataset_name':'BraTS2021',
        'dataset_modality':'MRI'}, torch.device('cuda'))

    infer_app.app_configs()

    from monai.transforms import LoadImaged, Orientationd, EnsureChannelFirstd, Compose 
    import nibabel as nib 

    input_dict = {'image':'/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_binarised/imagesTs/BraTS2021_00266.nii.gz'}
    load_and_transf = Compose([LoadImaged(keys=['image']), EnsureChannelFirstd(keys=['image']), Orientationd(keys=['image'], axcodes='RAS')])

    final_loaded_im = load_and_transf(input_dict)
    meta = {'original_affine': torch.from_numpy(final_loaded_im['image_meta_dict']['original_affine']).to(dtype=torch.float64), 'affine': torch.from_numpy(final_loaded_im['image_meta_dict']['affine']).to(dtype=torch.float64)}
    input_metatensor = MetaTensor(x=torch.from_numpy(final_loaded_im['image']).to(dtype=torch.float64), meta=meta) #affine=torch.from_numpy(final_loaded_im['image_meta_dict']['affine']).to(dtype=torch.float64))
    # MetaTensor(x=torch.from_numpy(final_loaded_im['image']).to(dtype=torch.float64), meta=final_loaded_im['image_meta_dict'], affine=torch.from_numpy(final_loaded_im['image_meta_dict']['affine']).to(dtype=torch.float64))
    request = {
        'image':{
            'metatensor': input_metatensor,
            'meta_dict':{'affine':input_metatensor.affine}
        },
        # 'model':'IS_interactive_edit',
        'model': 'IS_interactive_init',
        'config_labels_dict':{'background':0, 'tumor':1},
        'im':
        
        # {'Automatic Init': None}
        {'Interactive Init':{
            'interaction_torch_format': {
                'interactions': {
                    'points': None, #[torch.tensor([[40, 103, 43]]), torch.tensor([[62, 62, 39]])], #None
                    'scribbles': None, 
                    'bboxes': [torch.Tensor([[56,30,17, 92, 76, 51]]).to(dtype=torch.int64)] #None 
                    },
                'interactions_labels': {
                    'points_labels': None,#[torch.tensor([0]), torch.tensor([1])], #None,#[torch.tensor([0]), torch.tensor([1])], 
                    'scribbles_labels': None, 
                    'bboxes_labels': [torch.Tensor([1]).to(dtype=torch.int64)] #None
                    }
                    },
          
            'interaction_dict_format': {
            'points': {'background': [[40, 103, 43]],
            'tumor': [[62, 62, 39]]
            },
            # 'points': None,
            'scribbles': None,
            'bboxes': None, #{'background': [], 'tumor': [[56,30,17, 92, 76, 51]]} #None
            },
            'prev_probs': {'metatensor': None, 'meta_dict': None}, 
            'prev_pred': {'metatensor': None, 'meta_dict': None}}
        },
    }
    output = infer_app(request)
    print('halt')