from argparse import Namespace
# from radioa.datasets_preprocessing.conversion_utils import load_any_to_nib
from pathlib import Path
# from radioa.datasets_preprocessing.conversion_utils import load_any_to_nib
# from radioa.model.inferer import Inferer
# from radioa.prompts.prompt import Boxes3D, Points, PromptStep
# from radioa.utils.SegVol_segment_anything.network.model import SegVol
import torch
import os
import sys
# from radioa.utils.SegVol_segment_anything.monai_inferers_utils import (
#     build_binary_points,
#     build_binary_cube,
#     logits2roi_coor,
#     sliding_window_inference,
# )
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import monai.transforms as transforms
import copy
from monai.data import MetaTensor
import warnings 
import re 
# from radioa.utils.SegVol_segment_anything import sam_model_registry
# from radioa.utils.paths import get_model_path
# from radioa.utils.transforms import resample_to_shape_sparse

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

        #SegVol initialisations.

        self.prev_mask = None
        self.inputs = None
        self.mask_threshold = 0
        self.infer_overlap = 0.5
        self.start_coord = None
        self.end_coord = None

        self.spatial_size = (32, 256, 256)
        self.redo_map_zoomout_to_fg = True 

        #This is a set of transforms which takes an image and encoded prompt (analogous to the manner in which SegVol implements their mappings for zoom-in using 
        # image array representations) and extracts the region of interest according to the image. 

        ##NOTE: Possible weaknesses, this will eliminate background prompts without integer-code remapping, #and will eliminate prompts that would fall outside of the 
        # foreground patch. Here we are implicitly being uncharitable and adding code based off the current approach taken where possible.
        
        self.transform_input = transforms.Compose(
            [
                transforms.Orientationd(
                    keys=["image", "seg"], axcodes="RAS"
                ),  # Doesn't actually do anything since the meta data is never used by SegVol in their preprocessing.
                ForegroundNormalization(keys=["image"]),
                DimTranspose(keys=["image", "seg"]),
                MinMaxNormalization(),
                transforms.SpatialPadd(keys=["image", "seg"], spatial_size=(32, 256, 256), mode="constant"),
                transforms.CropForegroundd(keys=["image", "seg"], source_key="image"),
                transforms.ToTensord(keys=["image", "seg"]),
                transforms.ToDeviced(keys=["image", "seg"], device=self.infer_device)
            ]
        )

        # self.transform_prompts = transforms.Compose(
        #     [
        #         transforms.Orientationd(
        #             keys=["seg"], axcodes="RAS"
        #         ),  # Doesn't actually do anything since metadata is discarded. Kept for comparability to original
        #         DimTranspose(keys=["seg"]),
        #         transforms.SpatialPadd(keys=["seg"], spatial_size=(32, 256, 256), mode="constant"),
        #         transforms.CropForegroundd(keys=["image", "seg"], source_key="image"),
        #         transforms.ToTensord(keys=["seg"]),
        #     ]
        # )

        #NOTE: It is unclear, according to the authors, which approach would be better for resizing the prompt map. Here we 
        self.zoom_out_transform = transforms.Resized(
            keys=["image", "seg"], spatial_size=self.spatial_size, mode="nearest-exact"
        )
        # self.img_loader = transforms.LoadImage()

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
            'IS_autoseg':{'binary_predict':self.binary_predict},
            'IS_interactive_init': {'binary_predict':self.binary_predict},
            'IS_interactive_edit': {'binary_predict':self.binary_predict}
            }
        
    def binary_prop_to_model(self, im_dict: dict, is_state: dict | None):
        
        input_dom_img = im_dict['metatensor']
        input_dom_affine = im_dict['meta_dict']['affine']
        input_dom_shape = input_dom_img.shape

        if bool(is_state):
            #Placing the prompts into a tensor, we will be doing this in the same capacity as the demo implementation of SegVol which will inevitably lead to 
            # information loss.
            p_dict = (is_state['interaction_torch_format']['interactions'], is_state['interaction_torch_format']['interactions_labels'])
            
            coords = labels = input_p_mask = None
            
            #Determine the prompt type from the input prompt dictionaries: Not sure if intersection is optimal for catching exceptions here.
            provided_ptypes = list(set([k for k,v in p_dict[0].items() if v is not None]) & set([k for k,v in p_dict[1].items() if v is not None]))
            if not len(provided_ptypes) == 1:
                raise Exception('Only one valid prompt is permitted for SegVol otherwise the zoom-in mechanism is non-functional.')
            
            if provided_ptypes[0] == "points":
                
                #NOTE: The strategy employed by SegVol when working with image representations of prompt inputs will inevitably lead to the deletion of background prompts 
                # as they only retain the 1s. (Whatever that is depends on the definition here, but typically it will be some arbitary foreground.)
                coords = torch.cat(p_dict[0]['points'], dim=0)
                labels = torch.cat(p_dict[1]['points'], dim=0)
                # points_input = (coords.unsqueeze(0).to(device=self.infer_device), labels.unsqueeze(0).to(device=self.infer_device))
                input_p_mask = build_binary_points(coords, labels, input_dom_shape[1:]).unsqueeze(0)
                input_p_mask = input_p_mask

            elif provided_ptypes[0] == "bboxes":
                #NOTE: The strategy employed by SegVol when working with image representations of prompt inputs will inevitably lead to the deletion of background prompts 
                # as they only retain the 1s. (Whatever that is depends on the definition here, but typically it will be some arbitary foreground.)
                #NOTE: We can typically assume that the background probably won't have a bbox because that doesn't really have an inherent meaning.... 

                coords = torch.cat(p_dict[0]['bboxes'], dim=0)
                labels = torch.stack(p_dict[1]['bboxes'])

                #Extracting the set of coordinate info by picking only the foreground bbox as segvol does.
                idxs = torch.argwhere(labels == 1)[:,0].tolist()
                input_bbox = coords[idxs, :]
                if input_bbox.shape[0] > 1:
                    raise Exception('Cannot handle more than one foreground bounding box at a given time.')
                elif input_bbox.shape[0] == 0:
                    warnings.warn('There was no foreground bounding box provided for this given class (class=foreground if binary segmentation task.)')
                    input_p_mask = torch.zeros_like(input_dom_img)
                else:
                    #Creating the image array representation if we have one bbox!
                    input_p_mask = build_binary_cube(input_bbox, input_dom_shape[1:]).unsqueeze(0)
        
            else:
                raise Exception('No other prompting types are supported in SegVol.')

            if input_p_mask is None:
                raise Exception('BUG: Prompt mask was not generated despite the fact that there was a valid input prompt, even if it was empty due to handling of binary classes..') 

        else:
            #Handling empty prompt dict and/or Autosegmentation.
            input_p_mask = torch.zeros_like(input_dom_img)
            provided_ptypes = [None]
                
        (img_fg_dom, img_zoomout_dom), (prompt_fg_dom, prompt_zoomout_dom), (start_coord, end_coord) = self.input_forward_map(
            input_dom_img, input_p_mask
        )
        
        return {
            'img_fg_dom': img_fg_dom,
            'img_zoomout_dom': img_zoomout_dom,
            'fg_dom_shape': img_fg_dom.shape,
            'prompt_fg_dom': prompt_fg_dom,
            'prompt_zoom_dom': prompt_zoomout_dom,
            'prompt_type': provided_ptypes[0], 
            'start_coord': start_coord,
            'end_coord': end_coord,
            'input_dom_affine': input_dom_affine,
            'input_dom_shape': input_dom_shape,
        }
    def input_forward_map(
        self, img: torch.Tensor, prompt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        item = {}
        #Converts to numpy as this is the standard obj type for SegVol data processing.
        item["image"] = img.numpy()
        item["seg"] = prompt.numpy()
        item = self.transform_input(item)

        start_coord = item["foreground_start_coord"]  # Store coordinates for reinsertion of segmented foreground patch.
        end_coord = item["foreground_end_coord"]

        item_zoom_out = self.zoom_out_transform(item)
        item["zoom_out_image"] = item_zoom_out["image"]
        item["zoom_out_seg"] = item_zoom_out["seg"]
        image_fg_dom, image_zoomout_dom = item["image"].float().unsqueeze(0), item["zoom_out_image"].float().unsqueeze(0)
        prompt_fg_dom, prompt_zoomout_dom = item["seg"].float().unsqueeze(0), item["zoom_out_seg"].float().unsqueeze(0)
        
        image_fg_dom = image_fg_dom[0, 0]
        prompt_fg_dom = prompt_fg_dom[0, 0]
                  
        return (image_fg_dom, image_zoomout_dom), (prompt_fg_dom, prompt_zoomout_dom), (torch.from_numpy(start_coord), torch.from_numpy(end_coord))

    def map_to_sparse_prompt(self, mapped_inputs:dict):
        #Function which maps the prompts from a (here zoom-out domain's) image-scale representation, to a sparse representation for performing inference.

        #Checking if there are any actual input prompts to begin with?:
        if mapped_inputs['prompt_type'] is None:
            warnings.warn('Be careful, there is no prompt provided in the "api"-call and SegVol is not trained for this.')
            return None, None, None 

        #Currently not looking to simulate text prompt, hence it will be switched off here.
        text_prompt = None
        #Here we follow the demo's treatment of spatio-visual prompts, and assume that the zoom-in mechanism is being used. Hence the points and bbox prompts cannot be provided at the same time!
        if mapped_inputs['prompt_type'] == 'points':
            box_prompt = None 
            nonzero_indices = torch.nonzero(mapped_inputs['prompt_zoom_dom'][0,0])
            if nonzero_indices.shape[0] == 0:
                point_prompt = None 
                warnings.warn('The mapping to model-domain has left no remaining input points despite there initially being some in the request!!')
            else:
                point_prompt_coords = nonzero_indices.unsqueeze(0)
                point_prompt_lbs = torch.ones(point_prompt_coords.shape[:-1])
                point_prompt = (point_prompt_coords, point_prompt_lbs)

        elif mapped_inputs['prompt_type'] == 'bboxes':
            point_prompt = None 
            nonzero_indices = torch.nonzero(mapped_inputs['prompt_zoom_dom'][0,0])
            if nonzero_indices.shape[0] == 0:
                box_prompt = None 
            else:
                min_d, max_d = nonzero_indices[:, 0].min(), nonzero_indices[:, 0].max()
                min_h, max_h = nonzero_indices[:, 1].min(), nonzero_indices[:, 1].max()
                min_w, max_w = nonzero_indices[:, 2].min(), nonzero_indices[:, 2].max()
                
                box_prompt = torch.tensor([min_d, min_h, min_w, max_d, max_h, max_w]).unsqueeze(0)
        else:
            raise Exception('There was an unsupported prompt type inputted by the request!')
        
        assert text_prompt is None
        assert point_prompt is None or point_prompt.numel()
        assert box_prompt is None or box_prompt.numel() 

        return text_prompt, point_prompt, box_prompt 
    
    def postprocess_mask(self, mask, return_logits):
        """
        Postprocessing steps:
            - TODO
        """
        if not return_logits:
            mask = (mask > 0).cpu().numpy().astype(np.uint8)

        return mask

    def inv_trans_dense(self, mask: np.ndarray) -> nib.Nifti1Image:
        # Invert transform
        shape_after_dimtranspose = self.orig_shape[::-1]
        # undo crop foreground
        padded_segmenation_shape = np.maximum(self.spatial_size, shape_after_dimtranspose)
        padded_segmentation = np.zeros(
            padded_segmenation_shape, dtype=np.uint8
        )  # Should be created in 32,256,256 shape
        padded_segmentation[
            self.start_coord[0] : self.end_coord[0],
            self.start_coord[1] : self.end_coord[1],
            self.start_coord[2] : self.end_coord[2],
        ] = mask  #  stick into 32,256,256 shape then undo pad # ToDo needs fixing, try MS instances:

        # undo spatial pad
        total_pads = np.maximum(np.array(self.spatial_size) - np.array(shape_after_dimtranspose), 0)
        pad_starts = total_pads // 2

        transposed_segmentation = padded_segmentation[
            pad_starts[0] : shape_after_dimtranspose[0] + pad_starts[0],
            pad_starts[1] : shape_after_dimtranspose[1] + pad_starts[1],
            pad_starts[2] : shape_after_dimtranspose[2] + pad_starts[2],
        ]

        # undo dim_transpose
        segmentation = np.transpose(transposed_segmentation, (2, 1, 0))

        segmentation = nib.Nifti1Image(segmentation, self.orig_affine)

        return segmentation


    @torch.no_grad()
    def binary_zoom_out_predict(self, mapped_inputs:dict):
        # Performing zoom-out inference and mapping back to fg.

        #Mapping the image-scale representation of the prompts into the sparse format for inputting.


        text_zoomout_input, points_zoomout_input, box_zoomout_input = self.map_to_sparse_prompt(mapped_inputs)
        
        logits_global_zoom_out = self.model(
            mapped_inputs['img_zoomout_dom'], text=text_zoomout_input, boxes=box_zoomout_input, points=points_zoomout_input
        )

        # resize back global logits to the fg domain.
        logits_fg = F.interpolate(logits_global_zoom_out.cpu(), size=mapped_inputs['fg_dom_shape'], mode="nearest")[
            0
        ][0]

        return logits_fg #text_zoomout_input, points_zoomout_input, box_zoomout_input, logits_global_zoom_out

    @torch.no_grad()
    def binary_predict(self, request):
        
        #Callbacks which will be what is used to process the input requests for zoomout inference (and to store the original image domain relevant info for pasting back
        # segmentation.

        mapped_inputs = self.binary_subject_prep(request=request)

        #Performing inference on the zoom-out image and mapping back to fg.

        logits_fg = self.binary_zoom_out_predict(mapped_inputs)


        #Extracting the region of interest for zoom-in, also checks if there was any foreground estimated....:
        min_d, min_h, min_w, max_d, max_h, max_w = logits2roi_coor(spatial_size=mapped_inputs['fg_dom_shape'], logits_global_single=logits_fg)

        if min_d is None:
            warnings.warn('Warning, for one reason or another, no foreground was detected and skipping zoom-in. Results may be very poor if the target actually exists....')
        else:
            #Otherwise, there is not much wrong here, just continue as segvol does.. mapping image, prompts, pred to the zoom-in.
        
            # Crop roi for zoom-in from the foreground region cropped roi.
            img_zoomin_dom = mapped_inputs['img_fg_dom'][min_d:max_d+1, min_h:max_h+1, min_w:max_w+1].unsqueeze(0).unsqueeze(0)
            coarse_pred_zoomin_dom = (torch.sigmoid(logits_fg[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1])>0.5).long()

            if self.redo_map_zoomout_to_fg:
            #Using image-representation mapping of prompts in same capacity as segvol from zoom-out domain to foreground domain (even though we already have this....).
            # to be able to build prompt reflection for zoom-in
                
                #We have a generic mask representation in the format used by SegVol for their resizing already.
                prompt_fg_dom = F.interpolate(
                    mapped_inputs['prompt_zoom_dom'].float(),
                    size=mapped_inputs['fg_dom_shape'], mode='nearest')[0][0]
            else:
                prompt_fg_dom = mapped_inputs['prompt_fg_dom']
                #We already have the prompts in the foreground domain representation, removing additional lossy transforms probably will prevent loss of information.

            prompt_reflection = None

            prompt_zoomin_dom = prompt_fg_dom[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1]
            prompt_reflection = (
                prompt_zoomin_dom.unsqueeze(0).unsqueeze(0),
                coarse_pred_zoomin_dom.unsqueeze(0).unsqueeze(0),
            )

            assert img_zoomin_dom.shape[2:] == coarse_pred_zoomin_dom.shape == prompt_zoomin_dom.shape 
            ## inference
            logits_zoomin_dom = sliding_window_inference(
                img_zoomin_dom,
                prompt_reflection,
                self.spatial_size,
                1,
                self.model,
                self.infer_overlap,
                text=None,
                use_box=(mapped_inputs['prompt_type'] == "bboxes"),
                use_point=(mapped_inputs['prompt_type'] == "points"),
            ).cpu().squeeze()

            logits_fg[min_d:max_d + 1, min_h:max_h+1, min_w:max_w+1] = logits_zoomin_dom
            
        
        #Here we will perform the re-insertion back into the original image size. Probably can convert to probabilistic map here because we can't pad with -inf to represent
        #the background.


    # def predict(
    #     self,
    #     prompt: PromptStep | Boxes3D,
    #     text_prompt=None,
    #     return_logits=False,
    #     prev_seg=None,
    #     promptstep_in_model_coord_system=False,
    #     seed=1,
    # ):
    #     if self.loaded_image is None:
    #         raise RuntimeError("Must first set image!")

    #     if not isinstance(prompt, (Boxes3D, PromptStep)):
    #         raise TypeError(
    #             "Prompts must be 3d bounding boxes or points, and must be supplied as an instance of Boxes3D or PromptStep"
    #         )

    #     prompt_type = "box" if isinstance(prompt, Boxes3D) else "point"

    #     if prompt_type == "point":
    #         torch.manual_seed(
    #             seed
    #         )  # New points are sampled in the zoom-in section of zoom-out zoom-in inference leaving some randomness even after a point prompt is fixed.

    #     image_single, image_single_resize = self.img, self.img_zoom_out

    #     prompt = deepcopy(prompt)
    #     if not promptstep_in_model_coord_system:
    #         prompt = self.transform_promptstep_to_model_coords(prompt)
    #     prompt = self.preprocess_prompt(prompt, prompt_type, text_prompt)

    #     res = self.segment(image_single, image_single_resize, prompt, prompt_type)

    #     segmentation = self.postprocess_mask(res[-1], return_logits)

    #     segmentation_model_arr = segmentation

    #     # Turn into Nifti object in original space
    #     segmentation_orig_nib = self.inv_trans_dense(segmentation)

    #     low_res_logits = None  # low_res_logits aren't easily accessed nor used since segvol isn't interactive

    #     return segmentation_orig_nib, low_res_logits, segmentation_model_arr

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
            # assert self.image_embeddings, 'Image embeddings must exist and be stored for editing'
            raise Exception('SegVol, by default, is not configured to be used with iterative refinement approaches.')
        

        elif request['model'] == 'IS_interactive_init':
            key = 'Interactive Init' 
            is_state = request['im'][key]

            # self.image_info = None 
            #Mapping the image and prompts into the model coordinate domain (foreground crop region) 
            
            # NOTE: In order to disentangle the validation framework from inference apps 
            # this is always assumed to be handled within the inference app.
            
            mapped_input = self.binary_prop_to_model(request['image'], is_state)  

        elif request['model'] == 'IS_autoseg':
            key = 'Automatic Init'
            is_state = request['im'][key]
            if is_state is not None:
                raise Exception('Autoseg should not have any interaction info.')
            # self.image_info = None 
            #Extracting the image in the model's coordinate space. NOTE: In order to disentangle the validation framework from inference apps 
            # this is always assumed to be handled within the inference app.
            mapped_input = self.binary_prop_to_model(request['image'], is_state)        

        return mapped_input 

    def __call__(self, request:dict):

        if len(request['config_labels_dict']) == 2:
            class_type = 'binary'
        elif len(request['config_labels_dict']) > 2:
            class_type = 'multi'
            raise NotImplementedError 
        else:
            raise Exception('Should not have received less than two class labels at minimum')
        
        request = copy.deepcopy(request) 

        app = self.infer_apps[request['model']][f'{class_type}_predict']

        self.return_logits=True #False, we change this from the implementation by RadioActive, since we would like to generate a probabilistic map from the api call.

        probs_tensor, pred, affine = app(request=request)

        assert probs_tensor.shape[1:] == request['image']['metatensor'].shape[1:]
        assert pred.shape[1:] == request['image']['metatensor'].shape[1:] 
        assert torch.all(affine == request['image']['metatensor'].meta['affine'])
        assert isinstance(probs_tensor, torch.Tensor) 
        assert isinstance(pred, torch.Tensor) 

        output = {
            'probs':{
                'metatensor':probs_tensor.to(device='cpu'),
                'meta_dict':{'affine': affine.to(device='cpu')}
            },
            'pred':{
                'metatensor':pred.to(device='cpu'),
                'meta_dict':{'affine': affine.to(device='cpu')}
            },
        }
        return output 

if __name__ == '__main__':
   
    infer_app = InferApp(
        {'dataset_name':'BraTS2021',
        'dataset_modality':'MRI'}, torch.device('cuda'))

    infer_app.app_configs()

    from monai.transforms import LoadImaged, Orientationd, EnsureChannelFirstd, Compose  

    input_dict = {'image':'/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_binarised/imagesTs/BraTS2021_00266.nii.gz'}
    load_and_transf = Compose([LoadImaged(keys=['image']), EnsureChannelFirstd(keys=['image']), Orientationd(keys=['image'], axcodes='RAS')])

    final_loaded_im = load_and_transf(input_dict)
    request = {
        'image':{
            'metatensor': torch.from_numpy(final_loaded_im['image']),
            'meta_dict':{'affine':torch.from_numpy(final_loaded_im['image_meta_dict']['affine']).to(dtype=torch.float64)}
        },
        # 'model':'IS_interactive_edit',
        'model': 'IS_interactive_init',
        'config_labels_dict':{'background':0, 'tumor':1},
        'im':
        
        # {'Automatic Init': None}
        {'Interactive Init':{
            'interaction_torch_format': {
                'interactions': {
                    'points': None, #[torch.tensor([[40, 103, 43]]), torch.tensor([[62, 62, 39]])], #None, #[torch.tensor([[40, 103, 43]]), torch.tensor([[62, 62, 39]])], 
                    'scribbles': None, 
                    'bboxes': [torch.Tensor([[56,30,17, 92, 76, 51]]).to(dtype=torch.int64)] #None 
                    },
                'interactions_labels': {
                    'points': None, #[torch.tensor([0]), torch.tensor([1])], #None,#[torch.tensor([0]), torch.tensor([1])], 
                    'scribbles': None, 
                    'bboxes': [torch.Tensor([1]).to(dtype=torch.int64)] #None
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
    infer_app(request)
    print('halt')