
import os
import cv2
import torch
import joblib
import tqdm
import numpy as np
from loguru import logger
from yolov3.yolo import YOLOv3
from multi_person_tracker import MPT
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from . import constants

import pickle
from train.utils.geometry import batch_euler2matrix
from train.utils.train_utils import load_pretrained_model
from train.utils.vibe_image_utils import get_single_image_crop_demo
from collections import OrderedDict
from ..utils.renderer_pyrd import Renderer
from ..models.hmr import HMR
from .config import update_hparams, SMPL_MEAN_PARAMS
from ..utils.renderer_cam import render_image_group
from ..utils.image_utils import transform, crop_ul_br

from ..utils.image_utils import transform
from ..models.head.smplx_head_cam_full import SMPLXHeadCamFull
from ..models.hand import Hand
from ..models.hmrx import HMRX
from kornia.geometry.transform.imgwarp import (
    warp_perspective, get_perspective_transform, warp_affine
)
SCALE_FACTOR_HAND_BBOX=3.0
MIN_NUM_FRAMES = 0

class SMPLXTrainer(pl.LightningModule):
    def __init__(self, hparams, config_tune=None):
        super(SMPLXTrainer, self).__init__()
        self.hparams.update(hparams)

        self.body_model = HMR(
            backbone=self.hparams.MODEL.BACKBONE,
            img_res=self.hparams.DATASET.IMG_RES,
            pretrained_ckpt=self.hparams.TRAINING.PRETRAINED_CKPT,
            hparams=self.hparams,
        )
        self.hand_model = Hand(
            backbone=self.hparams.MODEL.BACKBONE,
            img_res=self.hparams.DATASET.IMG_RES,
            pretrained_ckpt=self.hparams.TRAINING.PRETRAINED_CKPT,
            hparams=self.hparams,
        )
        self.fullbody_model = HMRX(backbone=self.hparams.MODEL.BACKBONE,
                                   img_res=self.hparams.DATASET.IMG_RES,
                                   pretrained_ckpt=self.hparams.TRAINING.PRETRAINED_CKPT,
                                   hparams=self.hparams)
                    
def get_bbox_valid(joints, img_height, img_width, rescale):
    #Get bbox using keypoints

    img_width = img_width.unsqueeze(-1).unsqueeze(-1)
    img_height = img_height.unsqueeze(-1).unsqueeze(-1)
    # valid_mask= ((joints[:,:,[0]]>=0) & (joints[:,:,[1]]>=0) & (joints[:,:,[0]]<=img_width) & (joints[:,:,[1]]<img_height))
    # joints_for_min = joints.masked_fill(valid_mask==0,float('inf'))
    # joints_for_max = joints.masked_fill(valid_mask==0,-float('inf'))

    min_coords, _ = torch.min(joints, dim=1)
    xmin, ymin = min_coords[:, 0], min_coords[:, 1]
    max_coords, _ = torch.max(joints, dim=1)
    xmax, ymax = max_coords[:, 0], max_coords[:, 1]
    center = torch.stack([xmax + xmin, ymax + ymin], dim=-1) * 0.5
    width = (xmax - xmin)
    height = (ymax - ymin)
    # Convert the bounding box to a square box
    scale = torch.max(width, height).unsqueeze(-1)
    scale *= rescale
    return center, scale


import scipy.signal as signal

# from https://github.com/open-mmlab/mmhuman3d/tree/main
class SGFilter:
    """savgol_filter lib is from:
    https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.signal.savgol_filter.html.

    Args:
        window_size (float):
                    The length of the filter window
                    (i.e., the number of coefficients).
                    window_length must be a positive odd integer.
        polyorder (int):
                    The order of the polynomial used to fit the samples.
                    polyorder must be less than window_length.

    Returns:
        smoothed poses (np.ndarray, torch.tensor)
    """

    def __init__(self, window_size=11, polyorder=2):
        super(SGFilter, self).__init__()

        # 1-D Savitzky-Golay filter
        self.window_size = window_size
        self.polyorder = polyorder

    def __call__(self, x=None):
        # x.shape: [t,k,c]
        if self.window_size % 2 == 0:
            window_size = self.window_size - 1
        else:
            window_size = self.window_size
        if window_size > x.shape[0]:
            window_size = x.shape[0]
        if window_size <= self.polyorder:
            polyorder = window_size - 1
        else:
            polyorder = self.polyorder
        assert polyorder > 0
        assert window_size > polyorder
        if len(x.shape) != 3:
            warnings.warn('x should be a tensor or numpy of [T*M,K,C]')
        assert len(x.shape) == 3
        x_type = x
        if isinstance(x, torch.Tensor):
            if x.is_cuda:
                x = x.cpu().numpy()
            else:
                x = x.numpy()
        smooth_poses = np.zeros_like(x)
        # smooth at different axis
        C = x.shape[-1]
        for i in range(C):
            smooth_poses[..., i] = signal.savgol_filter(
                x[..., i], window_size, polyorder, axis=0)

        if isinstance(x_type, torch.Tensor):
            # we also return tensor by default
            if x_type.is_cuda:
                smooth_poses = torch.from_numpy(smooth_poses).cuda()
            else:
                smooth_poses = torch.from_numpy(smooth_poses)
        return smooth_poses


def smooth_process(x):

    smooth_func = SGFilter()

    if x.ndim == 4:
        for i in range(x.shape[1]):
            x[:, i] = smooth_func(x[:, i])
    elif x.ndim == 3:
        x = smooth_func(x)

    return x


from scipy.interpolate import interp1d
import numpy as np
import copy

def perform_motion_interpolation(smpl_joints, pred_vert_arr, detection_all):

    print("Do motion interpolation.")

    smpl_joints_fill = np.copy(smpl_joints)
    smpl_vertices_fill = np.copy(pred_vert_arr)
    detection_all_fill = np.copy(detection_all)

    person_count = len(set(detection_all[:, -1]))

    for person in range(person_count):
        choose_frame, choose_index, choose_joints, choose_vertices = [], [], [], []

        for i in range(len(detection_all)):
            frame_id, person_id = detection_all[i][0], detection_all[i][-1]
            if person_id == person:
                choose_frame.append(int(frame_id))
                choose_index.append(len(choose_index))
                choose_joints.append(smpl_joints[i])
                choose_vertices.append(pred_vert_arr[i])

        if len(choose_frame) < 3:
            continue

        existed_list = copy.copy(choose_frame)
        interval = 10
        choose_frame = choose_frame[0::interval]
        choose_index = choose_index[0::interval]

        choose_joints = np.stack(choose_joints, axis=0)  # (N, J_NUM, 3)
        choose_vertices = np.stack(choose_vertices, axis=0)  # (N, V_NUM, 3)

        choose_joints = interp1d(
            choose_frame,
            choose_joints[np.array(choose_index), :, :].transpose(1, 2, 0),
            kind='linear'
        )(range(int(min(choose_frame)), int(max(choose_frame)))).transpose(2, 0, 1)

        choose_vertices = interp1d(
            choose_frame,
            choose_vertices[np.array(choose_index), :, :].transpose(1, 2, 0),
            kind='linear'
        )(range(int(min(choose_frame)), int(max(choose_frame)))).transpose(2, 0, 1)

        print(f"Do motion smooth on person {person}.")
        choose_joints = smooth_process(
            choose_joints
        )

        infill_frame_ids = [
            frame_id
            for frame_id in range(int(min(choose_frame)), int(max(choose_frame)))
            if frame_id not in existed_list
        ]

        print(f"Infill {len(infill_frame_ids)} frames for person {person}")
        infill_seq = list(range(int(min(choose_frame)), int(max(choose_frame))))

        for infill_frame_id in infill_frame_ids:
            smpl_joints_fill_item = choose_joints[infill_seq.index(infill_frame_id)][np.newaxis, :]
            smpl_vertices_fill_item = choose_vertices[infill_seq.index(infill_frame_id)][np.newaxis, :]
            detection_all_fill_item = np.array([infill_frame_id, 0, 0, 0, 0, 0, 0, person])[np.newaxis, :]

            smpl_joints_fill = np.append(smpl_joints_fill, smpl_joints_fill_item, axis=0)
            smpl_vertices_fill = np.append(smpl_vertices_fill, smpl_vertices_fill_item, axis=0)
            detection_all_fill = np.append(detection_all_fill, detection_all_fill_item, axis=0)

    return smpl_joints_fill, smpl_vertices_fill, detection_all_fill


def crop_tensor(image, center, bbox_size, crop_size, interpolation = 'bilinear', align_corners=False):

    dtype = image.dtype
    device = image.device
    batch_size = image.shape[0]
    # points: top-left, top-right, bottom-right, bottom-left    
    src_pts = torch.zeros([4,2], dtype=dtype, device=device).unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    src_pts[:, 0, :] = center - bbox_size*0.5  # / (self.crop_size - 1)
    src_pts[:, 1, 0] = center[:, 0] + bbox_size[:, 0] * 0.5
    src_pts[:, 1, 1] = center[:, 1] - bbox_size[:, 0] * 0.5
    src_pts[:, 2, :] = center + bbox_size * 0.5
    src_pts[:, 3, 0] = center[:, 0] - bbox_size[:, 0] * 0.5
    src_pts[:, 3, 1] = center[:, 1] + bbox_size[:, 0] * 0.5

    DST_PTS = torch.tensor([[
        [0, 0],
        [crop_size - 1, 0],
        [crop_size - 1, crop_size - 1],
        [0, crop_size - 1],
    ]], dtype=dtype, device=device).expand(batch_size, -1, -1)
    # estimate transformation between points
    dst_trans_src = get_perspective_transform(src_pts, DST_PTS)
    # simulate broadcasting
    # dst_trans_src = dst_trans_src.expand(batch_size, -1, -1)

    # warp images 
    cropped_image = warp_affine(
        image, dst_trans_src[:, :2, :], (crop_size, crop_size),
        mode=interpolation, align_corners=align_corners)

    tform = torch.transpose(dst_trans_src, 2, 1)
    return cropped_image, tform

def load_valid(model, pretrained_file, skip_list=None):

    pretrained_dict = torch.load(pretrained_file)['state_dict']
    pretrained_dict = strip_prefix_if_present(pretrained_dict, prefix='model')
    pretrained_dict = strip_prefix_if_present(pretrained_dict, prefix='fullbody_model')
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict1 = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict1)
    model.load_state_dict(model_dict)

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not any(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix+'.', '')] = value
    return stripped_state_dict


def j2d_processing(kp, center, scale, img_res):
    """Process gt 2D keypoints and apply all augmentation transforms."""
    nparts = kp.shape[0]
    for i in range(nparts):
        kp[i,0:2] = transform(kp[i,0:2] + 1, center, scale,
                                [img_res, img_res])
    # convert to normalized coordinates
    kp[:,:-1] = 2. * kp[:,:-1] / img_res - 1.
    # flip the x coordinates

    kp = kp.astype('float32')
    return kp

class Tester:
    def __init__(self, args):
        self.args = args
        self.model_cfg = update_hparams(args.cfg)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

        self._build_model()
        self._load_pretrained_model()
        self.model.eval()

        self.smpl_cam_head = SMPLXHeadCamFull(img_res=self.model_cfg.DATASET.IMG_RES).to(self.device)

        self.flip_vector = torch.ones((1, 9), dtype=torch.float32)
        self.flip_vector[:, [1, 2, 3, 6]] *= -1
        self.flip_vector = self.flip_vector.reshape(1, 3, 3).cuda()
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)


    def _build_model(self):
        # ========= Define SPEC model ========= #
        self.hparams = self.model_cfg
        self.model = SMPLXTrainer(hparams=self.hparams).to(self.device)

    def _load_pretrained_model(self):
        # ========= Load pretrained weights ========= #
        logger.info(f'Loading pretrained model from {self.args.ckpt}')
        ckpt = torch.load(self.args.ckpt)['state_dict']
        load_pretrained_model(self.model, ckpt, overwrite_shape_mismatch=True, remove_lightning=True)
        logger.info(f'Loaded pretrained weights from \"{self.args.ckpt}\"')

    def run_detector(self, all_image_folder):
        # run multi object tracker
        mot = MPT(
            device=self.device,
            batch_size=self.args.tracker_batch_size,
            display=False,
            detector_type=self.args.detector,
            output_format='dict',
            yolo_img_size=self.args.yolo_img_size,
        )
        bboxes = []
        for fold_id, image_folder in enumerate(all_image_folder):
            bboxes.append(mot.detect(image_folder))

        # Save bbox files 
        # for fold_id, image_folder in enumerate(all_image_folder):
        #     all_bbox = mot.detect(image_folder)
        #     bboxes.append(all_bbox)
        #     image_file_names = [
        #         os.path.join(image_folder, x)
        #         for x in os.listdir(image_folder)
        #         if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
        #     ]
        #     image_file_names = (sorted(image_file_names))
        #     out_folder = os.path.join(image_folder, 'bbox')
        #     os.makedirs(out_folder, exist_ok=True)
        #     for img_idx, img_fname in (enumerate(image_file_names)):
        #         out_filename = os.path.join(out_folder, os.path.basename(img_fname)+'.txt')
        #         np.savetxt(out_filename, all_bbox[img_idx])

        return bboxes

    @torch.no_grad()
    def run_on_image_folder(self, all_image_folder, detections, output_folder, visualize_proj=False, save_result=False, eval_dataset='', from_one_video=False):

        for fold_idx, image_folder in enumerate(all_image_folder):
            image_file_names = [
                os.path.join(image_folder, x)
                for x in os.listdir(image_folder)
                if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
            ]
            image_file_names = (sorted(image_file_names))

            if from_one_video:
                smpl_joints_fill = []
                pred_vert_arr = []
                detection_all = []
                pose_arr = []
                transpose_arr = []
                shape_arr = []

            for img_idx, img_fname in tqdm.tqdm(enumerate(image_file_names)):
                dets = detections[fold_idx][img_idx]
                if len(dets) < 1:
                    continue
                # Load saved bbox files
                # dets = np.loadtxt(os.path.join(image_folder, 'bbox',os.path.basename(img_fname)+'.txt'))
                # if len(dets) < 1:
                #     continue

                if len(dets.shape)==1:
                    dets = np.expand_dims(dets, 0)
                img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
                rgb_img_full = np.transpose(img.astype('float32'),(2,0,1))/255.0

                orig_height, orig_width = img.shape[:2]

                inp_images = torch.zeros(len(dets), 3, self.model_cfg.DATASET.IMG_RES,
                                        self.model_cfg.DATASET.IMG_RES, device=self.device, dtype=torch.float)

                batch_size = inp_images.shape[0]

                bbox_scale = []
                bbox_center = []
 
                for det_idx, det in enumerate(dets):
                    bbox = det
                    bbox_scale.append(bbox[2] / 200.)
                    bbox_center.append([bbox[0], bbox[1]])
                    rgb_img, ul, br = crop_ul_br(img, bbox_center[-1], bbox_scale[-1],[self.model_cfg.DATASET.IMG_RES, self.model_cfg.DATASET.IMG_RES])
                    rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
                    rgb_img = torch.from_numpy(rgb_img)
                    norm_img = self.normalize_img(rgb_img)
                    inp_images[det_idx] = norm_img.float().to(self.device)

                    x1, y1, x2, y2 = bbox

                    detection_all.append([img_idx, x1, y1, x2, y2, 0, 0.99, det_idx])

                bbox_center = torch.tensor(bbox_center).cuda().float()
                bbox_scale = torch.tensor(bbox_scale).cuda().float()
                img_h = torch.tensor(orig_height).repeat(batch_size).cuda().float()
                img_w = torch.tensor(orig_width).repeat(batch_size).cuda().float()
                focal_length = ((img_w * img_w + img_h * img_h) ** 0.5).cuda().float()

                body_pred = self.model.body_model(inp_images, bbox_center=bbox_center, bbox_scale=bbox_scale, img_w=img_w, img_h=img_h)

                lhand_joints = body_pred['joints2d'][:, 25:40]
                rhand_joints = body_pred['joints2d'][:, 40:55]

                center_r, scale_r = get_bbox_valid(rhand_joints, img_h, img_w, SCALE_FACTOR_HAND_BBOX)

                rgb_img_full = torch.tensor(rgb_img_full).cuda().float()
                rgb_img_full = rgb_img_full.unsqueeze(0).expand(inp_images.shape[0],-1,-1,-1)
                right_hand_crop, _ = crop_tensor(rgb_img_full, center_r, scale_r, 224)
                right_hand_crop = self.normalize_img(right_hand_crop)
                right_hand_pred = self.model.hand_model(right_hand_crop)
                #Flip left hand image before feedint to hand network
                center_l, scale_l = get_bbox_valid(lhand_joints, img_h, img_w, SCALE_FACTOR_HAND_BBOX)
                left_hand_crop, _ = crop_tensor(rgb_img_full, center_l, scale_l, 224)
                left_hand_crop = self.normalize_img(left_hand_crop)
                left_hand_crop = torch.flip(left_hand_crop, [3])
                left_hand_pred = self.model.hand_model(left_hand_crop)
                #Flip predicted right hand pose to left hand 
                left_hand_pred['pred_pose'] = left_hand_pred['pred_pose'] * self.flip_vector.unsqueeze(0)

                full_body_pred = self.model.fullbody_model(body_pred['body_feat'], left_hand_pred['hand_feat'], right_hand_pred['hand_feat'], body_pred['pred_pose'], body_pred['pred_shape'], body_pred['pred_cam'],left_hand_pred['pred_pose'], right_hand_pred['pred_pose'],bbox_center, bbox_scale, img_w, img_h)
                
                cam_intrinsics = torch.eye(3).repeat(batch_size, 1, 1).cuda().float()
                cam_intrinsics[:, 0, 0]  = focal_length
                cam_intrinsics[:, 1, 1]  = focal_length
                cam_intrinsics[:, 0, 2] = img_w/2.
                cam_intrinsics[:, 1, 2] = img_h/2.

                output = self.smpl_cam_head(body_pose=full_body_pred['pred_pose'], lhand_pose = left_hand_pred['pred_pose'][:,1:], rhand_pose=right_hand_pred['pred_pose'][:,1:],
                            shape=body_pred['pred_shape'], cam=body_pred['pred_cam'], cam_intrinsics=cam_intrinsics, 
                            bbox_scale=bbox_scale,bbox_center=bbox_center, img_w=img_w, img_h=img_h, normalize_joints2d=False)

                del inp_images

                if from_one_video:
                    smpl_joints_fill.extend(output['joints3d'].detach().cpu().numpy())
                    vert = output['vertices'].detach().cpu().numpy()
                    trans = output['pred_cam_t'].detach().cpu().numpy()
                    vert = vert + np.expand_dims(trans, 1)

                    pose = full_body_pred['pred_pose'].detach().cpu().numpy()
                    shape = body_pred['pred_shape'].detach().cpu().numpy()

                    pose_arr.extend(pose)
                    transpose_arr.extend(trans)
                    shape_arr.extend(shape)
                    pred_vert_arr.extend(vert)


                if save_result and not from_one_video:
                    for out_ind, vertices in enumerate(output['vertices']):
                        out_dict = {}
                        out_dict['verts'] = output['vertices'][out_ind].detach().cpu().numpy()
                        out_dict['joints'] = output['joints2d'][out_ind][:24].detach().cpu().numpy()
                        out_dict['allSmplJoints3d'] = output['joints3d'][out_ind].detach().cpu().numpy()
                        if eval_dataset == 'agora':
                            imgname = os.path.basename(img_fname).replace('.png', '')
                            pickle.dump(out_dict, open(os.path.join(output_folder, imgname + '_personId_' + str(out_ind) + '.pkl'), 'wb'))
                        elif eval_dataset == 'bedlam':
                            imgname = img_fname.split('/')[-4] + '_frameID_' + img_fname.split('/')[-1]
                            imgname = imgname.replace('.png','')
                            pickle.dump(out_dict, open(os.path.join(output_folder, imgname + '_personId_' + str(out_ind) + '.pkl'), 'wb'))
                        else:
                            raise Exception('eval dataset can be either agora or bedlam')
                        

                if visualize_proj and not from_one_video:
                    img_h, img_w, _ = img.shape
                    focal_length = (img_w * img_w + img_h * img_h) ** 0.5

                    vertices = output['vertices'].detach().cpu().numpy()
                    translation = output['pred_cam_t'].detach().cpu().numpy()

                    pred_vertices_array = vertices + np.expand_dims(translation, 1)
                    renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                                        faces=self.smpl_cam_head.smplx.faces,
                                        same_mesh_color=False)
                    front_view = renderer.render_front_view(pred_vertices_array,
                                                            bg_img_rgb=img.copy())

                    # save rendering results
                    basename = img_fname.split('/')[-1]
                    filename = basename + "pred_%s.jpg" % 'bedlam'
                    filename_orig = basename + "orig_%s.jpg" % 'bedlam'
                    front_view_path = os.path.join(output_folder, filename)
                    orig_path = os.path.join(output_folder, filename_orig)
                    logger.info(f'Writing output files to {output_folder}')
                    cv2.imwrite(front_view_path, front_view[:, :, ::-1])
                    # cv2.imwrite(orig_path, img[:, :, ::-1])

                    renderer.delete()

            if from_one_video:
                pred_vert_arr = np.array(pred_vert_arr)
                smpl_joints_fill = np.array(smpl_joints_fill)
                detection_all = np.array(detection_all)
                pose_arr = np.array(pose_arr)
                transpose_arr = np.array(transpose_arr)
                shape_arr = np.array(shape_arr)

                smpl_joints, smpl_vertices, detection_all = perform_motion_interpolation(smpl_joints_fill, pred_vert_arr, detection_all)

                if save_result:
                    # Same format as the original Cliff paper demo
                    np.savez(os.path.join(output_folder, 'predicted_joints_bedlam.npy'),
                             pose=pose_arr,
                             shape=shape_arr,
                             global_t=transpose_arr,
                             pred_joints=smpl_joints,
                             detection_all=detection_all)

                if visualize_proj:

                    frames_id = detection_all[:, 0]

                    front_view_path = os.path.join(output_folder, 'result_video_bedlam.mp4')
                    frame_rate = 30  # Frames per second

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
                    img_h, img_w, _ = img.shape
                    video_writer = cv2.VideoWriter(front_view_path, fourcc, frame_rate, (img_w, img_h))

                    for img_idx, img_fname in tqdm.tqdm(enumerate(image_file_names)):

                        frames_id = detection_all[:, 0]

                        pred_vertices_array = smpl_vertices[frames_id == img_idx]

                        img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
                        img_h, img_w, _ = img.shape
                        focal_length = (img_w * img_w + img_h * img_h) ** 0.5

                        renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                                            faces=self.smpl_cam_head.smplx.faces,
                                            same_mesh_color=False)

                        front_view = renderer.render_front_view(pred_vertices_array,
                                                                bg_img_rgb=img.copy())
                        renderer.delete()
                        video_writer.write(front_view[:, :, ::-1])
                    video_writer.release()

