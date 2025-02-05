"""
=======================================================================
General Information
-------------------
This is a GPU-based python implementation of the following paper:
Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images
Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim
Proc. IEEE Computer Vision and Pattern Recognition (CVPR 2021, Oral)
Visit our project http://vclab.kaist.ac.kr/cvpr2021p1/ for more details.

Please cite this paper if you use this code in an academic publication.
Bibtex: 
@InProceedings{Meuleman_2021_CVPR,
    author = {Andreas Meuleman and Hyeonjoong Jang and Daniel S. Jeon and Min H. Kim},
    title = {Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images},
    booktitle = {CVPR},
    month = {June},
    year = {2021}
}
==========================================================================
License Information
-------------------
CC BY-NC-SA 3.0
Andreas Meuleman and Min H. Kim have developed this software and related documentation (the "Software"); confidential use in source form of the Software, without modification, is permitted provided that the following conditions are met:
Neither the name of the copyright holder nor the names of any contributors may be used to endorse or promote products derived from the Software without specific prior written permission.
The use of the software is for Non-Commercial Purposes only. As used in this Agreement, “Non-Commercial Purpose” means for the purpose of education or research in a non-commercial organisation only. “Non-Commercial Purpose” excludes, without limitation, any use of the Software for, as part of, or in any way in connection with a product (including software) or service which is sold, offered for sale, licensed, leased, published, loaned or rented. If you require a license for a use excluded by this agreement, please email [minhkim@kaist.ac.kr].
Warranty: KAIST-VCLAB MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. KAIST-VCLAB SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.
Please refer to license.txt for more details.
=======================================================================
"""
import logging
import math
import os.path
# import sys
import warnings
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
from log_utils import setupLogger
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

logger = setupLogger(__name__, logging.DEBUG)

class CamModel(ABC):
    def __init__(self, model: str, original_resolution: torch.Tensor, rt: torch.Tensor, matching_scale: torch.Tensor, device: torch.device | str ='cpu'):
        # self.id: int = -1
        self.model: str = model
        self.device: torch.device = torch.device(device)
        self.original_resolution: torch.Tensor = original_resolution.to(device)
        self.rt: torch.Tensor = rt.to(device)
        self.matching_scale: torch.Tensor = matching_scale.to(device)

    @abstractmethod
    def unproject(self, uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def project(self, point: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def vectorize_calibration(self) -> torch.Tensor:
        pass

class DoubleSphereModel(CamModel):
    def __init__(self, original_resolution: torch.Tensor, principal: torch.Tensor, fl: torch.Tensor, xi: float, alpha: float, rt: torch.Tensor, matching_scale: torch.Tensor, device: torch.device | str):
        """
        Args:
            original_resolution, principal, fl, xi, alpha: Double sphere intrinsics
            rt: [4, 4] camera pose
            matching_scale: [2] Scale to apply to the resolution, principal and fl 
                when working with images resized to the matching resolution
        """
        super().__init__('double_sphere', original_resolution, rt, matching_scale, device)
        self.principal: torch.Tensor = principal
        self.fl: torch.Tensor = fl
        self.xi: float = xi
        self.alpha: float = alpha
        logger.debug(f'Init cam model {self.model}:\n-device: {self.device}\n-original_resolution: {self.original_resolution}\n-rt: {self.rt}\n-matching_scale: {self.matching_scale}\n-principal: {self.principal}\n-fl: {self.fl}\n-xi: {self.xi}\n-alpha: {self.alpha}')


    def unproject(self, uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Unproject pixels to the unit sphere following the The Double Sphere Camera Model (https://arxiv.org/abs/1807.08957)
        Apply the self.matching_scale to fit the distance estimation resolution
        """
        m_xy = (uv - self.principal * self.matching_scale) / (self.fl * self.matching_scale)

        r2 = torch.sum(m_xy**2, dim=-1, keepdim=True)
        m_z = ((1 - self.alpha**2 * r2) 
               / (self.alpha * torch.sqrt(torch.clamp(1 - (2 * self.alpha - 1) * r2, min=0)) + 1 - self.alpha))

        point = torch.cat([m_xy, m_z], dim=-1)
        point = ((m_z * self.xi + torch.sqrt(m_z**2 + (1 - self.xi**2) * r2)) / (m_z**2 + r2)) * point
        point[..., 2] -= self.xi

        valid = (1 - (2 * self.alpha - 1) * r2 >= 0)
        return point, valid[..., 0]
    
    def project(self, point: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project a point in space to pixel coordinates (https://arxiv.org/abs/1807.08957)
        Apply the self.matching_scale to fit the distance estimation resolution
        """
        d1 = torch.norm(point, dim=-1, keepdim=True)

        c = self.xi * d1 + point[..., 2:3]
        d2 = torch.norm(torch.cat([point[..., :2], c], dim=-1), dim=-1, keepdim=True)
        norm = self.alpha * d2 + (1 - self.alpha) * c
        
        w1 = (1 - self.alpha) / self.alpha if self.alpha > 0.5 else self.alpha / (1 - self.alpha)
        w2 = (w1 + self.xi) / math.sqrt(2 * w1 * self.xi + self.xi**2 + 1)

        valid = point[..., 2:3] > - w2 * d1
        uv = (self.fl * self.matching_scale * point[..., :2]) / norm + self.principal * self.matching_scale
        return uv, valid[..., 0]

    def vectorize_calibration(self):
        """
        Convert the intrinsics into a continuous float vector that follows the Intrinsics' structure
        (See stitcher.cu for Intrinsics' definition)
        Scale the focal length and the principal point using the matching scale.
        """
        calibration_vector = torch.zeros([6], device=self.device)
        calibration_vector[0:2] = self.fl * self.matching_scale
        calibration_vector[2:4] = self.principal * self.matching_scale
        calibration_vector[4] = self.xi
        calibration_vector[5] = self.alpha
        return calibration_vector


class CVFisheyeModel(CamModel):
    def __init__(self, original_resolution: torch.Tensor, principal: torch.Tensor, fl: torch.Tensor, dist_params: torch.Tensor, rt: torch.Tensor, matching_scale: torch.Tensor, device: torch.device | str, max_theta: float = np.pi, unproj_crit: tuple[int, int, float] = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1e-8)):
        
        super().__init__('cv_fisheye', original_resolution, rt, matching_scale, device)
        self.principal = principal
        self.fl = fl
        self.dist_params = dist_params
        self.unproj_crit = unproj_crit
        self.max_theta = max_theta
        logger.debug(f'Init cam model {self.model}:\n-device: {self.device}\n-original_resolution: {self.original_resolution}\n-rt: {self.rt}\n-matching_scale: {self.matching_scale}\n-principal: {self.principal}\n-fl: {self.fl}\n-dist_params: {self.dist_params}\n-max_theta: {self.max_theta}\n-unproj_crit: {self.unproj_crit}')

    def unproject(self, uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m_xy = (uv - self.principal * self.matching_scale) / self.fl * self.matching_scale

        theta_d = torch.sqrt(torch.sum(m_xy**2, dim=-1, keepdim=True))
        theta_d = torch.min(torch.max(torch.tensor(-torch.pi * 0.5), theta_d), torch.tensor(torch.pi *0.5))
        theta = torch.clone(theta_d)
        scale = 0.0

        for _ in range(self.unproj_crit[1]):
            theta2 = theta.mul(theta)
            theta4 = theta2.mul(theta2)
            theta6 = theta4.mul(theta2)
            theta8 = theta6.mul(theta2)
            k0_theta2 = self.dist_params[0] * theta2
            k1_theta4 = self.dist_params[1] * theta4
            k2_theta6 = self.dist_params[2] * theta6
            k3_theta8 = self.dist_params[3] * theta8
            theta_fix = (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d) / (1.0 + 3*k0_theta2 + 5*k1_theta4 + 7*k2_theta6 + 9*k3_theta8)
            theta = theta - theta_fix

            if (torch.all(torch.abs(theta_fix) < self.unproj_crit[2])):
                break

        scale = torch.tan(theta) / theta_d

        theta_flipped = ((theta_d < 0) & (theta > 0)) | ((theta_d > 0) & (theta < 0))

        m_xy = m_xy * scale
        points = torch.cat([m_xy, torch.ones_like(m_xy[..., 0]).unsqueeze(-1)], dim=-1)
        points = points / torch.sqrt(torch.sum(points**2, dim=-1, keepdim=True))

        theta_sphere = torch.acos(points[..., 2]).unsqueeze(-1)
        
        points = torch.where(theta_flipped | (theta_sphere > self.max_theta), torch.nan, points)
        valid = torch.where(points == torch.nan, False, True)
        return points, valid[..., 0]
    
    def k(self):
        fl = (self.fl * self.matching_scale).cpu().numpy()
        principal = (self.principal * self.matching_scale).cpu().numpy()
        return np.array([[fl[0], 0, principal[0]], [0, fl[1], principal[1]], [0., 0., 1.]]).astype(np.float32)

    def d(self):
        dist_params = self.dist_params.cpu().numpy()
        return dist_params.astype(np.float32)



    def project(self, point: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        uv = point[..., :2] / point[..., 2].unsqueeze(-1)
        r = torch.norm(point, dim=-1, keepdim=True)
        theta = torch.atan(r)
        theta2 = theta.mul(theta)
        theta3 = theta2.mul(theta)
        theta4 = theta2.mul(theta2)
        theta5 = theta4.mul(theta)
        theta6 = theta3.mul(theta3)
        theta7 = theta6.mul(theta)
        theta8 = theta4.mul(theta4)
        theta9 = theta8.mul(theta)

        theta_d = theta + self.dist_params[0] * theta3 + self.dist_params[1] * theta5 + self.dist_params[2] * theta7 + self.dist_params[3] * theta9

        cdist = torch.where(r > 1e-8, theta_d * 1.0/r, 1.0)

        uv = (uv * cdist) * self.fl * self.matching_scale + self.principal * self.matching_scale
        valid = torch.where(theta > self.max_theta, False, True)

        return uv, valid[..., 0]
        
        #points = point.detach().squeeze().cpu().numpy()
        #norm = np.sqrt(points[..., 0]**2 + points[..., 1]**2) + sys.float_info.epsilon
        #theta = np.arctan2(-points[..., 2], norm)
        #theta = theta + np.pi / 2
        ##convert points to (n, 1, 3)
        #points = points.reshape(-1, 1, 3)
        #distorted_points, _ = cv2.fisheye.projectPoints(points, np.zeros((3, 1)), np.zeros((3, 1)), self.k(), self.d())
        #new_shape = list(point.shape[:-1])
        #new_shape.append(2)
        #distorted_points = distorted_points.reshape(new_shape)
        ## distorted_points[:,theta.squeeze() > self.max_theta] = -1.0
        #return torch.tensor(distorted_points, device=point.device), torch.tensor(theta.squeeze() <= self.max_theta, device=point.device).unsqueeze(0)

    def vectorize_calibration(self):
        """
        Convert the intrinsics into a continuous float vector that follows the Intrinsics' structure
        (See stitcher.cu for Intrinsics' definition)
        Scale the focal length and the principal point using the matching scale.
        """
        calibration_vector = torch.zeros([9], device=self.device)
        calibration_vector[0:2] = self.fl * self.matching_scale
        calibration_vector[2:4] = self.principal * self.matching_scale
        calibration_vector[4:8] = self.dist_params
        calibration_vector[8] = self.max_theta
        return calibration_vector


def parse_json_calib(raw_calibration, matching_resolution, device)->list[CamModel]:
    """
    Parse basalt-formated calibration file (https://gitlab.com/VladyslavUsenko/basalt/-/blob/master/doc/Calibration.md)
    Args:
        matching_resolution: [2] Resolution at which the fisheye images will be resize for distance estimation.
            It is used to obtain the matching_scale component of the calibration.
    """
    cam_models = []
    for extrinsics, intrinsics, original_resolution \
            in zip(raw_calibration['T_imu_cam'], raw_calibration['intrinsics'], raw_calibration['resolution'], strict=True):
        
        if(intrinsics["camera_type"] != "ds"):
            raise Exception("Unexpected camera model. The current implementation only support double sphere.")

        cam_intrinsics = intrinsics['intrinsics']

        r = R.from_quat([
            extrinsics['qx'],
            extrinsics['qy'],
            extrinsics['qz'],
            extrinsics['qw']
        ])

        t = torch.tensor([
            extrinsics['px'],
            extrinsics['py'],
            extrinsics['pz']
        ], device=device)

        rt = torch.eye(4, device=device)
        rt[:3, :3] = torch.tensor(r.as_matrix(), device=device)
        rt[:3, 3] = t

        cam_models.append(DoubleSphereModel(
            torch.tensor(original_resolution),
            torch.tensor([cam_intrinsics['cx'], cam_intrinsics['cy']], device=device),
            torch.tensor([cam_intrinsics['fx'], cam_intrinsics['fy']], device=device),
            cam_intrinsics['xi'],
            cam_intrinsics['alpha'],
            rt,
            torch.tensor([
                    matching_resolution[0] / original_resolution[0],
                    matching_resolution[1] / original_resolution[1]
                ], device=device),
            device
        ))

    return cam_models


def parse_json_calib_cv(file_path, matching_resolution, device, max_theta=2.0*np.pi) -> list[CamModel]:
    fs_config = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)

    num_cams = int(fs_config.getNode('nb_camera').real())

    cam_group = -1
    models = []
    cam_matrices = []
    cam_distortions = []
    image_sizes = []
    poses = []

    for i in range(num_cams):
        cam_cfg = fs_config.getNode(f'camera_{i}')
        if cam_cfg is None:
            logger.error(
                f'Failed to read camera config for camera {i} from multi-camera calibration config file {file_path}'
            )
            raise ValueError(
                f'Failed to read camera config for camera {i} from multi-camera calibration config file {file_path}'
            )

        # for now only fisheye cameras are supported
        if int(cam_cfg.getNode('distortion_type').real()) != 1:
            logger.error(f'Only fisheye cameras are supported. Camera {i} is not a fisheye camera.')
            raise ValueError(
                f'Only fisheye cameras are supported. Camera {i} is not a fisheye camera.'
            )
        models.append('cvfisheye')
        cam_matrices.append(cam_cfg.getNode('camera_matrix').mat())
        cam_distortions.append(cam_cfg.getNode('distortion_vector').mat())
        image_sizes.append(
            (int(cam_cfg.getNode('img_width').real()), int(cam_cfg.getNode('img_height').real()))
        )

        if i == 0:
            cam_group = int(cam_cfg.getNode('camera_group').real())
        else:
            if cam_group != int(cam_cfg.getNode('camera_group').real()):
                logger.error(
                    'All cameras in the multi-camera calibration config file must belong to the same camera group'
                )
                raise ValueError(
                    'All cameras in the multi-camera calibration config file must belong to the same camera group'
                )

        poses.append(cam_cfg.getNode('camera_pose_matrix').mat())

    # t_center = np.zeros((3, 1), poses[0].dtype)
    # for pose in poses:
    #     t_center += translation(pose)

    # t_center /= num_cams

    # for pose in poses:
    #     pose[:3, 3] += t_center.flatten()

    cam_models = []
    for image_size, cam_matrix, dist_params, rt in zip(
        image_sizes, cam_matrices, cam_distortions, poses, strict=True
    ):
        original_resolution = torch.tensor(image_size)
        cam_models.append(
            CVFisheyeModel(
                original_resolution,
                torch.tensor([cam_matrix[0, 2], cam_matrix[1, 2]], device=device, dtype=torch.float32),
                torch.tensor([cam_matrix[0, 0], cam_matrix[1, 1]], device=device, dtype=torch.float32),
                torch.tensor(dist_params.flatten(), device=device, dtype=torch.float32),
                torch.tensor(rt, device=device, dtype=torch.float32),
                torch.tensor(
                    [
                        matching_resolution[0] / original_resolution[0],
                        matching_resolution[1] / original_resolution[1],
                    ],
                    device=device,
                    dtype=torch.float32,
                ),
                device,
                max_theta,
            )
        )

    return cam_models


def rgb2yCbCr(rgb):
    rgb = rgb.float()
    yuv = torch.zeros_like(rgb)

    yuv[:, :, 0] = torch.clamp(16  + 0.1826 * rgb[:, :, 0] + 0.6142 * rgb[:, :, 1] + 0.062  * rgb[:, :, 2]
                               , min=16, max=235)
    yuv[:, :, 1] = torch.clamp(128 - 0.1006 * rgb[:, :, 0] - 0.3386 * rgb[:, :, 1] + 0.4392 * rgb[:, :, 2] 
                               , min=16, max=240)
    yuv[:, :, 2] = torch.clamp(128 + 0.4392 * rgb[:, :, 0] - 0.3989 * rgb[:, :, 1] - 0.0403 * rgb[:, :, 2]
                               , min=16, max=240)

    return yuv

def read_input_images(filename, dataset_path, matching_resolution, rgb_to_stitch_resolution, 
                      cam_models, references_indices):
    """
    Read and resize fisheye images 
    """
    images_to_match = []
    images_to_stitch = []
    valid_frame = True
    # Read input image for each camera
    for cam_index, cam_model in enumerate(cam_models):
        file_path = os.path.join(dataset_path, "cam" + str(cam_index)) + "/" + filename
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        # Type and innapropriate file handling
        if image is not None:
            if image.shape == (cam_model.original_resolution[1], cam_model.original_resolution[0], 3):
                # Map all types range to [0, 255] as float32
                if image.dtype == np.uint8:
                    image = image.astype(np.float32)
                elif image.dtype == np.uint16:
                    image = image.astype(np.float32) / 255
                elif image.dtype == np.float32:
                    if np.max(image) > 1:
                        image = np.clip(image, 0, 1)
                        warnings.warn("Image has out-of-range float values for file " 
                                      + file_path + ". Clipped for processing.")
                    image = image * 255
                else:
                    warnings.warn("Invalide image type for file " + file_path)
                    valid_frame = False
            else:
                warnings.warn("Invalid image size / channels for file " + file_path)
                valid_frame = False

        else:
            warnings.warn("Cannot read image for file " + file_path)
            valid_frame = False
        
        if valid_frame:
            # Keep references at higher resolution for stitching
            if cam_index in references_indices:
                image_to_stitch = cv2.resize(image, tuple(rgb_to_stitch_resolution), cv2.INTER_AREA)
                images_to_stitch.append(image_to_stitch)
            # Resize for matching and distance estimation
            image_to_match = cv2.resize(image, tuple(matching_resolution), cv2.INTER_AREA)
            images_to_match.append(image_to_match)

    return {"images_to_match": images_to_match, "images_to_stitch": images_to_stitch, "is_valid": valid_frame}

def evaluate_rgbd_panorama(rgbd_panoramas, filename, dataset_path, bad_px_ratio_thresholds, panorama_resolution):
    """
    Read ground truth in <dataset_path>/gt/
    Compute PSNR, SSIM, MAE, RMSE and bad pixel ratio on an RGB-D panorama.
    Args:
        rgbd_panoramas: dict[filename: dict['rgb': [rows, cols, 3] uint8, 'inv_distance': [rows, cols] float32]]
    """
    try:
        rgbd_panorama = rgbd_panoramas[filename]

        read_name = os.path.splitext(filename)[0]
        evaluated_rgb = rgbd_panorama["rgb"]
        gt_rgb = cv2.imread(os.path.join(dataset_path, "gt/rgb_" + read_name + ".png"), cv2.IMREAD_UNCHANGED)
        evaluated_distance = rgbd_panorama["inv_distance"]
        gt_distance = cv2.imread(os.path.join(dataset_path, "gt/inv_distance_" + read_name + ".exr"), 
                                 cv2.IMREAD_UNCHANGED)

        if(gt_rgb is not None and gt_rgb is not None 
                and gt_rgb.dtype == evaluated_rgb.dtype and gt_distance.dtype == evaluated_distance.dtype):
            
            gt_rgb = cv2.resize(gt_rgb, tuple(panorama_resolution), cv2.INTER_AREA)
            gt_distance = cv2.resize(gt_distance, tuple(panorama_resolution), cv2.INTER_AREA)
            
            gt_rgb = gt_rgb.astype(np.float32) / 255
            evaluated_rgb = evaluated_rgb.astype(np.float32) / 255
            ssim = structural_similarity(gt_rgb, evaluated_rgb, multichannel=True)
            psnr = peak_signal_noise_ratio(gt_rgb, evaluated_rgb)

            err = np.abs(evaluated_distance - gt_distance)
            mae = np.sum(err) / err.size
            
            err2 = err * err
            rmse = np.sqrt(np.sum(err2) / err2.size)
            
            bad_px_ratios = []
            for bad_px_ratio_threshold in bad_px_ratio_thresholds:
                bad_px_ratios.append(100 * np.sum(err > bad_px_ratio_threshold) / err.size)

            return {"ssim": ssim, "psnr": psnr, "mae": mae, "rmse": rmse, "bad_px_ratios": bad_px_ratios}
        else:
            warnings.warn("Invalid ground truth for file " + filename + ". Will be ignored for evaluation")
            return None

    except KeyError:
        warnings.warn("Invalid ground truth for file " + filename)
        return None

def save_rgbd_panorama(rgbd_panoramas, filename, dataset_path):
    try:
        rgbd_panorama = rgbd_panoramas[filename]
        save_name = os.path.splitext(filename)[0]
        cv2.imwrite(os.path.join(dataset_path, "output/rgb_" + save_name + ".png"), rgbd_panorama["rgb"])
        cv2.imwrite(os.path.join(dataset_path, "output/inv_distance_" + save_name + ".exr"), 
                    rgbd_panorama["inv_distance"])
    except KeyError:
        pass

def translation(transform: NDArray) -> NDArray:
    if transform.shape == (3, 4) or transform.shape == (4, 4):
        return transform[:3, 3].reshape((3, 1))
    else:
        logger.error(f"Invalid transform shape, {transform.shape}")
        raise ValueError(f"Invalid transform shape, {transform.shape}")
