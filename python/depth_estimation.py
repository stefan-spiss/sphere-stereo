  
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

import cv2
import numpy as np
import torch
from isb_filter import ISB_Filter
from log_utils import LOG_DEBUG
from stitcher import Stitcher
from utils import CamModel, rgb2yCbCr


class RGBD_Estimator:
    def __init__(self, cam_models: list[CamModel], min_dist: float, max_dist: float, candidate_count: int, references_indices: list[int], reprojection_viewpoint: torch.Tensor, 
                 masks: list[torch.Tensor], matching_resolution: list[int], rgb_to_stitch_resolution: list[int], panorama_resolution: list[int], sigma_i: float, sigma_s: float, device: torch.device | str):
        """
        Prepare RGB-D estimation from fisheye images. 
        Perform camera selection for adaptive matching, initialize filters and stitcher 
        Args:
            cam_models: [number of cameras] calibration parameters following the double sphere model for each camera
            min_dist, max_dist: minimum and maximum distance for the sphere sweep volume computation
            candidate_count: Number of distance candidates between min_dist and max_dist (included)
            references_indices: [number of references] Indices of the cameras where distance estimation is performed before stitching 
            reprojection_viewpoint: [3] Reference viewpoint where the RGB-D panorama will be created
            masks: [number of cameras][matching_rows, matching_cols] Mask of the valid area in the captured fisheye image.
                one represents a reliable area while zero-pixels are ignored for matching.
                Typically, the camera body and the outskirt of a fisheye image are inexploitable for stereo
            matching_resolution: Resolution (cols, rows) used for matching. May be lower than original to save computation
            rgb_to_stitch_resolution: Resolution (cols, rows) of the colour images sampled during stitching. 
                May have a higher resolution as it has a negligible impact on performance.
            panorama_resolution: Resolution (cols, rows) of the output RGB-D panoramas
            sigma_i: Edge preservation parameter. Lower values preserve edges during cost volume filtering
            sigma_s: Smoothing parameter. Higher values give more weight to coarser scales during filtering
            device: CUDA-enabled GPU used for processing
        """
        self.cam_models: list[CamModel] = cam_models
        self.min_dist: float = min_dist
        self.max_dist: float = max_dist
        self.candidate_count: int = candidate_count
        self.references_indices: list[int] = references_indices
        self.reprojection_viewpoint: torch.Tensor = reprojection_viewpoint
        self.matching_resolution: list[int] = matching_resolution
        self.device: torch.device | str = device
        self.sigma_i: float = sigma_i 
        self.sigma_s: float = sigma_s

        self.cost_filter = ISB_Filter(candidate_count, matching_resolution, device)
        self.distance_filter = ISB_Filter(1, matching_resolution, device)

        calibrations_for_stitch = [cam_models[reference_index] for reference_index in references_indices]
        masks_for_stitching = [masks[reference_index] for reference_index in references_indices]
        self.fishey_stitcher = Stitcher(calibrations_for_stitch, reprojection_viewpoint, 
                                        masks_for_stitching, min_dist, max_dist, 
                                        matching_resolution, rgb_to_stitch_resolution, panorama_resolution, device)
        
        self.select_camera(masks)

    def select_camera(self, masks):
        """
        Select the cameras for adaptive matching (see Section 3.1)
        """
        self.selected_cameras = []
        for reference_index in self.references_indices:
            reference_cam_model = self.cam_models[reference_index]
            selected_camera = -torch.ones(self.matching_resolution[::-1], dtype=int, device=self.device).unsqueeze(0)
            max_displacement = torch.ones(self.matching_resolution[::-1], device=self.device).unsqueeze(0)

            u, v = torch.meshgrid([torch.arange(0, self.matching_resolution[1], device=self.device), 
                torch.arange(0, self.matching_resolution[0], device=self.device)])
            pt_unit, reference_valid = reference_cam_model.unproject(torch.stack([v, u], dim=-1).unsqueeze(0))

            LOG_DEBUG(f'nan entries in pt_unit: {torch.isnan(pt_unit).any()}')
            LOG_DEBUG(f'all not valid: {torch.any(~reference_valid)}')

            # Go through all the matched cameras and select the best one per pixel
            for cam_index, (cam_model, mask) in enumerate(zip(self.cam_models, masks, strict=True)):
                pt_near = pt_unit * self.min_dist
                pt_far = pt_unit * self.max_dist

                # points in the matched camera's point of view
                rt = torch.matmul(torch.inverse(cam_model.rt), reference_cam_model.rt)
                pt_near = torch.matmul(torch.cat([pt_near, torch.ones_like(pt_near[..., :1])], dim=-1), rt.T)
                pt_far = torch.matmul(torch.cat([pt_far, torch.ones_like(pt_near[..., :1])], dim=-1), rt.T)
                pt_near = pt_near[..., :3] / torch.norm(pt_near[..., :3], dim=-1, keepdim=True)
                pt_far = pt_far[..., :3] / torch.norm(pt_far[..., :3], dim=-1, keepdim=True)

                uv_near, valid_near = cam_model.project(pt_near)
                uv_far, valid_far = cam_model.project(pt_far)  

                # Evaluate the displacement from a given change in distance
                displacement = torch.norm(uv_near - uv_far, dim=-1)
  
                # Check the validity mask of the reprojected pixels
                uv_near = ((uv_near + 0.5) / torch.tensor([self.matching_resolution[0], 
                                                   self.matching_resolution[1]], device=self.device)) * 2 - 1
                uv_far = ((uv_far + 0.5) / torch.tensor([self.matching_resolution[0], 
                                                 self.matching_resolution[1]], device=self.device)) * 2 - 1

                mask_near = torch.nn.functional.grid_sample(mask.unsqueeze(0), uv_near, align_corners=False)[0]
                mask_far = torch.nn.functional.grid_sample(mask.unsqueeze(0), uv_far, align_corners=False)[0]

                # mask_near_vis = mask_near.cpu().numpy().squeeze()
                # LOG_DEBUG(f'ref-cam-{reference_index}/cam-{cam_index}: mask_near_vis: shape: {mask_near_vis.shape}, min: {mask_near_vis.min()}, max: {mask_near_vis.max()}')
                # mask_near_vis = cv2.normalize(mask_near_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                # cv2.imshow(f'ref-cam-{reference_index}/cam-{cam_index}: mask_near', mask_near_vis)
                # mask_far_vis = mask_far.cpu().numpy().squeeze()
                # LOG_DEBUG(f'ref-cam-{reference_index}/cam-{cam_index}: mask_far_vis: shape: {mask_far_vis.shape}, min: {mask_far_vis.min()}, max: {mask_far_vis.max()}')
                # mask_far_vis = cv2.normalize(mask_far_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                # cv2.imshow(f'ref-cam-{reference_index}/cam-{cam_index}: mask_far', mask_far_vis)
                # disp_vis = displacement.cpu().numpy().squeeze()
                # LOG_DEBUG(f'ref-cam-{reference_index}/cam-{cam_index}: disp_vis: shape: {disp_vis.shape}, min: {disp_vis.min()}, max: {disp_vis.max()}')
                # disp_vis = cv2.normalize(disp_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                # cv2.imshow(f'ref-cam-{reference_index}/cam-{cam_index}: disp', disp_vis)
                
                # Update the selected best camera
                LOG_DEBUG(f'displacement shape = {displacement.shape}, max_displacement shape = {max_displacement.shape}')
                LOG_DEBUG(f'nan displacement: {torch.isnan(displacement).any()}')
                LOG_DEBUG(f'inf displacement: {torch.isinf(displacement).any()}')
                LOG_DEBUG(f'nan max_displacement: {torch.isnan(max_displacement).any()}')
                LOG_DEBUG(f'inf max_displacement: {torch.isinf(max_displacement).any()}')

                current_best = ((displacement > max_displacement)
                                * reference_valid
                                * valid_near * valid_far
                                * (masks[reference_index] >= 0.9) * (mask_near >= 0.9) * (mask_far >= 0.9))

                max_displacement[current_best] = displacement[current_best]
                # max_disp_vis = max_displacement.cpu().numpy().squeeze()
                LOG_DEBUG(f'ref-cam-{reference_index}/cam-{cam_index}: max_disp_vis: shape: {max_displacement.shape}, min: {max_displacement.min()}, max: {max_displacement.max()}')
                # max_disp_vis = cv2.normalize(max_disp_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                # cv2.imshow(f'ref-cam-{reference_index}/cam-{cam_index}: max_disp', max_disp_vis)

                selected_camera[current_best] = cam_index
                LOG_DEBUG(f'ref-cam-{reference_index}/cam-{cam_index}: selected_camera_vis: shape: {selected_camera.shape}, min: {selected_camera.min()}, max: {selected_camera.max()}')
                # selected_camera_numpy = selected_camera.cpu().numpy().squeeze()
                # mask_0 = selected_camera_numpy == 0
                # mask_1 = selected_camera_numpy == 1
                # mask_2 = selected_camera_numpy == 2 
                # mask_3 = selected_camera_numpy == 3 
                # mask_4 = selected_camera_numpy == 4 
                # mask_5 = selected_camera_numpy == 5 
                # selected_camera_vis = np.zeros((*selected_camera_numpy.shape, 3), dtype=np.uint8)
                # selected_camera_vis[mask_0] = [255, 0, 0]  # Red for camera 0
                # selected_camera_vis[mask_1] = [0, 255, 0]  # Green for camera 1
                # selected_camera_vis[mask_2] = [0, 0, 255]  # Blue for camera 2
                # selected_camera_vis[mask_3] = [255, 255, 0]  # Yellow for camera 3
                # selected_camera_vis[mask_4] = [255, 0, 255]  # Magenta for camera 4
                # selected_camera_vis[mask_5] = [0, 255, 255]  # Cyan for camera 5
                # cv2.cvtColor(selected_camera_vis, cv2.COLOR_RGB2BGR, selected_camera_vis)
                # cv2.imshow(f'ref-cam-{reference_index}/cam-{cam_index}: selected_camera_vis', selected_camera_vis)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
    
            self.selected_cameras.append(selected_camera)
            selected_camera_numpy = selected_camera.cpu().numpy().squeeze()
            LOG_DEBUG(f'ref-cam-{reference_index}: selected_camera_vis: shape: {selected_camera_numpy.shape}, min: {selected_camera_numpy.min()}, max: {selected_camera_numpy.max()}')
            mask_0 = selected_camera_numpy == 0
            mask_1 = selected_camera_numpy == 1
            mask_2 = selected_camera_numpy == 2 
            mask_3 = selected_camera_numpy == 3 
            mask_4 = selected_camera_numpy == 4 
            mask_5 = selected_camera_numpy == 5 
            selected_camera_vis = np.zeros((*selected_camera_numpy.shape, 3), dtype=np.uint8)
            selected_camera_vis[mask_0] = [255, 0, 0]  # Red for camera 0
            selected_camera_vis[mask_1] = [0, 255, 0]  # Green for camera 1
            selected_camera_vis[mask_2] = [0, 0, 255]  # Blue for camera 2
            selected_camera_vis[mask_3] = [255, 255, 0]  # Yellow for camera 3
            selected_camera_vis[mask_4] = [255, 0, 255]  # Magenta for camera 4
            selected_camera_vis[mask_5] = [0, 255, 255]  # Cyan for camera 5
            cv2.cvtColor(selected_camera_vis, cv2.COLOR_RGB2BGR, selected_camera_vis)
            cv2.imshow(f'ref-cam-{reference_index}: selected_camera_vis', selected_camera_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def estimate_fisheye_distance(self, reference_image, guide, reference_cam_model, selected_camera, images):
        """
        Estimate distance on a fisheye image using the images from the other cameras
        """
        u, v = torch.meshgrid([torch.arange(0, self.matching_resolution[1], device=self.device), 
                               torch.arange(0, self.matching_resolution[0], device=self.device)])
        pt_unit, _ = reference_cam_model.unproject(torch.stack([v, u], dim=-1).unsqueeze(0))
        
        distance_candidates = 1 / torch.linspace(1 / self.min_dist, 1 / self.max_dist, 
                                               self.candidate_count, device=self.device)
        point_volume = (distance_candidates.view(self.candidate_count, 1, 1, 1) * 
                        pt_unit.view(1, self.matching_resolution[1], self.matching_resolution[0], 3))
        
        sweeping_volume = torch.zeros(
            [1, 3, self.candidate_count, self.matching_resolution[1], self.matching_resolution[0]], 
            device=self.device)

        # Sweeping volume computation, with a different camera for each pixel following adaptive spherical matching 
        for cam_index, cam_model in enumerate(self.cam_models):
            rt = torch.matmul(torch.inverse(cam_model.rt), reference_cam_model.rt)
            point_volume_in_cam = torch.matmul(torch.cat([point_volume, torch.ones_like(point_volume[..., :1])], dim=-1), 
                                               rt.T)
            uv, _ = cam_model.project(point_volume_in_cam[..., :3])
            uv = ((uv + 0.5) / torch.tensor([self.matching_resolution[0], 
                                     self.matching_resolution[1]], device=self.device)) * 2 - 1
            uv = uv.unsqueeze(0)
            uv = torch.cat([uv, torch.zeros_like(uv[..., :1])], dim=-1)

            image = images[cam_index]
            sweeping_volume_for_cam = torch.nn.functional.grid_sample(image, uv, align_corners=False)
            
            selected_mask = selected_camera==cam_index
            selected_mask = selected_mask.repeat(1, 3, self.candidate_count, 1, 1)
            sweeping_volume[selected_mask] = sweeping_volume_for_cam[selected_mask]

        # Raw cost computation from difference between the sweeping volume and the image
        cost_volume = torch.sum(torch.abs(sweeping_volume - reference_image), dim=1).squeeze(0)
        # Cost volume filtering
        cost_volume = torch.clamp(cost_volume, max = 500)

        cost_volume, _ = self.cost_filter.apply(guide.clone(), cost_volume.clone(), self.sigma_i, self.sigma_s)

        # Distance selection
        min_cost, selected_index_map = torch.min(cost_volume, dim=0, keepdim=True)
        max_cost, _ = torch.max(cost_volume, dim=0, keepdim=True)

        # Quadratic fitting for sub-candidate accuracy
        left_cost = torch.gather(cost_volume, 0, 
                                 torch.clamp(selected_index_map - 1, min=0, max=self.candidate_count - 1))
        right_cost = torch.gather(cost_volume, 0, 
                                  torch.clamp(selected_index_map + 1, min=0, max=self.candidate_count - 1))
        variation = 0.5 * (left_cost - right_cost) / ((left_cost + right_cost) - 2. * min_cost + 1e-8)
        variation = torch.clamp(variation, min=-0.5, max=0.5)
        variation[selected_index_map == self.candidate_count - 1] = 0
        variation[selected_index_map == 0] = 0
        selected_index_map = selected_index_map.float() + variation
        selected_index_map[max_cost == min_cost] = self.candidate_count - 1

        # Index to distance conversion
        distance_map = distance_candidates[0] / ((distance_candidates[0] / distance_candidates[-1] - 1) 
                                                 * selected_index_map / (self.candidate_count - 1) + 1)
        distance_map[torch.abs(max_cost - min_cost) < 1e-8] = distance_candidates[-1]

        # Distance map post filtering, with higher edge preservation.
        filtered_distance, _ = self.distance_filter.apply(guide.clone(), distance_map.clone(), 
                                                          self.sigma_i/2, self.sigma_s/2)
        return filtered_distance[0]

    def estimate_RGBD_panorama(self, images_to_match, images_to_stitch):
        """
        Estimate depth on the reference fisheye images (specified when instantiating)
        Then stitch the fisheye images to produce a complete RGB-D panorama
        Args:
            images_to_match: [number of cameras][matching_rows, matching_cols, 3] Set of fisheye images for distance estimation.
                Their resolutions may be lower than original to save computation. Should be float32 with [0, 255] range
            images_to_stitch: [number of references][rgb_stitching_rows, rgb_stitching_cols, 3] Fisheye images used for colour stitching. 
                They may have a higher resolution as it has a negligible impact on performance. 
                Should be float32 with [0, 255] range
        Returns:
            rgb: [rows, cols, 3] colour panorama as uint8
            distance: [rows, cols] estimated distance panorama as float32
        """
        
        
        for i, img in enumerate(images_to_match):
            cv2.imshow(f"image {i}", img.cpu().numpy().astype(np.uint8))

        cv2.waitKey(0)
        # Evaluate distance for each of the reference fisheye images
        images_to_match_permuted = [image.unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(2)
                                    for image in images_to_match]
        
        distance_maps = []
        for reference_index, selected_camera in zip(self.references_indices, self.selected_cameras):
            guide = rgb2yCbCr(images_to_match[reference_index]).type(torch.uint8)
            distance_maps.append(
                self.estimate_fisheye_distance(
                    images_to_match_permuted[reference_index], 
                    guide,
                    self.cam_models[reference_index], 
                    selected_camera, 
                    images_to_match_permuted, 
                )
            )
        for i, img in enumerate(distance_maps):
            cv2.imshow(f"dist {i}", img.cpu().numpy().astype(np.uint8))

        cv2.waitKey(0)

        # Stitch in a disparity aware manner to create complete panoramas
        images_to_stitch = [reference_image.type(torch.uint8)
                            for reference_image in images_to_stitch]
        rgb, distance = self.fishey_stitcher.stitch(images_to_stitch, distance_maps)

        return rgb, distance
