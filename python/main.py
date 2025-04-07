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
import argparse
import json
import os.path
from contextlib import suppress
from pathlib import Path

import cv2
import numpy as np
import torch
from depth_estimation import RGBD_Estimator
from joblib import Parallel, delayed
from log_utils import (
    LOG_DEBUG,
    LOG_ERROR,
    LOG_INFO,
    initLogging,
)
from utils import (
    evaluate_rgbd_panorama,
    parse_json_calib,
    parse_json_calib_kb_fisheye,
    read_input_images,
    save_rgbd_panorama,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="evaluation_dataset", help="Path to the dataset folder")
    parser.add_argument('--references_indices', nargs="*", type=int, default=[0, 2], help="Indices of the reference cameras to be used for the RGB-D panorama.")
    parser.add_argument('--min_dist', type=float, default=0.55, help="Minimum distance of spherical sweeping volume (depends on unit of the camera calibration).")
    parser.add_argument('--max_dist', type=float, default=100, help="Maximum distance of spherical sweeping volume (depends on unit of the camera calibration).")
    parser.add_argument('--candidate_count', type=int, default=32, help="Number of depth candidates used for spherical sweeping (number of spheres between min and max distance).")
    parser.add_argument('--search_steps_min_dist', type=int, default=-1, help="Number of search steps to find minimum valid distance for each pixel in each reference image. If -1, min_dist is used for each pixel.")
    parser.add_argument('--sigma_i', type=float, default=10, help="Edge preservation parameter for bilateral filter weights used for edge-preserving downsampling (depth estimation).")
    parser.add_argument('--sigma_s', type=float, default=25, help="Smoothness parameter for gaussian filter weights used for edge-preserving upsampling (depth estimation).")
    parser.add_argument('--matching_resolution', nargs=2, type=int, default=[1024, 1024], help="Resolution of the input images to be used for matching.")
    parser.add_argument('--rgb_to_stitch_resolution', nargs=2, type=int, default=[1216, 1216], help="Resolution of the input images to be used for RGB stitching.")
    parser.add_argument('--panorama_resolution', nargs=2, type=int, default=[2048, 1024], help="Resolution of the output panorama.")
    parser.add_argument('--device', type=str, default="cuda:0", help="Device to use for computation (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument('--saving', action=argparse.BooleanOptionalAction, default=True, help="Save the RGB-D panorama to disk.")
    parser.add_argument('--visualize', action=argparse.BooleanOptionalAction, default=False, help="Visualize the RGB-D panorama.")
    parser.add_argument('--evaluate', action=argparse.BooleanOptionalAction, default=False, help="Evaluate the RGB-D panorama.")
    parser.add_argument('--bad_px_ratio_thresholds', type=float, default=[0.1, 0.4], help="Thresholds for bad pixel ratio evaluation.")
    parser.add_argument('--kb_fisheye', action=argparse.BooleanOptionalAction, default=False, help="Use OpenCV fisheye model.")
    parser.add_argument('--use_perspective_reproj', action=argparse.BooleanOptionalAction, default=False, help="If kb_fisheye is used, use perspective reprojection in addition to Kannala-Brandt reprojection (as done in OpenCV fisheye model).")
    parser.add_argument('--recalculate_fov', action=argparse.BooleanOptionalAction, default=True, help="If kb_fisheye is used, recalculate the field of view (FOV) for the fisheye camera model.")
    parser.add_argument('--max_theta', type=float, default=90, help="If kb_fisheye is used, maximum theta value for the fisheye camera model (in degrees).")
    parser.add_argument('--load_all_frames_parallel', action=argparse.BooleanOptionalAction, default=False, help="Process all images in parallel (for large datasets, this requires a lot of memory).")
    args = parser.parse_args()
    
    initLogging()

    if args.kb_fisheye:
        cam_models = parse_json_calib_kb_fisheye(os.path.join(args.dataset_path, "calibrated_cameras_data.yml"), args.matching_resolution, args.use_perspective_reproj, args.device, np.deg2rad(args.max_theta), args.recalculate_fov)

    else:
        with open(os.path.join(args.dataset_path, "calibration.json")) as f:
            raw_calibration = json.load(f)['value0']
            cam_models = parse_json_calib(raw_calibration, args.matching_resolution, args.device)

    if len(cam_models) < 2:
        LOG_ERROR("Only one or no camera model found. Please check the calibration file.")
        raise RuntimeError("Only one or no camera model found. Please check the calibration file.")

    # # Reference viewpoint for the estimated RGB-D panorama is the center of all cameras
    # reprojection_viewpoint = torch.zeros([3], device=args.device)
    # for cam in cam_models:
    #     reprojection_viewpoint += cam.rt[:3, 3] / len(cam_models)

    # Reference viewpoint for the estimated RGB-D panorama is the center of the references
    reprojection_viewpoint = torch.zeros([3], device=args.device)
    for references_index in args.references_indices:
        reprojection_viewpoint += cam_models[references_index].rt[:3, 3] / len(args.references_indices)

    # Read masks
    masks = []
    for cam_index in range(len(cam_models)):
        if os.path.isfile(os.path.join(args.dataset_path, "cam" + str(cam_index)) + "/" + "mask.png"):
            mask = cv2.imread(os.path.join(args.dataset_path, "cam" + str(cam_index)) + "/" + "mask.png", 
                              cv2.IMREAD_UNCHANGED)
            mask = cv2.resize(mask, tuple(args.matching_resolution), cv2.INTER_AREA)
            masks.append(torch.tensor(mask, device=args.device, dtype=torch.float32).unsqueeze(0)/255)
        else:
            masks.append(torch.ones(args.matching_resolution, device=args.device).T.unsqueeze(0))

    # Initialize distance estimator and stitcher
    rgbd_estimator = RGBD_Estimator(cam_models, args.min_dist, args.max_dist, args.candidate_count, args.search_steps_min_dist,
                                    args.references_indices, reprojection_viewpoint, masks, 
                                    args.matching_resolution, args.rgb_to_stitch_resolution, args.panorama_resolution, 
                                    args.sigma_i, args.sigma_s, args.device)


    filenames = os.listdir(os.path.join(args.dataset_path, "cam0/"))
    LOG_DEBUG("Found %d images in the dataset.", len(filenames))

    with suppress(ValueError): # mask is not mandatory
        filenames.remove("mask.png")

        # process all images in parallel (for large datasets, this requires a lot of memory)
    if args.load_all_frames_parallel:
        LOG_INFO("Loading all frames into memory in parallel.")
        all_fisheye_images = Parallel(n_jobs=-1, backend="threading")(
            delayed(read_input_images)(
                filename, args.dataset_path, args.matching_resolution, args.rgb_to_stitch_resolution, 
                cam_models, args.references_indices) 
            for filename in filenames)

        rgbd_panoramas = {}
        for frame_index, filename in enumerate(filenames):
            LOG_INFO(f"Processing frame {frame_index}: {filename}")
            fisheye_images = all_fisheye_images[frame_index]["images_to_match"]
            reference_fisheye_images = all_fisheye_images[frame_index]["images_to_stitch"]
            valid_frame = all_fisheye_images[frame_index]["is_valid"]

            if valid_frame:
                fisheye_images = [torch.tensor(fisheye_image, device=args.device) for fisheye_image in fisheye_images]
                reference_fisheye_images = [torch.tensor(reference_fisheye_image, device=args.device) 
                                            for reference_fisheye_image in reference_fisheye_images]
                rgb, distance = rgbd_estimator.estimate_RGBD_panorama(fisheye_images, reference_fisheye_images)
                
                rgbd_panoramas[filename] = {"rgb": rgb.cpu().numpy(), "inv_distance": 1 / distance.cpu().numpy()}

                if args.visualize:
                    # Map inverse distance to [0, 255] and display
                    distance_map = 1 / distance.cpu().numpy()
                    distance_map = ((rgbd_panoramas[filename]["inv_distance"] - 1 / args.max_dist) 
                                    / (1 / args.min_dist - 1 / args.max_dist))
                    distance_map = np.clip(255 * distance_map, 0, 255).astype(np.uint8)
                    distance_map = cv2.applyColorMap(distance_map, cv2.COLORMAP_MAGMA)
                    cv2.imshow("distance_map", distance_map)
                    cv2.imshow("rgb", rgbd_panoramas[filename]["rgb"])
                    key = cv2.waitKey(0)
                    
                    if key == 27:  # ESC key
                        break

        if args.saving:
            LOG_INFO("Saving all resulting panoramas in parallel.")
            Path(os.path.join(args.dataset_path, "output")).mkdir(parents=True, exist_ok=True)
            Parallel(n_jobs=-1, backend="threading")(
                delayed(save_rgbd_panorama)(rgbd_panoramas, filename, args.dataset_path) 
                for filename in filenames)


        if args.evaluate:
            LOG_INFO("Evaluation of all resulting panoramas in parallel.")
            evaluations = Parallel(n_jobs=-1, backend="threading")(
                delayed(evaluate_rgbd_panorama)(rgbd_panoramas, filename, args.dataset_path, 
                                                args.bad_px_ratio_thresholds, args.panorama_resolution) 
                for filename in filenames)

            # Average the evaluation metrics
            psnr = 0
            ssim = 0
            rmse = 0
            mae = 0
            bad_px_ratios = [0] * len(args.bad_px_ratio_thresholds)
            evaluation_count = 0

            for evaluation in evaluations:
                if evaluation is not None:
                    psnr += evaluation["psnr"]
                    ssim += evaluation["ssim"]
                    rmse += evaluation["rmse"]
                    mae += evaluation["mae"]
                    bad_px_ratios = [bad_px_ratio + current_bad_px_ratio 
                        for  bad_px_ratio, current_bad_px_ratio in zip(bad_px_ratios, evaluation["bad_px_ratios"], strict=True)]
                    evaluation_count += 1

            if evaluation_count > 0:
                LOG_INFO("PSNR = ", psnr / evaluation_count)
                LOG_INFO("SSIM = ", ssim / evaluation_count)
                for bad_px_ratio, bad_px_ratio_threshold in zip(bad_px_ratios, args.bad_px_ratio_thresholds, strict=True):
                    LOG_INFO(">", bad_px_ratio_threshold, " = ", bad_px_ratio / evaluation_count)
                LOG_INFO("MAE = ", mae / evaluation_count)
                LOG_INFO("RMSE = ", rmse / evaluation_count)
    else:
        evaluations = []
        for frame_index, filename in enumerate(filenames):
            LOG_INFO(f"Processing frame {frame_index}: {filename}")
            input_imgs = read_input_images(
                filename,
                args.dataset_path,
                args.matching_resolution,
                args.rgb_to_stitch_resolution,
                cam_models,
                args.references_indices,
            )
            fisheye_images = input_imgs["images_to_match"]
            reference_fisheye_images = input_imgs["images_to_stitch"]
            valid_frame = input_imgs["is_valid"]
                    
            if valid_frame:
                fisheye_images = [torch.tensor(fisheye_image, device=args.device) for fisheye_image in fisheye_images]
                reference_fisheye_images = [torch.tensor(reference_fisheye_image, device=args.device) 
                                            for reference_fisheye_image in reference_fisheye_images]
                rgb, distance = rgbd_estimator.estimate_RGBD_panorama(fisheye_images, reference_fisheye_images)
                
                rgbd_panorama = {}
                rgbd_panorama[filename] = {"rgb": rgb.cpu().numpy(), "inv_distance": 1 / distance.cpu().numpy()}

                if args.visualize:
                    # Map inverse distance to [0, 255] and display
                    distance_map = 1 / distance.cpu().numpy()
                    distance_map = ((rgbd_panorama[filename]["inv_distance"] - 1 / args.max_dist) 
                                    / (1 / args.min_dist - 1 / args.max_dist))
                    distance_map = np.clip(255 * distance_map, 0, 255).astype(np.uint8)
                    distance_map = cv2.applyColorMap(distance_map, cv2.COLORMAP_MAGMA)
                    cv2.imshow("distance_map", distance_map)
                    cv2.imshow("rgb", rgbd_panorama[filename]["rgb"])
                    key = cv2.waitKey(0)
                    
                    if key == 27:  # ESC key
                        break
                if args.saving:
                    LOG_INFO("Saving the resulting panorama.")
                    Path(os.path.join(args.dataset_path, "output")).mkdir(parents=True, exist_ok=True)
                    save_rgbd_panorama(rgbd_panorama, filename, args.dataset_path)
                if args.evaluate:
                    LOG_INFO("Evaluation of the resulting panorama.")
                    evaluations.append(
                        evaluate_rgbd_panorama(
                            rgbd_panorama, filename, args.dataset_path, 
                            args.bad_px_ratio_thresholds, args.panorama_resolution
                        )
                    )
                    LOG_INFO("Evaluation result:\n\tPSNR = {evaluation['psnr']}\n\tSSIM = {evaluation['ssim']}\n\tMAE = {evaluation['mae']}\n\tRMSE = {evaluation['rmse']}")
                
        if args.evaluate:
            LOG_INFO("Calculating average evaluation metrics.")
            # Average the evaluation metrics
            psnr = 0
            ssim = 0
            rmse = 0
            mae = 0
            bad_px_ratios = [0] * len(args.bad_px_ratio_thresholds)
            evaluation_count = 0

            for evaluation in evaluations:
                if evaluation is not None:
                    psnr += evaluation["psnr"]
                    ssim += evaluation["ssim"]
                    rmse += evaluation["rmse"]
                    mae += evaluation["mae"]
                    bad_px_ratios = [bad_px_ratio + current_bad_px_ratio 
                        for  bad_px_ratio, current_bad_px_ratio in zip(bad_px_ratios, evaluation["bad_px_ratios"], strict=True)]
                    evaluation_count += 1

            if evaluation_count > 0:
                LOG_INFO("PSNR = ", psnr / evaluation_count)
                LOG_INFO("SSIM = ", ssim / evaluation_count)
                for bad_px_ratio, bad_px_ratio_threshold in zip(bad_px_ratios, args.bad_px_ratio_thresholds, strict=True):
                    LOG_INFO(">", bad_px_ratio_threshold, " = ", bad_px_ratio / evaluation_count)
                LOG_INFO("MAE = ", mae / evaluation_count)
                LOG_INFO("RMSE = ", rmse / evaluation_count)