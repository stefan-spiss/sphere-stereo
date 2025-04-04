import argparse
import json
import math
import os.path

import cupy
import cv2
import numpy as np
import open3d as o3
import torch
from joblib import Parallel, delayed
from log_utils import LOG_DEBUG, LOG_ERROR, LOG_INFO, LOG_WARNING, initLogging
from utils import (
    CamModel,
    parse_json_calib,
    parse_json_calib_kb_fisheye,
    read_input_images,
    translation,
)


def makeSphericalRays(equirect_size: tuple[int, int], device: str | torch.device,
                      phi_deg: float, phi2_deg: float = -1.0) -> torch.Tensor:
    """
    Generate spherical rays for a given equirectangular image size.
    [TODO] - Add reference to omnimvs repo.
    Args:
        equirect_size (tuple[int, int]): The width and height of the equirectangular image.
        device: The device on which the tensor will be allocated (e.g., 'cpu' or 'cuda').
        phi_deg (float): The vertical field of view in degrees.
        phi2_deg (float, optional): The second vertical field of view in degrees. Defaults to -1.0.
    Returnsu:
        torch.Tensor: A tensor containing the spherical rays.
    """
    w, h = equirect_size
    xs, ys = np.meshgrid(range(w), range(h)) # row major
    w_2, h_2 = w / 2.0, (h - 1) / 2.0
    xs = (xs - w_2) / w_2 * np.pi + (np.pi / 2.0)
    if phi2_deg > 0.0:
        med = np.deg2rad((phi2_deg - phi_deg) / 2.0)
        med2 = np.deg2rad((phi2_deg + phi_deg) / 2.0)
        ys = (ys - h_2) / h_2 * med2 - med
    else:
        ys = (ys - h_2) / h_2 * np.deg2rad(phi_deg)
    
    X = -np.cos(ys) * np.cos(xs)
    Y = np.sin(ys) # sphere
    # Y = np.sin(ys) / np.cos(ys) # cylinder
    # Y = ys / np.deg2rad(phi_deg) # perspective cylinder
    Z = np.cos(ys) * np.sin(xs)
    # rays = np.concatenate((np.reshape(X, [1, -1]),
    #     np.reshape(Y, [1,-1]), np.reshape(Z, [1,-1]))).astype(np.float64)
    rays = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).astype(np.float32).T
    if np.isnan(rays).any():
        LOG_WARNING(f'Some rays contain nan entries: #of nan entries: {np.isnan(rays)}')
    return torch.tensor(rays, device=device)

def transformRays(rays: torch.Tensor, rt: torch.Tensor) -> torch.Tensor:
    """
    Transforms a set of rays using a given rotation-translation matrix.

    Args:
        rays (torch.Tensor): A tensor of shape (N, 3) representing N rays in 3D space.
        rt (torch.Tensor): A tensor of shape (4, 4) representing the rotation-translation matrix.

    Returns:
        torch.Tensor: A tensor of shape (N, 3) representing the transformed rays.
    """
    ones = torch.ones((rays.shape[0], 1), device=rays.device)
    rays_hom = torch.cat((rays, ones), dim=1)
    rays_trans = rt @ rays_hom.T
    return rays_trans[:3, :].T
    # return (rt[:3, :3] @ rays + rt[:3, 3:]).float()

def pixelToGrid(pts: torch.Tensor, target_resolution: tuple[int, int], source_resolution: tuple[int, int]):
    """
    Converts pixel coordinates to grid coordinates.

    Args:
        pts (torch.Tensor): A tensor of shape (N, 2) containing pixel coordinates.
        target_resolution (tuple[int, int]): The target resolution as a tuple (width, height).
        source_resolution (tuple[int, int]): The source resolution as a tuple (width, height).

    Returns:
        torch.Tensor: A tensor of shape (target_height, target_width, 2) containing grid coordinates.
    """
    w, h = target_resolution
    width, height = source_resolution
    xs = (pts[:,0]) / (width - 1) * 2 - 1
    ys = (pts[:,1]) / (height - 1) * 2 - 1
    xs = xs.reshape((h, w, 1))
    ys = ys.reshape((h, w, 1))
    return torch.cat((xs, ys), dim=2)

def interp2D(img: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """
    Perform bilinear interpolation on a 2D image using a given grid.

    Args:
        img (torch.Tensor): The input image tensor of shape (C, H, W), where C is the number of channels,
                            H is the height, and W is the width.
        grid (torch.Tensor): The grid tensor of shape (H, W, 2) containing the normalized coordinates
                             at which to sample the input image.

    Returns:
        torch.Tensor: The interpolated image tensor of shape (C, H, W).
    """
    # LOG_DEBUG(f'img_shape: {img.shape}')
    # LOG_DEBUG(f'grid_shape: {grid.shape}')
    grid = grid.unsqueeze(0)
    out = torch.nn.functional.grid_sample(img, grid, mode='bilinear', align_corners=True).squeeze()
    # LOG_DEBUG(f'out_shape: {out.shape}')
    return out


def getPanoramasInputImgs(imgs, cams: list[CamModel], rays, equirect_size):
    panos = []
    for cam, img in zip(cams, imgs, strict=True):
        is_rgb = len(img.shape) == 3
        # img[cam.invalid_mask] *= 0

        # LOG_DEBUG(f'rays_shape: {rays.shape}')
        P2 = transformRays(rays, cam.rt.inverse())
        # LOG_DEBUG(f'P2_shape: {P2.shape}')
        p, valid = cam.project(P2)
        LOG_DEBUG(f'p_shape: {p.shape}')
        grid = pixelToGrid(p, equirect_size, cam.original_resolution * cam.matching_scale)
        LOG_DEBUG(f'grid_shape: {grid.shape}')
        # valid = (theta <= cam.max_theta).reshape(self.equirect_size)
        valid = valid.reshape((equirect_size[1], equirect_size[0])).detach().cpu().numpy()
        LOG_DEBUG(f'valid_shape: {valid.shape}')

        img_tensor = torch.tensor(img, device=grid.device).permute(2, 0, 1).unsqueeze(0)
        LOG_DEBUG(f'img_tensor_shape: {img_tensor.shape}')
        equi_img = interp2D(img_tensor, grid).detach().cpu().numpy()
        LOG_DEBUG(f'equi_img_shape: {equi_img.shape}')
        if is_rgb:
            equi_img = np.moveaxis(equi_img, 0, -1)
            pano = np.zeros((equirect_size[1], equirect_size[0], 3))
        else:
            pano = np.zeros((equirect_size[1], equirect_size[0]))
        pano[valid] = equi_img[valid]
        panos.append(pano.astype(np.uint8))
    return panos

def getPanoramasInputImgsCupy(imgs, cams: list[CamModel], equirect_size):
    panos = []
    
    with open('python/vec_utils.cuh') as f:
        utils_source = f.read()
    
    with open('python/utils.cuh') as f:
        utils_source = utils_source + f.read()
        
    with open('python/kb_fisheye.cuh') as f:
        cuda_source = utils_source + f.read()

    with open('python/reprojection.cu') as f:
        cuda_source = cuda_source + f.read()
        cuda_source = cuda_source.replace("MAX_ITER", str(cams[0].unproj_crit[0]))
        cuda_source = cuda_source.replace("UNPROJ_CRIT_0", str(cams[0].unproj_crit[1]))
        cuda_source = cuda_source.replace("UNPROJ_CRIT_1", str(cams[0].unproj_crit[2]))
        cuda_source = cuda_source.replace("MIN_LIMIT", str(cams[0].proj_crit))

        cuda_source = cuda_source.replace("PANO_COLS", str(equirect_size[0]))
        cuda_source = cuda_source.replace("PANO_ROWS", str(equirect_size[1]))
        cuda_source = cuda_source.replace("COLS", str(cams[0].original_resolution[0].item()))
        cuda_source = cuda_source.replace("ROWS", str(cams[0].original_resolution[1].item()))
        cuda_source = cuda_source.replace("REFERENCES_COUNT", str(len(cams)))


    LOG_DEBUG(f"Cuda source:\n{cuda_source}")

    kernel_names = ['reprojectToPanorama']
    ext = '<FisheyeKB>' if not cam_models[0].use_perspective_reproj else '<FisheyeKBPerspectiveProjection>'
    for i in range(len(kernel_names)):
        kernel_names[i] += ext

    try:
        # module = cupy.RawModule(code=cuda_source, options=("-std=c++11", "-Xptxas"))
        # module = cupy.RawModule(code=cuda_source, options=("-std=c++11", "-rdc=true"))
        module = cupy.RawModule(code=cuda_source, options=("-std=c++11",), name_expressions=kernel_names)
        module.compile()
        LOG_INFO("CUDA compilation successful")
    except cupy.cuda.compiler.CompileException as e:
        LOG_ERROR(f"Compilation failed: {e}")
        raise e

    reprojection_fct = module.get_function(kernel_names[0])
    
    pano_lookup= torch.zeros([len(cams), equirect_size[1], equirect_size[0], 2], device='cuda:0')
    valid_lookup = torch.zeros([len(cams), equirect_size[1], equirect_size[0]], dtype=torch.bool, device='cuda:0')
    calibration_vectors = []
    translation_vectors = []
    rotation_vectors = []
    for cam in cams:
        calibration_vectors.append(cam.vectorize_calibration())
        inv_rt = cam.rt.inverse()
        translation_vectors.append(inv_rt[:3, 3])
        rotation_vectors.append(inv_rt[:3, :3])

    calibration_vectors = torch.cat(calibration_vectors, dim=0).contiguous()
    translations = torch.cat(translation_vectors, dim=0).contiguous()
    rotations = torch.cat(rotation_vectors, dim=0).contiguous()
    
    block_size = 256
    grid_size = math.ceil((equirect_size[0] * equirect_size[1]) / block_size)
    
    LOG_DEBUG(f'block_size: {block_size}, grid_size: {grid_size}')
    
    reprojection_fct(
        block=(block_size,), grid=(grid_size,),
        args=(
            pano_lookup.data_ptr(),
            valid_lookup.data_ptr(),
            calibration_vectors.data_ptr(),
            rotations.data_ptr(),
            translations.data_ptr()
        )
    )
    # cupy.cuda.get_current_stream().synchronize()
    
    for cam, img, pano_lut, valid in zip(cams, imgs, pano_lookup, valid_lookup, strict=True):
        is_rgb = len(img.shape) == 3
        # img[cam.invalid_mask] *= 0

        # LOG_DEBUG(f'p_shape: {p.shape}')
        p_vis = pano_lut.cpu().numpy()
        p = pano_lut.reshape((equirect_size[1] * equirect_size[0], 2))
        grid = pixelToGrid(p, equirect_size, cam.original_resolution * cam.matching_scale)
        # LOG_DEBUG(f'grid_shape: {grid.shape}')
        # valid = (theta <= cam.max_theta).reshape(self.equirect_size)
        # valid = valid_lookup.reshape((equirect_size[1], equirect_size[0]).get())
        valid = valid.cpu().numpy()
        # LOG_DEBUG(f'valid_shape: {valid.shape}')

        img_tensor = torch.tensor(img, device=grid.device).permute(2, 0, 1).unsqueeze(0)
        # LOG_DEBUG(f'img_tensor_shape: {img_tensor.shape}')
        equi_img = interp2D(img_tensor, grid).cpu().numpy()
        # LOG_DEBUG(f'equi_img_shape: {equi_img.shape}')
        if is_rgb:
            equi_img = np.moveaxis(equi_img, 0, -1)
            pano = np.zeros((equirect_size[1], equirect_size[0], 3))
        else:
            pano = np.zeros((equirect_size[1], equirect_size[0]))
        pano[valid] = equi_img[valid]
        panos.append(pano.astype(np.uint8))
    return panos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="evaluation_dataset")
    # parser.add_argument('--references_indices', nargs="*", type=int, default=[0, 2])
    parser.add_argument('--matching_resolution', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('--panorama_resolution', nargs=2, type=int, default=[2048, 1024])
    # parser.add_argument('--min_dist', type=float, default=0.55)
    # parser.add_argument('--max_dist', type=float, default=100)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--kb_fisheye', type=bool, default=False)
    parser.add_argument('--use_cupy', type=bool, default=False)
    args = parser.parse_args()
    
    initLogging()

    if args.kb_fisheye:
        use_perspective_reproj = False
        recalculate_fov = True
        cam_models = parse_json_calib_kb_fisheye(os.path.join(args.dataset_path, "calibrated_cameras_data.yml"), args.matching_resolution, use_perspective_reproj, args.device, np.pi / 2.0, recalculate_fov)
    else:
        f = open(os.path.join(args.dataset_path, "calibration.json"))
        raw_calibration = json.load(f)['value0']
        cam_models = parse_json_calib(raw_calibration, args.matching_resolution, args.device)

    # Reference viewpoint for the estimated RGB-D panorama is the center of the references
    reprojection_viewpoint = torch.zeros([3], device=args.device)
    for cam in cam_models:
        if args.kb_fisheye:
            cam.rt[:3, 3] *= 0.001
        reprojection_viewpoint += cam.rt[:3, 3]
    reprojection_viewpoint /= len(cam_models)

    for cam in cam_models:
        cam.rt[:3, 3] -= reprojection_viewpoint
    
    cam_centers = []
    cam_centers.append(o3.geometry.TriangleMesh.create_coordinate_frame(size=0.15))
    for i, cam in enumerate(cam_models):
        rt = cam.rt.cpu().numpy()
        # rt[:3, 3] *= (1 / args.min_dist - 1 / args.max_dist)
        cam_centers.append(o3.geometry.TriangleMesh.create_coordinate_frame(size=0.1).transform(rt))
        if i == 0:
            cam_centers[-1].paint_uniform_color([1, 0, 0])
        elif i == 1:
            cam_centers[-1].paint_uniform_color([0, 1, 0])

    o3.visualization.draw_geometries(cam_centers)
    
    filenames = os.listdir(os.path.join(args.dataset_path, "cam0/"))
    from contextlib import suppress

    with suppress(ValueError):
        filenames.remove("mask.png")

    all_fisheye_images = Parallel(n_jobs=-1, backend="threading")(
        delayed(read_input_images)(
            filename, args.dataset_path, args.matching_resolution, args.matching_resolution, 
            cam_models, range(len(cam_models))) 
        for filename in filenames)

    if args.use_cupy and args.kb_fisheye:
        panos = getPanoramasInputImgsCupy(all_fisheye_images[0]['images_to_match'], cam_models, args.panorama_resolution)
    else:
        rays = makeSphericalRays(args.panorama_resolution, args.device, 90.0)
        panos = getPanoramasInputImgs(all_fisheye_images[0]['images_to_match'], cam_models, rays, args.panorama_resolution)

    vis_pano_img = np.concatenate([img for img in panos], axis=0)
    cv2.namedWindow('pano_imgs', cv2.WINDOW_NORMAL)
    cv2.imshow('pano_imgs', vis_pano_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()