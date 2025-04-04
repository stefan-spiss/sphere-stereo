/**
=======================================================================
General Information
-------------------
This is a GPU-based implementation of the Kannala and Brandt fisheye model. The code supports two versions of the model:
    - The original model
    - A model including a perspective projection using the identity camera matrix following the OpenCV fisheye model
Author: Stefan Spiss
==========================================================================
License Information
-------------------
CC BY-NC-SA 3.0
Andreas Meuleman and Min H. Kim have developed this software and related documentation (the "Software"); confidential use in source form of the Software, without modification, is permitted provided that the following conditions are met:
Neither the name of the copyright holder nor the names of any contributors may be used to endorse or promote products derived from the Software without specific prior written permission.
The use of the software is for Non-Commercial Purposes only. As used in this Agreement, "Non-Commercial Purpose" means for the purpose of education or research in a non-commercial organisation only. "Non-Commercial Purpose" excludes, without limitation, any use of the Software for, as part of, or in any way in connection with a product (including software) or service which is sold, offered for sale, licensed, leased, published, loaned or rented. If you require a license for a use excluded by this agreement, please email [minhkim@kaist.ac.kr].
Warranty: KAIST-VCLAB MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. KAIST-VCLAB SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.
Please refer to license.txt for more details.
=======================================================================
**/

struct __align__(16) IntrinsicsKB {
    float2 fl, principal;
    float4 ks;
    float max_theta;
};

/**
 * Function calculates theta given r_theta using Newton-Raphson method
 */
inline __device__ float2 solve_for_theta(float r_theta, IntrinsicsKB calib, int max_iter, float eps) {
    float theta = r_theta;
    float theta_fix = 0.0f;

    bool should_iterate =
        (r_theta > eps);  // helps to avoid divergence at warp level (according to chatgpt)

    if (should_iterate) {
        float k1 = calib.ks.x;
        float k2 = calib.ks.y;
        float k3 = calib.ks.z;
        float k4 = calib.ks.w;

        for (int i = 0; i < max_iter; i++) {
            float theta2 = theta * theta;
            float func =
                fmaf(fmaf(fmaf(fmaf(k4, theta2, k3), theta2, k2), theta2, k1), theta2, 1) * theta;

            float d_func_d_theta =
                fmaf(fmaf(fmaf(fmaf(9 * k4, theta2, 7 * k3), theta2, 5 * k2), theta2, 3 * k1),
                     theta2, 1);

            theta_fix = (func - r_theta) / d_func_d_theta;

            theta = theta - theta_fix;

            if ((theta_fix * theta_fix) <= (eps * eps)) {
                break;
            }
        }
    }

    return make_float2(theta, theta_fix);
}

/**
 * Unproject pixels to the unit sphere using the Kannala and Brandt model.
 */
struct FisheyeKB {
    inline __device__ float3 unproject(float2 uv, IntrinsicsKB calib, bool& valid) {
#ifdef DEBUG
        int indexIn = blockIdx.x * blockDim.x + threadIdx.x;
        if (indexIn == 0)
            printf("FisheyeKB unproject\n");
#endif
        // float2 pi = uv;
        float2 pw = (uv - calib.principal) / calib.fl;

        float r_theta = sqrtf(pw.x * pw.x + pw.y * pw.y);

        float2 result = solve_for_theta(r_theta, calib, MAX_ITER, UNPROJ_CRIT_0);
        float theta = result.x;
        float theta_residual = result.y;

        float scale = sin(theta) / r_theta;

        // theta is monotonously increasing or decreasing depending on the sign of theta. If theta
        // has flipped, it might converge due to the symmetry, but on the wrong side of the camera
        // center. Here we check if the sign of theta has flipped during optimization bool
        // theta_flipped = (signbit(r_theta) != signbit(theta)); bool theta_flipped = ((r_theta < 0
        // && theta > 0) || (r_theta > 0 && theta < 0));
        bool theta_flipped = (sign(r_theta) != sign(theta));
        bool theta_converged = ((theta_residual * theta_residual) <= (UNPROJ_CRIT_1 * UNPROJ_CRIT_1));
        bool theta_in_range = ((theta * theta) <= (calib.max_theta * calib.max_theta));
        
        valid &= !theta_flipped && theta_converged && theta_in_range;


#ifdef NON_VALID_TO_NAN
        float3 point = make_float3(nanf(""), nanf(""), nanf(""));
        // bool valid = !theta_flipped && theta_converged && theta_in_range;
        if (valid)
        {
            point = make_float3(pw.x * scale, pw.y * scale, cosf(theta));
            point = point / length(point);
        }
#else
        float3 point = make_float3(pw.x * scale, pw.y * scale, cosf(theta));
        point = point / length(point);
#endif
        return point;
    }

    /**
     * Project a point in space to pixel coordinates
     */
    inline __device__ float2 project(float3 point, IntrinsicsKB calib) {
        bool valid = true;
        float2 out = project(point, calib, valid);

        return out;
    }

    /**
     * Project a point in space to pixel coordinates and set valid to false if the 3D point
     * is out of the cv fisheye model's scope (fov > calib.max_theta)
     */
    inline __device__ float2 project(float3 point, IntrinsicsKB calib, bool& valid) {
#ifdef DEBUG
        int indexIn = blockIdx.x * blockDim.x + threadIdx.x;
        if (indexIn == 0)
            printf("FisheyeKB project\n");
#endif
        float2 uv = make_float2(point.x, point.y);
        float z = point.z;

        float r = length(uv);

        float theta = atan2f(r, z);

        float k1 = calib.ks.x;
        float k2 = calib.ks.y;
        float k3 = calib.ks.z;
        float k4 = calib.ks.w;

        float theta2 = theta * theta;

        float r_theta = fmaf(fmaf(fmaf(fmaf(k4, theta2, k3), theta2, k2), theta2, k1), theta2, 1) * theta;

        bool r_valid = (r > MIN_LIMIT);
        bool z_valid = ((z * z) > (MIN_LIMIT * MIN_LIMIT));
        float inv_r = r_valid ? 1.f / r : z_valid ? 1.f / z : 1.f;

        float cdist = r_theta * inv_r;

        float2 out = uv * cdist * calib.fl + calib.principal;

        bool theta_in_range = ((theta * theta) <= (calib.max_theta * calib.max_theta));
        valid &= ((r_valid || z_valid) && theta_in_range);

#ifdef NON_VALID_TO_NAN
        if (!valid)
        {
            float2 out = make_float2(nanf(""), nanf(""));
        }
#endif

        return out;
    }
};

/**
 * Unproject pixels to the unit sphere using the Kannala and Brandt model with a perspective
 * projection using the identity camera matrix following the OpenCV fisheye model. Code very similar
 * to OpenCV fisheye::undistortPoints
 */
struct FisheyeKBPerspectiveProjection {
    inline __device__ float3 unproject(float2 uv, IntrinsicsKB calib, bool& valid) {
#ifdef DEBUG
        int indexIn = blockIdx.x * blockDim.x + threadIdx.x;
        if (indexIn == 0)
            printf("FisheyeKBPerspectiveProjection unproject\n");
#endif
        // float2 pi = uv;
        float2 pw = (uv - calib.principal) / calib.fl;

        float r_theta = sqrtf(pw.x * pw.x + pw.y * pw.y);
        r_theta = fmin(fmax(0.0f, r_theta), PI * 0.5f);

        float2 result = solve_for_theta(r_theta, calib, MAX_ITER, UNPROJ_CRIT_0);
        float theta = result.x;
        float theta_residual = result.y;

        theta = fmin(fmax(-PI * 0.5f, theta), PI * 0.5f);

        float scale = tanf(theta) / r_theta;

        // theta is monotonously increasing or decreasing depending on the sign of theta. If theta
        // has flipped, it might converge due to the symmetry, but on the wrong side of the camera
        // center. Here we check if the sign of theta has flipped during optimization
        bool theta_flipped = (sign(r_theta) != sign(theta));
        bool theta_converged = ((theta_residual * theta_residual) <= (UNPROJ_CRIT_1 * UNPROJ_CRIT_1));
        bool theta_in_range = ((theta * theta) <= (calib.max_theta * calib.max_theta));

        valid &= !theta_flipped && theta_converged && theta_in_range;

#ifdef NON_VALID_TO_NAN
        float3 point = make_float3(nanf(""), nanf(""), nanf(""));
        // bool valid = !theta_flipped && theta_converged && theta_in_range;
        if (valid)
        {
            float3 point = make_float3(pw.x * scale, pw.y * scale, 1.0f);
            point = point / length(point);
        }
#else
        float3 point = make_float3(pw.x * scale, pw.y * scale, 1.0f);
        point = point / length(point);
#endif

        return point;
    }

    /**
     * Project a point in space to pixel coordinates
     */
    inline __device__ float2 project(float3 point, IntrinsicsKB calib) {
        bool valid = true;
        float2 out = project(point, calib, valid);

        return out;
    }

    /**
     * Project a point in space to pixel coordinates and set valid to false if the 3D point
     * is out of the cv fisheye model's scope (fov > calib.max_theta)
     */
    inline __device__ float2 project(float3 point, IntrinsicsKB calib, bool& valid) {
#ifdef DEBUG
        int indexIn = blockIdx.x * blockDim.x + threadIdx.x;
        if (indexIn == 0)
            printf("FisheyeKBPerspectiveProjection project\n");
#endif
        float z_abs = fabsf(point.z);
        float2 uv = make_float2(point.x / z_abs, point.y / z_abs);

        float z = 1.0f * sign(point.z);

        float r = length(uv);

        float theta = atan2f(r, z);

        float k1 = calib.ks.x;
        float k2 = calib.ks.y;
        float k3 = calib.ks.z;
        float k4 = calib.ks.w;

        float theta2 = theta * theta;

        float r_theta = fmaf(fmaf(fmaf(fmaf(k4, theta2, k3), theta2, k2), theta2, k1), theta2, 1) * theta;

        bool r_valid = (r > MIN_LIMIT);
        float inv_r = r_valid ? 1.f / r : 1.f;

        float cdist = r_theta * inv_r;

        float2 out = uv * cdist * calib.fl + calib.principal;

        bool theta_in_range = ((theta * theta) <= (calib.max_theta * calib.max_theta));
        valid &= theta_in_range;

#ifdef NON_VALID_TO_NAN
        if (!valid)
        {
            float2 out = make_float2(nanf(""), nanf(""));
        }
#endif

        return out;
    }
};