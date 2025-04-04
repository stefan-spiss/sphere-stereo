/**
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
Andreas Meuleman and Min H. Kim have developed this software and related documentation (the
"Software"); confidential use in source form of the Software, without modification, is permitted
provided that the following conditions are met: Neither the name of the copyright holder nor the
names of any contributors may be used to endorse or promote products derived from the Software
without specific prior written permission. The use of the software is for Non-Commercial Purposes
only. As used in this Agreement, "Non-Commercial Purpose" means for the purpose of education or
research in a non-commercial organisation only. "Non-Commercial Purpose" excludes, without
limitation, any use of the Software for, as part of, or in any way in connection with a product
(including software) or service which is sold, offered for sale, licensed, leased, published, loaned
or rented. If you require a license for a use excluded by this agreement, please email
[minhkim@kaist.ac.kr]. Warranty: KAIST-VCLAB MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE
SUITABILITY OF THE SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. KAIST-VCLAB
SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR
DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES. Please refer to license.txt for more details.
=======================================================================
File information
-----------------
File newly created to include general functions required for multiple cuda files.
author: Stefan Spiss
=================================================================================
**/

#define PI 3.14159265f

struct Rotation {
    float r[3][3];
};

inline __device__ float3 matMul3x3(const float r[3][3], float3 vect) {
    return make_float3(r[0][0] * vect.x + r[0][1] * vect.y + r[0][2] * vect.z,
                       r[1][0] * vect.x + r[1][1] * vect.y + r[1][2] * vect.z,
                       r[2][0] * vect.x + r[2][1] * vect.y + r[2][2] * vect.z);
}

inline __device__ int sign(float x) { return (x > 0) - (x < 0); }

/**
 * Linear interpolation and type conversion in image.
 * Does not perform out of image boundaries check.
 */
inline __device__ float3 interp(const uchar3* sampled, float2 uv, int columns = COLS) {
    int u1, u2, v1, v2;
    u1 = __float2int_rd(uv.x);
    v1 = __float2int_rd(uv.y);

    u2 = u1 + 1;
    v2 = v1 + 1;

    float w1, w2, w3, w4;
    float u1f = (float)u1;
    float u2f = (float)u2;
    float v1f = (float)v1;
    float v2f = (float)v2;

    w1 = (u2f - uv.x) * (v2f - uv.y);
    w2 = (u2f - uv.x) * (uv.y - v1f);
    w3 = (uv.x - u1f) * (v2f - uv.y);
    w4 = (uv.x - u1f) * (uv.y - v1f);

    float3 p1, p2, p3, p4;
    p1 = uchar3Tofloat3(sampled[v1 * columns + u1]);
    p2 = uchar3Tofloat3(sampled[v2 * columns + u1]);
    p3 = uchar3Tofloat3(sampled[v1 * columns + u2]);
    p4 = uchar3Tofloat3(sampled[v2 * columns + u2]);

    return (w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4);
}

/**
 * Linear interpolation and type conversion in float map.
 * Does not perform out of image boundaries check.
 */
inline __device__ float interpF(const float* sampled, float2 uv, int columns = COLS) {
    int u1, u2, v1, v2;
    u1 = __float2int_rd(uv.x);
    v1 = __float2int_rd(uv.y);

    u2 = u1 + 1;
    v2 = v1 + 1;

    float w1, w2, w3, w4;
    float u1f = (float)u1;
    float u2f = (float)u2;
    float v1f = (float)v1;
    float v2f = (float)v2;

    w1 = (u2f - uv.x) * (v2f - uv.y);
    w2 = (u2f - uv.x) * (uv.y - v1f);
    w3 = (uv.x - u1f) * (v2f - uv.y);
    w4 = (uv.x - u1f) * (uv.y - v1f);

    float p1, p2, p3, p4;
    p1 = (sampled[v1 * columns + u1]);
    p2 = (sampled[v2 * columns + u1]);
    p3 = (sampled[v1 * columns + u2]);
    p4 = (sampled[v2 * columns + u2]);

    return (w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4);
}

/**
 * Linear interpolation and type conversion in float map.
 * Does perform if out of image boundaries and returns the default value if this is the case.
 */
inline __device__ float interpFwithBoundCheck(const float* sampled, float2 uv, int offset,
                                      int columns = COLS, int rows = ROWS, float defaultValue = 0.0f) {
    int u1, u2, v1, v2;
    u1 = __float2int_rd(uv.x);
    v1 = __float2int_rd(uv.y);

    u2 = u1 + 1;
    v2 = v1 + 1;

    bool inside = (u1 >= 0 && u1 < columns && u2 >= 0 && u2 < columns && v1 >= 0 && v1 < rows &&
                   v2 >= 0 && v2 < rows);
    if (!inside) {
        return defaultValue;
    }

    float w1, w2, w3, w4;
    float u1f = (float)u1;
    float u2f = (float)u2;
    float v1f = (float)v1;
    float v2f = (float)v2;

    w1 = (u2f - uv.x) * (v2f - uv.y);
    w2 = (u2f - uv.x) * (uv.y - v1f);
    w3 = (uv.x - u1f) * (v2f - uv.y);
    w4 = (uv.x - u1f) * (uv.y - v1f);

    float p1, p2, p3, p4;
    p1 = (sampled[offset + v1 * columns + u1]);
    p2 = (sampled[offset + v2 * columns + u1]);
    p3 = (sampled[offset + v1 * columns + u2]);
    p4 = (sampled[offset + v2 * columns + u2]);

    return (w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4);
}