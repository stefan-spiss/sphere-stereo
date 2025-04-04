
// #define DEBUG 1
// #define NON_VALID_TO_NAN 1


template <typename Op>
__global__ void reprojectToPanorama(float2* panorama, bool* valid_out, const IntrinsicsKB* calibs, const Rotation* rotations, const float3* translations) {
    int indexIn = blockIdx.x * blockDim.x + threadIdx.x;

    if (indexIn < PANO_ROWS * PANO_COLS) {
        // Get the sphere point corresponding to the current pixel
        float2 pixel = {float(indexIn % PANO_COLS), float(indexIn / PANO_COLS)};
        float phi = (float(pixel.y) + 0.5f) * PI / PANO_ROWS - PI / 2.f;
        float theta = (float(pixel.x) + 0.5f) * 2.f * PI / PANO_COLS - PI;
        float3 unitPointPanorama = 
        {
            cosf(phi) * sinf(theta),
            sinf(phi),
            cosf(phi) * cosf(theta)
        };

        // float2 size_2 = make_float2(float(PANO_COLS) / 2.f, float(PANO_ROWS-1) / 2.f);
        // float theta = (pixel.x - size_2.x) / size_2.x * PI + PI / 2.f;
        // float phi = (pixel.y - size_2.y) / size_2.y * PI / 2.f;
        // float3 unitPointPanorama = 
        // {
        //     -cosf(phi) * cosf(theta),
        //     sinf(phi),
        //     cosf(phi) * sinf(theta)
        // };

        Op camModel;

        for (int referenceIndex= 0; referenceIndex < REFERENCES_COUNT; referenceIndex++) {
            float3 unitInFisheye = matMul3x3(rotations[referenceIndex].r, unitPointPanorama) + translations[referenceIndex];
            
#ifdef DEBUG
            if (indexIn == 0) {
                printf("calibs:\n\t%f %f\n", calibs[referenceIndex].fl.x, calibs[referenceIndex].fl.y);
                printf("\t%f %f\n", calibs[referenceIndex].principal.x, calibs[referenceIndex].principal.y);
                printf("\t%f %f %f %f\n", calibs[referenceIndex].ks.x, calibs[referenceIndex].ks.y, calibs[referenceIndex].ks.z, calibs[referenceIndex].ks.w);
                printf("\t%f\n", calibs[referenceIndex].max_theta);
            }
#endif
            bool valid = true;
            float2 uv = camModel.project(unitInFisheye, calibs[referenceIndex], valid);
            
            panorama[referenceIndex * PANO_ROWS * PANO_COLS + indexIn] = uv;
            if (!valid) {
                panorama[referenceIndex * PANO_ROWS * PANO_COLS + indexIn] = make_float2(-1.f, -1.f);
            }
            valid_out[referenceIndex * PANO_ROWS * PANO_COLS + indexIn] = valid;
        }
    }
}
