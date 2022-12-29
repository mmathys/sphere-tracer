
#ifndef __DEBUG_H
#define __DEBUG_H

#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>
// #include "scene/obj_kind.h"
// #include "scene/scene_config.h"
#include "../src/geometry.h"

void print_vec(Vec v) {
    printf("[%f, %f, %f]\n", v.x, v.y, v.z);
}

void print_mat(Mat4 M) {
    float *A = M.A;

    for (int i = 0; i < 4; i++) {
        printf("[%.4f, %.4f, %.4f, %.4f]\n", A[i * 4], A[i * 4 + 1], A[i * 4 + 2], A[i * 4 + 3]);
    }
}

void print_avx_vec_f(__m256 v){
    // float* f = (float*)&v;
    float f[8];
    store_avx(v, f);
    printf("%f %f %f %f %f %f %f %f\n", f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
}
void print_avx_vec_d(__m256 v){
    // int* f = (int*)&v;
    int f[8];
    store_avx_int((__m256i)v, f);   // is this correct?
    printf("%d %d %d %d %d %d %d %d\n", f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);

}

void print_avx_vec_int(__m256i v){
    // int* f = (int*)&v;
    int f[8];
    store_avx_int(v, f);
    printf("%d %d %d %d %d %d %d %d\n", f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
}


void print_directions(Vec intersection_point, Vec ray_direction, Vec ray_origin) {
    printf("ray_origin: %f, %f, %f \t intersection point: %f, %f, %f \t ray_direction: %f. %f. %f \n", ray_origin.x,
           ray_origin.y, ray_origin.z, intersection_point.x, intersection_point.y, intersection_point.z,
           ray_direction.x, ray_direction.y, ray_direction.z);

}

void compare_normal(Vec old_normal, Vec new_normal){
    printf("old normal \t new normal \n");
    printf("%f \t %f \n", old_normal.x, new_normal.x);
    printf("%f \t %f \n", old_normal.y, new_normal.y);
    printf("%f \t %f \n", old_normal.z, new_normal.z);
    printf("\n");
}

#endif //TEAM07_DEBUG_H

