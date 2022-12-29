#ifndef __GEOMETRY_H_
#define __GEOMETRY_H_

#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

#include "scene/obj_kind.h"
#include "scene/scene_config.h"

typedef struct Vec{
    float x;
    float y;
    float z;
} Vec;

typedef struct Mat3{
    float A[9];
} Mat3;


typedef struct Mat4{
    float A[16];
} Mat4;

__m256 load_avx(float *x_ptr);

__m256 masked_load_avx(float *x_ptr, __m256i mask);

void store_avx(__m256 x_vec, float *addr);

void masked_store_avx(__m256 x_vec, float *addr, __m256i mask);

void store_avx_int(__m256i x_vec, int *addr);

__m256i compute_mask(int i);

void compute_mask_values(int i, int * out);

// vectorized operations
void vec_absV(__m256 x_in, __m256 y_in, __m256 z_in, __m256 *x_out, __m256 *y_out, __m256 *z_out);

void vec_maxV(__m256 x_in, __m256 y_in, __m256 z_in, __m256 q, __m256 *x_out, __m256 *y_out, __m256 *z_out);

void vec_minV(__m256 x_in, __m256 y_in, __m256 z_in, __m256 q, __m256 *x_out, __m256 *y_out, __m256 *z_out);

void vec_mult(__m256 x_in, __m256 y_in, __m256 z_in, __m256 x2_in, __m256 y2_in, __m256 z2_in, __m256 *x_out, __m256 *y_out, __m256 *z_out);

void vec_add(__m256 x_in, __m256 y_in, __m256 z_in, __m256 x2_in, __m256 y2_in, __m256 z2_in, __m256 *x_out, __m256 *y_out, __m256 *z_out);

void vec_sub(__m256 x_in, __m256 y_in, __m256 z_in, __m256 x2_in, __m256 y2_in, __m256 z2_in, __m256 *x_out, __m256 *y_out, __m256 *z_out);

void vec_smax(__m256 x_in, __m256 y_in, __m256 z_in, float s, __m256 *x_out, __m256 *y_out, __m256 *z_out);

void vec_sadd(__m256 x_in, __m256 y_in, __m256 z_in, __m256 scalar, __m256 *x_out, __m256 *y_out, __m256 *z_out);

void vec_ssub(__m256 x_in, __m256 y_in, __m256 z_in, __m256 scalar, __m256 *x_out, __m256 *y_out, __m256 *z_out);

void vec_sdiv(__m256 x_in, __m256 y_in, __m256 z_in, float s, __m256 *x_out, __m256 *y_out, __m256 *z_out);

void vec_smult(__m256 x_in, __m256 y_in, __m256 z_in, float s, __m256 *x_out, __m256 *y_out, __m256 *z_out);

void vec_smult2(__m256 x_in, __m256 y_in, __m256 z_in, __m256 s, __m256 *x_out, __m256 *y_out, __m256 *z_out);

void vec_and(__m256 x_in, __m256 y_in, __m256 z_in, __m256 mask, __m256 *x_out, __m256 *y_out, __m256 *z_out);

void vec_scalar_product(__m256 x_in, __m256 y_in, __m256 z_in,__m256 x2_in, __m256 y2_in, __m256 z2_in, __m256 *out);

void vec_fmadd(__m256 x_in, __m256 y_in, __m256 z_in, __m256 x2_in, __m256 y2_in, __m256 z2_in, __m256 x3_in, __m256 y3_in, __m256 z3_in, __m256 *x_out, __m256 *y_out, __m256 *z_out);

void vec_squared_norm(__m256 x, __m256 y, __m256 z, __m256 *out);

/* Normalize, such that sqrt(x^2 + y^2 + z^2) = 1.0*/
void vec_normalize(__m256 x_in, __m256 y_in, __m256 z_in, __m256 *x_out, __m256 *y_out, __m256 *z_out);

void vec_rotate(__m256 x_in, __m256 y_in, __m256 z_in, __m256 *x_out, __m256 *y_out, __m256 *z_out,  obj_prop *props, uint32_t idx);

void vec_reflect(__m256 dir_to_origin_x, __m256 dir_to_origin_y, __m256 dir_to_origin_z,
                 __m256 normal_x, __m256 normal_y, __m256 normal_z,
                 __m256 *x_out, __m256 *y_out, __m256 *z_out);

void vec_pow(__m256 x, __m256 y, __m256 *out);


void ray_vec_rotate(__m256 x_in, __m256 y_in, __m256 z_in, __m256 *x_out, __m256 *y_out, __m256 *z_out,  obj_prop *props, uint32_t idx);

void vec_mult_dir_matrix(Mat4 A, __m256 x, __m256 y, __m256 z, __m256 *out_x, __m256 *out_y, __m256 *out_z);

// non vectorized operations
float squared_norm(Vec v);

Vec absV(Vec v);

Vec maxV(Vec v, float q);

Vec mult(Vec v1, Vec v2);

Vec add(Vec v1, Vec v2);

Vec sub(Vec v1, Vec v2);

Vec sadd(Vec v, float s);

Vec ssub(Vec v, float s);

Vec sdiv(Vec v1, float s);

Vec smult(Vec v1, float s);

float scalar_product(Vec a, Vec b);

Vec fmadd(Vec a, float b, Vec c);

Vec normalize(Vec x);

Vec cross_product(Vec a, Vec b);

Vec vec(float x, float y, float z);

Vec mult_vec_matrix(Mat4 A, Vec x);

Vec mult_dir_matrix(Mat4 A, Vec x);

Vec load_vec(float *arr);

float deg_to_rad(float angle);

void compute_rotation_matrix(float x_deg, float y_deg, float z_deg, obj_prop *prop, uint32_t idx);

Vec rotate(Vec v,  obj_prop *props, uint32_t idx);

void get_camera_matrix(Vec euler_angles_deg, Vec from, Mat4 *M);

#endif
