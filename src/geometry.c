#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <stdio.h>

#include "scene/obj_kind.h"
#include "geometry.h"
#include "scene/scene_config.h"

// debugging
void print_avx(__m256 x){
    printf("[%f \t %f\t %f\t %f \t %f \t %f\t %f\t %f] \n", x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
}

__m256 load_avx(float *x_ptr){
    __m256 x = _mm256_loadu_ps((const float *) x_ptr);
    return x;
}

__m256 masked_load_avx(float *x_ptr, __m256i mask){
    __m256 x = _mm256_maskload_ps((const float *) x_ptr, mask);
    return x;
}

void store_avx(__m256 x_vec, float *addr){
    _mm256_storeu_ps(addr, x_vec);
}

void masked_store_avx(__m256 x_vec, float *addr, __m256i mask){
    _mm256_maskstore_ps(addr, mask, x_vec);
}

void store_avx_int(__m256i x_vec, int *addr){
    _mm256_storeu_si256((__m256i *)addr, x_vec);
}

// compute masked used for mask_load/mask_store
__m256i compute_mask(int i){
    // pre: i \in [1,7]
    // post: the i right parts should be 1111...1
    switch (i)
    {
    case 0:
        return _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0);
    case 1:
        return _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1);
    case 2:
        return _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
    case 3:
        return _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1);
    case 4:
        return _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1);
    case 5:
        return _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1);
    case 6:
        return _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1);
    case 7:
        return _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1);
    default:
        printf("wrong input for compute_mask\n");
        exit(1);
    }
}

void compute_mask_values(int i, int * out){
    // pre: i \in [1,7]
    // post: the i right parts should be 1111...1
    int k = 0;
    for (; k < i; k++)
    {
        out[k] = -1;
    }
    for (; k < 8; k++)
    {
        out[k] = 0;
    }
}


// vectorized operations

void vec_absV(__m256 x_in, __m256 y_in, __m256 z_in, __m256 *x_out, __m256 *y_out, __m256 *z_out){   
    // mask should be 1000 1000 1000 ... to get rid of the sign bits
    __m256 mask = _mm256_set1_ps(-0.0f);
    *x_out = _mm256_andnot_ps(mask, x_in);
    *y_out = _mm256_andnot_ps(mask, y_in);
    *z_out = _mm256_andnot_ps(mask, z_in);
}

void vec_maxV(__m256 x_in, __m256 y_in, __m256 z_in, __m256 q, __m256 *x_out, __m256 *y_out, __m256 *z_out) {
    *x_out = _mm256_max_ps(x_in, q);
    *y_out = _mm256_max_ps(y_in, q);
    *z_out = _mm256_max_ps(z_in, q);
}

void vec_minV(__m256 x_in, __m256 y_in, __m256 z_in, __m256 q, __m256 *x_out, __m256 *y_out, __m256 *z_out) {
    *x_out = _mm256_min_ps(x_in, q);
    *y_out = _mm256_min_ps(y_in, q);
    *z_out = _mm256_min_ps(z_in, q);
}

void vec_mult(__m256 x_in, __m256 y_in, __m256 z_in, __m256 x2_in, __m256 y2_in, __m256 z2_in, __m256 *x_out, __m256 *y_out, __m256 *z_out){
    *x_out = _mm256_mul_ps(x_in, x2_in);
    *y_out = _mm256_mul_ps(y_in, y2_in);
    *z_out = _mm256_mul_ps(z_in, z2_in);
}

void vec_add(__m256 x_in, __m256 y_in, __m256 z_in, __m256 x2_in, __m256 y2_in, __m256 z2_in, __m256 *x_out, __m256 *y_out, __m256 *z_out){
    *x_out = _mm256_add_ps(x_in, x2_in);
    *y_out = _mm256_add_ps(y_in, y2_in);
    *z_out = _mm256_add_ps(z_in, z2_in);
}

void vec_sub(__m256 x_in, __m256 y_in, __m256 z_in, __m256 x2_in, __m256 y2_in, __m256 z2_in, __m256 *x_out, __m256 *y_out, __m256 *z_out){
    *x_out = _mm256_sub_ps(x_in, x2_in);
    *y_out = _mm256_sub_ps(y_in, y2_in);
    *z_out = _mm256_sub_ps(z_in, z2_in);
}


void vec_smax(__m256 x_in, __m256 y_in, __m256 z_in, float s, __m256 *x_out, __m256 *y_out, __m256 *z_out){
    __m256 scalar = _mm256_set1_ps(s);
    *x_out =_mm256_max_ps (x_in, scalar);
    *y_out =_mm256_max_ps (y_in, scalar);
    *z_out =_mm256_max_ps (z_in, scalar);
}

void vec_sadd(__m256 x_in, __m256 y_in, __m256 z_in, __m256 scalar, __m256 *x_out, __m256 *y_out, __m256 *z_out){
    *x_out = _mm256_add_ps(x_in ,scalar);
    *y_out = _mm256_add_ps(y_in ,scalar);
    *z_out = _mm256_add_ps(z_in ,scalar);
}

void vec_ssub(__m256 x_in, __m256 y_in, __m256 z_in, __m256 scalar, __m256 *x_out, __m256 *y_out, __m256 *z_out){
    *x_out = _mm256_sub_ps(x_in ,scalar);
    *y_out = _mm256_sub_ps(y_in ,scalar);
    *z_out = _mm256_sub_ps(z_in ,scalar);
}

void vec_sdiv(__m256 x_in, __m256 y_in, __m256 z_in, float s, __m256 *x_out, __m256 *y_out, __m256 *z_out){
    __m256 scalar = _mm256_set1_ps(s);
    *x_out = _mm256_div_ps(x_in ,scalar);
    *y_out = _mm256_div_ps(y_in ,scalar);
    *z_out = _mm256_div_ps(z_in ,scalar);
}

void vec_smult(__m256 x_in, __m256 y_in, __m256 z_in, float s, __m256 *x_out, __m256 *y_out, __m256 *z_out){
    __m256 scalar = _mm256_set1_ps(s);
    *x_out = _mm256_mul_ps(x_in ,scalar);
    *y_out = _mm256_mul_ps(y_in ,scalar);
    *z_out = _mm256_mul_ps(z_in ,scalar);
}

void vec_smult2(__m256 x_in, __m256 y_in, __m256 z_in, __m256 s, __m256 *x_out, __m256 *y_out, __m256 *z_out) {
    *x_out = _mm256_mul_ps(x_in, s);
    *y_out = _mm256_mul_ps(y_in, s);
    *z_out = _mm256_mul_ps(z_in, s);
}

void vec_and(__m256 x_in, __m256 y_in, __m256 z_in, __m256 mask, __m256 *x_out, __m256 *y_out, __m256 *z_out) {
    *x_out = _mm256_and_ps(x_in, mask);
    *y_out = _mm256_and_ps(y_in, mask);
    *z_out = _mm256_and_ps(z_in, mask);
}

void vec_scalar_product(__m256 x_in, __m256 y_in, __m256 z_in,__m256 x2_in, __m256 y2_in, __m256 z2_in, __m256 *out){
    __m256 tmp_x = _mm256_mul_ps(x_in, x2_in);
    __m256 tmp_xy = _mm256_fmadd_ps(y_in, y2_in, tmp_x); // x1 * x2 + y1 * y2
    *out = _mm256_fmadd_ps(z_in, z2_in, tmp_xy);
}

void vec_fmadd(__m256 x_in, __m256 y_in, __m256 z_in, __m256 x2_in, __m256 y2_in, __m256 z2_in, __m256 x3_in, __m256 y3_in, __m256 z3_in, __m256 *x_out, __m256 *y_out, __m256 *z_out){
    *x_out = _mm256_fmadd_ps(x_in, x2_in, x3_in);
    *y_out = _mm256_fmadd_ps(y_in, y2_in, y3_in); 
    *z_out = _mm256_fmadd_ps(z_in, z2_in, z3_in); 
}

void vec_squared_norm(__m256 x, __m256 y, __m256 z, __m256 *out){
    vec_scalar_product(x,y,z, x,y,z, out);
}

void vec_normalize(__m256 x_in, __m256 y_in, __m256 z_in, __m256 *x_out, __m256 *y_out, __m256 *z_out){
    __m256 norms;
    vec_squared_norm(x_in, y_in, z_in, &norms);

    __m256 norm_factor = _mm256_sqrt_ps(norms);
    *x_out = _mm256_div_ps(x_in, norm_factor);
    *y_out = _mm256_div_ps(y_in, norm_factor);
    *z_out = _mm256_div_ps(z_in, norm_factor);
}

void vec_rotate(__m256 x_in, __m256 y_in, __m256 z_in, __m256 *x_out, __m256 *y_out, __m256 *z_out,  obj_prop *props, uint32_t idx){
    float *m0 = props->rotation_0;
    float *m1 = props->rotation_1;
    float *m2 = props->rotation_2;
    float *m3 = props->rotation_3;
    float *m4 = props->rotation_4;
    float *m5 = props->rotation_5;
    float *m6 = props->rotation_6;
    float *m7 = props->rotation_7;
    float *m8 = props->rotation_8;  

    // compute x_out = m0*x + m3*y + m6*z
    __m256 vec_m0 = _mm256_loadu_ps(m0 + idx);      // TODO: reuse these, or use new ones?
    __m256 vec_m3 = _mm256_loadu_ps(m3 + idx);
    __m256 vec_m6 = _mm256_loadu_ps(m6 + idx);

    __m256 x_acc = _mm256_mul_ps(vec_m0, x_in);
    x_acc = _mm256_fmadd_ps(vec_m3, y_in, x_acc);
    *x_out = _mm256_fmadd_ps(vec_m6, z_in, x_acc);

    // compute y_out = m1*x + m4*y + m7*z
    __m256 vec_m1 = _mm256_loadu_ps(m1 + idx);
    __m256 vec_m4 = _mm256_loadu_ps(m4 + idx);
    __m256 vec_m7 = _mm256_loadu_ps(m7 + idx);

    __m256 y_acc = _mm256_mul_ps(vec_m1, x_in);
    y_acc = _mm256_fmadd_ps(vec_m4, y_in, y_acc);
    *y_out = _mm256_fmadd_ps(vec_m7, z_in, y_acc);

    // compute z_out = m2*x + m5*y + m8*z
    __m256 vec_m2 = _mm256_loadu_ps(m2 + idx);
    __m256 vec_m5 = _mm256_loadu_ps(m5 + idx);
    __m256 vec_m8 = _mm256_loadu_ps(m8 + idx);

    __m256 z_acc = _mm256_mul_ps(vec_m2, x_in);
    z_acc = _mm256_fmadd_ps(vec_m5, y_in, z_acc);
    *z_out = _mm256_fmadd_ps(vec_m8, z_in, z_acc);

    return;
}

void vec_reflect(__m256 dir_to_origin_x, __m256 dir_to_origin_y, __m256 dir_to_origin_z,
                 __m256 normal_x, __m256 normal_y, __m256 normal_z,
                 __m256 *x_out, __m256 *y_out, __m256 *z_out) {
    // Vec reflected_ray = sub(smult(normal, 2 * scalar_product(direction_to_origin, normal)), direction_to_origin);
    // scalar_product(direction_to_origin, normal))
    __m256 scalar_products;
    vec_scalar_product(dir_to_origin_x, dir_to_origin_y, dir_to_origin_z, normal_x, normal_y, normal_z, &scalar_products);
    // 2 * scalar_product(direction_to_origin, normal))
    scalar_products = _mm256_add_ps(scalar_products, scalar_products);        // times 2
    // smult(normal, 2 * scalar_product(direction_to_origin, normal))
    __m256 n_x;
    __m256 n_y;
    __m256 n_z;
    vec_smult2(normal_x, normal_y, normal_z, scalar_products, &n_x, &n_y, &n_z);
    // sub(smult(normal, 2 * scalar_product(direction_to_origin, normal)), direction_to_origin);
    __m256 reflected_x;
    __m256 reflected_y;
    __m256 reflected_z;
    vec_sub(n_x, n_y, n_z, dir_to_origin_x, dir_to_origin_y, dir_to_origin_z, &reflected_x, &reflected_y, &reflected_z);

    // return normalize(reflected_ray);
    vec_normalize(reflected_x, reflected_y, reflected_z, x_out, y_out, z_out);
}

void vec_pow(__m256 x, __m256 y, __m256 *out) {
    float x_arr[8], y_arr[8], out_arr[8];
    store_avx(x, x_arr);
    store_avx(y, y_arr);
    for(int i = 0; i < 8; i++) {
        out_arr[i] = pow(x_arr[i], y_arr[i]);
    }
    *out = load_avx(out_arr);
}

void ray_vec_rotate(__m256 x_in, __m256 y_in, __m256 z_in, __m256 *x_out, __m256 *y_out, __m256 *z_out,  obj_prop *props, uint32_t idx){
    float *m0 = props->rotation_0;
    float *m1 = props->rotation_1;
    float *m2 = props->rotation_2;
    float *m3 = props->rotation_3;
    float *m4 = props->rotation_4;
    float *m5 = props->rotation_5;
    float *m6 = props->rotation_6;
    float *m7 = props->rotation_7;
    float *m8 = props->rotation_8;

    // compute x_out = m0*x + m3*y + m6*z
    __m256 vec_m0 = _mm256_set1_ps(*(m0 + idx));      // TODO: reuse these, or use new ones?
    __m256 vec_m3 = _mm256_set1_ps(*(m3 + idx));
    __m256 vec_m6 = _mm256_set1_ps(*(m6 + idx));

    __m256 x_acc = _mm256_mul_ps(vec_m0, x_in);
    x_acc = _mm256_fmadd_ps(vec_m3, y_in, x_acc);
    *x_out = _mm256_fmadd_ps(vec_m6, z_in, x_acc);

    // compute y_out = m1*x + m4*y + m7*z
    __m256 vec_m1 = _mm256_set1_ps(*(m1 + idx));
    __m256 vec_m4 = _mm256_set1_ps(*(m4 + idx));
    __m256 vec_m7 = _mm256_set1_ps(*(m7 + idx));

    __m256 y_acc = _mm256_mul_ps(vec_m1, x_in);
    y_acc = _mm256_fmadd_ps(vec_m4, y_in, y_acc);
    *y_out = _mm256_fmadd_ps(vec_m7, z_in, y_acc);

    // compute z_out = m2*x + m5*y + m8*z
    __m256 vec_m2 = _mm256_set1_ps(*(m2 + idx));
    __m256 vec_m5 = _mm256_set1_ps(*(m5 + idx));
    __m256 vec_m8 = _mm256_set1_ps(*(m8 + idx));

    __m256 z_acc = _mm256_mul_ps(vec_m2, x_in);
    z_acc = _mm256_fmadd_ps(vec_m5, y_in, z_acc);
    *z_out = _mm256_fmadd_ps(vec_m8, z_in, z_acc);

    return;
}


// non-vectorized operations

float squared_norm(Vec v){
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

Vec absV(Vec v){
    Vec res;

    res.x = fabs(v.x);
    res.y = fabs(v.y);
    res.z = fabs(v.z);
    return res;
}

Vec maxV(Vec v, float q){
    Vec res;

    res.x = fmaxf(v.x, q);
    res.y = fmaxf(v.y, q);
    res.z = fmaxf(v.z, q);
    return res;
}

Vec add(Vec v1, Vec v2){
    Vec res;

    res.x = v1.x + v2.x;
    res.y = v1.y + v2.y;
    res.z = v1.z + v2.z;
    return res;
}

Vec sub(Vec v1, Vec v2){
    Vec res;

    res.x = v1.x - v2.x;
    res.y = v1.y - v2.y;
    res.z = v1.z - v2.z;

    return res;
}

Vec sadd(Vec v, float s){
    Vec res;

    res.x = v.x + s;
    res.y = v.y + s;
    res.z = v.z + s;
    return res;
}

Vec ssub(Vec v, float s){
    Vec res;

    res.x = v.x - s;
    res.y = v.y - s;
    res.z = v.z - s;
    return res;
}


Vec smult(Vec v1, float s){
    Vec res;

    res.x = v1.x * s;
    res.y = v1.y * s;
    res.z = v1.z * s;

    return res;
}

float scalar_product(Vec a, Vec b){
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec fmadd(Vec a, float b, Vec c){
    Vec mult = smult(a, b);
    return add(mult, c);
    
}

Vec mult(Vec v1, Vec v2){
    Vec res;
    
    res.x = v1.x  * v2.x;
    res.y = v1.y * v2.y;
    res.z = v1.z * v2.z;

    return res;
}

Vec sdiv(Vec v1, float s){
    Vec res;
    
    res.x = v1.x  / s;
    res.y = v1.y  / s;
    res.z = v1.z  / s;

    return res;
}

Vec normalize(Vec x){
    float n = squared_norm(x);

    float factor = 1. / sqrtf(n);
    return smult(x, factor);
}

Vec cross_product(Vec a, Vec b){
    Vec res;
    res.x = a.y * b.z - a.z * b.y;
    res.y = a.z * b.x - a.x * b.z;
    res.z = a.x * b.y - a.y * b.x;

    return res;
}

Vec vec(float x, float y, float z){
    Vec res;
    res.x = x;
    res.y = y;
    res.z = z;
    return res;
}

Vec mult_vec_matrix(Mat4 A, Vec x)
{
    float a, b, c, w;

    a = x.x * A.A[0] + x.y * A.A[0 + 4] + x.z * A.A[0 + 8] + A.A[0 + 12];
    b = x.x * A.A[1] + x.y * A.A[0 + 5] + x.z * A.A[0 + 9] + A.A[0 + 13];
    c = x.x * A.A[2] + x.y * A.A[0 + 6] + x.z * A.A[0 + 10] + A.A[0 + 14];
    w = x.x * A.A[3] + x.y * A.A[0 + 7] + x.z * A.A[0 + 11] + A.A[0 + 15];
    
    Vec res = vec(a/w, b/w, c/w);

    return res;
}

Vec mult_dir_matrix(Mat4 A, Vec x)
{
    float a, b, c;

    a = x.x * A.A[0] + x.y * A.A[0 + 4] + x.z * A.A[0 + 8];
    b = x.x * A.A[1] + x.y * A.A[0 + 5] + x.z * A.A[0 + 9];
    c = x.x * A.A[2] + x.y * A.A[0 + 6] + x.z * A.A[0 + 10];
    
    Vec res = vec(a, b, c);

    return res;
}

void vec_mult_dir_matrix(Mat4 A, __m256 x, __m256 y, __m256 z, __m256 *out_x, __m256 *out_y, __m256 *out_z){
  __m256 A0 = _mm256_set1_ps(A.A[0]);
  __m256 A1 = _mm256_set1_ps(A.A[1]);
  __m256 A2 = _mm256_set1_ps(A.A[2]);
  __m256 A4 = _mm256_set1_ps(A.A[4]);
  __m256 A5 = _mm256_set1_ps(A.A[5]);
  __m256 A6 = _mm256_set1_ps(A.A[6]);
  __m256 A8 = _mm256_set1_ps(A.A[8]);
  __m256 A9 = _mm256_set1_ps(A.A[9]);
  __m256 A10 = _mm256_set1_ps(A.A[10]);

  
  *out_x = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(A0, x),_mm256_mul_ps(A4, y)), _mm256_mul_ps(A8, z));  
  *out_y = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(A1, x),_mm256_mul_ps(A5, y)), _mm256_mul_ps(A9, z));
  *out_z = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(A2, x),_mm256_mul_ps(A6, y)), _mm256_mul_ps(A10, z));  
}

Vec load_vec(float *arr){
    Vec res;
    res.x = arr[0];
    res.y = arr[1];
    res.z = arr[2];
    return res;
}

float deg_to_rad(float angle) { 
    return angle * 0.0174532925199;
}

void compute_rotation_matrix(float x_deg, float y_deg, float z_deg, obj_prop *prop, uint32_t idx){
    // degrees to radians // We need the minus, because we rotate the current point and not the object (therefore we need the inverse rotation)
    float x = deg_to_rad(-x_deg);
    float y = deg_to_rad(-y_deg);
    float z = deg_to_rad(-z_deg);

    prop->rotation_0[idx] = cosf(z) * cosf(y);
    prop->rotation_1[idx] = sinf(z) * cosf(y);
    prop->rotation_2[idx] = -sinf(y);

    prop->rotation_3[idx] = cosf(z) * sinf(y) * sinf(x) - sinf(z) * cosf(x);
    prop->rotation_4[idx] = sinf(z) * sinf(y) * sinf(x) + cosf(z) * cosf(x);
    prop->rotation_5[idx] = cosf(y) * sinf(x);

    prop->rotation_6[idx] = cosf(z) * sinf(y) * cosf(x) + sinf(z) * sinf(x);
    prop->rotation_7[idx] = sinf(z) * sinf(y) * cosf(x) - cosf(z) * sinf(x);
    prop->rotation_8[idx] = cosf(y) * cosf(x);
}

Vec rotate(Vec v,  obj_prop *props, uint32_t idx){
    Vec r;

    r.x = props->rotation_0[idx] * v.x + props->rotation_3[idx] * v.y + props->rotation_6[idx] * v.z;
    r.y = props->rotation_1[idx] * v.x + props->rotation_4[idx] * v.y + props->rotation_7[idx] * v.z;
    r.z = props->rotation_2[idx] * v.x + props->rotation_5[idx] * v.y + props->rotation_8[idx] * v.z;

    return r;
}

void get_camera_matrix(Vec from, Vec euler_angles_deg, Mat4 *M){
    Vec r;

    r.x = deg_to_rad(euler_angles_deg.x);
    r.y = deg_to_rad(euler_angles_deg.y);
    r.z = deg_to_rad(euler_angles_deg.z);

    M->A[0] = cosf(r.z) * cosf(r.y);
    M->A[1] = sinf(r.z) * cosf(r.y);
    M->A[2] = -sinf(r.y);
    M->A[3] = 0;

    M->A[4 + 0] = cosf(r.z) * sinf(r.y) * sinf(r.x) - sinf(r.z) * cosf(r.x);
    M->A[4 + 1] = sinf(r.z) * sinf(r.y) * sinf(r.x) + cosf(r.z) * cosf(r.x);
    M->A[4 + 2] = cosf(r.y) * sinf(r.x);
    M->A[4 + 3] = 0;

    M->A[2*4 + 0] = cosf(r.z) * sinf(r.y) * cosf(r.x) + sinf(r.z) * sinf(r.x);
    M->A[2*4 + 1] = sinf(r.z) * sinf(r.y) * cosf(r.x) - cos(r.z) * sinf(r.x); 
    M->A[2*4 + 2] = cosf(r.y) * cosf(r.x);
    M->A[2*4 + 3] = 0;

    M->A[3*4 + 0] = from.x;
    M->A[3*4 + 1] = from.y;
    M->A[3*4 + 2] = from.z;
    M->A[3*4 + 3] = 1;
}
