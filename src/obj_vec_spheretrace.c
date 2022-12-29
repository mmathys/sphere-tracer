#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>

#include "spheretrace.h"
#include "obj_vec_spheretrace.h"
#include "scene/obj_kind.h"
#include "scene/scene_config.h"
#include "geometry.h"

float obj_vec_get_distance_sphere(Vec from, scene_config *scene, uint32_t obj_index) {
    Vec center = vec(scene->objs.sphere_props.pos.x[obj_index],
                     scene->objs.sphere_props.pos.y[obj_index],
                     scene->objs.sphere_props.pos.z[obj_index]);
    float radius = scene->objs.sphere_radius[obj_index];

    return sqrtf(squared_norm(sub(from, center))) - radius;
}

__m256 vec_get_distance_sphere(__m256 x_from, __m256 y_from, __m256 z_from, scene_config *scene, uint32_t obj_index) {
    __m256 x_center = load_avx(scene->objs.sphere_props.pos.x + obj_index);
    __m256 y_center = load_avx(scene->objs.sphere_props.pos.y + obj_index);
    __m256 z_center = load_avx(scene->objs.sphere_props.pos.z + obj_index);
    __m256 radius = load_avx(scene->objs.sphere_radius + obj_index);

    vec_sub(x_from, y_from, z_from, x_center, y_center, z_center, &x_from, &y_from, &z_from);
    __m256 res;
    vec_squared_norm(x_from, y_from, z_from, &res);
    res = _mm256_sqrt_ps(res);
    res = _mm256_sub_ps(res, radius);

    return res;
}

__m256 masked_vec_get_distance_sphere(__m256 x_from, __m256 y_from, __m256 z_from, scene_config *scene, uint32_t obj_index, __m256i mask) {
    __m256 x_center = masked_load_avx(scene->objs.sphere_props.pos.x + obj_index, mask);
    __m256 y_center = masked_load_avx(scene->objs.sphere_props.pos.y + obj_index, mask);
    __m256 z_center = masked_load_avx(scene->objs.sphere_props.pos.z + obj_index, mask);
    __m256 radius = masked_load_avx(scene->objs.sphere_radius + obj_index, mask);

    vec_sub(x_from, y_from, z_from, x_center, y_center, z_center, &x_from, &y_from, &z_from);
    __m256 res;
    vec_squared_norm(x_from, y_from, z_from, &res);
    res = _mm256_sqrt_ps(res);
    res = _mm256_sub_ps(res, radius);
    
    return res;
}

__m256 vec_get_distance_torus(__m256 x_from, __m256 y_from, __m256 z_from, scene_config *scene, uint32_t obj_index) {
    __m256 r1 = load_avx(scene->objs.torus_r1 + obj_index);
    __m256 r2 = load_avx(scene->objs.torus_r2 + obj_index);

    __m256 x_center = load_avx(scene->objs.torus_props.pos.x + obj_index);
    __m256 y_center = load_avx(scene->objs.torus_props.pos.y + obj_index);
    __m256 z_center = load_avx(scene->objs.torus_props.pos.z + obj_index);

    // transform "from" to object space
    __m256 x_from_translated;
    __m256 y_from_translated;
    __m256 z_from_translated;

    vec_sub(x_from, y_from, z_from,
            x_center, y_center, z_center,
            &x_from_translated, &y_from_translated, &z_from_translated);

    // rotation
    __m256 x_from_in_obj_space;
    __m256 y_from_in_obj_space;
    __m256 z_from_in_obj_space;

    vec_rotate(x_from_translated, y_from_translated, z_from_translated,
               &x_from_in_obj_space, &y_from_in_obj_space, &z_from_in_obj_space,
               &scene->objs.torus_props, obj_index);

    __m256 x_squared = _mm256_mul_ps(x_from_in_obj_space, x_from_in_obj_space);
    __m256 z_squared = _mm256_mul_ps(z_from_in_obj_space, z_from_in_obj_space);
    __m256 squared_sum = _mm256_add_ps(x_squared, z_squared);
    __m256 tmpx = _mm256_sqrt_ps(squared_sum);
    tmpx = _mm256_sub_ps(tmpx, r1);

    __m256 tmpy = y_from_in_obj_space;

    __m256 squared_tmpx = _mm256_mul_ps(tmpx, tmpx);
    __m256 squared_tmpy = _mm256_mul_ps(tmpy, tmpy);
    __m256 squared_sum_tmp = _mm256_add_ps(squared_tmpx, squared_tmpy);
    __m256 normalized = _mm256_sqrt_ps(squared_sum_tmp);

    __m256 distance = _mm256_sub_ps(normalized, r2);

    return distance;
}

__m256 masked_vec_get_distance_torus(__m256 x_from, __m256 y_from, __m256 z_from, scene_config *scene, uint32_t obj_index, __m256i mask) {
    __m256 r1 = masked_load_avx(scene->objs.torus_r1 + obj_index, mask);
    __m256 r2 = masked_load_avx(scene->objs.torus_r2 + obj_index, mask);

    __m256 x_center = masked_load_avx(scene->objs.torus_props.pos.x + obj_index, mask);
    __m256 y_center = masked_load_avx(scene->objs.torus_props.pos.y + obj_index, mask);
    __m256 z_center = masked_load_avx(scene->objs.torus_props.pos.z + obj_index, mask);

    // transform "from" to object space
    __m256 x_from_translated;
    __m256 y_from_translated;
    __m256 z_from_translated;

    vec_sub(x_from, y_from, z_from,
            x_center, y_center, z_center,
            &x_from_translated, &y_from_translated, &z_from_translated);

    // rotation
    __m256 x_from_in_obj_space;
    __m256 y_from_in_obj_space;
    __m256 z_from_in_obj_space;

    vec_rotate(x_from_translated, y_from_translated, z_from_translated,
               &x_from_in_obj_space, &y_from_in_obj_space, &z_from_in_obj_space,
               &scene->objs.torus_props, obj_index);

    __m256 x_squared = _mm256_mul_ps(x_from_in_obj_space, x_from_in_obj_space);
    __m256 z_squared = _mm256_mul_ps(z_from_in_obj_space, z_from_in_obj_space);
    __m256 squared_sum = _mm256_add_ps(x_squared, z_squared);
    __m256 tmpx = _mm256_sqrt_ps(squared_sum);
    tmpx = _mm256_sub_ps(tmpx, r1);

    __m256 tmpy = y_from_in_obj_space;

    __m256 squared_tmpx = _mm256_mul_ps(tmpx, tmpx);
    __m256 squared_tmpy = _mm256_mul_ps(tmpy, tmpy);
    __m256 squared_sum_tmp = _mm256_add_ps(squared_tmpx, squared_tmpy);
    __m256 normalized = _mm256_sqrt_ps(squared_sum_tmp);

    __m256 distance = _mm256_sub_ps(normalized, r2);

    return distance;
}

__m256 vec_get_distance_plane(__m256 x_from, __m256 y_from, __m256 z_from, scene_config *scene, uint32_t obj_index) {
    __m256 x_normal = load_avx(scene->objs.plane_normals.x + obj_index);
    __m256 y_normal = load_avx(scene->objs.plane_normals.y + obj_index);
    __m256 z_normal = load_avx(scene->objs.plane_normals.z + obj_index);

    __m256 x_point = load_avx(scene->objs.plane_props.pos.x + obj_index);
    __m256 y_point = load_avx(scene->objs.plane_props.pos.y + obj_index);
    __m256 z_point = load_avx(scene->objs.plane_props.pos.z + obj_index);

    __m256 x_sub;
    __m256 y_sub;
    __m256 z_sub;
    __m256 res;

    vec_sub(x_from, y_from, z_from, x_point, y_point, z_point, &x_sub, &y_sub, &z_sub);
    vec_scalar_product(x_normal, y_normal, z_normal, x_sub, y_sub, z_sub, &res);

    return res;
}

__m256 masked_vec_get_distance_plane(__m256 x_from, __m256 y_from, __m256 z_from, scene_config *scene, uint32_t obj_index, __m256i mask) {
    __m256 x_normal = masked_load_avx(scene->objs.plane_normals.x + obj_index, mask);
    __m256 y_normal = masked_load_avx(scene->objs.plane_normals.y + obj_index, mask);
    __m256 z_normal = masked_load_avx(scene->objs.plane_normals.z + obj_index, mask);

    __m256 x_point = masked_load_avx(scene->objs.plane_props.pos.x + obj_index, mask);
    __m256 y_point = masked_load_avx(scene->objs.plane_props.pos.y + obj_index, mask);
    __m256 z_point = masked_load_avx(scene->objs.plane_props.pos.z + obj_index, mask);

    __m256 x_sub;
    __m256 y_sub;
    __m256 z_sub;
    __m256 res;

    vec_sub(x_from, y_from, z_from, x_point, y_point, z_point, &x_sub, &y_sub, &z_sub);
    vec_scalar_product(x_normal, y_normal, z_normal, x_sub, y_sub, z_sub, &res);

    return res;
}

__m256 vec_get_distance_box(__m256 x_from, __m256 y_from, __m256 z_from, scene_config *scene, uint32_t obj_index) {
    __m256 x_pos = load_avx(scene->objs.box_props.pos.x + obj_index);
    __m256 y_pos = load_avx(scene->objs.box_props.pos.y + obj_index);
    __m256 z_pos = load_avx(scene->objs.box_props.pos.z + obj_index);
    // from = from - pos
    vec_sub(x_from, y_from, z_from, x_pos, y_pos, z_pos, &x_from, &y_from, &z_from);
    // from = rotate(from)
    vec_rotate(x_from, y_from, z_from, &x_from, &y_from, &z_from, &scene->objs.box_props, obj_index);
    // from = absV(from)
    vec_absV(x_from, y_from, z_from, &x_from, &y_from, &z_from);

    // q = from - extents
    __m256 x_extents = load_avx(scene->objs.box_extents.x + obj_index);
    __m256 y_extents = load_avx(scene->objs.box_extents.y + obj_index);
    __m256 z_extents = load_avx(scene->objs.box_extents.z + obj_index);
    __m256 x_q;
    __m256 y_q;
    __m256 z_q;
    vec_sub(x_from, y_from, z_from, x_extents, y_extents, z_extents, &x_q, &y_q, &z_q);

    // a = maxV(q, 0.0)
    __m256 x_a;
    __m256 y_a;
    __m256 z_a;
    __m256 zero = _mm256_setzero_ps();
    vec_maxV(x_q, y_q, z_q, zero, &x_a, &y_a, &z_a);

    // b = squared_norm(a)
    __m256 b;
    vec_squared_norm(x_a, y_a, z_a, &b);

    // b = sqrtf(b)
    b = _mm256_sqrt_ps(b);

    // c = max(q.x, max(q.y, q.z)) = max(q.x, q.y, q.z)
    __m256 c;
    c = _mm256_max_ps(y_q, z_q);
    c = _mm256_max_ps(x_q, c);
    c = _mm256_min_ps(c, zero);

    __m256 res = _mm256_add_ps(b, c);
    
    return res;
}

__m256 masked_vec_get_distance_box(__m256 x_from, __m256 y_from, __m256 z_from, scene_config *scene, uint32_t obj_index, __m256i mask) {
    __m256 x_pos = masked_load_avx(scene->objs.box_props.pos.x + obj_index, mask);
    __m256 y_pos = masked_load_avx(scene->objs.box_props.pos.y + obj_index, mask);
    __m256 z_pos = masked_load_avx(scene->objs.box_props.pos.z + obj_index, mask);
    // from = from - pos
    vec_sub(x_from, y_from, z_from, x_pos, y_pos, z_pos, &x_from, &y_from, &z_from);
    // from = rotate(from)
    vec_rotate(x_from, y_from, z_from, &x_from, &y_from, &z_from, &scene->objs.box_props, obj_index);
    // from = absV(from)
    vec_absV(x_from, y_from, z_from, &x_from, &y_from, &z_from);

    // q = from - extents
    __m256 x_extents = masked_load_avx(scene->objs.box_extents.x + obj_index, mask);
    __m256 y_extents = masked_load_avx(scene->objs.box_extents.y + obj_index, mask);
    __m256 z_extents = masked_load_avx(scene->objs.box_extents.z + obj_index, mask);
    __m256 x_q;
    __m256 y_q;
    __m256 z_q;
    vec_sub(x_from, y_from, z_from, x_extents, y_extents, z_extents, &x_q, &y_q, &z_q);

    // a = maxV(q, 0.0)
    __m256 x_a;
    __m256 y_a;
    __m256 z_a;
    __m256 zero = _mm256_setzero_ps();
    vec_maxV(x_q, y_q, z_q, zero, &x_a, &y_a, &z_a);

    // b = squared_norm(a)
    __m256 b;
    vec_squared_norm(x_a, y_a, z_a, &b);

    // b = sqrtf(b)
    b = _mm256_sqrt_ps(b);

    // c = max(q.x, max(q.y, q.z)) = max(q.x, q.y, q.z)
    __m256 c;
    c = _mm256_max_ps(y_q, z_q);
    c = _mm256_max_ps(x_q, c);
    c = _mm256_min_ps(c, zero);

    __m256 res = _mm256_add_ps(b, c);
    return res;
}

__m256 vec_get_distance_boxframe(__m256 x_from, __m256 y_from, __m256 z_from, scene_config *scene, uint32_t object_index) {
    __m256 x_pos = load_avx(scene->objs.boxframe_props.pos.x + object_index);
    __m256 y_pos = load_avx(scene->objs.boxframe_props.pos.y + object_index);
    __m256 z_pos = load_avx(scene->objs.boxframe_props.pos.z + object_index);
    __m256 x_extents = load_avx(scene->objs.boxframe_extents.x + object_index);
    __m256 y_extents = load_avx(scene->objs.boxframe_extents.y + object_index);
    __m256 z_extents = load_avx(scene->objs.boxframe_extents.z + object_index);
    __m256 thickness = load_avx(scene->objs.boxframe_thickness + object_index);

    //translation
    vec_sub(x_from, y_from, z_from, x_pos, y_pos, z_pos, &x_from, &y_from, &z_from);
    //rotation
    vec_rotate(x_from, y_from, z_from, &x_from, &y_from, &z_from, &(scene->objs.boxframe_props), object_index);
    // absV(p)
    vec_absV(x_from, y_from, z_from, &x_from, &y_from, &z_from);
    // sub(absV(p), extents)
    vec_sub(x_from, y_from, z_from, x_extents, y_extents, z_extents, &x_from, &y_from, &z_from);

    // sadd(p, e)
    __m256 q_x, q_y, q_z;
    vec_sadd(x_from, y_from, z_from, thickness, &q_x, &q_y, &q_z);

    // absV(sadd(p, e))
    vec_absV(q_x, q_y, q_z, &q_x, &q_y, &q_z);

    // ssub(absV(sadd(p, e)), e);
    vec_ssub(q_x, q_y, q_z, thickness, &q_x, &q_y, &q_z);

    __m256 zero = _mm256_set1_ps(0.0f);

    // last line of return
    __m256 temp_x, temp_y, temp_z;
    vec_maxV(q_x, q_y, z_from, zero, &temp_x, &temp_y, &temp_z);
    __m256 temp_squared_norm;
    vec_squared_norm(temp_x, temp_y, temp_z, &temp_squared_norm);

    __m256 temp_sqrt = _mm256_sqrt_ps (temp_squared_norm);
    __m256 temp_RHS = _mm256_min_ps(_mm256_max_ps(q_x, _mm256_max_ps(q_y, z_from)), zero);
    __m256 temp_bottom = _mm256_add_ps(temp_sqrt, temp_RHS);

    // middle line of return
    vec_maxV(q_x, y_from, q_z, zero, &temp_x, &temp_y, &temp_z);
    vec_squared_norm(temp_x, temp_y, temp_z, &temp_squared_norm);

    temp_sqrt = _mm256_sqrt_ps(temp_squared_norm);
    temp_RHS = _mm256_min_ps(_mm256_max_ps(q_x, _mm256_max_ps(y_from, q_z)), zero);
    __m256 temp_middle = _mm256_add_ps(temp_sqrt, temp_RHS);

    // top most line of return
    vec_maxV(x_from, q_y, q_z, zero, &temp_x, &temp_y, &temp_z);
    vec_squared_norm(temp_x, temp_y, temp_z, &temp_squared_norm);

    temp_sqrt = _mm256_sqrt_ps(temp_squared_norm);
    temp_RHS = _mm256_min_ps(_mm256_max_ps (x_from, _mm256_max_ps(q_y, q_z)), zero);
    __m256 temp_top = _mm256_add_ps(temp_sqrt, temp_RHS);

    // combine
    __m256 result = _mm256_min_ps(_mm256_min_ps(temp_top, temp_middle), temp_bottom);
    
    return result;
}

__m256 masked_vec_get_distance_boxframe(__m256 x_from, __m256 y_from, __m256 z_from, scene_config *scene, uint32_t object_index, __m256i mask) {
    __m256 x_pos = masked_load_avx(scene->objs.boxframe_props.pos.x + object_index, mask);
    __m256 y_pos = masked_load_avx(scene->objs.boxframe_props.pos.y + object_index, mask);
    __m256 z_pos = masked_load_avx(scene->objs.boxframe_props.pos.z + object_index, mask);
    __m256 x_extents = masked_load_avx(scene->objs.boxframe_extents.x + object_index, mask);
    __m256 y_extents = masked_load_avx(scene->objs.boxframe_extents.y + object_index, mask);
    __m256 z_extents = masked_load_avx(scene->objs.boxframe_extents.z + object_index, mask);
    __m256 thickness = masked_load_avx(scene->objs.boxframe_thickness + object_index, mask);

    //translation
    vec_sub(x_from, y_from, z_from, x_pos, y_pos, z_pos, &x_from, &y_from, &z_from);
    //rotation
    vec_rotate(x_from, y_from, z_from, &x_from, &y_from, &z_from, &(scene->objs.boxframe_props), object_index);
    // absV(p)
    vec_absV(x_from, y_from, z_from, &x_from, &y_from, &z_from);
    // sub(absV(p), extents)
    vec_sub(x_from, y_from, z_from, x_extents, y_extents, z_extents, &x_from, &y_from, &z_from);

    // sadd(p, e)
    __m256 q_x, q_y, q_z;
    vec_sadd(x_from, y_from, z_from, thickness, &q_x, &q_y, &q_z);

    // absV(sadd(p, e))
    vec_absV(q_x, q_y, q_z, &q_x, &q_y, &q_z);

    // ssub(absV(sadd(p, e)), e);
    vec_ssub(q_x, q_y, q_z, thickness, &q_x, &q_y, &q_z);

    __m256 zero = _mm256_set1_ps(0.0f);

    // last line of return
    __m256 temp_x, temp_y, temp_z;
    vec_maxV(q_x, q_y, z_from, zero, &temp_x, &temp_y, &temp_z);
    __m256 temp_squared_norm;
    vec_squared_norm(temp_x, temp_y, temp_z, &temp_squared_norm);

    __m256 temp_sqrt = _mm256_sqrt_ps (temp_squared_norm);
    __m256 temp_RHS = _mm256_min_ps(_mm256_max_ps(q_x, _mm256_max_ps(q_y, z_from)), zero);
    __m256 temp_bottom = _mm256_add_ps(temp_sqrt, temp_RHS);

    // middle line of return
    vec_maxV(q_x, y_from, q_z, zero, &temp_x, &temp_y, &temp_z);
    vec_squared_norm(temp_x, temp_y, temp_z, &temp_squared_norm);

    temp_sqrt = _mm256_sqrt_ps(temp_squared_norm);
    temp_RHS = _mm256_min_ps(_mm256_max_ps(q_x, _mm256_max_ps(y_from, q_z)), zero);
    __m256 temp_middle = _mm256_add_ps(temp_sqrt, temp_RHS);

    // top most line of return
    vec_maxV(x_from, q_y, q_z, zero, &temp_x, &temp_y, &temp_z);
    vec_squared_norm(temp_x, temp_y, temp_z, &temp_squared_norm);

    temp_sqrt = _mm256_sqrt_ps(temp_squared_norm);
    temp_RHS = _mm256_min_ps(_mm256_max_ps (x_from, _mm256_max_ps(q_y, q_z)), zero);
    __m256 temp_top = _mm256_add_ps(temp_sqrt, temp_RHS);

    // combine
    __m256 result = _mm256_min_ps(_mm256_min_ps(temp_top, temp_middle), temp_bottom);

    return result;
}

/* reflect the incoming vector with respect to the surface normal*/
Vec reflect(Vec direction_to_origin, Vec normal) {
    Vec reflected_ray = sub(smult(normal, 2 * scalar_product(direction_to_origin, normal)), direction_to_origin);
    return normalize(reflected_ray);
}

/* Compute the surface normal at a certain point */
Vec compute_normal(Vec intersection_point, obj_kind kind, uint32_t object_index, scene_config *scene) {
    float delta = 10e-5;
    Vec d1 = vec(delta, 0, 0);
    Vec d2 = vec(0, delta, 0);
    Vec d3 = vec(0, 0, delta);

    Vec p1 = add(intersection_point, d1);
    Vec p2 = add(intersection_point, d2);
    Vec p3 = add(intersection_point, d3);

    Vec pm1 = sub(intersection_point, d1);
    Vec pm2 = sub(intersection_point, d2);
    Vec pm3 = sub(intersection_point, d3);

    float x, y, z;
    x = get_distance_single_point(scene, kind, object_index, p1) - get_distance_single_point(scene, kind, object_index, pm1);
    y = get_distance_single_point(scene, kind, object_index, p2) - get_distance_single_point(scene, kind, object_index, pm2);
    z = get_distance_single_point(scene, kind, object_index, p3) - get_distance_single_point(scene, kind, object_index, pm3);
    Vec n = vec(x, y, z);

    return normalize(n);
}

float vec_get_min(__m256 vec, int *idx) {
    // swaps top and bottom halves
    __m256 swapped_halves = _mm256_permute2f128_ps(vec, vec, 1); 
    __m256 min_values = _mm256_min_ps(vec, swapped_halves);
    // minimum is in first half of last computation
    __m256 swapped_lower_half = _mm256_permute_ps(min_values, 0b01001110);
    min_values = _mm256_min_ps(min_values, swapped_lower_half);
    // minimum is now one of the lower two values
    __m256 min = _mm256_permute_ps(min_values, 0b10110001); 
    min_values = _mm256_min_ps(min, min_values);

    __m256 mask  = _mm256_cmp_ps(vec, min_values, 0);
    
    *idx  = _tzcnt_u32(_mm256_movemask_ps(mask));

    return _mm256_cvtss_f32(min_values);
}

float masked_vec_get_min(__m256 vec, __m256i mask, int *idx) {
    __m256 infinity = _mm256_set1_ps(FLOAT_INFINITY);
    __m256 masked_vec = _mm256_blendv_ps(infinity, vec, (__m256) mask);

    return vec_get_min(masked_vec, idx);
}

inline int obj_vec_update_min_dist(Vec from, __m256 x_from, __m256 y_from, __m256 z_from, 
                            scene_config *scene,
                            int N, 
                            __m256(*obj_vec_distance_func)(__m256 x_from, __m256 y_from, __m256 z_from, scene_config *scene, uint32_t obj_index),
                            __m256(*mask_obj_vec_distance_func)(__m256 x_from, __m256 y_from, __m256 z_from, scene_config *scene, uint32_t obj_index, __m256i mask),
                            float t_times_threshold, uint32_t *obj_hit_index, obj_prop **min_dist_obj_prop, obj_kind *min_obj_kind, float *min_distance, 
                            float cutoff, obj_kind kind){

    obj_prop *kind_props = get_props(kind, scene);
    __m256i kind_mask = get_mask(kind, scene);
    uint32_t kind_remaining = get_remaining(kind, scene);

    int i;
    for(i = 0; i < (int) N-7; i+=8) {
        __m256 dists = obj_vec_distance_func(x_from, y_from, z_from, scene, i);
        int min_idx;
        float d = vec_get_min(dists, &min_idx);

        if(d < *min_distance){
            *min_distance = d;
        }

        if(d <= t_times_threshold){
            *min_dist_obj_prop = kind_props;
            *obj_hit_index = i + min_idx;
            *min_obj_kind = kind;

            return 1;
        }
    }


    if((int) kind_remaining > cutoff){
        __m256 dists = mask_obj_vec_distance_func(x_from, y_from, z_from, scene, i, kind_mask);
        int min_idx;
        float d = masked_vec_get_min(dists, kind_mask, &min_idx);

        if(d < *min_distance){
            *min_distance = d;
        }

        if(d <= t_times_threshold){
            *min_dist_obj_prop = kind_props;
            *obj_hit_index = i + min_idx;
            *min_obj_kind = kind;
            return 1;
        }
    }
    else if(kind_remaining > 0){
        for(; i < (int)N; i++){
            float d = get_distance_single_point(scene, kind, i, from); // get_distance_plane(from, scene, i);
            if(d < *min_distance){
                *min_distance = d;
            }

            if(d <= t_times_threshold){
                *min_dist_obj_prop = kind_props;
                *obj_hit_index = i;
                *min_obj_kind = kind;
                return 1;
            }
        }
    }
    return 0;
}

inline int obj_vec_update_min_dist_shadow(Vec from, __m256 x_from, __m256 y_from, __m256 z_from, 
                            scene_config *scene,
                            int N, 
                            __m256(*obj_vec_distance_func)(__m256 x_from, __m256 y_from, __m256 z_from, scene_config *scene, uint32_t obj_index),
                            __m256(*mask_obj_vec_distance_func)(__m256 x_from, __m256 y_from, __m256 z_from, scene_config *scene, uint32_t obj_index, __m256i mask),
                            float t_times_threshold, float *min_distance, float cutoff, obj_kind kind){
    
    
    __m256i kind_mask = get_mask(kind, scene);
    uint32_t kind_remaining = get_remaining(kind, scene);

    int i;
    for(i = 0; i < (int)N-7; i+=8) {
        __m256 dists = obj_vec_distance_func(x_from, y_from, z_from, scene, i);
        
        int min_idx;
        float d = vec_get_min(dists, &min_idx);

        if(d < *min_distance){
            *min_distance = d;
        }

        if(d <= t_times_threshold){
            return 1;
        }
    }

    if((int) kind_remaining > cutoff) {
        // mask = _mm256_loadu_si256((const __m256i_u *) scene->objs.sphere_mask_values);
        // masked_vec_get_distance_sphere(x_from, y_from, z_from, scene, i, ds, mask);
        __m256 dists = mask_obj_vec_distance_func(x_from, y_from, z_from, scene, i, kind_mask);
                
        int min_idx;
        float d = masked_vec_get_min(dists, kind_mask, &min_idx);

        if(d < *min_distance){
            *min_distance = d;
        }

        if(d <= t_times_threshold){
            return 1;
        }
    }
    else if((int) kind_remaining > 0) {
        for(; i < (int) N; i++){
            float d = get_distance_single_point(scene, kind, i, from);

            if(d < *min_distance){
                *min_distance = d;
            }

            if(d <= t_times_threshold){
                return 1;
            }
        }
    }

    return 0;
}

uint32_t sphere_trace_shadow(
        Vec ray_origin,
        Vec ray_direction,
        float max_dist,
        scene_config *scene
) {
    float threshold = 10e-5;
    float t = threshold;

    int cutoff = 4;     // if number of remaining shapes <= cutoff, then compute without vector instructions

    while (t < max_dist) {
        float min_distance = FLOAT_INFINITY;
        float t_times_threshold = t * threshold;

        Vec from = fmadd(ray_direction, t, ray_origin);

        // we allocate only 3 registers (3x from)
        __m256 x_from = _mm256_set1_ps(from.x);
        __m256 y_from = _mm256_set1_ps(from.y);
        __m256 z_from = _mm256_set1_ps(from.z);


        if (obj_vec_update_min_dist_shadow(from, x_from, y_from, z_from, scene,
            scene->objs.plane_len, vec_get_distance_plane, masked_vec_get_distance_plane, 
            t_times_threshold, &min_distance, cutoff, Plane)){
            return 1;
        }

        if (obj_vec_update_min_dist_shadow(from, x_from, y_from, z_from, scene,
            scene->objs.sphere_len, vec_get_distance_sphere, masked_vec_get_distance_sphere, 
            t_times_threshold, &min_distance, cutoff, Sphere)){
            return 1;
        }

        if (obj_vec_update_min_dist_shadow(from, x_from, y_from, z_from, scene,
            scene->objs.box_len, vec_get_distance_box, masked_vec_get_distance_box, 
            t_times_threshold, &min_distance, cutoff, Box)){
            return 1;
        }

        if (obj_vec_update_min_dist_shadow(from, x_from, y_from, z_from, scene,
            scene->objs.boxframe_len, vec_get_distance_boxframe, masked_vec_get_distance_boxframe, 
            t_times_threshold, &min_distance, cutoff, Boxframe)){
            return 1;
        }

        if (obj_vec_update_min_dist_shadow(from, x_from, y_from, z_from, scene,
            scene->objs.torus_len, vec_get_distance_torus, masked_vec_get_distance_torus, 
            t_times_threshold, &min_distance, cutoff, Torus)){
            return 1;
        }

        t += min_distance;
    }
    return 0;
}


Vec shade(
        Vec ray_origin,
        Vec ray_direction,
        float t,
        obj_prop *obj_prop,
        uint32_t obj_index,
        Vec n,
        scene_config *scene
) {

    float ks = obj_prop->reflection[obj_index];
    float kd = 1 - ks;

    float ns = obj_prop->shininess[obj_index]; //shininess

    // intersection point
    Vec p = fmadd(ray_direction, t, ray_origin);

    // viewing direction
    Vec viewing_direction = smult(ray_direction, (-1));

    Vec obj_color = vec(obj_prop->color.x[obj_index],
                        obj_prop->color.y[obj_index],
                        obj_prop->color.z[obj_index]);

    Vec R = vec(0, 0, 0);

    float lights_count = scene->lights.len;
    float *lights_pos = scene->lights.pos;
    float *lights_color = scene->lights.emission;

    Vec diffuse_contribution = vec(0, 0, 0);
    Vec specular_contribution = vec(0, 0, 0);

    for (int i = 0; i < lights_count; i++) {
        Vec light_i = load_vec(lights_pos + 3 * i);
        Vec light_dir = sub(light_i, p);

        if (scalar_product(light_dir, n) > 0) {
            float dist2 = squared_norm(light_dir);
            light_dir = normalize(light_dir);

            // phong: R = (N*L)*N - L
            Vec reflected_direction = reflect(light_dir, n);

            int shadow = 1 - sphere_trace_shadow(p, light_dir, sqrt(dist2), scene);

            float dot = scalar_product(light_dir, n);

            Vec light_i_color = load_vec(lights_color + 3 * i);
            Vec light_i_intensity = smult(light_i_color, shadow / (4 * M_PI * dist2));

            Vec light_i_diffuse_contribution = smult(mult(obj_color, light_i_intensity), dot);
            Vec light_i_specular_contribution = smult(light_i_intensity,
                                                      pow(fmax(scalar_product(reflected_direction, viewing_direction),
                                                               0), ns));
            /* sum up contributions of all light sources */
            diffuse_contribution = add(diffuse_contribution, light_i_diffuse_contribution);
            specular_contribution = add(specular_contribution, light_i_specular_contribution);
        }
    }

    // R = kd * (diffuse) + ks * (specular) 
    R = add(smult(diffuse_contribution, kd), smult(specular_contribution, ks));

    return R;
}

Vec sphere_trace(
        Vec ray_origin,
        Vec ray_direction,
        scene_config *scene,
        uint32_t depth
) {
    uint32_t max_depth = 3;
    float threshold = 10e-6;
    float max_dist = 100;
    // maximum number of antialias contributions per object, set to 0xffffffff for no limit
    float t = threshold;

    uint32_t num_steps = 0;

    uint32_t obj_hit_index = 0;
    obj_prop *min_dist_obj_prop = NULL;

    Vec pixel_color = vec(0, 0, 0);
    float remaining_pixel_coverage = 1.0f;

    int cutoff = 4; // if number of remaining shapes <= cutoff, then compute without vector instructions

    while (t < max_dist) {
        float min_distance = FLOAT_INFINITY;

        float t_times_threshold = t * threshold;

        Vec from = fmadd(ray_direction, t, ray_origin);

        // we allocate only 3 registers (3x from)
        __m256 x_from = _mm256_set1_ps(from.x);
        __m256 y_from = _mm256_set1_ps(from.y);
        __m256 z_from = _mm256_set1_ps(from.z);

        obj_kind min_obj_kind;
        
        if (obj_vec_update_min_dist(from, x_from, y_from, z_from, scene,
            scene->objs.plane_len, vec_get_distance_plane, masked_vec_get_distance_plane, 
            t_times_threshold, &obj_hit_index, &min_dist_obj_prop, &min_obj_kind, &min_distance, cutoff, Plane)){
            
            goto intersection_found;
        }

        if (obj_vec_update_min_dist(from, x_from, y_from, z_from, scene,
            scene->objs.sphere_len, vec_get_distance_sphere, masked_vec_get_distance_sphere, 
            t_times_threshold, &obj_hit_index, &min_dist_obj_prop, &min_obj_kind, &min_distance, cutoff, Sphere)){
            
            goto intersection_found;
        }

        if (obj_vec_update_min_dist(from, x_from, y_from, z_from, scene,
            scene->objs.box_len, vec_get_distance_box, masked_vec_get_distance_box, 
            t_times_threshold, &obj_hit_index, &min_dist_obj_prop, &min_obj_kind, &min_distance, cutoff, Box)){
        
            goto intersection_found;
        }

        if (obj_vec_update_min_dist(from, x_from, y_from, z_from, scene,
            scene->objs.boxframe_len, vec_get_distance_boxframe, masked_vec_get_distance_boxframe, 
            t_times_threshold, &obj_hit_index, &min_dist_obj_prop, &min_obj_kind, &min_distance, cutoff, Boxframe)){
        
            goto intersection_found;
        }

        if (obj_vec_update_min_dist(from, x_from, y_from, z_from, scene,
            scene->objs.torus_len, vec_get_distance_torus, masked_vec_get_distance_torus, 
            t_times_threshold, &obj_hit_index, &min_dist_obj_prop, &min_obj_kind, &min_distance, cutoff, Torus)){
        
            goto intersection_found;
        }

intersection_found:
        if (min_distance <= t_times_threshold) {
		  
		    Vec intersection_point = fmadd(ray_direction, t, ray_origin);
            Vec normal = compute_normal(intersection_point, min_obj_kind, obj_hit_index, scene);
            Vec pixel = shade(ray_origin, ray_direction, t, min_dist_obj_prop, obj_hit_index, normal, scene);
            float r = min_dist_obj_prop->reflection[obj_hit_index];

            // compute reflection component
            if (depth < max_depth && r > 0) {
                // reflect the ray using phong formula
                Vec secondary_ray_direction = reflect(smult(ray_direction, -1), normal);

                Vec secondary_ray_color = sphere_trace(intersection_point, secondary_ray_direction, scene, depth + 1);
                pixel = add(smult(secondary_ray_color, r), smult(pixel, 1.0 - r));
            }
            pixel_color = add(pixel_color, smult(pixel, remaining_pixel_coverage));

            return pixel_color;
        }

        t += min_distance;
        num_steps++;
    }

    return pixel_color;
}

// returns list of vec3fs
void obj_vec_render(Vec *buffer,
            const uint32_t width,
            const uint32_t height,
            scene_config *scene) {

    float fov = scene->cam_config.fov;

    Vec camera_position = vec(scene->cam_config.posx, scene->cam_config.posy, scene->cam_config.posz);
    Vec camera_rotation = vec(scene->cam_config.rotx, scene->cam_config.roty, scene->cam_config.rotz);

    const float ratio = width / (float) (height);

    float angle = tan(deg_to_rad(fov * 0.5));

    Vec zeros = vec(0, 0, 0);

    Mat4 cam_to_world;
    get_camera_matrix(camera_position, camera_rotation, &cam_to_world);

    Vec ray_origin = mult_vec_matrix(cam_to_world, zeros);
    

    uint32_t supersample_window = 2;

    uint32_t supersample_width = supersample_window * width;
    uint32_t supersample_height = supersample_window * height;

    float ratio_times_angle = ratio * angle;
    float reciprocal_supersample_width_times_2_times_ratio_times_angle = ratio_times_angle* 2.f/(float) (supersample_width);
    float reciprocal_supersample_height_times_2_times_angle = angle * 2.f/(float) (supersample_height);

    for (uint32_t j = 0; j < height; ++j) {
        for (uint32_t i = 0; i < width; ++i) {
            Vec pixel_color = vec(0, 0, 0);

            for (uint32_t offset_height = 0; offset_height < supersample_window; offset_height ++){
                for (uint32_t offset_width = 0; offset_width < supersample_window; offset_width ++){

                    float super_sample_i = supersample_window * i + offset_width;
                    float super_sample_j = supersample_window * j + offset_height;

                    float x = super_sample_i * reciprocal_supersample_width_times_2_times_ratio_times_angle - ratio_times_angle;
                    float y = angle - super_sample_j * reciprocal_supersample_height_times_2_times_angle;

                    Vec ray_direction;
                    Vec normalized_x_y_m1 = vec(x, y, -1);
                    normalized_x_y_m1 = normalize(normalized_x_y_m1);

                    ray_direction = mult_dir_matrix(cam_to_world, normalized_x_y_m1);

                    pixel_color = add(pixel_color, sphere_trace(ray_origin, ray_direction, scene, 0));
                }
            }

            pixel_color = smult(pixel_color, 1./(supersample_window * supersample_window));

            buffer[(width * j + i)].x = pixel_color.x;
            buffer[(width * j + i)].y = pixel_color.y;
            buffer[(width * j + i)].z = pixel_color.z;
        }
    }
}

