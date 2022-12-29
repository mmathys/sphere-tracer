#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>

#include "spheretrace.h"
#include "ray_vec_spheretrace.h"
#include "obj_vec_spheretrace.h"

#include "scene/obj_kind.h"
#include "scene/scene_config.h"
#include "geometry.h"

#define RAY_VEC_CUTOFF_COUNT 2

#ifdef COLLECT_STATS

// actual rays which will be used
long stats_active_rays_in_mask_counter[8] = {0, 0, 0, 0, 0, 0, 0, 0};
// all the rays existing; even those which aren't even used
long stats_total_rays_computed = 0;

#endif

/* Load 8 reflection parameters into an avx register*/
void get_reflection_parameters(obj_kind *min_obj_kind, int *obj_index, scene_config *scene, __m256 *param, __m256 *inv_param){
    float r[8];
    float rm[8];
    for (int i = 0; i < 8; i++){
        obj_kind kind = min_obj_kind[i];
        r[i] = get_props(kind, scene)->reflection[obj_index[i]];
        rm[i] = 1.0 - r[i];
    }

    *param = _mm256_loadu_ps(r);
    *inv_param = _mm256_loadu_ps(rm);
}


inline void ray_vec_update_min_dist(__m256 x_from, __m256 y_from, __m256 z_from, 
                            scene_config *scene,
                            int N, 
                            __m256(*ray_vec_distance_func)(__m256 x_from, __m256 y_from, __m256 z_from, scene_config *scene, uint32_t obj_index),
                            __m256 *t_times_threshold, __m256i *obj_hit_index, __m256i *min_obj_kind, __m256 *min_distance, obj_kind kind){
        for(int i = 0; i < (int) N; i++) {
            __m256 dists = ray_vec_distance_func(x_from, y_from, z_from, scene, i);
            
            *min_distance = _mm256_min_ps(*min_distance, dists);  // update min_distance if the new object is closer

            __m256 dists_less_threshold  = _mm256_cmp_ps(dists, *t_times_threshold, _CMP_LE_OS); 
            __m256i updated_obj_index = _mm256_set1_epi32(i);
            
            // if d <= t*threshold, update the object hit index.
            *obj_hit_index = _mm256_blendv_epi8(*obj_hit_index, updated_obj_index, _mm256_castps_si256(dists_less_threshold));
            *min_obj_kind = _mm256_blendv_epi8(*min_obj_kind, _mm256_set1_epi32(kind), _mm256_castps_si256(dists_less_threshold));
		}
}

__m256 ray_vec_get_distance_sphere(__m256 x_from, __m256 y_from, __m256 z_from, scene_config *scene, uint32_t obj_index) {
    __m256 x_center = _mm256_set1_ps(scene->objs.sphere_props.pos.x[obj_index]);
    __m256 y_center = _mm256_set1_ps(scene->objs.sphere_props.pos.y[obj_index]);
    __m256 z_center = _mm256_set1_ps(scene->objs.sphere_props.pos.z[obj_index]);
    __m256 radius = _mm256_set1_ps(scene->objs.sphere_radius[obj_index]);

    vec_sub(x_from, y_from, z_from, x_center, y_center, z_center, &x_from, &y_from, &z_from);
    __m256 res;
    vec_squared_norm(x_from, y_from, z_from, &res);
    res = _mm256_sqrt_ps(res);
    res = _mm256_sub_ps(res, radius);

    return res;
}

__m256 ray_vec_get_distance_torus(__m256 x_from, __m256 y_from, __m256 z_from, scene_config *scene, uint32_t obj_index) {
    __m256 r1 = _mm256_set1_ps(*(scene->objs.torus_r1 + obj_index));
    __m256 r2 = _mm256_set1_ps(*(scene->objs.torus_r2 + obj_index));

    __m256 x_center = _mm256_set1_ps(*(scene->objs.torus_props.pos.x + obj_index));
    __m256 y_center = _mm256_set1_ps(*(scene->objs.torus_props.pos.y + obj_index));
    __m256 z_center = _mm256_set1_ps(*(scene->objs.torus_props.pos.z + obj_index));

    // transform "from" to object space

    // translation
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

    ray_vec_rotate(x_from_translated, y_from_translated, z_from_translated,
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

__m256 ray_vec_get_distance_plane(__m256 x_from, __m256 y_from, __m256 z_from, scene_config *scene, uint32_t obj_index) {
    __m256 x_normal = _mm256_set1_ps(*(scene->objs.plane_normals.x + obj_index));
    __m256 y_normal = _mm256_set1_ps(*(scene->objs.plane_normals.y + obj_index));
    __m256 z_normal = _mm256_set1_ps(*(scene->objs.plane_normals.z + obj_index));

    __m256 x_point = _mm256_set1_ps(*(scene->objs.plane_props.pos.x + obj_index));
    __m256 y_point = _mm256_set1_ps(*(scene->objs.plane_props.pos.y + obj_index));
    __m256 z_point = _mm256_set1_ps(*(scene->objs.plane_props.pos.z + obj_index));

    __m256 x_sub;
    __m256 y_sub;
    __m256 z_sub;
    __m256 res;

    vec_sub(x_from, y_from, z_from, x_point, y_point, z_point, &x_sub, &y_sub, &z_sub);
    vec_scalar_product(x_normal, y_normal, z_normal, x_sub, y_sub, z_sub, &res);

    return res;
}

__m256 ray_vec_get_distance_box(__m256 x_from, __m256 y_from, __m256 z_from, scene_config *scene, uint32_t obj_index) {
    __m256 x_pos = _mm256_set1_ps(*(scene->objs.box_props.pos.x + obj_index));
    __m256 y_pos = _mm256_set1_ps(*(scene->objs.box_props.pos.y + obj_index));
    __m256 z_pos = _mm256_set1_ps(*(scene->objs.box_props.pos.z + obj_index));
    // from = from - pos
    vec_sub(x_from, y_from, z_from, x_pos, y_pos, z_pos, &x_from, &y_from, &z_from);
    // from = rotate(from)
    ray_vec_rotate(x_from, y_from, z_from, &x_from, &y_from, &z_from, &scene->objs.box_props, obj_index);
    // from = absV(from)
    vec_absV(x_from, y_from, z_from, &x_from, &y_from, &z_from);

    // q = from - extents
    __m256 x_extents = _mm256_set1_ps(*(scene->objs.box_extents.x + obj_index));
    __m256 y_extents = _mm256_set1_ps(*(scene->objs.box_extents.y + obj_index));
    __m256 z_extents = _mm256_set1_ps(*(scene->objs.box_extents.z + obj_index));
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

__m256 ray_vec_get_distance_boxframe(__m256 x_from, __m256 y_from, __m256 z_from, scene_config *scene, uint32_t object_index) {
    __m256 x_pos = _mm256_set1_ps(scene->objs.boxframe_props.pos.x[object_index]);
    __m256 y_pos = _mm256_set1_ps(scene->objs.boxframe_props.pos.y[object_index]);
    __m256 z_pos = _mm256_set1_ps(scene->objs.boxframe_props.pos.z[object_index]);
    __m256 x_extents = _mm256_set1_ps(scene->objs.boxframe_extents.x[object_index]);
    __m256 y_extents = _mm256_set1_ps(scene->objs.boxframe_extents.y[object_index]);
    __m256 z_extents = _mm256_set1_ps(scene->objs.boxframe_extents.z[object_index]);
    __m256 thickness = _mm256_set1_ps(scene->objs.boxframe_thickness[object_index]);

    //translation
    vec_sub(x_from, y_from, z_from, x_pos, y_pos, z_pos, &x_from, &y_from, &z_from);
    //rotation
    ray_vec_rotate(x_from, y_from, z_from, &x_from, &y_from, &z_from, &(scene->objs.boxframe_props), object_index);
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


void ray_vec_sphere_trace_shadow(__m256 ray_origin_x, __m256 ray_origin_y, __m256 ray_origin_z,
                             __m256 ray_direction_x, __m256 ray_direction_y, __m256 ray_direction_z,
                             __m256 max_dist, scene_config *scene, __m256 *result) {
    __m256 threshold = _mm256_set1_ps(10e-5);
    __m256 t = _mm256_set1_ps(10e-5);

    __m256 t_times_threshold = _mm256_mul_ps(t, threshold);

    __m256i obj_hit_index = _mm256_set1_epi32(3);
    __m256i min_obj_kind = _mm256_set1_epi32(0);

    int ones = -1;
    float *as_float = (float *)&ones;
    __m256 mask_alive = _mm256_set1_ps(*as_float);
    __m256 min_distance_less_threshold = _mm256_setzero_ps();

    while (1) {
        __m256 min_distance = _mm256_set1_ps(FLOAT_INFINITY);

        __m256 x_from = _mm256_fmadd_ps(ray_direction_x, t, ray_origin_x);
        __m256 y_from = _mm256_fmadd_ps(ray_direction_y, t, ray_origin_y);
        __m256 z_from = _mm256_fmadd_ps(ray_direction_z, t, ray_origin_z);
		
        ray_vec_update_min_dist(x_from, y_from, z_from, scene,
            scene->objs.plane_len, ray_vec_get_distance_plane, 
            &t_times_threshold, &obj_hit_index, &min_obj_kind, &min_distance, Plane);
   
        ray_vec_update_min_dist(x_from, y_from, z_from, scene,
            scene->objs.sphere_len, ray_vec_get_distance_sphere, 
            &t_times_threshold, &obj_hit_index, &min_obj_kind, &min_distance, Sphere);
   
        ray_vec_update_min_dist(x_from, y_from, z_from, scene,
            scene->objs.boxframe_len, ray_vec_get_distance_boxframe, 
            &t_times_threshold, &obj_hit_index, &min_obj_kind, &min_distance, Boxframe);

        ray_vec_update_min_dist(x_from, y_from, z_from, scene,
            scene->objs.box_len, ray_vec_get_distance_box, 
            &t_times_threshold, &obj_hit_index, &min_obj_kind, &min_distance, Box);

        ray_vec_update_min_dist(x_from, y_from, z_from, scene,
            scene->objs.torus_len, ray_vec_get_distance_torus, 
            &t_times_threshold, &obj_hit_index, &min_obj_kind, &min_distance, Torus);

        t = _mm256_add_ps(t, min_distance);
        
        t_times_threshold = _mm256_mul_ps(t, threshold);

        __m256 min_distance_less_threshold_temp = _mm256_and_ps(mask_alive, _mm256_cmp_ps(min_distance, t_times_threshold, _CMP_LE_OS));
        min_distance_less_threshold = _mm256_or_ps(min_distance_less_threshold, min_distance_less_threshold_temp);
        mask_alive = _mm256_andnot_ps(min_distance_less_threshold, mask_alive);     // update the new-intersected objects
        
        __m256 t_less_max_dist = _mm256_cmp_ps(t, max_dist, _CMP_LT_OS);     // check if t < max_dist
        mask_alive = _mm256_and_ps(mask_alive, t_less_max_dist);

		int alive_mask = _mm256_movemask_ps(mask_alive);
		
		int count = 0;
		for(int i = 0; i < 8; i++){
		  count += (alive_mask >> i) & 0x01;
		}

		if(count <= RAY_VEC_CUTOFF_COUNT){
          __m256 one = _mm256_set1_ps(1);
          *result = _mm256_and_ps(min_distance_less_threshold, one);

		  // compute remaining ones if there are some
		  if(count > 0){
			float vec_orig_x[8];
			store_avx(x_from, vec_orig_x);

			float vec_orig_y[8];
			store_avx(y_from, vec_orig_y);
        
			float vec_orig_z[8];
			store_avx(z_from, vec_orig_z);

			float vec_max_dist[8];
			store_avx(max_dist, vec_max_dist);
		  
			float vec_t[8];
			store_avx(t, vec_t);
		  
			float vec_dir_x[8];
			store_avx(ray_direction_x, vec_dir_x);

			float vec_dir_y[8];
			store_avx(ray_direction_y, vec_dir_y);

			float vec_dir_z[8];
			store_avx(ray_direction_z, vec_dir_z);

			float shadows[8];
			store_avx(*result, shadows);

			for(int i = 0; i < 8; i++){
			  if((alive_mask >> i) & 0x01){
				// compute result
				Vec ray_origin = {vec_orig_x[i], vec_orig_y[i], vec_orig_z[i]};
				Vec ray_direction = {vec_dir_x[i], vec_dir_y[i], vec_dir_z[i]};
                        
			    float shadow = sphere_trace_shadow(ray_origin, ray_direction, vec_max_dist[i] - vec_t[i], scene);
				shadows[i] = shadow;
			  }
			}

			*result = _mm256_loadu_ps(shadows);
		  }
		  return;
		}
    }
}

/*
 * - intersection point (x, y, z) ("ray_origin" and "t" are not needed)
 * - ray direction (x, y, z)
 * - object props (as array) and object indices (as vector)
 * - normal vector
 * - scene
 * - radiance R
 */
void ray_vec_shade(__m256 intersection_x, __m256 intersection_y, __m256 intersection_z,
               __m256 ray_direction_x, __m256 ray_direction_y, __m256 ray_direction_z,
               __m256 obj_color_x, __m256 obj_color_y, __m256 obj_color_z, __m256 obj_ks, __m256 obj_ns,
               __m256 normal_x, __m256 normal_y, __m256 normal_z,
               scene_config *scene,
               __m256 *R_x, __m256 *R_y, __m256 *R_z) {

    __m256 viewing_direction_x, viewing_direction_y, viewing_direction_z;
    __m256 light_x, light_y, light_z;
    __m256 light_dir_x, light_dir_y, light_dir_z;
    __m256 obj_kd;
    __m256 scalar_products;
    __m256 one;
    __m256 zero;
    __m256 mask_alive;
    __m256 dist2, sqrt_dist2;
    __m256 light_dir_x_n, light_dir_y_n, light_dir_z_n;
    __m256 reflected_direction_x, reflected_direction_y, reflected_direction_z;
    __m256 res_shadow, shadow;
    __m256 dot;
    __m256 light_color_x, light_color_y, light_color_z;
    __m256 divisor, divisor2, scalar;
    __m256 light_intensity_x, light_intensity_y, light_intensity_z;
    __m256 diffuse_lhs_x, diffuse_lhs_y, diffuse_lhs_z;
    __m256 light_diffuse_contribution_x, light_diffuse_contribution_y, light_diffuse_contribution_z;
    __m256 spec_rhs, spec_rhs2, spec_rhs3;
    __m256 light_specular_contribution_x, light_specular_contribution_y, light_specular_contribution_z;
    __m256 masked_diff_contr_x, masked_diff_contr_y, masked_diff_contr_z;
    __m256 masked_spec_contr_x, masked_spec_contr_y, masked_spec_contr_z;

    vec_smult(ray_direction_x, ray_direction_y, ray_direction_z, (-1),
              &viewing_direction_x, &viewing_direction_y, &viewing_direction_z);

    __m256 diffuse_contribution_x = _mm256_setzero_ps();
    __m256 diffuse_contribution_y = _mm256_setzero_ps();
    __m256 diffuse_contribution_z = _mm256_setzero_ps();
    __m256 specular_contribution_x = _mm256_setzero_ps();
    __m256 specular_contribution_y = _mm256_setzero_ps();
    __m256 specular_contribution_z = _mm256_setzero_ps();

    int lights_count = scene->lights.len;
    float *lights_pos = scene->lights.pos;
    float *lights_color = scene->lights.emission;

    for (int i = 0; i < lights_count; i++) {
        Vec light_i = load_vec(lights_pos + 3 * i);
        light_x = _mm256_set1_ps(light_i.x);
        light_y = _mm256_set1_ps(light_i.y);
        light_z = _mm256_set1_ps(light_i.z);

        // Vec light_dir = sub(light_i, p);
        // light_dir = light - intersection
        vec_sub(light_x, light_y, light_z, intersection_x, intersection_y, intersection_z, &light_dir_x, &light_dir_y, &light_dir_z);

        // if_mask = scalar_product(light_dir, normal)
        vec_scalar_product(light_dir_x, light_dir_y, light_dir_z, normal_x, normal_y, normal_z, &scalar_products);
        zero = _mm256_setzero_ps();
        mask_alive = _mm256_cmp_ps(scalar_products, zero, _CMP_GT_OS);

        // vec_normalize calls vec_squared_norm anyway --> room for optimization :))
        vec_squared_norm(light_dir_x, light_dir_y, light_dir_z, &dist2);
        vec_normalize(light_dir_x, light_dir_y, light_dir_z, &light_dir_x_n, &light_dir_y_n, &light_dir_z_n);

        vec_reflect(light_dir_x_n, light_dir_y_n, light_dir_z_n, normal_x, normal_y, normal_z, &reflected_direction_x, &reflected_direction_y, &reflected_direction_z);

        sqrt_dist2 = _mm256_sqrt_ps(dist2);
        ray_vec_sphere_trace_shadow(intersection_x, intersection_y, intersection_z, light_dir_x_n, light_dir_y_n, light_dir_z_n, sqrt_dist2, scene, &res_shadow);

        one = _mm256_set1_ps(1.0);
        shadow = _mm256_sub_ps(one, res_shadow);
        vec_scalar_product(light_dir_x_n, light_dir_y_n, light_dir_z_n, normal_x, normal_y, normal_z, &dot);

        Vec light_i_color = load_vec(lights_color + 3 * i);
        light_color_x = _mm256_set1_ps(light_i_color.x);
        light_color_y = _mm256_set1_ps(light_i_color.y);
        light_color_z = _mm256_set1_ps(light_i_color.z);

        divisor = _mm256_set1_ps(4 * M_PI);
        divisor2 = _mm256_mul_ps(divisor, dist2);
        scalar = _mm256_div_ps(shadow, divisor2);
        vec_smult2(light_color_x, light_color_y, light_color_z, scalar, &light_intensity_x, &light_intensity_y, &light_intensity_z);

        // light diffuse contribution
        vec_mult(obj_color_x, obj_color_y, obj_color_z,
                 light_intensity_x, light_intensity_y, light_intensity_z,
                 &diffuse_lhs_x, &diffuse_lhs_y, &diffuse_lhs_z);
        vec_smult2(diffuse_lhs_x, diffuse_lhs_y, diffuse_lhs_z,
                   dot,
                   &light_diffuse_contribution_x, &light_diffuse_contribution_y, &light_diffuse_contribution_z);

        // light specular contribution
        vec_scalar_product(reflected_direction_x, reflected_direction_y, reflected_direction_z,
                           viewing_direction_x, viewing_direction_y, viewing_direction_z,
                           &spec_rhs);
        spec_rhs2 = _mm256_max_ps(spec_rhs, zero);
        vec_pow(spec_rhs2, obj_ns, &spec_rhs3);
        vec_smult2(light_intensity_x, light_intensity_y, light_intensity_z, spec_rhs3,
                   &light_specular_contribution_x, &light_specular_contribution_y, &light_specular_contribution_z);

        // mask diffuse_contribution and specular_contribution with "mask_alive"
        vec_and(light_diffuse_contribution_x, light_diffuse_contribution_y, light_diffuse_contribution_z, mask_alive,
                   &masked_diff_contr_x, &masked_diff_contr_y, &masked_diff_contr_z);
        vec_and(light_specular_contribution_x, light_specular_contribution_y, light_specular_contribution_z, mask_alive,
                   &masked_spec_contr_x, &masked_spec_contr_y, &masked_spec_contr_z);

        // add the masked contributions to total contributions
        vec_add(masked_diff_contr_x, masked_diff_contr_y, masked_diff_contr_z,
                    diffuse_contribution_x, diffuse_contribution_y, diffuse_contribution_z,
                    &diffuse_contribution_x, &diffuse_contribution_y, &diffuse_contribution_z);
        vec_add(masked_spec_contr_x, masked_spec_contr_y, masked_spec_contr_z,
                    specular_contribution_x, specular_contribution_y, specular_contribution_z,
                    &specular_contribution_x, &specular_contribution_y, &specular_contribution_z);
    }
    obj_kd = _mm256_sub_ps(one, obj_ks);

    // R = kd * (diffuse) + ks * (specular)
    vec_smult2(diffuse_contribution_x, diffuse_contribution_y, diffuse_contribution_z,
                obj_kd,
                &diffuse_contribution_x, &diffuse_contribution_y, &diffuse_contribution_z);
    vec_smult2(specular_contribution_x, specular_contribution_y, specular_contribution_z,
                obj_ks,
                &specular_contribution_x, &specular_contribution_y, &specular_contribution_z);
    vec_add(diffuse_contribution_x, diffuse_contribution_y, diffuse_contribution_z,
                specular_contribution_x, specular_contribution_y, specular_contribution_z,
                R_x, R_y, R_z);
}

void ray_vec_compute_normal(__m256 intersection_x, __m256 intersection_y, __m256 intersection_z, __m256i v_kind,
							__m256i v_object_index, scene_config *scene, __m256 *normal_x, __m256 *normal_y, __m256 *normal_z, int alive_mask) {
    float delta = 10e-5;

    int kind[8];
    store_avx_int(v_kind, kind);
    
    int object_index[8];
    store_avx_int(v_object_index, object_index);

    float x[8];
    store_avx(intersection_x, x);
    float y[8];
    store_avx(intersection_y, y);
    float z[8];
    store_avx(intersection_z, z);

    store_avx(intersection_y, y);
    store_avx(intersection_z, z);

    __m256 d = _mm256_set1_ps(delta);
    __m256 intersection_x_p = _mm256_add_ps(intersection_x, d);
    __m256 intersection_y_p = _mm256_add_ps(intersection_y, d);
    __m256 intersection_z_p = _mm256_add_ps(intersection_z, d);
    __m256 intersection_x_pm = _mm256_sub_ps(intersection_x, d);
    __m256 intersection_y_pm = _mm256_sub_ps(intersection_y, d);
    __m256 intersection_z_pm = _mm256_sub_ps(intersection_z, d);

    float x_p[8];
    store_avx(intersection_x_p, x_p);
    float y_p[8];
    store_avx(intersection_y_p, y_p);
    float z_p[8];
    store_avx(intersection_z_p, z_p);
    float x_pm[8];
    store_avx(intersection_x_pm, x_pm);
    float y_pm[8];
    store_avx(intersection_y_pm, y_pm);
    float z_pm[8];
    store_avx(intersection_z_pm, z_pm);

    float x_normal[8], y_normal[8], z_normal[8];
    for(int i = 0; i < 8; i++) {
	  if((alive_mask >> i) & 0x01){
        int k = kind[i];
        int idx = object_index[i];

        x_normal[i] = get_distance_single_point(scene, (obj_kind) k, idx, vec(x_p[i], y[i], z[i]))
		  - get_distance_single_point(scene, (obj_kind) k, idx, vec(x_pm[i], y[i], z[i]));
        y_normal[i] = get_distance_single_point(scene, (obj_kind) k, idx, vec(x[i], y_p[i], z[i]))
		  - get_distance_single_point(scene, (obj_kind) k, idx, vec(x[i], y_pm[i], z[i]));
        z_normal[i] = get_distance_single_point(scene, (obj_kind) k, idx, vec(x[i], y[i], z_p[i]))
		  - get_distance_single_point(scene, (obj_kind) k, idx, vec(x[i], y[i], z_pm[i]));
	  }
    }

    __m256 v_x = load_avx(x_normal);
    __m256 v_y = load_avx(y_normal);
    __m256 v_z = load_avx(z_normal);
    vec_normalize(v_x, v_y, v_z, normal_x, normal_y, normal_z);
}

void ray_vec_sphere_trace(
            __m256 ray_origin_x,
            __m256 ray_origin_y,
            __m256 ray_origin_z,
            __m256 ray_direction_x,
            __m256 ray_direction_y,
            __m256 ray_direction_z,
            __m256 *pixel_color_x,
            __m256 *pixel_color_y,
            __m256 *pixel_color_z,
            __m256 mask_alive,
            scene_config *scene,
            uint32_t depth,
			int perform_mixed_vectorization)
    {

    // get original alive mask to know if we have to reflect
    __m256 original_alive_mask = mask_alive;
	
    uint32_t max_depth = 3;
    __m256 max_dist = _mm256_set1_ps(100);

    __m256 threshold = _mm256_set1_ps(10e-6);
    __m256 t = _mm256_set1_ps(10e-6);

    __m256i obj_hit_index = _mm256_set1_epi32(0);
    __m256i min_obj_kind = _mm256_set1_epi32(0);    // dummy object

    __m256 min_distance_less_threshold = _mm256_setzero_ps();

    int alive_mask1 = _mm256_movemask_ps(mask_alive);

	if(perform_mixed_vectorization){
	  int count1 = 0;
	  for(int i = 0; i < 8; i++){
        count1 += (alive_mask1 >> i) & 0x01;
	  }

	  if(count1 <= RAY_VEC_CUTOFF_COUNT){
		float vec_orig_x[8];
		store_avx(ray_origin_x, vec_orig_x);

		float vec_orig_y[8];
		store_avx(ray_origin_y, vec_orig_y);
        
		float vec_orig_z[8];
		store_avx(ray_origin_z, vec_orig_z);

		float vec_dir_x[8];
		store_avx(ray_direction_x, vec_dir_x);

		float vec_dir_y[8];
		store_avx(ray_direction_y, vec_dir_y);

		float vec_dir_z[8];
		store_avx(ray_direction_z, vec_dir_z);

		float vec_col_x[8];
		store_avx(*pixel_color_x, vec_col_x);

		float vec_col_y[8];
		store_avx(*pixel_color_y, vec_col_y);

		float vec_col_z[8];
		store_avx(*pixel_color_z, vec_col_z);
        
		for(int i = 0; i < 8; i++){
		  if((alive_mask1 >> i) & 0x01){
			// compute result
			Vec ray_origin = {vec_orig_x[i], vec_orig_y[i], vec_orig_z[i]};
			Vec ray_direction = {vec_dir_x[i], vec_dir_y[i], vec_dir_z[i]};
                        
			Vec pix = sphere_trace(ray_origin, ray_direction, scene, depth);

			vec_col_x[i] = pix.x;
			vec_col_y[i] = pix.y;
			vec_col_z[i] = pix.z;
		  }
		}
            
		*pixel_color_x = load_avx(vec_col_x);
		*pixel_color_y = load_avx(vec_col_y);
		*pixel_color_z = load_avx(vec_col_z);

		return;
	  }
	}

	__m256 x_from = _mm256_fmadd_ps(ray_direction_x, t, ray_origin_x);
	__m256 y_from = _mm256_fmadd_ps(ray_direction_y, t, ray_origin_y);
	__m256 z_from = _mm256_fmadd_ps(ray_direction_z, t, ray_origin_z);

    while (1) {
        __m256 t_less_max_dist = _mm256_cmp_ps(t, max_dist, _CMP_LT_OS);     // check if t < max_dist

        mask_alive = _mm256_and_ps(mask_alive, t_less_max_dist);

        __m256 min_distance = _mm256_set1_ps(FLOAT_INFINITY);
        __m256 t_times_threshold = _mm256_mul_ps(t, threshold);

        ray_vec_update_min_dist(x_from, y_from, z_from, scene,
            scene->objs.plane_len, ray_vec_get_distance_plane, 
            &t_times_threshold, &obj_hit_index, &min_obj_kind, &min_distance, Plane);
   
        ray_vec_update_min_dist(x_from, y_from, z_from, scene,
            scene->objs.sphere_len, ray_vec_get_distance_sphere, 
            &t_times_threshold, &obj_hit_index, &min_obj_kind, &min_distance, Sphere);
   
        ray_vec_update_min_dist(x_from, y_from, z_from, scene,
            scene->objs.boxframe_len, ray_vec_get_distance_boxframe, 
            &t_times_threshold, &obj_hit_index, &min_obj_kind, &min_distance, Boxframe);

		ray_vec_update_min_dist(x_from, y_from, z_from, scene,
            scene->objs.box_len, ray_vec_get_distance_box, 
            &t_times_threshold, &obj_hit_index, &min_obj_kind, &min_distance, Box);

		ray_vec_update_min_dist(x_from, y_from, z_from, scene,
            scene->objs.torus_len, ray_vec_get_distance_torus, 
            &t_times_threshold, &obj_hit_index, &min_obj_kind, &min_distance, Torus);
		
        __m256 min_distance_less_threshold_temp  = _mm256_and_ps(mask_alive, _mm256_cmp_ps(min_distance, t_times_threshold, _CMP_LE_OS)); 
        min_distance_less_threshold = _mm256_or_ps(min_distance_less_threshold, min_distance_less_threshold_temp);
        mask_alive = _mm256_andnot_ps(min_distance_less_threshold, mask_alive);     // update the new-intersected objects
        t = _mm256_add_ps(t, min_distance);
        
        mask_alive = _mm256_and_ps(mask_alive, t_less_max_dist);

	    int alive_mask = _mm256_movemask_ps(mask_alive);
        
        int count = 0;
        for(int i = 0; i < 8; i++){
            count += (alive_mask >> i) & 0x01;
        }
        
        // if still computing on rays, do computations
        #ifdef COLLECT_STATS
		
		stats_active_rays_in_mask_counter[count-1] += 1;
		stats_total_rays_computed += 1;
	
        #endif

		//  update for next iteration
		x_from = _mm256_fmadd_ps(ray_direction_x, t, ray_origin_x);
		y_from = _mm256_fmadd_ps(ray_direction_y, t, ray_origin_y);
		z_from = _mm256_fmadd_ps(ray_direction_z, t, ray_origin_z);

		// if perform_mixed_vectorization is 0, then it's the same as alive_mask == 0
        if (count <= RAY_VEC_CUTOFF_COUNT * perform_mixed_vectorization){ 
             // if 0 > min_obj_kind: we continue. Otherwise there was no intersection and we can go to the next iteration    
            __m256i intersection_found_mask = _mm256_cmpgt_epi32(min_obj_kind, _mm256_setzero_si256());
            int intersection_found = _mm256_movemask_epi8(intersection_found_mask);
        
            if (alive_mask == 0 && intersection_found == 0){ 
                *pixel_color_x = _mm256_setzero_ps();
                *pixel_color_y = _mm256_setzero_ps();
                *pixel_color_z = _mm256_setzero_ps();
                return;
            }

            int min_obj_kind_ptr[8];
            store_avx_int(min_obj_kind, min_obj_kind_ptr);

            int obj_hit_index_ptr[8];
            store_avx_int(obj_hit_index, obj_hit_index_ptr);

            __m256 intersection_x = _mm256_fmadd_ps(ray_direction_x, t, ray_origin_x);
            __m256 intersection_y = _mm256_fmadd_ps(ray_direction_y, t, ray_origin_y);
            __m256 intersection_z = _mm256_fmadd_ps(ray_direction_z, t, ray_origin_z);

            __m256 normal_x, normal_y, normal_z;

            ray_vec_compute_normal(intersection_x, intersection_y, intersection_z, min_obj_kind, obj_hit_index, scene, &normal_x, &normal_y, &normal_z, ~alive_mask);
            
            float obj_color_x_arr[8], obj_color_y_arr[8], obj_color_z_arr[8], obj_ks_arr[8], obj_ns_arr[8];
            for (int j = 0; j < 8; j++){
			  if((~alive_mask >> j) & 0x01){
                int min_obj_kind_j = min_obj_kind_ptr[j];
                int hit_idx = obj_hit_index_ptr[j];
                obj_prop *prop = get_props((obj_kind) min_obj_kind_j, scene);
                obj_color_x_arr[j] = prop->color.x[hit_idx];
                obj_color_y_arr[j] = prop->color.y[hit_idx];
                obj_color_z_arr[j] = prop->color.z[hit_idx];
                obj_ks_arr[j] = prop->reflection[hit_idx];
                obj_ns_arr[j] = prop->shininess[hit_idx];
			  }
            }
            __m256 obj_color_x, obj_color_y, obj_color_z, obj_ks, obj_ns;
            obj_color_x = load_avx(obj_color_x_arr);
            obj_color_y = load_avx(obj_color_y_arr);
            obj_color_z = load_avx(obj_color_z_arr);
            obj_ks = load_avx(obj_ks_arr);
            obj_ns = load_avx(obj_ns_arr);

            __m256 shade_res_x, shade_res_y, shade_res_z;
            ray_vec_shade(intersection_x, intersection_y, intersection_z, ray_direction_x, ray_direction_y, ray_direction_z,
                        obj_color_x, obj_color_y, obj_color_z, obj_ks, obj_ns,
                        normal_x, normal_y, normal_z, scene, &shade_res_x, &shade_res_y, &shade_res_z);

            __m256 r;
            __m256 rm; // 1 - r
            __m256 mones = _mm256_set1_ps(-1);
            get_reflection_parameters((obj_kind *) min_obj_kind_ptr, obj_hit_index_ptr, scene, &r, &rm);
            
            __m256 bool_shoot_secondary_ray = _mm256_and_ps(original_alive_mask, _mm256_cmp_ps(_mm256_setzero_ps(), r, _CMP_LT_OS));
            int reflections_mask = _mm256_movemask_ps(bool_shoot_secondary_ray);

            if (reflections_mask != 0 && depth <= max_depth){
                __m256 secondary_ray_direction_x, secondary_ray_direction_y, secondary_ray_direction_z;
                vec_reflect(_mm256_mul_ps(mones, ray_direction_x), _mm256_mul_ps(mones, ray_direction_y) ,_mm256_mul_ps(mones, ray_direction_z), 
                            normal_x, normal_y, normal_z, 
                            &secondary_ray_direction_x,  &secondary_ray_direction_y, &secondary_ray_direction_z);

                __m256 secondary_ray_color_x = _mm256_setzero_ps();
                __m256 secondary_ray_color_y = _mm256_setzero_ps();
                __m256 secondary_ray_color_z = _mm256_setzero_ps();

				ray_vec_sphere_trace(intersection_x, intersection_y, intersection_z, 
									 secondary_ray_direction_x, secondary_ray_direction_y, secondary_ray_direction_z, 
									 &secondary_ray_color_x, &secondary_ray_color_y, &secondary_ray_color_z, 
									 bool_shoot_secondary_ray, scene, depth+1, perform_mixed_vectorization);

				__m256 reflection_contribution_x = _mm256_mul_ps(secondary_ray_color_x, r);
				__m256 reflection_contribution_y = _mm256_mul_ps(secondary_ray_color_y, r);
				__m256 reflection_contribution_z = _mm256_mul_ps(secondary_ray_color_z, r);

				__m256 old_pixel_contribution_x = _mm256_mul_ps(shade_res_x, rm);
				__m256 old_pixel_contribution_y = _mm256_mul_ps(shade_res_y, rm);
				__m256 old_pixel_contribution_z = _mm256_mul_ps(shade_res_z, rm);

				__m256 without_reflection_x = shade_res_x;
				__m256 without_reflection_y = shade_res_y;
				__m256 without_reflection_z = shade_res_z;

				__m256 with_reflection_x = _mm256_add_ps(old_pixel_contribution_x, reflection_contribution_x);
				__m256 with_reflection_y =  _mm256_add_ps(old_pixel_contribution_y, reflection_contribution_y);
				__m256 with_reflection_z =  _mm256_add_ps(old_pixel_contribution_z, reflection_contribution_z);

				*pixel_color_x = _mm256_and_ps(min_distance_less_threshold, _mm256_blendv_ps(without_reflection_x, with_reflection_x, bool_shoot_secondary_ray));
				*pixel_color_y = _mm256_and_ps(min_distance_less_threshold, _mm256_blendv_ps(without_reflection_y, with_reflection_y, bool_shoot_secondary_ray));
				*pixel_color_z = _mm256_and_ps(min_distance_less_threshold, _mm256_blendv_ps(without_reflection_z, with_reflection_z, bool_shoot_secondary_ray));
            }else{
			    *pixel_color_x = _mm256_and_ps(min_distance_less_threshold, shade_res_x);
				*pixel_color_y = _mm256_and_ps(min_distance_less_threshold, shade_res_y);
				*pixel_color_z = _mm256_and_ps(min_distance_less_threshold, shade_res_z);
			}

			if(count > 0){
                float vec_orig_x[8];
                store_avx(x_from, vec_orig_x);

                float vec_orig_y[8];
                store_avx(y_from, vec_orig_y);
                
                float vec_orig_z[8];
                store_avx(z_from, vec_orig_z);

                float vec_dir_x[8];
                store_avx(ray_direction_x, vec_dir_x);

                float vec_dir_y[8];
                store_avx(ray_direction_y, vec_dir_y);

                float vec_dir_z[8];
                store_avx(ray_direction_z, vec_dir_z);

                float vec_col_x[8];
                store_avx(*pixel_color_x, vec_col_x);

                float vec_col_y[8];
                store_avx(*pixel_color_y, vec_col_y);

                float vec_col_z[8];
                store_avx(*pixel_color_z, vec_col_z);
                
                for(int i = 0; i < 8; i++){
                    if((alive_mask >> i) & 0x01){
                        // compute result
                        Vec ray_origin = {vec_orig_x[i], vec_orig_y[i], vec_orig_z[i]};
                        Vec ray_direction = {vec_dir_x[i], vec_dir_y[i], vec_dir_z[i]};
                                
                        Vec pix = sphere_trace(ray_origin, ray_direction, scene, depth);

                        vec_col_x[i] = pix.x;
                        vec_col_y[i] = pix.y;
                        vec_col_z[i] = pix.z;
                    }
                }
                    
                *pixel_color_x = load_avx(vec_col_x);
                *pixel_color_y = load_avx(vec_col_y);
                *pixel_color_z = load_avx(vec_col_z);
            }

			return;
		}
    }
}

// returns list of vec3fs
void ray_vec_render(Vec *buffer,
            const uint32_t width,
            const uint32_t height,
		    scene_config *scene,
			const int perform_mixed_vectorization) {

    float fov = scene->cam_config.fov;

    Vec camera_position = vec(scene->cam_config.posx, scene->cam_config.posy, scene->cam_config.posz);
    Vec camera_rotation = vec(scene->cam_config.rotx, scene->cam_config.roty, scene->cam_config.rotz);

    const float ratio = width / (float) (height);

    float angle = tan(deg_to_rad(fov * 0.5));

    Vec zeros = vec(0, 0, 0);

    Mat4 cam_to_world;
    get_camera_matrix(camera_position, camera_rotation, &cam_to_world);

    Vec ray_origin = mult_vec_matrix(cam_to_world, zeros);

	__m256 ray_origin_x = _mm256_set1_ps(ray_origin.x);
	__m256 ray_origin_y = _mm256_set1_ps(ray_origin.y);
	__m256 ray_origin_z = _mm256_set1_ps(ray_origin.z);
		        
	uint32_t supersample_width = 2 * width;
    uint32_t supersample_height = 2 * height;

    float ratio_times_angle = ratio * angle;
    float width_scale = ratio_times_angle* 2.f/(float) (supersample_width);
    float height_scale = angle * 2.f/(float) (supersample_height);

	__m256 vec_width_scale = _mm256_set1_ps(width_scale);
	__m256 vec_height_scale = _mm256_set1_ps(height_scale);
	__m256 vec_ratio_times_angle = _mm256_set1_ps(ratio_times_angle);
	__m256 vec_angle = _mm256_set1_ps(angle);

    for (uint32_t h = 0; h < height; h++) {
	    for (uint32_t w = 0; w < width; w += 4) {
		    __m256 pix_x_aa = _mm256_setzero_ps();
			__m256 pix_y_aa = _mm256_setzero_ps();
			__m256 pix_z_aa = _mm256_setzero_ps();
		  
		    for(uint32_t h_offset = 0; h_offset < 2; h_offset++){
                int i = 2 * h + h_offset;
                int j = 2 * w;
                
                __m256 vec_x = _mm256_set_ps(j + 7, j + 6, j + 5, j + 4, j + 3, j + 2, j + 1, j);
                __m256 vec_y = _mm256_set1_ps(i);
                
                vec_x = _mm256_mul_ps(vec_x, vec_width_scale);
                vec_x = _mm256_sub_ps(vec_x, vec_ratio_times_angle);

                vec_y = _mm256_mul_ps(vec_y, vec_height_scale);
                vec_y = _mm256_sub_ps(vec_angle, vec_y);
                
                __m256 vec_normalized_x_y_m1_x;
                __m256 vec_normalized_x_y_m1_y;
                __m256 vec_normalized_x_y_m1_z;

                __m256 m1 = _mm256_set1_ps(-1);

                vec_normalize(vec_x, vec_y, m1, &vec_normalized_x_y_m1_x, &vec_normalized_x_y_m1_y, &vec_normalized_x_y_m1_z);

                __m256 ray_dir_x;
                __m256 ray_dir_y;
                __m256 ray_dir_z;
                
                vec_mult_dir_matrix(cam_to_world, vec_normalized_x_y_m1_x, vec_normalized_x_y_m1_y, vec_normalized_x_y_m1_z, &ray_dir_x, &ray_dir_y, &ray_dir_z);

                __m256 pix_x = _mm256_setzero_ps();
                __m256 pix_y = _mm256_setzero_ps();
                __m256 pix_z = _mm256_setzero_ps();

                int ones = -1;
                float *as_float = (float *)&ones;
                __m256 mask_alive = _mm256_set1_ps(*as_float);

                ray_vec_sphere_trace(ray_origin_x, ray_origin_y, ray_origin_z, ray_dir_x, ray_dir_y, ray_dir_z, &pix_x, &pix_y, &pix_z, mask_alive, scene, 0, perform_mixed_vectorization);
                
                pix_x_aa = _mm256_add_ps(pix_x_aa, pix_x);
                pix_y_aa = _mm256_add_ps(pix_y_aa, pix_y);
                pix_z_aa = _mm256_add_ps(pix_z_aa, pix_z);
			}

			// calculate antialiased average from supersampled cumulative results
			pix_x_aa = _mm256_mul_ps(pix_x_aa, _mm256_set1_ps(0.25));
			pix_y_aa = _mm256_mul_ps(pix_y_aa, _mm256_set1_ps(0.25));
			pix_z_aa = _mm256_mul_ps(pix_z_aa, _mm256_set1_ps(0.25));

			__m256 pix_x_aa_hadd = _mm256_hadd_ps(pix_x_aa, pix_x_aa);
			__m256 pix_y_aa_hadd = _mm256_hadd_ps(pix_y_aa, pix_y_aa);
			__m256 pix_z_aa_hadd = _mm256_hadd_ps(pix_z_aa, pix_z_aa);			
			
			float pix_x_array[8];
			float pix_y_array[8];
			float pix_z_array[8];

			_mm256_storeu_ps(pix_x_array, pix_x_aa_hadd);
			_mm256_storeu_ps(pix_y_array, pix_y_aa_hadd);
			_mm256_storeu_ps(pix_z_array, pix_z_aa_hadd);

			for(int v = 0; v < 4; v+=2){
			    buffer[(width * h + w + v)].x = pix_x_array[2 * v];
				buffer[(width * h + w + v)].y = pix_y_array[2 * v];
				buffer[(width * h + w + v)].z = pix_z_array[2 * v];

			    buffer[(width * h + w + v + 1)].x = pix_x_array[2 * v + 1];
				buffer[(width * h + w + v + 1)].y = pix_y_array[2 * v + 1];
				buffer[(width * h + w + v + 1)].z = pix_z_array[2 * v + 1];
			}
		}
    }
}
