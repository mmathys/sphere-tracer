
#include <stdio.h>

#include "spheretrace.h"
#include "obj_vec_spheretrace.h"
#include "ray_vec_spheretrace.h"

#include "scene/obj_kind.h"
#include "scene/scene_config.h"
#include "geometry.h"
#include "benchmark.h"


const float FLOAT_INFINITY =  (float) INFINITY;

float get_distance_single_point(scene_config *scene, obj_kind kind, uint32_t obj_index, Vec from){
    switch (kind) {
        case Sphere: {
            return get_distance_sphere(from, scene, obj_index);
        }
        case Plane: {
            return get_distance_plane(from, scene, obj_index);
        }
        case Torus: {
            return get_distance_torus(from, scene, obj_index);
        }
        case Boxframe: {
            return get_distance_boxframe(from, scene, obj_index);
        }
        case Box: {
            return get_distance_box(from, scene, obj_index);
        }
        case Dummy: {
            return 0; 
        }
        default:
            printf("wrong kind for get_distance_single_point");
            exit(1);
    }
}

float get_distance_box(Vec from, scene_config *scene, uint32_t obj_index){
    Vec extents = vec(scene->objs.box_extents.x[obj_index],
                      scene->objs.box_extents.y[obj_index],
                      scene->objs.box_extents.z[obj_index]);

    Vec pos = vec(scene->objs.box_props.pos.x[obj_index],
                  scene->objs.box_props.pos.y[obj_index],
                  scene->objs.box_props.pos.z[obj_index]);

    // transform "from" to object space
    // translation
    Vec from_translated = sub(from, pos);
    // rotation

    Vec from_in_obj_space = rotate(from_translated, &scene->objs.box_props, obj_index);

    Vec q = sub(absV(from_in_obj_space), extents);

    return sqrtf(squared_norm(maxV(q, 0.0))) + fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.0);
}

float get_distance_boxframe(Vec from, scene_config *scene, uint32_t obj_index){
    Vec extents = vec(scene->objs.boxframe_extents.x[obj_index],
                      scene->objs.boxframe_extents.y[obj_index],
                      scene->objs.boxframe_extents.z[obj_index]);

    Vec pos = vec(scene->objs.boxframe_props.pos.x[obj_index],
                  scene->objs.boxframe_props.pos.y[obj_index],
                  scene->objs.boxframe_props.pos.z[obj_index]);

    float e = scene->objs.boxframe_thickness[obj_index];

    // transform "from" to object space
    // translation
    Vec from_translated = sub(from, pos);
    // rotation
    Vec p = rotate(from_translated, &scene->objs.boxframe_props, obj_index);

    p = sub(absV(p), extents);
    Vec q = ssub(absV(sadd(p, e)), e);
    return fminf(fminf(
            sqrtf(squared_norm(maxV(vec(p.x, q.y, q.z), 0.0))) + fminf(fmaxf(p.x, fmaxf(q.y, q.z)), 0.0),
            sqrtf(squared_norm(maxV(vec(q.x, p.y, q.z), 0.0))) + fminf(fmaxf(q.x, fmaxf(p.y, q.z)), 0.0)),
                 sqrtf(squared_norm(maxV(vec(q.x, q.y, p.z), 0.0))) + fminf(fmaxf(q.x, fmaxf(q.y, p.z)), 0.0));
}

float get_distance_plane(Vec from, scene_config *scene, uint32_t obj_index) {

    Vec normal = vec(scene->objs.plane_normals.x[obj_index],
                     scene->objs.plane_normals.y[obj_index],
                     scene->objs.plane_normals.z[obj_index]);

    Vec point_on_plane = vec(scene->objs.plane_props.pos.x[obj_index],
                  scene->objs.plane_props.pos.y[obj_index],
                  scene->objs.plane_props.pos.z[obj_index]);


    return scalar_product(normal, sub(from, point_on_plane));
}

float get_distance_sphere(Vec from, scene_config *scene, uint32_t obj_index) {
    Vec center = vec(scene->objs.sphere_props.pos.x[obj_index],
                     scene->objs.sphere_props.pos.y[obj_index],
                     scene->objs.sphere_props.pos.z[obj_index]);
    float radius = scene->objs.sphere_radius[obj_index];

    return sqrtf(squared_norm(sub(from, center))) - radius;
}

float get_distance_torus(Vec from, scene_config *scene, uint32_t obj_index){
    float r1 = scene->objs.torus_r1[obj_index];
    float r2 = scene->objs.torus_r2[obj_index];

    Vec center = vec(scene->objs.torus_props.pos.x[obj_index],
                  scene->objs.torus_props.pos.y[obj_index],
                  scene->objs.torus_props.pos.z[obj_index]);

    // transform "from" to object space
    // translation
    Vec from_translated = sub(from, center);
    // rotation
    Vec from_in_obj_space = rotate(from_translated, &scene->objs.torus_props, obj_index);

    float tmpx = sqrtf(from_in_obj_space.x * from_in_obj_space.x + from_in_obj_space.z * from_in_obj_space.z) - r1;
    float tmpy = from_in_obj_space.y;

    return sqrtf(tmpx * tmpx + tmpy * tmpy) - r2;
}

obj_prop *get_props(obj_kind obj_kind, scene_config *scene){
    switch (obj_kind){
        case Sphere: {
            return &scene->objs.sphere_props;
        }
        case Plane: {
            return &scene->objs.plane_props;
        }
        case Torus: {
            return &scene->objs.torus_props;
        }
        case Boxframe: {
            return &scene->objs.boxframe_props;
        }
        case Box: {
            return &scene->objs.box_props;
        }
        case Dummy: {
            return &scene->objs.dummy_props;
        }
        default:
            exit(1);
    }
}

__m256i get_mask(obj_kind obj_kind, scene_config *scene){
    switch (obj_kind){
        case Sphere: {
            return scene->objs.sphere_mask;
        }
        case Plane: {
            return scene->objs.plane_mask;
        }
        case Torus: {
            return scene->objs.torus_mask;
        }
        case Boxframe: {
            return scene->objs.boxframe_mask;
        }
        case Box: {
            return scene->objs.box_mask;
        }
        case Dummy: {
            return _mm256_setzero_si256();
        }
        default:
            exit(1);
    }
}

uint32_t get_remaining(obj_kind obj_kind, scene_config *scene){
    switch (obj_kind){
        case Sphere: {
            return scene->objs.sphere_remaining;
        }
        case Plane: {
            return scene->objs.plane_remaining;
        }
        case Torus: {
            return scene->objs.torus_remaining;
        }
        case Boxframe: {
            return scene->objs.boxframe_remaining;
        }
        case Box: {
            return scene->objs.box_remaining;
        }
        case Dummy: {
            return 0;
        }
        default:
            exit(1);
    }
}

/*
 * MicroSample Autotuner
 *
 * Idea: 1) get scene, 2) render a small picture in both modes, and measure (with measure_whole)
 *       3) see which mode performs better and use this one
 * 
 */
render_mode determine_render_mode_sample(scene_config *scene){
    unsigned long long cycles_obj = 0;    
    unsigned long long cycles_ray = 0;

    int N = 40;
    int width = 4 * N;
    int height = 3 * N;
    Vec result_obj[width * height];
    Vec result_ray[width * height];
    int num_runs = 1;
    render_mode chosen_mode;

    // measure obj
    cycles_obj = measure_whole(render, result_obj, width, height, scene, num_runs, ObjectVectorization, &chosen_mode);
    
    // measure ray
    cycles_ray = measure_whole(render, result_ray, width, height, scene, num_runs, RayVectorization, &chosen_mode);

    if(cycles_obj < cycles_ray){
        return ObjectVectorization;
    }
    return RayVectorization;
}

/**
 *
 * Static Analysis Autotuning
 * 
 * If many objects, and many reflections, object vectorization will probably be better
 * than ray vectorization
 * This autotuner chooses between ray and object vectorization
 *
 */
render_mode determine_render_mode_obj_or_ray_static(scene_config *scene) {
    int num_spheres = scene->objs.sphere_len >= 16;
    int num_boxes = scene->objs.box_len >= 16;
    int num_boxframes = scene->objs.boxframe_len >= 16;
    int num_planes = scene->objs.plane_len >= 16;
    int num_torus = scene->objs.torus_len >= 16;

    int num_refl = 0;
    for(uint32_t i = 0; i < scene->objs.sphere_len; i++) {
        num_refl += scene->objs.sphere_props.reflection[i] > 0;
    }
    for(uint32_t i = 0; i < scene->objs.box_len; i++) {
        num_refl += scene->objs.box_props.reflection[i] > 0;
    }
    for(uint32_t i = 0; i < scene->objs.boxframe_len; i++) {
        num_refl += scene->objs.boxframe_props.reflection[i] > 0;
    }
    for(uint32_t i = 0; i < scene->objs.plane_len; i++) {
        num_refl += scene->objs.plane_props.reflection[i] > 0;
    }
    for(uint32_t i = 0; i < scene->objs.torus_len; i++) {
        num_refl += scene->objs.torus_props.reflection[i] > 0;
    }

    if ((num_spheres + num_boxes + num_boxframes + num_planes + num_torus) && ((double) num_refl / scene->objs.len > 0.9)){
        return ObjectVectorization;
    }
    return RayVectorization;
}

/**
 * Static Analysis Autotuning. 
 *  Chooses between ray and mixed vectorization
 * As soon as there are many object available, mixed vectorization will always be better than
 * simple ray vectorization
 */
render_mode determine_render_mode_mixed_or_ray(scene_config *scene) {
    int num_spheres = scene->objs.sphere_len >= 5;
    int num_boxes = scene->objs.box_len >= 5;
    int num_boxframes = scene->objs.boxframe_len >= 5;
    int num_planes = scene->objs.plane_len >= 5;
    int num_torus = scene->objs.torus_len >= 5;

    if (num_spheres + num_boxes + num_boxframes + num_planes + num_torus){
	  return MixedVectorization;
    }
    return RayVectorization;
}

/* Render function. This will decide on whether to call obj_vectorization or ray_vectorization
 * rendermode = 0: make decision based on scene
 * rendermode = 1: always render in ray-vectorization mode
 * rendermode = 2: always render in object-vectorization mode
    */
render_mode render(Vec *buffer,
            const uint32_t width,
            const uint32_t height,
            scene_config *scene,
            render_mode rendermode) {
    
    if (rendermode == Automatic) {
        rendermode = determine_render_mode_mixed_or_ray(scene);
    }

    if (rendermode == ObjectVectorization){
        obj_vec_render(buffer, width, height, scene);
    } else if (rendermode == RayVectorization){
	  ray_vec_render(buffer, width, height, scene, 0);
	}else if (rendermode == MixedVectorization){
	  ray_vec_render(buffer, width, height, scene, 1);
    } else {
        printf("invalid render mode\n");
        exit(1);
    }
    return rendermode;
}
