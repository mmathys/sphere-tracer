#ifndef __SCENE_CONFIG_H
#define __SCENE_CONFIG_H

#include <stdint.h>
#include <immintrin.h>

#include "obj_kind.h"

typedef struct camera_config {
    uint32_t fov;
    float posx;
    float posy;
    float posz;

    float rotx;
    float roty;
    float rotz;
} camera_config;

typedef struct point_lights {
    uint32_t len;
    float *pos;
    float *emission;
} point_lights;

typedef struct obj_coords{
    float *x;
    float *y;
    float *z;
} obj_coords;

typedef struct obj_prop{
    obj_coords pos;
    obj_coords color;
    float *shininess;
    float *reflection;

    // store the rotation matrix elementwise (each matrix consists of 1 element in each of the arrays)
    float *rotation_0;
    float *rotation_1;
    float *rotation_2;
    float *rotation_3;
    float *rotation_4;
    float *rotation_5;    
    float *rotation_6;
    float *rotation_7;
    float *rotation_8;

} obj_prop;

typedef struct objects {
    uint32_t len;

    // dummy
    obj_prop dummy_props;

  	// Boxes config
	uint32_t box_len;
    uint32_t box_remaining;
    __m256i box_mask;
	obj_prop box_props;
    obj_coords box_extents;

	// Planes
	uint32_t plane_len;
    uint32_t plane_remaining;
    __m256i plane_mask;
    obj_prop plane_props;
    obj_coords plane_normals;

	// torus	
	uint32_t torus_len;
    uint32_t torus_remaining;
    __m256i torus_mask;
    obj_prop torus_props;
	float *torus_r1;
	float *torus_r2;

	// sphere
	uint32_t sphere_len;
    uint32_t sphere_remaining;
    __m256i sphere_mask;
    obj_prop sphere_props;
    float *sphere_radius;

	// boxframe
	uint32_t boxframe_len;
    uint32_t boxframe_remaining;
    __m256i boxframe_mask;
    obj_prop boxframe_props;
    obj_coords boxframe_extents;
    float *boxframe_thickness;

} objects;

typedef struct scene_config {
    camera_config cam_config;

    point_lights lights;

    objects objs;
} scene_config;

#endif
