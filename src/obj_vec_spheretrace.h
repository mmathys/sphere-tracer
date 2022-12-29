#ifndef __OBJ_VEC_SPHERETRACE_H
#define __OBJ_VEC_SPHERETRACE_H

#include "geometry.h"
#include "scene/scene_config.h"

Vec sphere_trace(
        Vec ray_origin,
        Vec ray_direction,
        scene_config *scene,
        uint32_t depth);

uint32_t sphere_trace_shadow(
        Vec ray_origin,
        Vec ray_direction,
        float max_dist,
        scene_config *scene);

void obj_vec_render(Vec *buffer, const uint32_t width, const uint32_t height, scene_config *scene);

#endif
