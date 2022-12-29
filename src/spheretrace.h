#ifndef __SPHERETRACE_H
#define __SPHERETRACE_H

#include "geometry.h"
#include "scene/scene_config.h"
#include "scene/obj_kind.h"

#ifndef M_PI
#define M_PI 3.141592653589793f
#endif

#ifdef COLLECT_STATS

// actual rays which will be used
extern long stats_active_rays_in_mask_counter[8];
// all the rays existing
extern long stats_total_rays_computed;

#endif

extern const float FLOAT_INFINITY;

typedef enum render_mode {
    Automatic = -1,
    ObjectVectorization = 0,
    RayVectorization = 1,
	MixedVectorization = 2
} render_mode;

/* compute the istance of the vector from to the object at index obj_index of type kind*/
float get_distance_single_point(scene_config *scene, obj_kind kind, uint32_t obj_index, Vec from);

float get_distance_box(Vec from, scene_config *scene, uint32_t obj_index);

float get_distance_boxframe(Vec from, scene_config *scene, uint32_t obj_index);

float get_distance_plane(Vec from, scene_config *scene, uint32_t obj_index);

float get_distance_sphere(Vec from, scene_config *scene, uint32_t obj_index);

float get_distance_torus(Vec from, scene_config *scene, uint32_t obj_index);

obj_prop *get_props(obj_kind obj_kind, scene_config *scene);

__m256i get_mask(obj_kind obj_kind, scene_config *scene);

uint32_t get_remaining(obj_kind obj_kind, scene_config *scene);

render_mode render(Vec *buffer,
                   const uint32_t width,
                   const uint32_t height,
                   scene_config *scene,
                   render_mode rendermode
);

#endif
