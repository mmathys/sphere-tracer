#ifndef __RAY_VEC_SPHERETRACE_H
#define __RAY_VEC_SPHERETRACE_H

#include "scene/scene_config.h"
#include "geometry.h"

void ray_vec_render(Vec *buffer, const uint32_t width, const uint32_t height, scene_config *scene, const int perform_mixed_vectorization);

#endif
