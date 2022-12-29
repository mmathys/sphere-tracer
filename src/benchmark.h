/*
 * Benchmarking Infrastructure used to measure and compare the runtimes of the following functions of the referenced
 * and the optimized implementation:
 *  - Overall runtime in cycles
 *  - Runtime of calling shade()
 *  - Runtime of calling sphereTrace()
 *  - Runtime of calling sphereTraceShadow()
 * Note: When measuring the individual runtimes, be aware that each function calls all the lower functions
 *
 * How to use:
 *      - Each of the functions (shade, sphereTrace, sphereTraceShadow) has a separate prep function.
 *      - The prep function takes a pointer to a function of the specific type to be tested
 *          (e.g. prep_measureSphereTrace will take a pointer to a sphere tracing function)
 *      - Currently the code measures one optimized function and compres it to a reference implementation
 *      - The measurement functions all perform the whole sphere tracing process, but only measure the optimized part.
 *          The rest will be performed unoptimized.
 */

#ifndef __BENCHMARK_H
#define __BENCHMARK_H

#include <stdlib.h>
#include "geometry.h"
#include "scene/scene_config.h"

/* flags */
//#define MEASURE_DETAILED                    // comment this out to only perform overall measurement

/* time measurement parameters */
#define CYCLES_REQUIRED 1e8
#define FREQUENCY 3.1e9
#define CALIBRATE
#define REP 50

/* Measure the overall runtime of the rendering process */
unsigned long long measure_whole(render_mode (*render_func)(Vec *buffer,
                                         const uint32_t width,
                                         const uint32_t height,
                                         scene_config *scene,
                                         render_mode rendermode
                                         ),
                     Vec *buffer,
                     const uint32_t width,
                     const uint32_t height,
                     scene_config *scene,
                     int num_runs,
                     render_mode rendermode,
                     render_mode *chosen_mode
                     );


#endif //__BENCHMARK_H
