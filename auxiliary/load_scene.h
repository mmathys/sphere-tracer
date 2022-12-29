#ifndef TEAM07_LOAD_SCENE_H
#define TEAM07_LOAD_SCENE_H

#include <vector>
#include <string>
#include "../src/scene/scene_config.h"
#include "../src/spheretrace.h"

/**
 * Check for all files containing scenes
 * @return List of config file names
 */
std::vector<std::string> get_scene_configs(std::string test_scenes);

/*
 * Loads auxiliary for optimized implementation
 */
scene_config load_config(std::string test_scenes_dir, std::string file_name);

/**
 * Deallocates memory used for a scene
 */
void free_scene(scene_config conf);

/**
 * Prints the config of a scene
 * @param conf Scene config to print
 */
void print_scene_config(scene_config conf);

bool should_run(std::string test_name, std::vector<std::string> scenes);

void print_scene_info(std::vector<std::string> scenes, int render_mode, int mode);

render_mode get_render_mode();

#endif //TEAM07_LOAD_SCENE_H
