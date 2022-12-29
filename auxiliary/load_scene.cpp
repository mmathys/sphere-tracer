#include <vector>
#include <string>
#include <dirent.h>
#include <filesystem>
#include <sstream>
#include <iostream>
#include <fstream>
#include <immintrin.h>
#include <unistd.h>

#include "load_scene.h"
#include "json.hpp"
#include "../src/geometry.h"
#include "../src/spheretrace.h"

using namespace std;
using json = nlohmann::json;

/**
 * Assumes only correctly formatted configs
 * and no other files in directory
 *
 */
vector <string> get_scene_configs(string test_scenes) {
    DIR *path = opendir(test_scenes.c_str());
    vector <string> file_names;

    struct dirent *dp;

    string suffix = ".json";

    while ((dp = readdir(path)) != NULL) {
        string n(dp->d_name);

        // check if ending with suffix
        if (n.size() >= suffix.size()
            && equal(suffix.rbegin(), suffix.rend(), n.rbegin())) {
            file_names.push_back(n);
        }
    }

    closedir(path);
    return file_names;
}

void malloc_obj_coords(obj_coords *coords, uint32_t N) {
    coords->x = new float[N];
    coords->y = new float[N];
    coords->z = new float[N];
}

void free_obj_coords(obj_coords *coords) {
    delete[] coords->x;
    delete[] coords->y;
    delete[] coords->z;
}

void malloc_obj_props(obj_prop *props, uint32_t N) {
    malloc_obj_coords(&props->color, N);
    malloc_obj_coords(&props->pos, N);
    props->reflection = new float[N];    // this should initialize to 0
    props->shininess = new float[N];
    props->rotation_0 = new float[N];
    props->rotation_1 = new float[N];
    props->rotation_2 = new float[N];
    props->rotation_3 = new float[N];
    props->rotation_4 = new float[N];
    props->rotation_5 = new float[N];
    props->rotation_6 = new float[N];
    props->rotation_7 = new float[N];
    props->rotation_8 = new float[N];

}

void free_obj_props(obj_prop *props) {
    free_obj_coords(&props->color);
    free_obj_coords(&props->pos);
    delete[] props->reflection;
    delete[] props->shininess;
    delete[] props->rotation_0;
    delete[] props->rotation_1;
    delete[] props->rotation_2;
    delete[] props->rotation_3;
    delete[] props->rotation_4;
    delete[] props->rotation_5;
    delete[] props->rotation_6;
    delete[] props->rotation_7;
    delete[] props->rotation_8;
}

void fill_prop(obj_prop *prop, json obj, uint32_t i) {

    prop->pos.x[i] = obj["position"]["x"];
    prop->pos.y[i] = obj["position"]["y"];
    prop->pos.z[i] = obj["position"]["z"];

    prop->color.x[i] = obj["color"]["x"];
    prop->color.y[i] = obj["color"]["y"];
    prop->color.z[i] = obj["color"]["z"];

    prop->reflection[i] = obj["reflection"];
    prop->shininess[i] = obj["shininess"];

    float rot_x = obj["rotation"]["x"];
    float rot_y = obj["rotation"]["y"];
    float rot_z = obj["rotation"]["z"];

    compute_rotation_matrix(rot_x, rot_y, rot_z, prop, i);
}

void print_vec(float *vec, int num){
    for(int i = 0; i<num-1; i++){
        printf("%f, ", vec[i]);
    }
    printf("%f", vec[num-1]);

}

scene_config load_config(string test_scenes_dir, string file_name) {
    ifstream file;
    string path = test_scenes_dir + file_name;
    file.open(path);
    if(!file.is_open()) {
        cout << "could not open: " << path << endl;
        exit(1);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    string raw_json = buffer.str();

    json j_config = json::parse(raw_json);

    scene_config conf;

    // load camera conf
    conf.cam_config.fov = j_config["camera"]["fov"];
    conf.cam_config.posx = j_config["camera"]["position"]["x"];
    conf.cam_config.posy = j_config["camera"]["position"]["y"];
    conf.cam_config.posz = j_config["camera"]["position"]["z"];

    conf.cam_config.rotx = j_config["camera"]["rotation"]["x"];
    conf.cam_config.roty = j_config["camera"]["rotation"]["y"];
    conf.cam_config.rotz = j_config["camera"]["rotation"]["z"];

    auto lights = j_config["pointlights"];
    uint32_t N = lights.size();

    conf.lights.len = N;
    conf.lights.pos = new float[3 * N];
    conf.lights.emission = new float[3 * N];

    for (int i = 0; i < (int) N; i++) {
        conf.lights.pos[3 * i] = lights[i]["position"]["x"];
        conf.lights.pos[3 * i + 1] = lights[i]["position"]["y"];
        conf.lights.pos[3 * i + 2] = lights[i]["position"]["z"];

        conf.lights.emission[3 * i] = lights[i]["emission"]["x"];
        conf.lights.emission[3 * i + 1] = lights[i]["emission"]["y"];
        conf.lights.emission[3 * i + 2] = lights[i]["emission"]["z"];
    }

    auto objs = j_config["objects"];
    N = objs.size();

    conf.objs.len = N;

    conf.objs.plane_len = 0;
    conf.objs.torus_len = 0;
    conf.objs.box_len = 0;
    conf.objs.sphere_len = 0;
    conf.objs.boxframe_len = 0;

    uint32_t planes = 0;
    uint32_t tori = 0;
    uint32_t boxes = 0;
    uint32_t spheres = 0;
    uint32_t boxframes = 0;

    for (int i = 0; i < (int) N; i++) {
        string kind = objs[i]["kind"];

        if (kind == "plane") {
            planes++;
        } else if (kind == "box") {
            boxes++;
        } else if (kind == "sphere") {
            spheres++;
        } else if (kind == "boxframe") {
            boxframes++;
        } else if (kind == "torus") {
            tori++;
        }
    }

    malloc_obj_props(&conf.objs.dummy_props, 8);
    conf.objs.dummy_props.reflection = new float[8]();  // initialize to 0

    // sphere
    malloc_obj_props(&conf.objs.sphere_props, spheres);
    conf.objs.sphere_radius = new float[spheres];

    // torus
    malloc_obj_props(&conf.objs.torus_props, tori);
    conf.objs.torus_r1 = new float[tori];
    conf.objs.torus_r2 = new float[tori];

    // box
    malloc_obj_props(&conf.objs.box_props, boxes);
    malloc_obj_coords(&conf.objs.box_extents, boxes);

    // plane
    malloc_obj_props(&conf.objs.plane_props, planes);
    malloc_obj_coords(&conf.objs.plane_normals, planes);

    // boxframe
    malloc_obj_props(&conf.objs.boxframe_props, boxframes);
    malloc_obj_coords(&conf.objs.boxframe_extents, boxframes);
    conf.objs.boxframe_thickness = new float[boxframes];

    for (int i = 0; i < (int) N; i++) {
        string kind = objs[i]["kind"];

        if (kind == "plane") {
            uint32_t j = conf.objs.plane_len;
            fill_prop(&conf.objs.plane_props, objs[i], j);

            conf.objs.plane_normals.x[j] = objs[i]["params"]["normal"]["x"];
            conf.objs.plane_normals.y[j] = objs[i]["params"]["normal"]["y"];
            conf.objs.plane_normals.z[j] = objs[i]["params"]["normal"]["z"];

            conf.objs.plane_len++;
        } else if (kind == "sphere") {
            uint32_t j = conf.objs.sphere_len;
            fill_prop(&conf.objs.sphere_props, objs[i], j);
            conf.objs.sphere_radius[j] = objs[i]["params"]["radius"];

            conf.objs.sphere_len++;
        } else if (kind == "box") {
            uint32_t j = conf.objs.box_len;
            fill_prop(&conf.objs.box_props, objs[i], j);

            conf.objs.box_extents.x[j] = objs[i]["params"]["extents"]["x"];
            conf.objs.box_extents.y[j] = objs[i]["params"]["extents"]["y"];
            conf.objs.box_extents.z[j] = objs[i]["params"]["extents"]["z"];

            conf.objs.box_len++;
        } else if (kind == "torus") {
            uint32_t j = conf.objs.torus_len;
            fill_prop(&conf.objs.torus_props, objs[i], j);

            conf.objs.torus_r1[j] = objs[i]["params"]["r1"];
            conf.objs.torus_r2[j] = objs[i]["params"]["r2"];

            conf.objs.torus_len++;
        } else if (kind == "boxframe") {
            uint32_t j = conf.objs.boxframe_len;
            fill_prop(&conf.objs.boxframe_props, objs[i], j);

            conf.objs.boxframe_extents.x[j] = objs[i]["params"]["extents"]["x"];
            conf.objs.boxframe_extents.y[j] = objs[i]["params"]["extents"]["y"];
            conf.objs.boxframe_extents.z[j] = objs[i]["params"]["extents"]["z"];
            conf.objs.boxframe_thickness[j] = objs[i]["params"]["thickness"];

            conf.objs.boxframe_len++;
        }
    }

    // masked stuff
    conf.objs.box_remaining = conf.objs.box_len % 8;
    conf.objs.plane_remaining = conf.objs.plane_len % 8;
    conf.objs.torus_remaining = conf.objs.torus_len % 8;
    conf.objs.sphere_remaining = conf.objs.sphere_len % 8;
    conf.objs.boxframe_remaining = conf.objs.boxframe_len % 8;

    conf.objs.box_mask = compute_mask(conf.objs.box_remaining);
    conf.objs.plane_mask = compute_mask(conf.objs.plane_remaining);
    conf.objs.torus_mask = compute_mask(conf.objs.torus_remaining);
    conf.objs.sphere_mask = compute_mask(conf.objs.sphere_remaining);
    conf.objs.boxframe_mask = compute_mask(conf.objs.boxframe_remaining);

    return conf;
}

void free_scene(scene_config conf) {
    delete[] conf.lights.pos;
    delete[] conf.lights.emission;

    free_obj_props(&conf.objs.sphere_props);
    delete[] conf.objs.sphere_radius;

    free_obj_props(&conf.objs.box_props);
    free_obj_coords(&conf.objs.box_extents);

    free_obj_props(&conf.objs.plane_props);
    free_obj_coords(&conf.objs.plane_normals);

    free_obj_props(&conf.objs.torus_props);
    delete[] conf.objs.torus_r1;
    delete[] conf.objs.torus_r2;


    free_obj_props(&conf.objs.boxframe_props);
    free_obj_coords(&conf.objs.boxframe_extents);
    delete[] conf.objs.boxframe_thickness;
}

void print_scene_info(vector <string> scenes, int render_mode, int mode) {
    string verb = "";
    if (mode == 0) {
        verb = "testing";
    } else if (mode == 1) {
        verb = "benchmarking";
    }

    if (scenes.size() > 0) {
        cout << "only " << verb << " scene";
        if (scenes.size() > 1) cout << "s";
        for (string scene : scenes) {
            cout << " " << scene;
            if (scene != scenes.back()) cout << ","; // assuming all elements are distinct
        }
        cout << "." << endl;
    } else {
        cout << verb << " all scenes." << endl;
    }
    cout << "render mode: ";
    if (render_mode == ObjectVectorization) {
        cout << "object vectorization";
    } else if(render_mode == RayVectorization) {
        cout << "ray vectorization";
    } else if(render_mode == MixedVectorization) {
        cout << "mixed vectorization";
    } else {
        cout << "automatic";
    }
    cout << endl;
}

bool should_run(string test_name, vector <string> scenes) {
    if (scenes.size() == 0) return true;
    for (string scene : scenes) {
        if (scene == test_name) return true;
    }
    return false;
}

render_mode get_render_mode() {
    if (strcmp("auto", optarg) == 0) {
        return Automatic;
    } else if(strcmp("obj", optarg) == 0) {
        return ObjectVectorization;
    } else if(strcmp("ray", optarg) == 0) {
        return RayVectorization;
    } else if(strcmp("mixed", optarg) == 0) {
        return MixedVectorization;
    } else {
        cout << "rendermode should be \"auto\", \"obj\", \"ray\" or \"mixed\"" << endl;
        exit(1);
    }
}
