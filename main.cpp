#include <iostream>
#include <vector>
#include <string>
#include <sys/types.h>
#include <fstream>
#include <unistd.h>
#include "auxiliary/json.hpp"

#include "src/spheretrace.h"
#include "auxiliary/load_scene.h"

#define TEST_SCENES "test_scenes/"

using namespace std;

void export_buf(string filename, int32_t width, int32_t height, Vec *buffer) {
    std::ofstream ofs;
    ofs.open(filename);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (uint32_t i = 0; i < (uint32_t) (width * height); ++i) {
        unsigned char r = static_cast<unsigned char>(std::min(1.0f, buffer[i].x) * 255);
        unsigned char g = static_cast<unsigned char>(std::min(1.0f, buffer[i].y) * 255);
        unsigned char b = static_cast<unsigned char>(std::min(1.0f, buffer[i].z) * 255);
        ofs << r << g << b;
    }

    ofs.close();
}

#ifdef ONLY_SCENE_LOAD
void render_dummy(Vec *buffer, const uint32_t width, const uint32_t height, scene_config *scene){
	cout << buffer[0].x << scene->objs.len << endl;
}
#endif


int main(int argc, char** argv)
{
    
    /**
     * Renders and compares output from defined scenes
     */
    uint32_t width = 640;
    uint32_t height = 480;
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
    render_mode rendermode = Automatic;
#pragma GCC diagnostic pop

    std::string s;

    int c;

    // parse commmand line arguments
    while ((c = getopt(argc, argv, "s:w:h:m:")) != -1) {
        switch (c) {
            case 's':
                s = optarg;
                break;
            case 'w':
                width = stoi(optarg);
                break;
            case 'h':
                height = stoi(optarg);
                break;
            case 'm':
                rendermode = get_render_mode();
                break;
            default:
                cout << "Usage: -s scene_name [-w width] [-h height] [-m auto|obj|ray|mixed]" << endl;
                exit(1);
        }
    }

    if(s.empty()){
        cout << "no scene passed" << endl;
        exit(1);
    }


    scene_config conf = load_config("test_scenes/", s + ".json");
    // Get test name
    int lastindex = s.find_last_of('.');
    string scene_name = s.substr(0, lastindex);
    cout << scene_name << ":\t";

    Vec *impl_result = new Vec[width * height];


#ifdef COLLECT_STATS
	// actual rays which will be used
    memset(stats_active_rays_in_mask_counter, 0, sizeof stats_active_rays_in_mask_counter);
	// all the rays existing
	stats_total_rays_computed = 0;
#endif
	
#ifndef ONLY_SCENE_LOAD
    render(impl_result, width, height, &conf, rendermode);
#else
	render_dummy(impl_result, width, height, &conf);	
#endif

    export_buf(scene_name + ".ppm", width, height, impl_result);

#ifdef COLLECT_STATS

	if(stats_total_rays_computed == 0){
	  stats_total_rays_computed = 1;
	}
	
	cout << endl << "Ray vector usage: " << endl;

	for(int i = 0; i < 8; i++){
	  cout << "\t\t" << (i+1) << ": " << (100.*(double)stats_active_rays_in_mask_counter[i]/stats_total_rays_computed) << "%" << endl;
	}

#endif

    cout << "done" << endl;

    free_scene(conf);

	delete[] impl_result;
}
