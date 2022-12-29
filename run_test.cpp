#include <iostream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <fstream>
#include <iomanip>
#include <unistd.h>

#include "auxiliary/json.hpp"

#include "src/spheretrace.h"
#include "auxiliary/load_scene.h"

#define EPS_MSE 1

#define TEST_SCENES "test_scenes/"

using namespace std;

/**
 * 
 * Compare the reference buffer to the optimized buffer
 * Assumes array format: r0|g0|b0|r1|g1|b1|.....
 * 
 * Compares difference of single entries (EPS_INNER threshold)
 * and also a Mean-Squared Error between all of them (EPS_MSE)
 * 
 * Returns false if either of the checks fail
 * 
 */
bool eval_buffer(unsigned char *buf, unsigned char *ref, uint32_t len) {
    float MSE = 0;

    for (int i = 0; i < (int) len; i++) {
        int a = buf[i];
        int b = ref[i];
        int d = abs(a - b);

        MSE += d;
    }

    MSE /= len;

    if (MSE > EPS_MSE) {
        return false;
    }

    return true;
}

void create_ppm(int32_t width, int32_t height, Vec* buffer, unsigned char *ppm) {
    for (uint32_t i = 0; i < (uint32_t) width * height; ++i) {
        unsigned char r = static_cast<unsigned char>(std::min(1.0f, buffer[i].x) * 255);
        unsigned char g = static_cast<unsigned char>(std::min(1.0f, buffer[i].y) * 255);
        unsigned char b = static_cast<unsigned char>(std::min(1.0f, buffer[i].z) * 255);
        ppm[3*i] = r;
        ppm[3*i+1] = g;
        ppm[3*i+2] = b;
    }
}

string ppm_header(int32_t width, int32_t height) {
    return "P6\n" + to_string(width) + " " + to_string(height) + "\n255\n";
}

void save_ppm(string filename, int32_t width, int32_t height, unsigned char *ppm) {
    std::ofstream ofs;
    ofs.open(filename);
    ofs << ppm_header(width, height);
    for (uint32_t i = 0; i < (uint32_t) (width * height); ++i) {
        ofs << ppm[3*i] << ppm[3*i+1] << ppm[3*i+2];
    }
    ofs.close();
}

void read_ppm(string filename, unsigned char *ppm) {
    std::ifstream ifs;
    ifs.open(filename);
    string s;
    uint32_t width, height, i;
    ifs >> s >> width >> height >> i;
    if (s != "P6" || width == 0 || height == 0 || i != 255) {
        cout << "invalid format." << endl;
    } else {
        int seek_len = ppm_header(width, height).length();
        ifs.seekg(seek_len);
        unsigned char c;
        for(int i = 0; i < (int) (3 * width * height); i++) {
            c = ifs.get();
            ppm[i] = c;
        }
    }
    ifs.close();
}



int main(int argc, char *argv[]) {
    /**
     * Renders and compares output from defined scenes
     */
    uint32_t width = 640, height = 480;
    mkdir("./out", 0700);
    vector <string> configs = get_scene_configs(TEST_SCENES);

    int c;
    vector<string> scenes;
    render_mode rendermode = Automatic;

    while ((c = getopt(argc, argv, "s:m:")) != -1) {
        char *token;
        switch (c) {
            case 's':
                // split optarg by ","
                token = strtok(optarg, ",");
                while(token != NULL) {
                    scenes.push_back(token);
                    token = strtok(NULL, ",");
                }
                break;
            case 'm':
                rendermode = get_render_mode();
                break;
            default:
                cout << "wrong usage" << endl;
                exit(1);
        }
    }

    print_scene_info(scenes, rendermode, 0);

    for (string &s : configs) {
        scene_config conf = load_config(TEST_SCENES, s);
        // Get test name
        int lastindex = s.find_last_of('.');
        string test_name = s.substr(0, lastindex);
        if (!should_run(test_name, scenes)) continue;
        cout << std::setw (20) << test_name << ": ";
        fflush(stdout);

        Vec *impl_result = new Vec[width * height];
        render_mode chosen_mode = render(impl_result, width, height, &conf, rendermode);

        unsigned char *ppm = new unsigned char[3 * width * height];
        create_ppm(width, height, impl_result, ppm);
        save_ppm("./out/" + test_name + ".ppm", width, height, ppm);
        unsigned char *ref = new unsigned char[3 * width * height];

        read_ppm("./reference/" + test_name + ".ppm", ref);
        bool result = eval_buffer(ppm, ref, 3 * width * height);

        cout << (result ? "✅ done" : "❌ different, check output");

        string chosen_mode_str;
        if(chosen_mode == ObjectVectorization){
            chosen_mode_str = "object";
        }else if(chosen_mode == RayVectorization){
            chosen_mode_str = "ray";
        }else if(chosen_mode == MixedVectorization){
            chosen_mode_str = "mixed";
        }
        cout << " (used " << chosen_mode_str << " vectorization)" << endl << endl;

        free_scene(conf);
    }
}
