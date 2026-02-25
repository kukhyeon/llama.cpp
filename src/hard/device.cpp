#include "device.h"
#include <iostream>
#include <algorithm>

// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

Device::Device(const std::string& device_name) : device(device_name){
    // trim whitespace
    ltrim(this->device);
    rtrim(this->device);
    
    // std::cout << "[DEBUG] Device name: '" << this->device << "'" << std::endl;

    if (this->device == "S22_Ultra" || this->device == "Fold4" ||  this->device == "Pixel9"){
	    cluster_indices = {0, 4, 7};
    } else if (this->device == "S24"){
	    cluster_indices = {0, 4, 7, 9};
    } else if (this->device == "S25"){
        cluster_indices = {0, 6};
    } else {
        std::cerr << "[WARNING] Unknown device name: " << this->device << std::endl;
    }
}

const std::vector<int> Device::get_cluster_indices() const{
    return this->cluster_indices;
}

const std::string Device::get_device_name() const{
    return this->device;
}
