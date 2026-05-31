#include "record.h"
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace {

bool env_flag_enabled(const char * name, bool fallback = false) {
    const char * value = std::getenv(name);
    if (value == nullptr) {
        return fallback;
    }

    std::string s = value;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return s == "1" || s == "true" || s == "on" || s == "yes";
}

std::string env_string(const char * name, const std::string & fallback = {}) {
    const char * value = std::getenv(name);
    return value ? value : fallback;
}

int env_int(const char * name, int fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::atoi(value);
}

long long env_ll(const char * name, long long fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::atoll(value);
}

long long read_int_file(const std::string & path) {
    std::ifstream file(path);
    if (!file) {
        return -1;
    }

    long long value = -1;
    file >> value;
    return value;
}

std::string read_token_file(const std::string & path, const std::string & fallback = "0") {
    std::ifstream file(path);
    if (!file) {
        return fallback;
    }

    std::string value;
    file >> value;
    return value.empty() ? fallback : value;
}

std::string format_number(double value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

std::string read_scaled_file(const std::string & path, double divisor, const std::string & fallback = "0") {
    std::ifstream file(path);
    if (!file) {
        return fallback;
    }

    double value = 0.0;
    file >> value;
    if (!file) {
        return fallback;
    }

    return format_number(value / divisor);
}

std::string strip_colon(std::string value) {
    if (!value.empty() && value.back() == ':') {
        value.pop_back();
    }
    return value;
}

struct thermal_zone_info {
    std::string dir;
    std::string name;
};

std::vector<thermal_zone_info> list_thermal_zones(const DVFS & dvfs) {
    std::vector<thermal_zone_info> zones;
    const std::string thermal_root = "/sys/devices/virtual/thermal";

    std::vector<std::string> empty_thermal = dvfs.get_empty_thermal();

    std::error_code ec;
    for (const auto & entry : std::filesystem::directory_iterator(thermal_root, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_directory()) {
            continue;
        }

        const std::string dir = entry.path().string();
        const std::string base = entry.path().filename().string();
        if (base.rfind("thermal_zone", 0) != 0) {
            continue;
        }

        std::string name = read_token_file(dir + "/type", "");
        if (name.empty()) {
            continue;
        }

        if (name == "qcom,secure-non") {
            name = "secure-non";
        } else if (std::find(empty_thermal.begin(), empty_thermal.end(), name) != empty_thermal.end()) {
            continue;
        }

        zones.push_back({ dir, name });
    }

    std::sort(zones.begin(), zones.end(), [](const thermal_zone_info & a, const thermal_zone_info & b) {
        return a.dir < b.dir;
    });

    return zones;
}

std::vector<std::string> read_meminfo_names() {
    std::vector<std::string> names;
    std::ifstream file("/proc/meminfo");
    std::string key;
    while (file >> key) {
        std::string rest;
        std::getline(file, rest);
        names.push_back(strip_colon(key));
    }
    return names;
}

std::vector<std::string> read_meminfo_values_mib() {
    std::vector<std::string> values;
    std::ifstream file("/proc/meminfo");
    std::string key;
    double value_kib = 0.0;
    std::string unit;

    while (file >> key >> value_kib) {
        std::getline(file, unit);
        values.push_back(format_number(value_kib / 1024.0));
    }

    return values;
}

void append_value(std::vector<std::string> & records, const std::string & path) {
    records.push_back(read_token_file(path));
}

void append_scaled_value(std::vector<std::string> & records, const std::string & path, double divisor) {
    records.push_back(read_scaled_file(path, divisor));
}

void write_csv_row(std::ostream & out, const std::vector<std::string> & data) {
    for (const auto & v : data) {
        out << v << ",";
    }
    out << "\n";
}

void write_csv_row(std::ostream & out, const std::string & data) {
    out << data << "\n";
}

bool write_signal_file(const std::string & path, const std::string & value) {
    if (path.empty()) {
        return false;
    }

    std::ofstream file(path, std::ios::trunc);
    if (!file) {
        std::cerr << "[record] failed to write thermal signal file: " << path
                  << " (" << std::strerror(errno) << ")" << std::endl;
        return false;
    }

    file << value << "\n";
    return true;
}

struct thermal_signal_detector {
    bool enabled = false;
    bool latched = false;
    bool verbose = false;
    int prime_cpu = 6;
    int debounce = 1;
    int drop_count = 0;
    long long high_water_khz = 0;
    long long tolerance_khz = 0;
    std::string cur_path;
    std::string signal_file;
    std::string cool_state = "cool";
    std::string throttled_state = "throttling";

    explicit thermal_signal_detector(const DVFS & dvfs) {
        enabled = env_flag_enabled("IGNITE_THERMAL_SIGNAL", false);
        if (!enabled) {
            return;
        }

        const auto clusters = dvfs.get_cluster_indices();
        if (!clusters.empty()) {
            prime_cpu = clusters.back();
        }
        prime_cpu = env_int("IGNITE_THERMAL_PRIME_CPU", prime_cpu);
        debounce = std::max(1, env_int("IGNITE_THERMAL_DEBOUNCE", debounce));
        tolerance_khz = std::max(0LL, env_ll("IGNITE_THERMAL_TOLERANCE_KHZ", tolerance_khz));
        verbose = env_flag_enabled("IGNITE_THERMAL_VERBOSE", false);

        cur_path = env_string("IGNITE_THERMAL_CUR_FREQ_PATH",
                "/sys/devices/system/cpu/cpu" + std::to_string(prime_cpu) + "/cpufreq/scaling_cur_freq");
        signal_file = env_string("IGNITE_THERMAL_SIGNAL_FILE",
                env_string("LLAMA_BACKEND_POLICY_THERMAL_STATE_FILE", "./thermal_state.txt"));
        cool_state = env_string("IGNITE_THERMAL_STATE_COOL", cool_state);
        throttled_state = env_string("IGNITE_THERMAL_STATE_THROTTLED", throttled_state);

        if (env_flag_enabled("IGNITE_THERMAL_CLEAR_ON_START", true)) {
            write_signal_file(signal_file, cool_state);
        }

        if (verbose) {
            std::cerr << "[record] thermal signal enabled: cpu=" << prime_cpu
                      << ", mode=scaling_cur_freq_drop"
                      << ", tolerance_khz=" << tolerance_khz
                      << ", debounce=" << debounce
                      << ", file=" << signal_file << std::endl;
        }
    }

    void poll() {
        if (!enabled || latched) {
            return;
        }

        const long long cur_khz = read_int_file(cur_path);
        if (cur_khz <= 0) {
            return;
        }

        if (high_water_khz <= 0 || cur_khz > high_water_khz) {
            high_water_khz = cur_khz;
            drop_count = 0;
            return;
        }

        if (cur_khz + tolerance_khz < high_water_khz) {
            drop_count++;
        } else {
            drop_count = 0;
        }

        if (drop_count >= debounce) {
            latched = true;
            write_signal_file(signal_file, throttled_state);
            std::cerr << "[record] thermal throttling detected on cpu" << prime_cpu
                      << ": cur_khz=" << cur_khz
                      << ", previous_high_khz=" << high_water_khz
                      << " -> " << signal_file << "=" << throttled_state << std::endl;
        }
    }
};

} // namespace

// test function
void get_cpu_info() {
    std::cout << read_scaled_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 1000.0) << std::endl;
    std::cout << read_scaled_file("/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq", 1000.0) << std::endl;
}

const std::string get_records_names(const DVFS& dvfs) {
    std::string names = "Time,";

    // thermal info
    for (const auto & zone : list_thermal_zones(dvfs)) {
        names += zone.name + ",";
    }

    // gpu info
    names += "gpu_min_clock,gpu_max_clock,";

    // cpu info
    for(const auto index : dvfs.get_cluster_indices()){
        names += std::string("cpu") + std::to_string(index) + std::string("_max_freq,cpu") + std::to_string(index) + "_cur_freq,";
    }

    // mem info
    for (const auto & name : read_meminfo_names()) {
        names += name + ",";
    }

    // power
    if (dvfs.get_device_name() == "Pixel9") names += "current_now,voltage_now,";
    else names += "power_now,current_now,voltage_now,";

    // RAM clock info
    names += "scaling_devfreq_max,scaling_devfreq_min,cur_freq,";

    // [ADD] LLCC clock info (Only for S25)
    if (dvfs.get_device_name() == "S25") {
        names += "llcc_prime_cur_freq,llcc_gold_cur_freq,llcc_gold_compute_cur_freq,llcc_cur_freq,bwmon_llcc_gold_cur_freq,bwmon_llcc_prime_cur_freq,";
    }

	return names;
}

/* 
 * GET HARD RECORDS function
 * - args 
 *   	- cluster_indices: an integer vector or array to contain cluster indices (ex. {0,4,7})
 * - task
 *   	- Get hard records such as thermal, power, etc.
 * - return
 *   	- A string vector to contain outputs
 * */
std::vector<std::string> get_hard_records(const DVFS& dvfs) {
    std::vector<int> cluster_indices = dvfs.get_cluster_indices();
    std::string device_name = dvfs.get_device_name();
    std::vector<std::string> records;

    // thermal info
    for (const auto & zone : list_thermal_zones(dvfs)) {
        append_scaled_value(records, zone.dir + "/temp", 1000.0);
    }

    // GPU clock info
    if (device_name == "Pixel9"){
        append_value(records, "/sys/devices/platform/1f000000.mali/scaling_min_freq");
        append_value(records, "/sys/devices/platform/1f000000.mali/scaling_max_freq");
    } else if (device_name == "S25") {
        append_value(records, "/sys/class/devfreq/3d00000.qcom,kgsl-3d0/min_freq");
        append_value(records, "/sys/class/devfreq/3d00000.qcom,kgsl-3d0/max_freq");
    } else { // S24
        append_value(records, "/sys/kernel/gpu/gpu_min_clock");
        append_value(records, "/sys/kernel/gpu/gpu_max_clock");
    }

    // CPU clock info
    for (std::size_t i=0; i<cluster_indices.size(); ++i){
        int idx = cluster_indices[i];
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(idx) + "/cpufreq/";
        append_scaled_value(records, base + "scaling_max_freq", 1000.0);
        append_scaled_value(records, base + "scaling_cur_freq", 1000.0);
    }

	// RAM info
    const auto meminfo_values = read_meminfo_values_mib();
    records.insert(records.end(), meminfo_values.begin(), meminfo_values.end());

    // Power Consumption info
    if (device_name == "Pixel9"){
        append_value(records, "/sys/class/power_supply/battery/current_now");
        append_value(records, "/sys/class/power_supply/battery/voltage_now");
    } else {
        append_value(records, "/sys/class/power_supply/battery/power_now"); // pixel does not contain
        append_value(records, "/sys/class/power_supply/battery/current_now");
        append_value(records, "/sys/class/power_supply/battery/voltage_now");
    }

    // RAM clock info (S24/Pixel9/S25)
    if (device_name == "Pixel9"){
        append_scaled_value(records, "/sys/devices/platform/17000010.devfreq_mif/devfreq/17000010.devfreq_mif/max_freq", 1000.0);
        append_scaled_value(records, "/sys/devices/platform/17000010.devfreq_mif/devfreq/17000010.devfreq_mif/scaling_devfreq_min", 1000.0);
        append_scaled_value(records, "/sys/devices/platform/17000010.devfreq_mif/devfreq/17000010.devfreq_mif/cur_freq", 1000.0);
    } else if (device_name == "S24") { // S24
        append_scaled_value(records, "/sys/devices/platform/17000010.devfreq_mif/devfreq/17000010.devfreq_mif/scaling_devfreq_max", 1000.0);
        append_scaled_value(records, "/sys/devices/platform/17000010.devfreq_mif/devfreq/17000010.devfreq_mif/scaling_devfreq_min", 1000.0);
        append_scaled_value(records, "/sys/devices/platform/17000010.devfreq_mif/devfreq/17000010.devfreq_mif/cur_freq", 1000.0);
    } else if (device_name == "S25") { // S25 is held
        append_scaled_value(records, "/sys/devices/system/cpu/bus_dcvs/DDR/soc:qcom,memlat:ddr:prime-latfloor/max_freq", 1000.0);
        append_scaled_value(records, "/sys/devices/system/cpu/bus_dcvs/DDR/soc:qcom,memlat:ddr:prime-latfloor/min_freq", 1000.0);
        append_scaled_value(records, "/sys/devices/system/cpu/bus_dcvs/DDR/cur_freq", 1000.0);

        // [ADD] LLCC info for S25
        append_scaled_value(records, "/sys/devices/system/cpu/bus_dcvs/LLCC/soc:qcom,memlat:llcc:prime/cur_freq", 1000.0);
        append_scaled_value(records, "/sys/devices/system/cpu/bus_dcvs/LLCC/soc:qcom,memlat:llcc:gold/cur_freq", 1000.0);
        append_scaled_value(records, "/sys/devices/system/cpu/bus_dcvs/LLCC/soc:qcom,memlat:llcc:gold-compute/cur_freq", 1000.0);
        append_scaled_value(records, "/sys/devices/system/cpu/bus_dcvs/LLCC/cur_freq", 1000.0);
        append_scaled_value(records, "/sys/devices/system/cpu/bus_dcvs/LLCC/240b3400.qcom,bwmon-llcc-gold/cur_freq", 1000.0);
        append_scaled_value(records, "/sys/devices/system/cpu/bus_dcvs/LLCC/240b7400.qcom,bwmon-llcc-prime/cur_freq", 1000.0);
    }

    return records;
}

std::vector<std::string> get_hard_records_wo_systime(const DVFS& dvfs){
    return get_hard_records(dvfs);
}

void write_file(const std::vector<std::string>& data, std::string output){
    
    // open file append mode
	std::ofstream file(output, std::ios::app);

	// check file open
	if (!file){
		std::cerr << "failed to open file: " << HARD_RECORD_FILE << std::endl;
		return;
	}

	// wrtie file
	for (const auto & v : data){
		file << v << ",";
	}
	file << "\n";

	// close file
	file.close();
}

void write_file(const std::string& data, std::string output){
	// open file append mode
	std::ofstream file(output, std::ios::app);

	// check file open
	if (!file){
		std::cerr << "failed to open file: " << HARD_RECORD_FILE << std::endl;
		return;
	}

	// wrtie file
	file << data << "\n";

	// close file
	file.close();
}


/*
 * 
 * ### This function should be called by background process! 
 * ### sigterm should be true after experiment completion
 * 
 * */
void record_hard(std::atomic<bool>& sigterm, const DVFS& dvfs){
    // pin_current({2}); // generally silver core on mobile
    sigterm = false;
    std::string filename = dvfs.output_filename;
    thermal_signal_detector thermal_signal(dvfs);


	// insert hard names
    std::ofstream file(filename, std::ios::app);
    if (!file) {
        std::cerr << "failed to open file: " << filename << std::endl;
        return;
    }

    write_csv_row(file, get_records_names(dvfs));

    std::vector<std::string> records;
    auto start_sys_time = std::chrono::system_clock::now();
    if (dvfs.control_start_point != dvfs.zero_start_point) {
        start_sys_time = dvfs.control_start_point;
    }

    do{
        thermal_signal.poll();

        // get records
		records = get_hard_records(dvfs);
        auto now = std::chrono::system_clock::now();
		auto sys_time = std::chrono::duration_cast<std::chrono::milliseconds>(now-start_sys_time).count(); // ms base
		records.insert(records.begin(), std::to_string(sys_time)); // insert systime into firstrecord element
        // File write record
        write_csv_row(file, records);
        // wait
        //std::this_thread::sleep_for(std::chrono::milliseconds(170));

    }while(sigterm != true);
}
