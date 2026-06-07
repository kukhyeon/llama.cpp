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

struct cooling_device_info {
    std::string dir;
    std::string name;
    int index = -1;
};

int parse_prefixed_index(const std::string & value, const std::string & prefix) {
    if (value.rfind(prefix, 0) != 0 || value.size() == prefix.size()) {
        return -1;
    }
    const auto first_digit = value.begin() + prefix.size();
    if (!std::all_of(first_digit, value.end(), [](unsigned char c) {
        return std::isdigit(c);
    })) {
        return -1;
    }
    return std::atoi(value.c_str() + prefix.size());
}

std::string sanitize_column_token(const std::string & value) {
    std::string out;
    bool last_was_underscore = false;

    for (unsigned char c : value) {
        if (std::isalnum(c)) {
            out.push_back((char) std::tolower(c));
            last_was_underscore = false;
        } else if (!last_was_underscore) {
            out.push_back('_');
            last_was_underscore = true;
        }
    }

    while (!out.empty() && out.back() == '_') {
        out.pop_back();
    }
    while (!out.empty() && out.front() == '_') {
        out.erase(out.begin());
    }

    return out.empty() ? "unknown" : out;
}

bool wildcard_match(const char * pattern, const char * text) {
    const char * star = nullptr;
    const char * retry = nullptr;

    while (*text != '\0') {
        if (*pattern == *text) {
            pattern++;
            text++;
        } else if (*pattern == '*') {
            star = pattern++;
            retry = text;
        } else if (star != nullptr) {
            pattern = star + 1;
            text = ++retry;
        } else {
            return false;
        }
    }

    while (*pattern == '*') {
        pattern++;
    }

    return *pattern == '\0';
}

bool thermal_zone_filtered(const std::string & name, const std::vector<std::string> & filters) {
    for (const auto & filter : filters) {
        if (wildcard_match(filter.c_str(), name.c_str())) {
            return true;
        }
    }
    return false;
}

struct cpu_stat_sample {
    std::string name;
    unsigned long long idle = 0;
    unsigned long long total = 0;
};

bool is_cpu_stat_name(const std::string & name) {
    if (name == "cpu") {
        return true;
    }
    if (name.rfind("cpu", 0) != 0 || name.size() == 3) {
        return false;
    }
    return std::all_of(name.begin() + 3, name.end(), [](unsigned char c) {
        return std::isdigit(c);
    });
}

std::vector<cpu_stat_sample> read_cpu_stat_samples() {
    std::vector<cpu_stat_sample> samples;
    std::ifstream file("/proc/stat");
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string name;
        iss >> name;
        if (!is_cpu_stat_name(name)) {
            break;
        }

        unsigned long long user = 0;
        unsigned long long nice = 0;
        unsigned long long system = 0;
        unsigned long long idle = 0;
        unsigned long long iowait = 0;
        unsigned long long irq = 0;
        unsigned long long softirq = 0;
        unsigned long long steal = 0;

        iss >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
        const unsigned long long idle_all = idle + iowait;
        const unsigned long long total = user + nice + system + idle + iowait + irq + softirq + steal;
        samples.push_back({ name, idle_all, total });
    }

    return samples;
}

std::vector<std::string> read_cpu_util_names() {
    std::vector<std::string> names;
    for (const auto & sample : read_cpu_stat_samples()) {
        if (sample.name == "cpu") {
            names.push_back("cpu_util_pct");
        } else {
            names.push_back(sample.name + "_util_pct");
        }
    }
    return names;
}

std::vector<std::string> read_cpu_util_values() {
    static std::vector<cpu_stat_sample> prev;

    const auto cur = read_cpu_stat_samples();
    std::vector<std::string> values;
    values.reserve(cur.size());

    if (prev.empty()) {
        values.assign(cur.size(), "0");
        prev = cur;
        return values;
    }

    for (const auto & now : cur) {
        auto it = std::find_if(prev.begin(), prev.end(), [&](const cpu_stat_sample & old) {
            return old.name == now.name;
        });

        double util = 0.0;
        if (it != prev.end() && now.total > it->total) {
            const unsigned long long total_delta = now.total - it->total;
            const unsigned long long idle_delta = now.idle >= it->idle ? now.idle - it->idle : 0;
            util = 100.0 * (1.0 - (double) idle_delta / (double) total_delta);
            if (util < 0.0) {
                util = 0.0;
            } else if (util > 100.0) {
                util = 100.0;
            }
        }

        values.push_back(format_number(util));
    }

    prev = cur;
    return values;
}

std::string read_first_token_file(const std::vector<std::string> & paths, const std::string & fallback = "0") {
    for (const auto & path : paths) {
        std::ifstream file(path);
        if (!file) {
            continue;
        }

        std::string value;
        file >> value;
        if (!value.empty()) {
            return value;
        }
    }

    return fallback;
}

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
        } else if (thermal_zone_filtered(name, empty_thermal)) {
            continue;
        }

        zones.push_back({ dir, name });
    }

    std::sort(zones.begin(), zones.end(), [](const thermal_zone_info & a, const thermal_zone_info & b) {
        return a.dir < b.dir;
    });

    return zones;
}

const std::vector<cooling_device_info> & list_cooling_devices() {
    static const std::vector<cooling_device_info> devices = []() {
        std::vector<cooling_device_info> result;
        const std::string thermal_root = "/sys/class/thermal";

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
            if (base.rfind("cooling_device", 0) != 0) {
                continue;
            }

            const std::string name = read_token_file(dir + "/type", "");
            if (name.empty()) {
                continue;
            }

            result.push_back({ dir, name, parse_prefixed_index(base, "cooling_device") });
        }

        std::sort(result.begin(), result.end(), [](const cooling_device_info & a, const cooling_device_info & b) {
            if (a.index != b.index) {
                return a.index < b.index;
            }
            return a.dir < b.dir;
        });

        return result;
    }();

    return devices;
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

    // cooling device mitigation state
    if (env_flag_enabled("IGNITE_RECORD_COOLING", false)) {
        for (const auto & device : list_cooling_devices()) {
            names += "cdev" + std::to_string(device.index) + "_"
                   + sanitize_column_token(device.name) + "_cur_state,";
        }
    }

    // gpu info
    names += "gpu_min_clock,gpu_max_clock,gpu_busy_pct,gpu_load,";

    // cpu info
    for(const auto index : dvfs.get_cluster_indices()){
        names += std::string("cpu") + std::to_string(index) + std::string("_max_freq,cpu") + std::to_string(index) + "_cur_freq,";
    }
    for (const auto & name : read_cpu_util_names()) {
        names += name + ",";
    }

    // mem info
    if (env_flag_enabled("IGNITE_RECORD_MEMINFO", true)) {
        for (const auto & name : read_meminfo_names()) {
            names += name + ",";
        }
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

    // cooling device mitigation state
    if (env_flag_enabled("IGNITE_RECORD_COOLING", false)) {
        for (const auto & device : list_cooling_devices()) {
            append_value(records, device.dir + "/cur_state");
        }
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
    records.push_back(read_first_token_file({
        "/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage",
        "/sys/devices/platform/soc/3d00000.qcom,kgsl-3d0/kgsl/kgsl-3d0/gpu_busy_percentage",
    }));
    records.push_back(read_first_token_file({
        "/sys/class/kgsl/kgsl-3d0/devfreq/gpu_load",
        "/sys/class/devfreq/3d00000.qcom,kgsl-3d0/gpu_load",
        "/sys/devices/platform/soc/3d00000.qcom,kgsl-3d0/devfreq/3d00000.qcom,kgsl-3d0/gpu_load",
    }));

    // CPU clock info
    for (std::size_t i=0; i<cluster_indices.size(); ++i){
        int idx = cluster_indices[i];
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(idx) + "/cpufreq/";
        append_scaled_value(records, base + "scaling_max_freq", 1000.0);
        append_scaled_value(records, base + "scaling_cur_freq", 1000.0);
    }
    const auto cpu_util_values = read_cpu_util_values();
    records.insert(records.end(), cpu_util_values.begin(), cpu_util_values.end());

	// RAM info
    if (env_flag_enabled("IGNITE_RECORD_MEMINFO", true)) {
        const auto meminfo_values = read_meminfo_values_mib();
        records.insert(records.end(), meminfo_values.begin(), meminfo_values.end());
    }

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
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

    }while(sigterm != true);
}
