#include "dvfs.h"

namespace {

long long read_first_positive_integer(const std::vector<std::string> & candidates) {
    for (const auto & path : candidates) {
        std::ifstream file(path);
        long long value = -1;
        if (file && (file >> value) && value > 0) {
            return value;
        }
    }
    return -1;
}

} // namespace

// DVFS --------------------------------------
const std::map<std::string, std::map<int, std::vector<int>>> DVFS::cpufreq = {
    { "S22_Ultra", {
        { 0, { 307200, 403200, 518400, 614400, 729600, 844800, 960000, 1075200, 1171200, 1267200, 1363200, 1478400, 1574400, 1689600, 1785600 } },
        { 4, { 633600, 768000, 883200, 998400, 1113600, 1209600, 1324800, 1440000, 1555200, 1651200, 1766400, 1881600, 1996800, 2112000, 2227200, 2342400, 2419200 } },
        { 7, { 806400, 940800, 1056000, 1171200, 1286400, 1401600, 1497600, 1612800, 1728000, 1843200, 1958400, 2054400, 2169600, 2284800, 2400000, 2515200, 2630400, 2726400, 2822400, 2841600 } }
    }},
    { "S24", {
        { 0, { 400000, 576000, 672000, 768000, 864000, 960000, 1056000, 1152000, 1248000, 1344000, 1440000, 1536000, 1632000, 1728000, 1824000, 1920000, 1959000 } },
        { 4, { 672000, 768000, 864000, 960000, 1056000, 1152000, 1248000, 1344000, 1440000, 1536000, 1632000, 1728000, 1824000, 1920000, 2016000, 2112000, 2208000, 2304000, 2400000, 2496000, 2592000 } },
        { 7, { 672000, 768000, 864000, 960000, 1056000, 1152000, 1248000, 1344000, 1440000, 1536000, 1632000, 1728000, 1824000, 1920000, 2016000, 2112000, 2208000, 2304000, 2400000, 2496000, 2592000, 2688000, 2784000, 2880000, 2900000 } },
        { 9, { 672000, 768000, 864000, 960000, 1056000, 1152000, 1248000, 1344000, 1440000, 1536000, 1632000, 1728000, 1824000, 1920000, 2016000, 2112000, 2208000, 2304000, 2400000, 2496000, 2592000, 2688000, 2784000, 2880000, 2976000, 3072000, 3207000 } }
    }},
    { "S25", {
        { 0, { 384000, 556800, 748800, 960000, 1152000, 1363200, 1555200, 1785600, 1996800, 2227200, 2400000, 2745600, 2918400, 3072000, 3321600, 3532800 } },
        { 6, { 1017600, 1209600, 1401600, 1689600, 1958400, 2246400, 2438400, 2649600, 2841600, 3072000, 3283200, 3513600, 3840000, 4089600, 4281600, 4473600 } }
    }},
	{ "Fold4", {
		{ 0, { 300000, 441600, 556800, 691200, 806400, 940800, 1056000, 1132800, 1228800, 1324800, 1440000, 1555200, 1670400, 1804800, 1920000, 2016000} },
		{ 4, { 633600, 768000, 883200, 998400, 1113600, 1209600, 1324800, 1440000, 1555200, 1651200, 1766400, 1881600, 1996800, 2112000, 2227200, 2342400, 2457600, 2572800, 2649600, 2745600 } },
		{ 7, { 787200, 921600, 1036800, 1171200, 1286400, 1401600, 1536000, 1651200, 1766400, 1881600, 1996800, 2131200, 2246400, 2361600, 2476800, 2592000, 2707200, 2822400, 2918400, 2995200 } }
	}},
	{ "Pixel9", {
		{ 0, { 820000, 955000, 1098000, 1197000, 1328000, 1425000, 1548000, 1696000, 1849000, 1950000 } },
		{ 4, { 357000, 578000, 648000, 787000, 910000, 1065000, 1221000, 1328000, 1418000, 1549000, 1795000, 1945000, 2130000, 2245000, 2367000, 2450000, 2600000 } },
		{ 7, { 700000, 1164000, 1396000, 1557000, 1745000, 1885000, 1999000, 2147000, 2294000, 2363000, 2499000, 2687000, 2802000, 2914000, 2943000, 2970000, 3015000, 3105000 } }
	}}
};

const std::map<std::string, std::vector<int>> DVFS::ddrfreq = {
    { "S22_Ultra", { 547000, 768000, 1555000, 1708000, 2092000, 2736000, 3196000 } },
    { "S24", { 421000, 676000, 845000, 1014000, 1352000, 1539000, 1716000, 2028000, 2288000, 2730000, 3172000, 3738000, 4206000 } },
    { "S25", { 547000, 1353000, 1555000, 1708000, 2092000, 2736000, 3187000, 3686000, 4224000, 4761000 } },
    
    { "Fold4", { 547000, 768000, 1555000, 1708000, 2092000, 2736000, 3196000 } },
    { "Pixel9", { 421000, 546000, 676000, 845000, 1014000, 1352000, 1539000, 1716000, 2028000, 2288000, 2730000, 3172000, 3744000 } }
};

const std::map<std::string, std::vector<int>> DVFS::gpufreq = {
    { "S25", { 160000000, 222000000, 342000000, 389000000, 443000000, 525000000, 607000000, 660000000, 734000000, 832000000, 900000000, 967000000, 1050000000, 1100000000, 1200000000 } }
};


const std::map<std::string, std::vector<std::string>> DVFS::empty_thermal = {
    { "S22_Ultra", { "sdr0-pa0", "sdr1-pa0", "pm8350b_tz", "pm8350b-ibat-lvl0", "pm8350b-ibat-lvl1", "pm8350b-bcl-lvl0", "pm8350b-bcl-lvl1", "pm8350b-bcl-lvl2", "socd", "pmr735b_tz"}},
    { "Fold4", { "sdr0-pa0", "sdr1-pa0", "pm8350b_tz", "pm8350b-ibat-lvl0", "pm8350b-ibat-lvl1", "pm8350b-bcl-lvl0", "pm8350b-bcl-lvl1", "pm8350b-bcl-lvl2", "socd", "pmr735b_tz", "qcom,secure-non"}},
    { "S24", {}},
    { "S25", { "camera*", "video*", "pm8550-bcl-lvl*", "mdmss*", "mmw*", "sdr*" }},
    { "Pixel9", {}}
};


// consturctor
DVFS::DVFS(const std::string& device_name) : Device(device_name) { output_filename = ""; }
DVFS::~DVFS() {
    close_s25_clock_snapshot_cache();
    close_fd_cache();
}


const std::map<int, std::vector<int>>& DVFS::get_cpu_freq() const {
    return cpufreq.at(device);
}
const std::vector<std::string>& DVFS::get_empty_thermal() const {
    return empty_thermal.at(device);
}

const std::vector<int>& DVFS::get_ddr_freq() const {
    return ddrfreq.at(device);
}

const std::vector<int>& DVFS::get_gpu_freq() const {
    return gpufreq.at(device);
}

std::vector<int> DVFS::get_cpu_freqs_conf(int prime_cpu_index){
    int prime_cluster_id = this->cluster_indices[this->cluster_indices.size()-1];
    int max_prime_cluster_idx = this->get_cpu_freq().at(prime_cluster_id).size()-1;
    
    // integrity check
    if (prime_cpu_index > max_prime_cluster_idx ){
        std::cerr << "[WARNING] Too big prime_cpu_index: " << prime_cpu_index << " > " << max_prime_cluster_idx << std::endl;
    }


    // generate frequency configuration
    std::vector<int> freq_conf = {};
    for (auto cluster_idx : this->cluster_indices){
        int max_idx = this->get_cpu_freq().at(cluster_idx).size()-1;
        int idx = static_cast<int>(
            std::round(((double)prime_cpu_index/(double)max_prime_cluster_idx)*(double)max_idx)
        );

        freq_conf.push_back(idx);
    }

    return freq_conf;
}

bool DVFS::read_s25_clock_snapshot(S25ClockSnapshot & snapshot) const {
    // Always clear the output first so a failed read cannot leave stale clocks
    // that could accidentally select a runtime FFN profile.
    snapshot = {};

    if (get_device_name() != "S25") {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(s25_clock_read_mu);
        if (s25_clock_read_fds.ready) {
            snapshot.cpu_gold_khz =
                read_fd_positive_integer(s25_clock_read_fds.cpu_gold_fd);
            snapshot.cpu_prime_khz =
                read_fd_positive_integer(s25_clock_read_fds.cpu_prime_fd);
            snapshot.gpu_hz = read_fd_positive_integer(s25_clock_read_fds.gpu_fd);
            return snapshot.cpu_gold_khz > 0 &&
                   snapshot.cpu_prime_khz > 0 &&
                   snapshot.gpu_hz > 0;
        }
    }

    snapshot.cpu_gold_khz = read_first_positive_integer({
        "/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq",
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
        "/sys/devices/system/cpu/cpufreq/policy0/cpuinfo_cur_freq",
        "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq",
    });
    snapshot.cpu_prime_khz = read_first_positive_integer({
        "/sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq",
        "/sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq",
        "/sys/devices/system/cpu/cpufreq/policy6/cpuinfo_cur_freq",
        "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_cur_freq",
    });
    snapshot.gpu_hz = read_first_positive_integer({
        "/sys/class/devfreq/3d00000.qcom,kgsl-3d0/cur_freq",
        "/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq",
        "/sys/devices/platform/soc/3d00000.qcom,kgsl-3d0/devfreq/3d00000.qcom,kgsl-3d0/cur_freq",
    });

    return snapshot.cpu_gold_khz > 0 &&
           snapshot.cpu_prime_khz > 0 &&
           snapshot.gpu_hz > 0;
}

bool DVFS::try_open_read_first(
        const std::vector<std::string> & candidates,
        int & out_fd) {
    for (const auto & path : candidates) {
        const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd >= 0) {
            out_fd = fd;
            return true;
        }
    }
    out_fd = -1;
    return false;
}

long long DVFS::read_fd_positive_integer(int fd) {
    if (fd < 0) {
        return -1;
    }

    char buffer[64];
    ssize_t n;
    do {
        n = pread(fd, buffer, sizeof(buffer) - 1, 0);
    } while (n < 0 && errno == EINTR);
    if (n <= 0) {
        return -1;
    }
    buffer[n] = '\0';

    char * end = nullptr;
    errno = 0;
    const long long value = std::strtoll(buffer, &end, 10);
    return errno == 0 && end != buffer && value > 0 ? value : -1;
}

bool DVFS::init_s25_clock_snapshot_cache() {
    std::lock_guard<std::mutex> lock(s25_clock_read_mu);
    close_s25_clock_snapshot_cache_nolock();

    if (get_device_name() != "S25") {
        return false;
    }

    const bool gold_ok = try_open_read_first({
        "/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq",
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
        "/sys/devices/system/cpu/cpufreq/policy0/cpuinfo_cur_freq",
        "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq",
    }, s25_clock_read_fds.cpu_gold_fd);
    const bool prime_ok = try_open_read_first({
        "/sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq",
        "/sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq",
        "/sys/devices/system/cpu/cpufreq/policy6/cpuinfo_cur_freq",
        "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_cur_freq",
    }, s25_clock_read_fds.cpu_prime_fd);
    const bool gpu_ok = try_open_read_first({
        "/sys/class/devfreq/3d00000.qcom,kgsl-3d0/cur_freq",
        "/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq",
        "/sys/devices/platform/soc/3d00000.qcom,kgsl-3d0/devfreq/3d00000.qcom,kgsl-3d0/cur_freq",
    }, s25_clock_read_fds.gpu_fd);

    if (!gold_ok || !prime_ok || !gpu_ok) {
        close_s25_clock_snapshot_cache_nolock();
        return false;
    }

    s25_clock_read_fds.ready = true;
    return true;
}

void DVFS::close_s25_clock_snapshot_cache() {
    std::lock_guard<std::mutex> lock(s25_clock_read_mu);
    close_s25_clock_snapshot_cache_nolock();
}

void DVFS::close_s25_clock_snapshot_cache_nolock() {
    close_fd(s25_clock_read_fds.cpu_gold_fd);
    close_fd(s25_clock_read_fds.cpu_prime_fd);
    close_fd(s25_clock_read_fds.gpu_fd);
    s25_clock_read_fds.ready = false;
}

bool DVFS::get_s25_clock_targets(
        int cpu_gold_idx,
        int cpu_prime_idx,
        int gpu_idx,
        S25ClockSnapshot & targets) const {
    targets = {};
    if (get_device_name() != "S25" ||
            cpu_gold_idx < 0 || cpu_prime_idx < 0 || gpu_idx < 0) {
        return false;
    }

    const auto & cpu = get_cpu_freq();
    const auto gold_it = cpu.find(0);
    const auto prime_it = cpu.find(6);
    const auto & gpu = get_gpu_freq();
    if (gold_it == cpu.end() || prime_it == cpu.end() ||
            cpu_gold_idx >= (int) gold_it->second.size() ||
            cpu_prime_idx >= (int) prime_it->second.size() ||
            gpu_idx >= (int) gpu.size()) {
        return false;
    }

    targets.cpu_gold_khz = gold_it->second[cpu_gold_idx];
    targets.cpu_prime_khz = prime_it->second[cpu_prime_idx];
    targets.gpu_hz = gpu[gpu_idx];
    return true;
}
// -------------------------------------------


// Collector ----------------------------------
Collector::Collector(const std::string& device_name) : Device(device_name) {}

// pixel9
// BIG: thermal/thermal_zone0
// MID: thermal/thermal_zone1
const std::map<std::string, std::vector<std::string>> Collector::thermal_zones_cpu = {
    { "Pixel9", { /*BIG*/ "/sys/devices/virtual/thermal/thermal_zone0", /*MID*/ "/sys/devices/virtual/thermal/thermal_zone1" } }
};

double Collector::collect_high_temp(){
    if (this->device != "Pixel9") return 0.0;

    std::string command = "su -c \"";
    for (auto zone_path : this->thermal_zones_cpu.at(this->device)){
        command += std::string("awk '{print \\$1/1000}' ")+zone_path+std::string("/temp; ");
    }
    command += "\""; // closing quote

    std::string output = execute_cmd(command.c_str());
    std::vector<std::string> temps = split_string(output);

    // print high temperature
    std::vector<double> temp_vals = {};
    for (auto t_str : temps){
        temp_vals.push_back(std::stod(t_str));
    }

    return std::max_element(temp_vals.begin(), temp_vals.end())[0];
}

// -------------------------------------------


int DVFS::open_wr(const std::string& path) {
    // open with O_CLOEXEC to prevent FD leak to child processes
    int fd = open(path.c_str(), O_WRONLY | O_CLOEXEC);
    if (fd < 0) {
        fprintf(stderr, "[DVFS] open failed: %s (%s)\n", path.c_str(), strerror(errno));
    }
    return fd;
}

void DVFS::close_fd(int& fd) {
    // close fd
    if (fd >= 0) {
        close(fd);
        fd = -1;
    }
}

bool DVFS::try_open_first(const std::vector<std::string>& candidates, int& out_fd) {
    // try open files in candidates sequentially
    for (const auto& p : candidates) {
        int fd = open_wr(p);
        if (fd >= 0) {
            out_fd = fd;
            return true;
        }
    }
    out_fd = -1;
    return false;
}

int DVFS::write_fd_int(int fd, long long v) {
    // write integer status to fd
    
    if (fd < 0) return -1;

    char buf[64];
    int len = snprintf(buf, sizeof(buf), "%lld\n", v);
    if (len <= 0) return -2;

    // sysfs: offset 0 write is safe
    (void)lseek(fd, 0, SEEK_SET);

    const char* p = buf;
    int left = len;
    while (left > 0) {
        ssize_t n = write(fd, p, left);
        if (n < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "[DVFS] write failed (fd=%d): %s\n", fd, strerror(errno));
            return -3;
        }
        p += n;
        left -= (int)n;
    }
    return 0;
}

// 1) FD cache initialization
int DVFS::init_fd_cache() {
    std::lock_guard<std::mutex> lk(io_mu);

    close_fd_cache_nolock(); // if already opened, close first

    // CPU policy fds
    cpu_fds.clear();
    cpu_fds.reserve(cluster_indices.size());

    for (int idx : cluster_indices) {
        CpuPolicyFD p;
        p.policy_idx = idx;

        //  Pixel9 and S24 have same path structure
        const std::string base = "/sys/devices/system/cpu/cpufreq/policy" + std::to_string(idx);
        p.max_fd = open_wr(base + "/scaling_max_freq");
        p.min_fd = open_wr(base + "/scaling_min_freq");

        if (p.max_fd < 0 || p.min_fd < 0) {
            fprintf(stderr, "[DVFS] policy%d open incomplete (need root?)\n", idx);
            close_fd(p.max_fd);
            close_fd(p.min_fd);
            // if failure, close all and return error
            close_fd_cache();
            fd_ready = false;
            return -1;
        }

        cpu_fds.push_back(p);
    }

    // MIF(devfreq) fds (RAM)
    if (get_device_name() == "S25") {
        // S25 uses bus_dcvs RAM voters. Keep the chmod-dependent prime-latfloor nodes
        // excluded, but cache the remaining writable nodes for direct FD-based control.
        s25_ram_fds.ddr_boost_fd = open_wr("/sys/devices/system/cpu/bus_dcvs/DDR/boost_freq");
        s25_ram_fds.ddrqos_boost_fd = open_wr("/sys/devices/system/cpu/bus_dcvs/DDRQOS/boost_freq");

        s25_ram_fds.ddr_gold_min_fd = open_wr("/sys/devices/system/cpu/bus_dcvs/DDR/soc:qcom,memlat:ddr:gold/min_freq");
        s25_ram_fds.ddr_gold_max_fd = open_wr("/sys/devices/system/cpu/bus_dcvs/DDR/soc:qcom,memlat:ddr:gold/max_freq");
        s25_ram_fds.ddr_gold_compute_min_fd = open_wr("/sys/devices/system/cpu/bus_dcvs/DDR/soc:qcom,memlat:ddr:gold-compute/min_freq");
        s25_ram_fds.ddr_gold_compute_max_fd = open_wr("/sys/devices/system/cpu/bus_dcvs/DDR/soc:qcom,memlat:ddr:gold-compute/max_freq");
        s25_ram_fds.ddr_prime_min_fd = open_wr("/sys/devices/system/cpu/bus_dcvs/DDR/soc:qcom,memlat:ddr:prime/min_freq");
        s25_ram_fds.ddr_prime_max_fd = open_wr("/sys/devices/system/cpu/bus_dcvs/DDR/soc:qcom,memlat:ddr:prime/max_freq");

        s25_ram_fds.ddrqos_gold_min_fd = open_wr("/sys/devices/system/cpu/bus_dcvs/DDRQOS/soc:qcom,memlat:ddrqos:gold/min_freq");
        s25_ram_fds.ddrqos_gold_max_fd = open_wr("/sys/devices/system/cpu/bus_dcvs/DDRQOS/soc:qcom,memlat:ddrqos:gold/max_freq");
        s25_ram_fds.ddrqos_prime_min_fd = open_wr("/sys/devices/system/cpu/bus_dcvs/DDRQOS/soc:qcom,memlat:ddrqos:prime/min_freq");
        s25_ram_fds.ddrqos_prime_max_fd = open_wr("/sys/devices/system/cpu/bus_dcvs/DDRQOS/soc:qcom,memlat:ddrqos:prime/max_freq");

        gpu_fds.base = "/sys/class/devfreq/3d00000.qcom,kgsl-3d0";
        gpu_fds.min_fd = open_wr(gpu_fds.base + "/min_freq");
        gpu_fds.max_fd = open_wr(gpu_fds.base + "/max_freq");

        if (s25_ram_fds.ddr_boost_fd < 0 ||
            s25_ram_fds.ddrqos_boost_fd < 0 ||
            s25_ram_fds.ddr_gold_min_fd < 0 ||
            s25_ram_fds.ddr_gold_max_fd < 0 ||
            s25_ram_fds.ddr_gold_compute_min_fd < 0 ||
            s25_ram_fds.ddr_gold_compute_max_fd < 0 ||
            s25_ram_fds.ddr_prime_min_fd < 0 ||
            s25_ram_fds.ddr_prime_max_fd < 0 ||
            s25_ram_fds.ddrqos_gold_min_fd < 0 ||
            s25_ram_fds.ddrqos_gold_max_fd < 0 ||
            s25_ram_fds.ddrqos_prime_min_fd < 0 ||
            s25_ram_fds.ddrqos_prime_max_fd < 0 ||
            gpu_fds.min_fd < 0 ||
            gpu_fds.max_fd < 0) {
            fprintf(stderr, "[DVFS] S25 RAM/GPU voter open failed (need root?)\n");
            close_fd_cache_nolock();
            fd_ready = false;
            return -2;
        }

        fd_ready = true;
        return 0;
    }

    // Pixel 9 and S24 have same base path
    mif_fds.base = "/sys/devices/platform/17000010.devfreq_mif/devfreq/17000010.devfreq_mif";
    {
        // Depending on device and kernel, the min/max path differs
        std::vector<std::string> min_candidates = {
            mif_fds.base + "/scaling_devfreq_min", // S24
            mif_fds.base + "/min_freq", // Pixel9
            mif_fds.base + "/scaling_min_freq"
        };

        std::vector<std::string> max_candidates;
        if (get_device_name() == "Pixel9") {
            max_candidates = {
                mif_fds.base + "/max_freq", // Pixel9 preferred
                mif_fds.base + "/scaling_devfreq_max"
            };
        } else {
            max_candidates = {
                mif_fds.base + "/scaling_devfreq_max", // S24 preferred
                mif_fds.base + "/max_freq"
            };
        }

        if (!try_open_first(min_candidates, mif_fds.min_fd)) {
            fprintf(stderr, "[DVFS] MIF min open failed (need root? path mismatch)\n");
            close_fd_cache();
            fd_ready = false;
            return -2;
        }
        if (!try_open_first(max_candidates, mif_fds.max_fd)) {
            fprintf(stderr, "[DVFS] MIF max open failed (need root? path mismatch)\n");
            close_fd_cache();
            fd_ready = false;
            return -3;
        }
    }

    fd_ready = true;
    return 0;
}

// 2) FD cache cleanup
void DVFS::close_fd_cache() {
    std::lock_guard<std::mutex> lk(io_mu);
    close_fd_cache_nolock(); 
}

void DVFS::close_fd_cache_nolock() {
    // close all cached fds with no lock
    // assume io_mu is already locked
    // to avoid deadlock
    for (auto& p : cpu_fds) {
        close_fd(p.max_fd);
        close_fd(p.min_fd);
    }
    cpu_fds.clear();

    close_fd(mif_fds.min_fd);
    close_fd(mif_fds.max_fd);
    close_fd(gpu_fds.min_fd);
    close_fd(gpu_fds.max_fd);
    close_fd(s25_ram_fds.ddr_boost_fd);
    close_fd(s25_ram_fds.ddrqos_boost_fd);
    close_fd(s25_ram_fds.ddr_gold_min_fd);
    close_fd(s25_ram_fds.ddr_gold_max_fd);
    close_fd(s25_ram_fds.ddr_gold_compute_min_fd);
    close_fd(s25_ram_fds.ddr_gold_compute_max_fd);
    close_fd(s25_ram_fds.ddr_prime_min_fd);
    close_fd(s25_ram_fds.ddr_prime_max_fd);
    close_fd(s25_ram_fds.ddrqos_gold_min_fd);
    close_fd(s25_ram_fds.ddrqos_gold_max_fd);
    close_fd(s25_ram_fds.ddrqos_prime_min_fd);
    close_fd(s25_ram_fds.ddrqos_prime_max_fd);

    fd_ready = false;
}

// 3) set/unset: directly write if FD cache is ready
int DVFS::set_cpu_freq(const std::vector<int>& freq_indices) {
    if ((int)cluster_indices.size() != (int)freq_indices.size()) return 1;

    std::lock_guard<std::mutex> lk(io_mu);

    if (!fd_ready) {
        fprintf(stderr, "[DVFS] fd cache not ready. call init_fd_cache() first.\n");
        return 2;
    }

    // max first, min last (protect min > max being set)
    for (int i = 0; i < (int)cluster_indices.size(); ++i) {
        int policy = cluster_indices[i];
        int freq_idx = freq_indices[i];

        const auto& table = cpufreq.at(device).at(policy);
        if (freq_idx < 0 || freq_idx >= (int)table.size()) return 3;

        int clk = table[freq_idx];

        // search corresponding policy fd (if sequence is identical, cpu_fds[i] can be used directly)
        // safe policy match
        CpuPolicyFD* fdp = nullptr;
        for (auto& p : cpu_fds) if (p.policy_idx == policy) { fdp = &p; break; }
        if (!fdp) return 4;

        if (write_fd_int(fdp->max_fd, clk) != 0) return 5;
        if (write_fd_int(fdp->min_fd, clk) != 0) return 6;
    }
    return 0;
}

int DVFS::unset_cpu_freq() {
    // unset to default (min: lowest, max: highest)

    std::lock_guard<std::mutex> lk(io_mu);

    if (!fd_ready) {
        fprintf(stderr, "[DVFS] fd cache not ready. call init_fd_cache() first.\n");
        return 2;
    }

    for (int policy : cluster_indices) {
        const auto& table = cpufreq.at(device).at(policy);
        int min_clk = table.front();
        int max_clk = table.back();

        CpuPolicyFD* fdp = nullptr;
        for (auto& p : cpu_fds) if (p.policy_idx == policy) { fdp = &p; break; }
        if (!fdp) return 4;

        if (write_fd_int(fdp->max_fd, max_clk) != 0) return 5;
        if (write_fd_int(fdp->min_fd, min_clk) != 0) return 6;
    }
    return 0;
}

int DVFS::set_ram_freq(const int freq_idx) {
    std::lock_guard<std::mutex> lk(io_mu);

    if (!fd_ready) {
        fprintf(stderr, "[DVFS] fd cache not ready. call init_fd_cache() first.\n");
        return 2;
    }

    const auto& table = get_ddr_freq();
    if (freq_idx < 0 || freq_idx >= (int)table.size()) return 1;

    int clk = table[freq_idx];

    if (this->get_device_name() == "S25") {
        if (write_fd_int(s25_ram_fds.ddr_boost_fd, clk) != 0) return 3;
        if (write_fd_int(s25_ram_fds.ddrqos_boost_fd, 0) != 0) return 4;

        if (write_fd_int(s25_ram_fds.ddr_gold_min_fd, clk) != 0) return 5;
        if (write_fd_int(s25_ram_fds.ddr_gold_max_fd, clk) != 0) return 6;
        if (write_fd_int(s25_ram_fds.ddr_gold_compute_min_fd, clk) != 0) return 7;
        if (write_fd_int(s25_ram_fds.ddr_gold_compute_max_fd, clk) != 0) return 8;
        if (write_fd_int(s25_ram_fds.ddr_prime_min_fd, clk) != 0) return 9;
        if (write_fd_int(s25_ram_fds.ddr_prime_max_fd, clk) != 0) return 10;

        if (write_fd_int(s25_ram_fds.ddrqos_gold_min_fd, 1) != 0) return 11;
        if (write_fd_int(s25_ram_fds.ddrqos_gold_max_fd, 1) != 0) return 12;
        if (write_fd_int(s25_ram_fds.ddrqos_prime_min_fd, 1) != 0) return 13;
        if (write_fd_int(s25_ram_fds.ddrqos_prime_max_fd, 1) != 0) return 14;
        return 0;
    }

    // max first, min last (policy-dependent, but this form is generally safe)
    if (write_fd_int(mif_fds.max_fd, clk) != 0) return 3;
    if (write_fd_int(mif_fds.min_fd, clk) != 0) return 4;
    return 0;
}

int DVFS::unset_ram_freq() {
    std::lock_guard<std::mutex> lk(io_mu);

    if (!fd_ready) {
        fprintf(stderr, "[DVFS] fd cache not ready. call init_fd_cache() first.\n");
        return 2;
    }

    const auto& table = get_ddr_freq();
    int min_clk = table.front();
    int max_clk = table.back();

    if (this->get_device_name() == "S25") {
        if (write_fd_int(s25_ram_fds.ddr_boost_fd, min_clk) != 0) return 3;
        if (write_fd_int(s25_ram_fds.ddrqos_boost_fd, 0) != 0) return 4;

        if (write_fd_int(s25_ram_fds.ddr_gold_min_fd, min_clk) != 0) return 5;
        if (write_fd_int(s25_ram_fds.ddr_gold_max_fd, max_clk) != 0) return 6;
        if (write_fd_int(s25_ram_fds.ddr_gold_compute_min_fd, min_clk) != 0) return 7;
        if (write_fd_int(s25_ram_fds.ddr_gold_compute_max_fd, max_clk) != 0) return 8;
        if (write_fd_int(s25_ram_fds.ddr_prime_min_fd, min_clk) != 0) return 9;
        if (write_fd_int(s25_ram_fds.ddr_prime_max_fd, max_clk) != 0) return 10;

        if (write_fd_int(s25_ram_fds.ddrqos_gold_min_fd, 0) != 0) return 11;
        if (write_fd_int(s25_ram_fds.ddrqos_gold_max_fd, 1) != 0) return 12;
        if (write_fd_int(s25_ram_fds.ddrqos_prime_min_fd, 0) != 0) return 13;
        if (write_fd_int(s25_ram_fds.ddrqos_prime_max_fd, 1) != 0) return 14;
        return 0;
    }

    if (write_fd_int(mif_fds.max_fd, max_clk) != 0) return 3;
    if (write_fd_int(mif_fds.min_fd, min_clk) != 0) return 4;
    return 0;
}

int DVFS::set_gpu_freq(const int freq_idx) {
    std::lock_guard<std::mutex> lk(io_mu);

    if (!fd_ready) {
        fprintf(stderr, "[DVFS] fd cache not ready. call init_fd_cache() first.\n");
        return 2;
    }

    auto it = gpufreq.find(device);
    if (it == gpufreq.end()) {
        fprintf(stderr, "[DVFS] GPU DVFS table not found for %s\n", device.c_str());
        return 1;
    }

    const auto& table = it->second;
    if (freq_idx < 0 || freq_idx >= (int)table.size()) return 3;
    if (gpu_fds.min_fd < 0 || gpu_fds.max_fd < 0) return 4;

    const int min_clk = table.front();
    const int max_clk = table.back();
    const int clk = table[freq_idx];

    // Open the constraint window before pinning the target. This avoids min > max
    // failures when switching from a high fixed clock to a lower one.
    if (write_fd_int(gpu_fds.min_fd, min_clk) != 0) return 5;
    if (write_fd_int(gpu_fds.max_fd, max_clk) != 0) return 6;
    if (write_fd_int(gpu_fds.max_fd, clk) != 0) return 7;
    if (write_fd_int(gpu_fds.min_fd, clk) != 0) return 8;
    return 0;
}

int DVFS::unset_gpu_freq() {
    std::lock_guard<std::mutex> lk(io_mu);

    if (!fd_ready) {
        fprintf(stderr, "[DVFS] fd cache not ready. call init_fd_cache() first.\n");
        return 2;
    }

    auto it = gpufreq.find(device);
    if (it == gpufreq.end()) {
        return 0;
    }

    const auto& table = it->second;
    if (gpu_fds.min_fd < 0 || gpu_fds.max_fd < 0) return 3;

    if (write_fd_int(gpu_fds.max_fd, table.back()) != 0) return 4;
    if (write_fd_int(gpu_fds.min_fd, table.front()) != 0) return 5;
    return 0;
}
