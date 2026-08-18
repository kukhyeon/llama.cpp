#include "battery-sync.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using clock_type = std::chrono::steady_clock;
using milliseconds = std::chrono::milliseconds;

std::string trim(std::string value) {
    const auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };

    value.erase(value.begin(), std::find_if(value.begin(), value.end(),
        [&](char ch) { return !is_space(static_cast<unsigned char>(ch)); }));
    value.erase(std::find_if(value.rbegin(), value.rend(),
        [&](char ch) { return !is_space(static_cast<unsigned char>(ch)); }).base(), value.end());
    return value;
}

bool path_exists(const fs::path & path) {
    std::error_code ec;
    return fs::exists(path, ec) && !ec;
}

bool read_first_line(const fs::path & path, std::string & value) {
    std::ifstream input(path);
    return input.is_open() && static_cast<bool>(std::getline(input, value));
}

battery_temperature_gate_result make_result(
        battery_temperature_gate_status status,
        const std::string & path,
        int updates_observed,
        const std::optional<int64_t> & initial_temperature,
        const std::optional<int64_t> & final_temperature,
        clock_type::time_point start,
        const battery_temperature_gate_now & now,
        std::string detail) {
    const auto elapsed = now() - start;
    return {
        status,
        path,
        updates_observed,
        initial_temperature,
        final_temperature,
        std::chrono::duration_cast<milliseconds>(
            std::max(elapsed, clock_type::duration::zero())),
        std::move(detail),
    };
}

} // namespace

const char * battery_temperature_gate_status_name(battery_temperature_gate_status status) {
    switch (status) {
        case battery_temperature_gate_status::ready:          return "ready";
        case battery_temperature_gate_status::timeout:        return "timeout";
        case battery_temperature_gate_status::unavailable:    return "unavailable";
        case battery_temperature_gate_status::io_error:       return "io_error";
        case battery_temperature_gate_status::interrupted:    return "interrupted";
        case battery_temperature_gate_status::invalid_config: return "invalid_config";
    }
    return "unknown";
}

std::string battery_temperature_resolve_path(const battery_temperature_gate_config & config) {
    if (!config.temperature_path.empty()) {
        return config.temperature_path;
    }

    std::vector<fs::path> thermal_zones;
    std::error_code ec;
    fs::directory_iterator iterator(config.thermal_class_path, ec);
    const fs::directory_iterator end;
    for (; !ec && iterator != end; iterator.increment(ec)) {
        thermal_zones.push_back(iterator->path());
    }
    std::sort(thermal_zones.begin(), thermal_zones.end());

    for (const auto & thermal_zone : thermal_zones) {
        std::string type;
        const fs::path type_path = thermal_zone / "type";
        const fs::path temp_path = thermal_zone / "temp";
        if (read_first_line(type_path, type) && trim(std::move(type)) == "battery" && path_exists(temp_path)) {
            return temp_path.string();
        }
    }

    const fs::path battery_root(config.battery_power_supply_path);
    const fs::path temp_path = battery_root / "temp";
    if (path_exists(temp_path)) {
        return temp_path.string();
    }

    const fs::path temperature_path = battery_root / "temperature";
    if (path_exists(temperature_path)) {
        return temperature_path.string();
    }

    return {};
}

battery_temperature_read_result battery_temperature_read_raw(const std::string & path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        if (!path_exists(path)) {
            return {battery_temperature_read_status::unavailable, 0,
                    "temperature path is unavailable: " + path};
        }
        return {battery_temperature_read_status::io_error, 0,
                "cannot open temperature path: " + path};
    }

    std::string text;
    if (!std::getline(input, text)) {
        return {battery_temperature_read_status::io_error, 0,
                "cannot read temperature value: " + path};
    }

    text = trim(std::move(text));
    if (text.empty()) {
        return {battery_temperature_read_status::io_error, 0,
                "empty temperature value: " + path};
    }

    errno = 0;
    char * end = nullptr;
    const long long value = std::strtoll(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || *end != '\0') {
        return {battery_temperature_read_status::io_error, 0,
                "invalid integer temperature value at: " + path};
    }

    return {battery_temperature_read_status::ok, static_cast<int64_t>(value), {}};
}

battery_temperature_gate_result battery_temperature_wait_for_updates(
        const battery_temperature_gate_config & config,
        const std::atomic<bool> * stop) {
    const battery_temperature_gate_hooks hooks {
        battery_temperature_read_raw,
        [] { return clock_type::now(); },
        [](milliseconds duration) { std::this_thread::sleep_for(duration); },
    };
    return battery_temperature_wait_for_updates(config, stop, hooks);
}

battery_temperature_gate_result battery_temperature_wait_for_updates(
        const battery_temperature_gate_config & config,
        const std::atomic<bool> * stop,
        const battery_temperature_gate_hooks & hooks) {
    const bool hooks_valid = static_cast<bool>(hooks.read_temperature) &&
                             static_cast<bool>(hooks.now) &&
                             static_cast<bool>(hooks.sleep_for);
    if (!hooks_valid || config.update_count <= 0 ||
            config.poll_interval <= milliseconds::zero() ||
            config.settle_delay < milliseconds::zero() ||
            config.timeout <= milliseconds::zero()) {
        const auto now = hooks.now ? hooks.now : [] { return clock_type::now(); };
        const auto start = now();
        return make_result(battery_temperature_gate_status::invalid_config, {}, 0,
                           std::nullopt, std::nullopt, start, now,
                           "invalid battery temperature gate configuration");
    }

    const auto start = hooks.now();
    const auto deadline = start + config.timeout;
    const std::string path = battery_temperature_resolve_path(config);
    if (path.empty()) {
        return make_result(battery_temperature_gate_status::unavailable, {}, 0,
                           std::nullopt, std::nullopt, start, hooks.now,
                           "no battery temperature path was found");
    }

    const auto stopped = [&] { return stop != nullptr && stop->load(std::memory_order_relaxed); };
    if (stopped()) {
        return make_result(battery_temperature_gate_status::interrupted, path, 0,
                           std::nullopt, std::nullopt, start, hooks.now,
                           "battery temperature gate was interrupted");
    }

    battery_temperature_read_result reading = hooks.read_temperature(path);
    if (reading.status != battery_temperature_read_status::ok) {
        const auto status = reading.status == battery_temperature_read_status::unavailable
            ? battery_temperature_gate_status::unavailable
            : battery_temperature_gate_status::io_error;
        return make_result(status, path, 0, std::nullopt, std::nullopt,
                           start, hooks.now, std::move(reading.detail));
    }

    const int64_t initial_temperature = reading.value;
    int64_t previous_temperature = reading.value;
    int updates_observed = 0;

    const auto finish = [&](battery_temperature_gate_status status, std::string detail) {
        return make_result(status, path, updates_observed, initial_temperature,
                           previous_temperature, start, hooks.now, std::move(detail));
    };

    const auto sleep_until = [&](clock_type::time_point target) {
        while (true) {
            if (stopped()) {
                return false;
            }
            const auto current = hooks.now();
            if (current >= target) {
                break;
            }
            const auto remaining = std::chrono::ceil<milliseconds>(target - current);
            hooks.sleep_for(std::min(config.poll_interval, remaining));
        }
        return !stopped();
    };

    while (true) {
        const auto now = hooks.now();
        if (now >= deadline) {
            return finish(battery_temperature_gate_status::timeout,
                          "timed out waiting for battery temperature updates");
        }

        if (!sleep_until(std::min(deadline, now + config.poll_interval))) {
            return finish(battery_temperature_gate_status::interrupted,
                          "battery temperature gate was interrupted");
        }

        reading = hooks.read_temperature(path);
        if (reading.status != battery_temperature_read_status::ok) {
            const auto status = reading.status == battery_temperature_read_status::unavailable
                ? battery_temperature_gate_status::unavailable
                : battery_temperature_gate_status::io_error;
            return finish(status, std::move(reading.detail));
        }

        if (reading.value != previous_temperature) {
            previous_temperature = reading.value;
            ++updates_observed;
        }

        if (updates_observed >= config.update_count) {
            const auto settle_target = hooks.now() + config.settle_delay;
            if (settle_target > deadline) {
                return finish(battery_temperature_gate_status::timeout,
                              "battery update was detected, but the settle delay exceeds the timeout");
            }
            if (!sleep_until(settle_target)) {
                return finish(battery_temperature_gate_status::interrupted,
                              "battery temperature gate was interrupted during settle delay");
            }
            return finish(battery_temperature_gate_status::ready,
                          "required battery temperature updates were observed");
        }

        if (hooks.now() >= deadline) {
            return finish(battery_temperature_gate_status::timeout,
                          "timed out waiting for battery temperature updates");
        }
    }
}
