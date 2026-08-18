#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>

enum class battery_temperature_read_status {
    ok,
    unavailable,
    io_error,
};

struct battery_temperature_read_result {
    battery_temperature_read_status status = battery_temperature_read_status::io_error;
    int64_t value = 0;
    std::string detail;
};

enum class battery_temperature_gate_status {
    ready,
    timeout,
    unavailable,
    io_error,
    interrupted,
    invalid_config,
};

struct battery_temperature_gate_config {
    // An empty path enables automatic discovery. Automatic discovery first
    // looks for a thermal zone whose type is exactly "battery", then falls
    // back to the Android power_supply battery attributes.
    std::string temperature_path;
    std::string thermal_class_path = "/sys/class/thermal";
    std::string battery_power_supply_path = "/sys/class/power_supply/battery";

    int update_count = 1;
    std::chrono::milliseconds poll_interval { 100 };
    std::chrono::milliseconds settle_delay { 200 };
    std::chrono::milliseconds timeout { 70000 };
};

struct battery_temperature_gate_result {
    battery_temperature_gate_status status = battery_temperature_gate_status::invalid_config;
    std::string temperature_path;
    int updates_observed = 0;
    std::optional<int64_t> initial_raw_temperature;
    std::optional<int64_t> final_raw_temperature;
    std::chrono::milliseconds elapsed { 0 };
    std::string detail;
};

using battery_temperature_reader =
    std::function<battery_temperature_read_result(const std::string & path)>;
using battery_temperature_gate_now =
    std::function<std::chrono::steady_clock::time_point()>;
using battery_temperature_gate_sleep =
    std::function<void(std::chrono::milliseconds duration)>;

struct battery_temperature_gate_hooks {
    battery_temperature_reader read_temperature;
    battery_temperature_gate_now now;
    battery_temperature_gate_sleep sleep_for;
};

const char * battery_temperature_gate_status_name(battery_temperature_gate_status status);

// Returns the explicit configured path unchanged. With no explicit path,
// returns an empty string when no supported battery temperature attribute can
// be found.
std::string battery_temperature_resolve_path(const battery_temperature_gate_config & config);

battery_temperature_read_result battery_temperature_read_raw(const std::string & path);

// The gate detects updates by observing raw value changes. A platform that
// publishes the same value on two consecutive battery updates cannot expose
// that update through a value-only sysfs attribute; in that case this function
// intentionally fails closed with timeout.
battery_temperature_gate_result battery_temperature_wait_for_updates(
        const battery_temperature_gate_config & config,
        const std::atomic<bool> * stop = nullptr);

// Dependency-injected overload for deterministic tests and non-sysfs readers.
battery_temperature_gate_result battery_temperature_wait_for_updates(
        const battery_temperature_gate_config & config,
        const std::atomic<bool> * stop,
        const battery_temperature_gate_hooks & hooks);

