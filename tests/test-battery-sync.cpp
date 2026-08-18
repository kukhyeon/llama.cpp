#include "hard/battery-sync.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

namespace {

namespace fs = std::filesystem;
using clock_type = std::chrono::steady_clock;
using milliseconds = std::chrono::milliseconds;

struct temporary_directory {
    fs::path path;

    temporary_directory() {
        const auto suffix = clock_type::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("llama-battery-sync-" + std::to_string(suffix));
        assert(fs::create_directories(path));
    }

    ~temporary_directory() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

void write_text(const fs::path & path, const std::string & value) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path);
    assert(output.is_open());
    output << value;
    assert(output.good());
}

struct fake_runtime {
    clock_type::time_point current {};
    std::vector<battery_temperature_read_result> readings;
    size_t read_index = 0;
    std::atomic<bool> * stop_on_sleep = nullptr;

    battery_temperature_gate_hooks hooks() {
        return {
            [this](const std::string &) {
                assert(!readings.empty());
                const size_t index = std::min(read_index++, readings.size() - 1);
                return readings[index];
            },
            [this] { return current; },
            [this](milliseconds duration) {
                current += duration;
                if (stop_on_sleep != nullptr) {
                    stop_on_sleep->store(true, std::memory_order_relaxed);
                }
            },
        };
    }
};

battery_temperature_read_result value(int64_t raw) {
    return {battery_temperature_read_status::ok, raw, {}};
}

} // namespace

int main() {
    {
        temporary_directory root;
        write_text(root.path / "thermal_zone0/type", "battery_charger\n");
        write_text(root.path / "thermal_zone0/temp", "39000\n");
        write_text(root.path / "thermal_zone1/type", "battery\n");
        write_text(root.path / "thermal_zone1/temp", "32000\n");

        battery_temperature_gate_config config;
        config.thermal_class_path = root.path.string();
        config.battery_power_supply_path = (root.path / "power_supply/battery").string();
        assert(battery_temperature_resolve_path(config) ==
               (root.path / "thermal_zone1/temp").string());
    }

    {
        temporary_directory root;
        write_text(root.path / "power_supply/battery/temperature", "321\n");

        battery_temperature_gate_config config;
        config.thermal_class_path = (root.path / "thermal").string();
        config.battery_power_supply_path = (root.path / "power_supply/battery").string();
        assert(battery_temperature_resolve_path(config) ==
               (root.path / "power_supply/battery/temperature").string());
    }

    {
        temporary_directory root;
        const auto path = root.path / "temp";
        write_text(path, "  -1234 \n");
        const auto reading = battery_temperature_read_raw(path.string());
        assert(reading.status == battery_temperature_read_status::ok);
        assert(reading.value == -1234);

        write_text(path, "32.5\n");
        assert(battery_temperature_read_raw(path.string()).status ==
               battery_temperature_read_status::io_error);
        assert(battery_temperature_read_raw((root.path / "missing").string()).status ==
               battery_temperature_read_status::unavailable);
    }

    {
        battery_temperature_gate_config config;
        config.temperature_path = "fake";
        config.update_count = 2;
        config.poll_interval = milliseconds(10);
        config.settle_delay = milliseconds(20);
        config.timeout = milliseconds(100);

        fake_runtime runtime;
        runtime.readings = {value(300), value(300), value(301), value(301), value(302)};
        const auto result = battery_temperature_wait_for_updates(config, nullptr, runtime.hooks());
        assert(result.status == battery_temperature_gate_status::ready);
        assert(result.updates_observed == 2);
        assert(result.initial_raw_temperature == 300);
        assert(result.final_raw_temperature == 302);
        assert(result.elapsed == milliseconds(60));
    }

    {
        battery_temperature_gate_config config;
        config.temperature_path = "fake";
        config.poll_interval = milliseconds(10);
        config.settle_delay = milliseconds(0);
        config.timeout = milliseconds(25);

        fake_runtime runtime;
        runtime.readings = {value(300)};
        const auto result = battery_temperature_wait_for_updates(config, nullptr, runtime.hooks());
        assert(result.status == battery_temperature_gate_status::timeout);
        assert(result.updates_observed == 0);
        assert(result.elapsed == milliseconds(25));
    }

    {
        battery_temperature_gate_config config;
        config.temperature_path = "fake";
        config.poll_interval = milliseconds(10);
        config.timeout = milliseconds(100);

        fake_runtime runtime;
        runtime.readings = {
            value(300),
            {battery_temperature_read_status::io_error, 0, "injected read failure"},
        };
        const auto result = battery_temperature_wait_for_updates(config, nullptr, runtime.hooks());
        assert(result.status == battery_temperature_gate_status::io_error);
        assert(result.detail == "injected read failure");
    }

    {
        battery_temperature_gate_config config;
        config.temperature_path = "fake";
        config.poll_interval = milliseconds(10);
        config.timeout = milliseconds(100);

        std::atomic<bool> stop { false };
        fake_runtime runtime;
        runtime.readings = {value(300)};
        runtime.stop_on_sleep = &stop;
        const auto result = battery_temperature_wait_for_updates(config, &stop, runtime.hooks());
        assert(result.status == battery_temperature_gate_status::interrupted);
    }

    {
        battery_temperature_gate_config config;
        config.temperature_path = "fake";
        config.update_count = 0;

        fake_runtime runtime;
        runtime.readings = {value(300)};
        const auto result = battery_temperature_wait_for_updates(config, nullptr, runtime.hooks());
        assert(result.status == battery_temperature_gate_status::invalid_config);
        assert(std::string(battery_temperature_gate_status_name(result.status)) == "invalid_config");
    }

    {
        temporary_directory root;
        battery_temperature_gate_config config;
        config.thermal_class_path = (root.path / "missing-thermal").string();
        config.battery_power_supply_path = (root.path / "missing-battery").string();

        const auto result = battery_temperature_wait_for_updates(config);
        assert(result.status == battery_temperature_gate_status::unavailable);
    }

    return 0;
}
