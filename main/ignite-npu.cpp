#include "arg.h"
#include "common.h"
#include "console.h"
#include "log.h"
#include "query-pacing.h"
#include "sampling.h"
#include "llama.h"
#include "chat.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <time.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <tuple>  // to accumulate json

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#include "nlohmann/json.hpp"


// dvfs library
#include "hard/record.h"
#include "hard/dvfs.h"
#include "hard/utils.h"
#include "hard/affinity.h"

using ignite_json = nlohmann::json;

static llama_context           ** g_ctx;
static llama_model             ** g_model;
static common_sampler          ** g_smpl;
static common_params            * g_params;
static std::vector<llama_token> * g_input_tokens;
static std::ostringstream       * g_output_ss;
static std::vector<llama_token> * g_output_tokens;
static bool is_interacting  = false;
static bool need_insert_eot = false;
std::atomic_bool sigterm(false);

static bool env_flag_enabled(const char * name) {
    const char * env = std::getenv(name);
    if (env == nullptr) {
        return false;
    }

    return std::strcmp(env, "1") == 0 ||
        std::strcmp(env, "on") == 0 || std::strcmp(env, "ON") == 0 ||
        std::strcmp(env, "yes") == 0 || std::strcmp(env, "YES") == 0 ||
        std::strcmp(env, "true") == 0 || std::strcmp(env, "TRUE") == 0;
}

struct runtime_route_clock_producer_state {
    const DVFS * dvfs = nullptr;
    S25ClockSnapshot requested_clocks;
    bool enabled = false;
    bool prefill_active = false;
    int32_t input_tokens = 0;
    size_t query_id = 0;
    uint32_t read_failure_streak = 0;
    uint32_t skip_boundaries = 0;
    uint64_t samples = 0;
    uint64_t read_failures = 0;
    uint64_t route_requests = 0;
};

static bool is_power_of_two(uint32_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

static void runtime_route_clock_layer_producer(
        llama_context * ctx,
        int32_t boundary_layer,
        int32_t ubatch_tokens,
        void * user_data) {
    auto * state = static_cast<runtime_route_clock_producer_state *>(user_data);
    if (state == nullptr || !state->enabled || !state->prefill_active ||
            state->dvfs == nullptr || state->input_tokens <= 0 ||
            ubatch_tokens <= 0) {
        return;
    }

    if (state->skip_boundaries > 0) {
        --state->skip_boundaries;
        return;
    }

    S25ClockSnapshot clocks;
    ++state->samples;
    if (!state->dvfs->read_s25_clock_snapshot(clocks)) {
        ++state->read_failures;
        ++state->read_failure_streak;

        // A missing sysfs node should not add three failed open/read sequences
        // to every remaining layer. Retry with an exponential layer-boundary
        // backoff, capped at one attempt per 16 boundaries, and log only the
        // first and power-of-two failures.
        const uint32_t exponent = std::min<uint32_t>(state->read_failure_streak - 1, 4);
        state->skip_boundaries = (1u << exponent) - 1;
        if (state->read_failure_streak == 1 ||
                is_power_of_two(state->read_failure_streak)) {
            LOG_WRN(
                    "runtime_routes: query=%zu failed to read layer clock snapshot "
                    "after layer=%d (streak=%u, retry_after=%u boundaries)\n",
                    state->query_id,
                    boundary_layer,
                    state->read_failure_streak,
                    state->skip_boundaries);
        }
        return;
    }

    if (state->read_failure_streak > 0) {
        LOG_DBG(
                "runtime_routes: layer clock sampling recovered for query=%zu "
                "after %u failures\n",
                state->query_id,
                state->read_failure_streak);
    }
    state->read_failure_streak = 0;
    state->skip_boundaries = 0;

    const std::string active_profile = llama_runtime_route_active_profile(ctx);
    const auto selection = llama_backend_policy_select_layer_route_profile(
            active_profile.c_str(),
            true,
            state->input_tokens,
            ubatch_tokens,
            clocks.cpu_gold_khz,
            clocks.cpu_prime_khz,
            clocks.gpu_hz);
    if (!selection.matched) {
        return;
    }

    // Publish every matched selection, including the currently active
    // profile. A different profile can still be pending after a min-dwell
    // rejection; republishing the active profile cancels that stale request
    // before the scheduler takes this boundary's single request snapshot.
    const bool requests_change = active_profile != selection.profile;
    const bool accepted = llama_runtime_route_request_profile(ctx, selection.profile);
    if (accepted && requests_change) {
        ++state->route_requests;
    }
    if (!requests_change) {
        if (!accepted) {
            LOG_DBG(
                    "runtime_routes: failed to refresh active profile=%s "
                    "at boundary_layer=%d\n",
                    selection.profile,
                    boundary_layer);
        }
        return;
    }
    const bool thermal_drop =
        clocks.cpu_gold_khz < state->requested_clocks.cpu_gold_khz ||
        clocks.cpu_prime_khz < state->requested_clocks.cpu_prime_khz ||
        clocks.gpu_hz < state->requested_clocks.gpu_hz;
    LOG_DBG(
            "runtime_routes: layer producer query=%zu boundary_layer=%d "
            "clock(gold=%lld prime=%lld gpu=%lld) profile=%s distance=%.6f "
            "thermal_drop=%s request=%s\n",
            state->query_id,
            boundary_layer,
            clocks.cpu_gold_khz,
            clocks.cpu_prime_khz,
            clocks.gpu_hz,
            selection.profile,
            selection.distance,
            thermal_drop ? "yes" : "no",
            accepted ? "accepted" : "rejected");
}

static bool module_bench_done(
        const common_params & params,
        const std::vector<llama_token> & embd,
        int n_consumed,
        const std::vector<llama_token> & embd_inp) {
    if (params.module_bench_type == LLAMA_MODULE_BENCH_OFF) {
        return false;
    }

    const bool module_both_done =
        params.module_bench_phase == LLAMA_MODULE_BENCH_PHASE_BOTH;
    const bool module_prefill_done =
        params.module_bench_phase == LLAMA_MODULE_BENCH_PHASE_PREFILL && embd.size() > 1;
    const bool module_decode_done =
        params.module_bench_phase == LLAMA_MODULE_BENCH_PHASE_DECODE &&
        embd.size() == 1 &&
        n_consumed >= (int) embd_inp.size();

    return module_both_done || module_prefill_done || module_decode_done;
}

static bool should_write_backend_profile_csv(const llama_igparams * ig) {
    return ig != nullptr && ig->backend_compute_profile;
}

static bool should_write_op_breakdown_csv(const llama_igparams * ig) {
    return should_write_backend_profile_csv(ig) &&
        (ig->backend_op_breakdown || env_flag_enabled("IGNITE_CSV_OP_BREAKDOWN"));
}

static bool should_write_op_load_csv() {
    return env_flag_enabled("CSV_OP_LOAD");
}

static bool should_write_op_load_breakdown_csv() {
    return env_flag_enabled("CSV_OP_LOAD_BREAKDOWN");
}

static bool should_write_sched_trace_csv() {
    return env_flag_enabled("SCHED_TRACE") || env_flag_enabled("GGML_SCHED_TRACE");
}

static bool should_write_ffn_worker_trace_csv() {
    return env_flag_enabled("GGML_FFN_WORKER_TRACE") || env_flag_enabled("FFN_WORKER_TRACE");
}

// Appends per-op CSV headers for the optional op breakdown section.
// Each ggml op contributes six columns:
// prefill_cpu, decode_cpu, prefill_htp, decode_htp, prefill_gpu, decode_gpu.
// TODO: move to utils if this CSV formatting is reused outside ignite-npu.
static void append_profile_csv_op_headers(std::ostream & os) {
    for (int op = 0; op < GGML_OP_COUNT; ++op) {
        const char * op_name = ggml_op_name((ggml_op) op);
        if (op_name == nullptr || op_name[0] == '\0') {
            op_name = "unknown";
        }
        os << ",prefill_cpu_op_" << op_name
           << ",decode_cpu_op_" << op_name
           << ",prefill_htp_op_" << op_name
           << ",decode_htp_op_" << op_name
           << ",prefill_gpu_op_" << op_name
           << ",decode_gpu_op_" << op_name;
    }
}

// Appends per-op CSV values matching append_profile_csv_op_headers().
// TODO: move to utils if this CSV formatting is reused outside ignite-npu.
static void append_profile_csv_op_values(std::ostream & os, const ggml_backend_sched_profile_data & prof) {
    for (int op = 0; op < GGML_OP_COUNT; ++op) {
        os << "," << prof.prefill_cpu_ops_by_type[op]
           << "," << prof.decode_cpu_ops_by_type[op]
           << "," << prof.prefill_htp_ops_by_type[op]
           << "," << prof.decode_htp_ops_by_type[op]
           << "," << prof.prefill_gpu_ops_by_type[op]
           << "," << prof.decode_gpu_ops_by_type[op];
    }
}

static void append_op_load_csv_breakdown_headers(std::ostream & os) {
    for (int timer = 0; timer < GGML_BACKEND_OP_LOAD_PROFILE_TIMER_COUNT; ++timer) {
        const char * timer_name = ggml_backend_op_load_profile_timer_name((ggml_backend_op_load_profile_timer) timer);
        os << ",prefill_htp_" << timer_name << "_ms"
           << ",prefill_gpu_" << timer_name << "_ms"
           << ",decode_htp_" << timer_name << "_ms"
           << ",decode_gpu_" << timer_name << "_ms";
    }
}

static void append_op_load_csv_breakdown_values(std::ostream & os, const ggml_backend_op_load_profile_data & prof) {
    for (int timer = 0; timer < GGML_BACKEND_OP_LOAD_PROFILE_TIMER_COUNT; ++timer) {
        os << "," << prof.prefill_htp_timer_ms[timer]
           << "," << prof.prefill_gpu_timer_ms[timer]
           << "," << prof.decode_htp_timer_ms[timer]
           << "," << prof.decode_gpu_timer_ms[timer];
    }
}

static void append_op_load_csv_headers(std::ostream & os, const ggml_backend_op_load_profile_data & prof, bool include_ops, bool include_breakdown) {
    if (include_ops) {
        for (int op = 0; op < GGML_OP_COUNT; ++op) {
            if (!prof.ops_enabled[op]) {
                continue;
            }

            const char * op_name = ggml_op_name((ggml_op) op);
            if (op_name == nullptr || op_name[0] == '\0') {
                op_name = "unknown";
            }

            os << ",prefill_cpu_" << op_name << "_ms"
               << ",prefill_cpu_" << op_name << "_count"
               << ",prefill_htp_" << op_name << "_ms"
               << ",prefill_htp_" << op_name << "_count"
               << ",prefill_gpu_" << op_name << "_ms"
               << ",prefill_gpu_" << op_name << "_count"
               << ",decode_cpu_" << op_name << "_ms"
               << ",decode_cpu_" << op_name << "_count"
               << ",decode_htp_" << op_name << "_ms"
               << ",decode_htp_" << op_name << "_count"
               << ",decode_gpu_" << op_name << "_ms"
               << ",decode_gpu_" << op_name << "_count";
        }

        for (uint32_t i = 0; i < prof.n_name_groups; ++i) {
            const char * group = prof.name_groups[i];
            os << ",prefill_cpu_node_" << group << "_ms"
               << ",prefill_cpu_node_" << group << "_count"
               << ",prefill_htp_node_" << group << "_ms"
               << ",prefill_htp_node_" << group << "_count"
               << ",prefill_gpu_node_" << group << "_ms"
               << ",prefill_gpu_node_" << group << "_count"
               << ",decode_cpu_node_" << group << "_ms"
               << ",decode_cpu_node_" << group << "_count"
               << ",decode_htp_node_" << group << "_ms"
               << ",decode_htp_node_" << group << "_count"
               << ",decode_gpu_node_" << group << "_ms"
               << ",decode_gpu_node_" << group << "_count";
        }
    }

    if (include_breakdown) {
        append_op_load_csv_breakdown_headers(os);
    }
}

static void append_op_load_csv_values(std::ostream & os, const ggml_backend_op_load_profile_data & prof, bool include_ops, bool include_breakdown) {
    if (include_ops) {
        for (int op = 0; op < GGML_OP_COUNT; ++op) {
            if (!prof.ops_enabled[op]) {
                continue;
            }

            os << "," << prof.prefill_cpu_ms_by_type[op]
               << "," << prof.prefill_cpu_count_by_type[op]
               << "," << prof.prefill_htp_ms_by_type[op]
               << "," << prof.prefill_htp_count_by_type[op]
               << "," << prof.prefill_gpu_ms_by_type[op]
               << "," << prof.prefill_gpu_count_by_type[op]
               << "," << prof.decode_cpu_ms_by_type[op]
               << "," << prof.decode_cpu_count_by_type[op]
               << "," << prof.decode_htp_ms_by_type[op]
               << "," << prof.decode_htp_count_by_type[op]
               << "," << prof.decode_gpu_ms_by_type[op]
               << "," << prof.decode_gpu_count_by_type[op];
        }

        for (uint32_t i = 0; i < prof.n_name_groups; ++i) {
            os << "," << prof.prefill_cpu_ms_by_name[i]
               << "," << prof.prefill_cpu_count_by_name[i]
               << "," << prof.prefill_htp_ms_by_name[i]
               << "," << prof.prefill_htp_count_by_name[i]
               << "," << prof.prefill_gpu_ms_by_name[i]
               << "," << prof.prefill_gpu_count_by_name[i]
               << "," << prof.decode_cpu_ms_by_name[i]
               << "," << prof.decode_cpu_count_by_name[i]
               << "," << prof.decode_htp_ms_by_name[i]
               << "," << prof.decode_htp_count_by_name[i]
               << "," << prof.decode_gpu_ms_by_name[i]
               << "," << prof.decode_gpu_count_by_name[i];
        }
    }

    if (include_breakdown) {
        append_op_load_csv_breakdown_values(os, prof);
    }
}

static void llama_op_load_profile_print_custom(
        const std::string & output_filename,
        int query_id,
        std::chrono::time_point<std::chrono::system_clock> start_sys_time,
        bool include_ops,
        bool include_breakdown) {
    auto now_sys_time = std::chrono::system_clock::now();
    auto sys_time = std::chrono::duration_cast<std::chrono::milliseconds>(now_sys_time - start_sys_time).count();

    std::ofstream file(output_filename, std::ios::app);
    if (!file.is_open()) {
        return;
    }

    const auto prof = ggml_backend_op_load_profile_get();
    file << std::to_string(sys_time) << "," << query_id;
    append_op_load_csv_values(file, prof, include_ops, include_breakdown);
    file << "\n";
}

void ctx_kv_cache_clear(struct llama_context * ctx) {
    //llama_kv_cache_clear(ctx); //deprecated
    auto* mem = llama_get_memory(ctx);
    llama_memory_clear(mem, true);
}

std::tuple<int, double, int, double> llama_perf_context_print_custom(const struct llama_context * ctx, const std::string & output_filename, std::chrono::time_point<std::chrono::system_clock> start_sys_time, const llama_igparams * ig) {
    const auto data = llama_perf_context(ctx);
    const double t_end_ms = 1e-3 * ggml_time_us();

    // ("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    // ("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
    //         __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    // LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
    //         __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    // LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));

    // Open the CSV file in append mode.
    // The fixed columns store aggregate throughput/timing counters. Backend
    // profiling columns are appended only when requested by the ignite options.

    // Convert time_point to time_t (seconds since epoch)
    auto now_sys_time = std::chrono::system_clock::now();
    auto sys_time = std::chrono::duration_cast<std::chrono::milliseconds>(now_sys_time-start_sys_time).count();
    const double prefill_speed = data.t_p_eval_ms > 0.0 && data.n_p_eval > 0 ? 1e3 / data.t_p_eval_ms * data.n_p_eval : 0.0;
    const double decode_speed  = data.t_eval_ms   > 0.0 && data.n_eval   > 0 ? 1e3 / data.t_eval_ms   * data.n_eval   : 0.0;

    // system time, prefill speed, decode speed, prefill tokens, decode tokens, ttft
    std::ofstream file(output_filename, std::ios::app);
    if (file.is_open()) {
        file << std::to_string(sys_time) << "," << prefill_speed << "," << decode_speed << ","
              << data.n_p_eval << ","<< data.n_eval << "," << (data.t_p_eval_ms);
        if (should_write_backend_profile_csv(ig)) {
            const auto prof = ggml_backend_sched_profile_get();
            file << "," << prof.prefill_cpu_layers
                 << "," << prof.prefill_htp_layers
                 << "," << prof.prefill_gpu_layers
                 << "," << prof.prefill_cpu_ms
                 << "," << prof.prefill_htp_ms
                 << "," << prof.prefill_gpu_ms
                 << "," << prof.decode_cpu_layers
                 << "," << prof.decode_htp_layers
                 << "," << prof.decode_gpu_layers
                 << "," << prof.decode_cpu_ms
                 << "," << prof.decode_htp_ms
                 << "," << prof.decode_gpu_ms
                 << "," << prof.total_ops
                 << "," << prof.prefill_cpu_ops
                 << "," << prof.decode_cpu_ops
                 << "," << prof.prefill_htp_ops
                 << "," << prof.decode_htp_ops
                 << "," << prof.prefill_gpu_ops
                 << "," << prof.decode_gpu_ops
                 << "," << prof.prefill_copy_ms
                 << "," << prof.prefill_wait_ms
                 << "," << prof.prefill_build_ms
                 << "," << prof.prefill_sampling_ms
                 << "," << prof.decode_copy_ms
                 << "," << prof.decode_wait_ms
                 << "," << prof.decode_build_ms
                 << "," << prof.decode_sampling_ms;
            if (should_write_op_breakdown_csv(ig)) {
                append_profile_csv_op_values(file, prof);
            }
        }
        file << "\n";
        file.close();
    } else {
        // LLAMA_LOG_INFO("Failed to open file: %s\n", output_filename.c_str());
    }

    return std::make_tuple(data.n_p_eval, prefill_speed, data.n_eval, decode_speed);
}

std::vector<std::string> loadQuestions(const std::string &filename) {
// A parsing function for "questions.json" with very simple way
// The following is JSON file type:
// {
//   "questions": [
//     "the first content of question",
//     "the second content of question",
//     "the third content of question"
//   ]
// }
    std::vector<std::string> questions;
    std::ifstream file(filename);
    
    if (!file) {
        std::cerr << "Failed to open " << filename << ". Exiting.\n";
        return questions;
    }

    try {
        ignite_json jsonData;
        file >> jsonData; // JSON parsing

        if (jsonData.contains("questions") && jsonData["questions"].is_array()) {
            for (const auto& item : jsonData["questions"]) {
                if (item.is_string()) {
                    questions.push_back(item.get<std::string>());
                }
            }
        } else {
            std::cerr << "Invalid JSON format: 'data' key missing or not an array\n";
        }
    } catch (const std::exception &e) {
        std::cerr << "JSON parsing error: " << e.what() << "\n";
    }

    return questions;
}

static void print_usage(int argc, char ** argv) {
    (void) argc;

    LOG("\nexample usage:\n");
    LOG("\n  text generation:     %s -m your_model.gguf -p \"I believe the meaning of life is\" -n 128 -no-cnv\n", argv[0]);
    LOG("\n  chat (conversation): %s -m your_model.gguf -sys \"You are a helpful assistant\"\n", argv[0]);
    LOG("\n");
}

static bool file_exists(const std::string & path) {
    std::ifstream f(path.c_str());
    return f.good();
}

static bool file_is_empty(const std::string & path) {
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    return f.tellg() == 0;
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!is_interacting && g_params->interactive) {
            is_interacting  = true;
            need_insert_eot = true;
        } else {
            console::cleanup();
            LOG("\n");
            common_perf_print(*g_ctx, *g_smpl);

            // make sure all logs are flushed
            LOG("Interrupted by user\n");
            common_log_pause(common_log_main());

            _exit(130);
        }
    }
}
#endif

int main(int argc, char ** argv) {
    common_params params;
    g_params = &params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMPLETION, print_usage)) {
        return 1;
    }

    common_init();

    auto & sparams = params.sampling;

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    if (params.embedding) {
        LOG_ERR("************\n");
        LOG_ERR("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        LOG_ERR("************\n\n");

        return 0;
    }

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_WRN("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_WRN("%s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        LOG_WRN("%s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
    }

    const bool csv_op_load = should_write_op_load_csv();
    const bool csv_op_load_breakdown = should_write_op_load_breakdown_csv();
    const bool csv_load = csv_op_load || csv_op_load_breakdown;
    const bool sched_trace = should_write_sched_trace_csv();
    const bool ffn_worker_trace = should_write_ffn_worker_trace_csv();
    const bool any_sched_trace = sched_trace || ffn_worker_trace;
    ggml_backend_op_load_profile_set_enabled(csv_op_load);
    ggml_backend_op_load_profile_set_breakdown_enabled(csv_op_load_breakdown);
    ggml_backend_op_load_profile_set_ops_from_env(std::getenv("CSV_OP_LOAD_OPS"));
    ggml_backend_op_load_profile_set_patterns_from_env(std::getenv("CSV_OP_LOAD_PATTERNS"));
    if (const char * path = std::getenv("SCHED_TRACE_PATH")) {
        ggml_backend_sched_trace_set_path(path);
    } else if (const char * path = std::getenv("GGML_SCHED_TRACE_PATH")) {
        ggml_backend_sched_trace_set_path(path);
    }
    ggml_backend_sched_trace_set_enabled(sched_trace);

    LOG_INF("%s: llama backend init\n", __func__);

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    common_sampler * smpl = nullptr;

    g_model = &model;
    g_ctx = &ctx;
    g_smpl = &smpl;

    std::vector<common_chat_msg> chat_msgs;

    // load the model and apply lora adapter, if any
    LOG_INF("%s: load the model and apply lora adapter, if any\n", __func__);

    auto llama_init = common_init_from_params(params);

    ctx   = llama_init->context();
    model = llama_init->model();
    smpl  = llama_init->sampler(0);

    if (ctx == NULL) {
        LOG_ERR("%s: error: unable to create context\n", __func__);
        return 1;
    }

    auto * ig = get_ignite_params(ctx);
    if (ig == nullptr) {
        LOG_ERR("%s: failed to get ignite params\n", __func__);
        return 1;
    }
    if (env_flag_enabled("IGNITE_CSV_OP_BREAKDOWN")) {
        ig->backend_compute_profile = true;
        ig->backend_op_breakdown = true;
        ggml_backend_sched_profile_set_enabled(true);
    }

    llama_memory_t mem = llama_get_memory(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // note: the time for chat template initialization is not negligible:
    auto chat_templates = common_chat_templates_init(model, params.chat_template);

    // start measuring performance timings from here
    llama_perf_context_reset(ctx);

    LOG_INF("%s: llama threadpool init, n_threads = %d\n", __func__, (int) params.cpuparams.n_threads);

    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        LOG_ERR("%s: no CPU backend found\n", __func__);
        return 1;
    }
    auto * reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto * ggml_threadpool_new_fn = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto * ggml_threadpool_free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
    decltype(ggml_threadpool_prewake) * ggml_threadpool_prewake_fn = nullptr;
    if (params.query_prewake_us > 0) {
        ggml_threadpool_prewake_fn = (decltype(ggml_threadpool_prewake) *)
            ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_prewake");
        if (ggml_threadpool_prewake_fn == nullptr) {
            LOG_WRN("%s: CPU backend does not provide thread-pool pre-wake; continuing without it\n", __func__);
        }
    }

    struct ggml_threadpool_params tpp_batch =
            ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
    struct ggml_threadpool_params tpp =
            ggml_threadpool_params_from_cpu_params(params.cpuparams);

    if (!set_process_priority(params.cpuparams.priority)) {
        LOG_ERR("%s: error: failed to set process priority\n", __func__);
        return 1;
    }

    struct ggml_threadpool * threadpool_batch = NULL;
    if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
        threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
        if (!threadpool_batch) {
            LOG_ERR("%s: batch threadpool create failed : n_threads %d\n", __func__, tpp_batch.n_threads);
            return 1;
        }

        // start the non-batch threadpool in the paused state
        tpp.paused = true;
    }

    struct ggml_threadpool * threadpool = ggml_threadpool_new_fn(&tpp);
    if (!threadpool) {
        LOG_ERR("%s: threadpool create failed : n_threads %d\n", __func__, tpp.n_threads);
        return 1;
    }

    struct ggml_threadpool * prefill_threadpool = threadpool_batch != nullptr ? threadpool_batch : threadpool;
    const uint32_t prefill_threadpool_poll = threadpool_batch != nullptr ? tpp_batch.poll : tpp.poll;
    const int prefill_threadpool_threads = threadpool_batch != nullptr ? tpp_batch.n_threads : tpp.n_threads;
    if (params.query_prewake_us > 0 &&
            (prefill_threadpool_poll == 0 || prefill_threadpool_threads <= 1)) {
        LOG_WRN(
                "%s: query pre-wake requires a prefill CPU pool with polling enabled "
                "and more than one thread; continuing without it (poll=%u, threads=%d)\n",
                __func__, prefill_threadpool_poll, prefill_threadpool_threads);
        ggml_threadpool_prewake_fn = nullptr;
    }

    llama_attach_threadpool(ctx, threadpool, threadpool_batch);

    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    if (n_ctx > n_ctx_train) {
        LOG_WRN("%s: model was trained on only %d context tokens (%d specified)\n", __func__, n_ctx_train, n_ctx);
    }

    // auto enable conversation mode if chat template is available
    const bool has_chat_template = common_chat_templates_was_explicit(chat_templates.get());
    if (params.conversation_mode == COMMON_CONVERSATION_MODE_AUTO) {
        if (has_chat_template) {
            LOG_INF("%s: chat template is available, enabling conversation mode (disable it with -no-cnv)\n", __func__);
            params.conversation_mode = COMMON_CONVERSATION_MODE_ENABLED;
        } else {
            params.conversation_mode = COMMON_CONVERSATION_MODE_DISABLED;
        }
    }

    // in case user force-activate conversation mode (via -cnv) without proper chat template, we show a warning
    if (params.conversation_mode && !has_chat_template) {
        LOG_WRN("%s: chat template is not available or is not supported. This may cause the model to output suboptimal responses\n", __func__);
    }

    // print chat template example in conversation mode
    if (params.conversation_mode) {
        if (params.enable_chat_template) {
            if (!params.prompt.empty() && params.system_prompt.empty()) {
                LOG_WRN("*** User-specified prompt will pre-start conversation, did you mean to set --system-prompt (-sys) instead?\n");
            }

            LOG_INF("%s: chat template example:\n%s\n", __func__, common_chat_format_example(chat_templates.get(), params.use_jinja, params.default_template_kwargs).c_str());
        } else {
            LOG_INF("%s: in-suffix/prefix is specified, chat template will be disabled\n", __func__);
        }
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    std::string path_session = params.path_prompt_cache;
    std::vector<llama_token> session_tokens;

    if (!path_session.empty()) {
        LOG_INF("%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());
        if (!file_exists(path_session)) {
            LOG_INF("%s: session file does not exist, will create.\n", __func__);
        } else if (file_is_empty(path_session)) {
            LOG_INF("%s: The session file is empty. A new session will be initialized.\n", __func__);
        } else {
            // The file exists and is not empty
            session_tokens.resize(n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_state_load_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                LOG_ERR("%s: failed to load session file '%s'\n", __func__, path_session.c_str());
                return 1;
            }
            session_tokens.resize(n_token_count_out);
            LOG_INF("%s: loaded a session with prompt size of %d tokens\n", __func__, (int)session_tokens.size());
        }
    }

    const bool add_bos = llama_vocab_get_add_bos(vocab) && !params.use_jinja;
    if (!llama_model_has_encoder(model)) {
        GGML_ASSERT(!llama_vocab_get_add_eos(vocab));
    }

    LOG_DBG("n_ctx: %d, add_bos: %d\n", n_ctx, add_bos);

    std::vector<llama_token> embd_inp;

    bool waiting_for_first_input = false;
    auto chat_add_and_format = [&chat_msgs, &chat_templates](const std::string & role, const std::string & content) {
        common_chat_msg new_msg;
        new_msg.role = role;
        new_msg.content = content;
        auto formatted = common_chat_format_single(chat_templates.get(), chat_msgs, new_msg, role == "user", g_params->use_jinja);
        chat_msgs.push_back(new_msg);
        LOG_DBG("formatted: '%s'\n", formatted.c_str());
        return formatted;
    };

    std::string prompt;
    {
        if (params.conversation_mode && params.enable_chat_template) {
            if (!params.system_prompt.empty()) {
                // format the system prompt (will use template default if empty)
                chat_add_and_format("system", params.system_prompt);
            }

            if (!params.prompt.empty()) {
                // format and append the user prompt
                chat_add_and_format("user", params.prompt);
            } else {
                waiting_for_first_input = true;
            }

            if (!params.system_prompt.empty() || !params.prompt.empty()) {
                common_chat_templates_inputs inputs;
                inputs.use_jinja = g_params->use_jinja;
                inputs.messages = chat_msgs;
                inputs.add_generation_prompt = !params.prompt.empty();

                prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
            }
        } else {
            // otherwise use the prompt as is
            prompt = params.prompt;
        }

        if (params.interactive_first || !prompt.empty() || session_tokens.empty()) {
            LOG_DBG("tokenize the prompt\n");
            embd_inp = common_tokenize(ctx, prompt, true, true);
        } else {
            LOG_DBG("use session tokens\n");
            embd_inp = session_tokens;
        }

        LOG_DBG("prompt: \"%s\"\n", prompt.c_str());
        LOG_DBG("tokens: %s\n", string_from(ctx, embd_inp).c_str());
    }

    // Should not run without any tokens
    if (!waiting_for_first_input && embd_inp.empty()) {
        if (add_bos) {
            embd_inp.push_back(llama_vocab_bos(vocab));
            LOG_WRN("embd_inp was considered empty and bos was added: %s\n", string_from(ctx, embd_inp).c_str());
        } else {
            LOG_ERR("input is empty\n");
            return -1;
        }
    }

    // Tokenize negative prompt
    if ((int) embd_inp.size() > n_ctx - 4) {
        LOG_ERR("%s: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }

    bool session_do_save = false;

    {
        size_t n_match = 0;

        if (!session_tokens.empty()) {
            for (llama_token id : session_tokens) {
                if (n_match >= embd_inp.size() || id != embd_inp[n_match]) {
                    break;
                }
                n_match++;
            }
            if (params.prompt.empty() && n_match == embd_inp.size()) {
                LOG_INF("%s: using full prompt from session file\n", __func__);
            } else if (n_match >= embd_inp.size()) {
                LOG_INF("%s: session file has exact match for prompt!\n", __func__);
            } else if (n_match < (embd_inp.size() / 2)) {
                LOG_WRN("%s: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                        __func__, n_match, embd_inp.size());
            } else {
                LOG_INF("%s: session file matches %zu / %zu tokens of prompt\n",
                        __func__, n_match, embd_inp.size());
            }

            if (session_tokens.size() == n_match) {
                // [TAG_CONTEXT_STATE_LOGITS]
                // in this case, we are going to reuse the logits from the session
                // if we ever decide to remove the logits from the session, we need to handle this somehow
                // ref: https://github.com/ggml-org/llama.cpp/pull/18862#issuecomment-3756330941
            }

            // remove any "future" tokens that we might have inherited from the previous session
            if (session_tokens.size() > n_match) {
                if (!llama_memory_seq_rm(mem, -1, n_match, -1)) {
                    LOG_WRN("%s: unable to resuse common prefix (for example, when the memory is recurrent)\n", __func__);
                    llama_memory_clear(mem, true);
                    session_tokens.clear();
                    n_match = 0;
                } else {
                    session_tokens.resize(n_match);
                }
            }
        }

        session_do_save = !path_session.empty() && n_match < embd_inp.size() && !params.prompt_cache_ro;
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size()) {
        params.n_keep = (int)embd_inp.size();
    } else {
        params.n_keep += add_bos; // always keep the BOS token
    }

    if (params.conversation_mode) {
        if (params.single_turn && !params.prompt.empty()) {
            params.interactive = false;
            params.interactive_first = false;
        } else {
            params.interactive_first = true;
        }
    }

    // enable interactive mode if interactive start is specified
    if (params.interactive_first) {
        params.interactive = true;
    }

    if (params.verbose_prompt) {
        LOG_INF("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            LOG_INF("%6d -> '%s'\n", embd_inp[i], common_token_to_piece(ctx, embd_inp[i]).c_str());
        }

        if (params.n_keep > add_bos) {
            LOG_INF("%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                LOG_CNT("%s", common_token_to_piece(ctx, embd_inp[i]).c_str());
            }
            LOG_CNT("'\n");
        }
        LOG_INF("\n");
    }

//------------------------------------------------
    // streaming initialization
    std::string json_path = params.json_path;
    std::cout << std::flush << "json_path: " << json_path << std::endl;
    std::string output_path_infer = params.output_dir + "/inference_stats.csv"; // deprecated in future. see L#982
    std::cout << std::flush << "output_path_infer: " << output_path_infer << std::endl;
    std::string output_path_load = params.output_dir + "/inference_load.csv";
    std::cout << std::flush << "output_path_load: " << output_path_load << std::endl;
    std::string output_path_ffn_profile = params.output_dir + "/ffn_profile_trace.csv";
    auto start_sys_time = std::chrono::system_clock::now();
    std::ofstream file(output_path_infer, std::ios::app);
    if (file.is_open() && output_path_infer!="/inference_stats.csv") {
        file << "sys_time,prefill_speed,decode_speed,prefill_token,decode_token,ttft";
        if (should_write_backend_profile_csv(ig)) {
            file << ",prefill_cpu_layers,prefill_htp_layers,prefill_gpu_layers";
            file << ",prefill_cpu_ms,prefill_htp_ms,prefill_gpu_ms";
            file << ",decode_cpu_layers,decode_htp_layers,decode_gpu_layers";
            file << ",decode_cpu_ms,decode_htp_ms,decode_gpu_ms";
            file << ",total_ops,prefill_cpu_ops,decode_cpu_ops,prefill_htp_ops,decode_htp_ops,prefill_gpu_ops,decode_gpu_ops";
            file << ",prefill_copy_ms,prefill_wait_ms,prefill_build_ms,prefill_sampling_ms";
            file << ",decode_copy_ms,decode_wait_ms,decode_build_ms,decode_sampling_ms";
            if (should_write_op_breakdown_csv(ig)) {
                append_profile_csv_op_headers(file);
            }
        }
        file << "\n";
        file.close();
    }
    if (csv_load && output_path_load != "/inference_load.csv") {
        std::ofstream load_file(output_path_load, std::ios::app);
        if (load_file.is_open()) {
            load_file << "sys_time,query_id";
            append_op_load_csv_headers(load_file, ggml_backend_op_load_profile_get(), csv_op_load, csv_op_load_breakdown);
            load_file << "\n";
            load_file.close();
        }
    }

    // dummy dvfs object
    const std::string device_name =
        std::strlen(ig->device_name) > 0 ? ig->device_name : "S25";

    DVFS dvfs(device_name);
    dvfs.control_start_point = start_sys_time;
    dvfs.output_filename = params.output_dir + "/hardware_stats.csv";

    const bool ffn_clock_switch_enabled = llama_backend_policy_ffn_clock_switch_enabled();
    const bool runtime_route_clock_enabled = llama_backend_policy_runtime_route_clock_enabled();
    S25ClockSnapshot requested_prefill_clocks;
    const bool requested_prefill_clocks_valid = dvfs.get_s25_clock_targets(
            params.cpu_gold_clk_idx_p,
            params.cpu_prime_clk_idx_p,
            ig->gpu_clk_idx_p,
            requested_prefill_clocks);
    const bool clock_snapshot_cache_ready =
        (ffn_clock_switch_enabled || runtime_route_clock_enabled) &&
        dvfs.init_s25_clock_snapshot_cache();
    if ((ffn_clock_switch_enabled || runtime_route_clock_enabled) &&
            !clock_snapshot_cache_ready) {
        LOG_WRN(
                "%s: failed to cache S25 current-clock descriptors; clock "
                "selection will use rate-limited fallback reads\n",
                __func__);
    }
    if (ffn_clock_switch_enabled && !requested_prefill_clocks_valid) {
        LOG_WRN(
                "%s: FFN clock switching requires valid S25 prefill Gold/Prime/GPU "
                "indices; keeping the base policy until targets are available\n",
                __func__);
    }
    if (runtime_route_clock_enabled && !requested_prefill_clocks_valid) {
        LOG_WRN(
                "%s: layer-route clock switching requires valid S25 prefill "
                "Gold/Prime/GPU indices; keeping the initial route\n",
                __func__);
    }

    runtime_route_clock_producer_state route_clock_producer;
    bool route_clock_producer_registered = false;
    if (runtime_route_clock_enabled && requested_prefill_clocks_valid) {
        route_clock_producer.dvfs = &dvfs;
        route_clock_producer.requested_clocks = requested_prefill_clocks;
        route_clock_producer.enabled = true;
        route_clock_producer_registered = llama_runtime_route_set_layer_producer(
                ctx, runtime_route_clock_layer_producer, &route_clock_producer);
        if (!route_clock_producer_registered) {
            route_clock_producer.enabled = false;
            LOG_WRN(
                    "%s: failed to register the prefill layer clock producer; "
                    "runtime routes will remain on their initial profile\n",
                    __func__);
        }
    }

    std::ofstream ffn_profile_trace;
    if (ffn_clock_switch_enabled && output_path_ffn_profile != "/ffn_profile_trace.csv") {
        ffn_profile_trace.open(output_path_ffn_profile, std::ios::app);
        if (ffn_profile_trace.is_open()) {
            if (ffn_profile_trace.tellp() == std::streampos(0)) {
                ffn_profile_trace
                    << "sys_time_ms,query_id,input_tokens,ubatch_tokens,"
                    << "requested_gold_idx,requested_prime_idx,requested_gpu_idx,"
                    << "requested_gold_khz,requested_prime_khz,requested_gpu_hz,"
                    << "actual_gold_khz,actual_prime_khz,actual_gpu_hz,"
                    << "thermal_throttled,matched,pending_changed,profile,distance\n";
                ffn_profile_trace.flush();
            }
        } else {
            LOG_WRN("%s: failed to open FFN profile trace '%s'\n",
                    __func__, output_path_ffn_profile.c_str());
        }
    }

    const bool want_prefill_cpu_cluster_dvfs =
        params.cpu_gold_clk_idx_p >= 0 || params.cpu_prime_clk_idx_p >= 0;
    const bool want_decode_cpu_cluster_dvfs =
        params.cpu_gold_clk_idx_d >= 0 || params.cpu_prime_clk_idx_d >= 0;
    const bool want_cpu_dvfs =
        ig->cpu_clk_idx_p >= 0 || ig->cpu_clk_idx_d >= 0 ||
        want_prefill_cpu_cluster_dvfs || want_decode_cpu_cluster_dvfs;
    const bool want_ram_dvfs = ig->ram_clk_idx_p >= 0 || ig->ram_clk_idx_d >= 0;
    const bool want_gpu_dvfs = ig->gpu_clk_idx_p >= 0 || ig->gpu_clk_idx_d >= 0;
    const bool want_prefill_dvfs =
        ig->cpu_clk_idx_p >= 0 || want_prefill_cpu_cluster_dvfs ||
        ig->ram_clk_idx_p >= 0 || ig->gpu_clk_idx_p >= 0;
    const bool want_decode_dvfs =
        ig->cpu_clk_idx_d >= 0 || want_decode_cpu_cluster_dvfs ||
        ig->ram_clk_idx_d >= 0 || ig->gpu_clk_idx_d >= 0;
    bool runtime_dvfs_ready = false;

    if (want_prefill_dvfs || want_decode_dvfs) {
        runtime_dvfs_ready = (dvfs.init_fd_cache() == 0);
        if (!runtime_dvfs_ready) {
            LOG_WRN("%s: failed to init DVFS for %s, continuing without runtime DVFS\n",
                    __func__, device_name.c_str());
        }
    }

    auto apply_dvfs = [&](int cpu_idx, int cpu_gold_idx, int cpu_prime_idx, int ram_idx, int gpu_idx) {
        if (!runtime_dvfs_ready) {
            return;
        }

        if (cpu_gold_idx >= 0 || cpu_prime_idx >= 0) {
            if (device_name != "S25") {
                LOG_WRN("%s: separate Gold/Prime CPU DVFS indices are only supported for S25\n", __func__);
            } else if (cpu_gold_idx < 0 || cpu_prime_idx < 0) {
                LOG_WRN("%s: Gold and Prime CPU DVFS indices must both be set\n", __func__);
            } else if (dvfs.set_cpu_freq({cpu_gold_idx, cpu_prime_idx}) != 0) {
                LOG_WRN("%s: failed to set Gold/Prime CPU DVFS indices %d/%d\n",
                        __func__, cpu_gold_idx, cpu_prime_idx);
            }
        } else if (cpu_idx >= 0) {
            auto conf = dvfs.get_cpu_freqs_conf(cpu_idx);
            if (dvfs.set_cpu_freq(conf) != 0) {
                LOG_WRN("%s: failed to set CPU DVFS index %d\n", __func__, cpu_idx);
            }
        }

        if (ram_idx >= 0) {
            if (dvfs.set_ram_freq(ram_idx) != 0) {
                LOG_WRN("%s: failed to set RAM DVFS index %d\n", __func__, ram_idx);
            }
        }

        if (gpu_idx >= 0) {
            if (dvfs.set_gpu_freq(gpu_idx) != 0) {
                LOG_WRN("%s: failed to set GPU DVFS index %d\n", __func__, gpu_idx);
            }
        }
    };

    auto reset_dvfs = [&]() {
        if (!runtime_dvfs_ready) {
            return;
        }
        if (want_cpu_dvfs) {
            dvfs.unset_cpu_freq();
        }
        if (want_ram_dvfs) {
            dvfs.unset_ram_freq();
        }
        if (want_gpu_dvfs) {
            dvfs.unset_gpu_freq();
        }
    };

    if (ig->is_ignite_active && want_prefill_dvfs) {
        apply_dvfs(
            ig->cpu_clk_idx_p,
            params.cpu_gold_clk_idx_p,
            params.cpu_prime_clk_idx_p,
            ig->ram_clk_idx_p,
            ig->gpu_clk_idx_p);
    }

    #if IGNITE_USE_SYSTEM_DVFS
    std::thread record_thread;
    if (params.hardware_stats) {
        sigterm = false;
        record_thread = std::thread(
            record_hard,
            std::ref(sigterm),
            std::cref(dvfs),
            params.hardware_stats_core);
    }
    #endif

    // Input json file instead of cli input
    std::vector<std::string> json_questions;
    size_t current_question_index = 0;
    if (params.interactive) {
        json_questions = loadQuestions(json_path);
        // if (json_questions.empty()) {
        //     LOG_ERR("No questions loaded from %s. Exiting interactive mode.\n", json_path.c_str());
        //     return 1;
        // }
    }
    bool custom_max_query = params.max_query_number == -1 ? false : true;
    size_t max_query_num = custom_max_query ?
        std::min((size_t) params.max_query_number, json_questions.size()) :
        json_questions.size();
    if (custom_max_query && (size_t) params.max_query_number > json_questions.size()) {
        LOG_WRN(
                "requested %d queries, but %zu JSON questions are available; "
                "limiting the run to %zu queries\n",
                params.max_query_number,
                json_questions.size(),
                max_query_num);
    }
    // JSON questions load done
//------------------------------------------------

    // ctrl+C handling
    {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

// ------------------------------------------------
    // timer variables
    std::chrono::steady_clock::time_point inference_start_time;
    bool inference_started = false;

    struct query_timing_sample {
        size_t query_id = 0;
        std::chrono::steady_clock::time_point scheduled_start;
        std::chrono::steady_clock::time_point actual_start;
        std::chrono::steady_clock::time_point inference_end;
        std::chrono::steady_clock::time_point postprocess_end;
        std::chrono::steady_clock::duration prewake_actual_lead = std::chrono::steady_clock::duration::zero();
        bool prewake_issued = false;
        bool valid = false;
    };

    struct query_timing_record {
        query_timing_sample sample;
        std::chrono::steady_clock::duration cooling;
        std::chrono::steady_clock::duration pacing_wait;
        std::chrono::steady_clock::duration overrun;
        bool deadline_miss = false;
        bool next_query_scheduled = false;
        const char * status = "";
    };

    const bool query_pacing_enabled = ig->query_interval > 0 && !json_questions.empty();
    const std::chrono::milliseconds query_period(std::max(ig->query_interval, 0));
    const std::chrono::microseconds query_prewake_lead(std::max(params.query_prewake_us, 0));
    const bool query_prewake_enabled = query_pacing_enabled &&
        query_prewake_lead > std::chrono::microseconds::zero() &&
        ggml_threadpool_prewake_fn != nullptr;
    const std::chrono::milliseconds query_wake_lateness_tolerance(10);
    std::optional<std::ofstream> query_timing_file;
    std::vector<query_timing_record> query_timing_records;
    std::chrono::steady_clock::time_point query_schedule_epoch;
    query_timing_sample query_timing;
    bool query_schedule_started = false;
    int query_pacing_exit_status = 0;

    if (query_pacing_enabled) {
        const std::string output_path_query_timing = params.output_dir + "/query_timing.csv";
        query_timing_records.reserve(max_query_num);
        query_timing_file.emplace(output_path_query_timing, std::ios::app);
        if (query_timing_file->is_open()) {
            if (query_timing_file->tellp() == std::streampos(0)) {
                // cooling_ms spans device completion to the next query start;
                // pacing_wait_ms is the subset after stats/trace processing.
                *query_timing_file
                    << "query_id,period_ms,scheduled_start_ms,actual_start_ms,"
                    << "inference_end_ms,postprocess_end_ms,service_ms,postprocess_ms,"
                    << "cooling_ms,pacing_wait_ms,start_lateness_ms,overrun_ms,"
                    << "deadline_miss,next_query_scheduled";
                if (query_prewake_enabled) {
                    // These fields describe the pre-wake for this row's own start.
                    *query_timing_file
                        << ",prewake_issued,prewake_requested_us,prewake_actual_lead_us";
                }
                *query_timing_file << ",status\n";
            }
        } else {
            LOG_WRN("query_pacing: failed to open timing log %s\n", output_path_query_timing.c_str());
        }
    }

    auto query_duration_ms = [](std::chrono::steady_clock::duration duration) {
        return std::chrono::duration<double, std::milli>(duration).count();
    };

    auto query_duration_us = [](std::chrono::steady_clock::duration duration) {
        return std::chrono::duration<double, std::micro>(duration).count();
    };

    auto queue_query_timing = [&](const query_timing_sample & sample,
                                  std::chrono::steady_clock::duration cooling,
                                  std::chrono::steady_clock::duration pacing_wait,
                                  std::chrono::steady_clock::duration overrun,
                                  bool deadline_miss,
                                  bool next_query_scheduled,
                                  const char * status) {
        if (!sample.valid || !query_timing_file || !query_timing_file->is_open()) {
            return;
        }
        query_timing_records.push_back({
                sample,
                cooling,
                pacing_wait,
                overrun,
                deadline_miss,
                next_query_scheduled,
                status});
    };

    if (query_pacing_enabled) {
        LOG_INF(
                "query_pacing: fixed JSON query start period=%lld ms; "
                "the run stops on work overrun or wake lateness above %lld ms\n",
                (long long) query_period.count(),
                (long long) query_wake_lateness_tolerance.count());
    }
    if (params.query_prewake_us > 0) {
        if (query_prewake_enabled) {
            LOG_INF(
                    "query_pacing: prefill CPU thread-pool pre-wake enabled at %lld us before each paced query\n",
                    (long long) query_prewake_lead.count());
        } else if (!query_pacing_enabled) {
            LOG_WRN("query_pacing: query pre-wake requested without an active query period; ignoring it\n");
        }
    }
    // prefill/decode detector variables
    bool generation_started = false;
    bool prefill_active = false;
    bool decode_active = false;
// ------------------------------------------------

    if (params.interactive) {
        LOG_INF("%s: interactive mode on.\n", __func__);

        if (!params.antiprompt.empty()) {
            for (const auto & antiprompt : params.antiprompt) {
                LOG_INF("Reverse prompt: '%s'\n", antiprompt.c_str());
                if (params.verbose_prompt) {
                    auto tmp = common_tokenize(ctx, antiprompt, false, true);
                    for (int i = 0; i < (int) tmp.size(); i++) {
                        LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                    }
                }
            }
        }

        if (params.input_prefix_bos) {
            LOG_INF("Input prefix with BOS\n");
        }

        if (!params.input_prefix.empty()) {
            LOG_INF("Input prefix: '%s'\n", params.input_prefix.c_str());
            if (params.verbose_prompt) {
                auto tmp = common_tokenize(ctx, params.input_prefix, true, true);
                for (int i = 0; i < (int) tmp.size(); i++) {
                    LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                }
            }
        }

        if (!params.input_suffix.empty()) {
            LOG_INF("Input suffix: '%s'\n", params.input_suffix.c_str());
            if (params.verbose_prompt) {
                auto tmp = common_tokenize(ctx, params.input_suffix, false, true);
                for (int i = 0; i < (int) tmp.size(); i++) {
                    LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                }
            }
        }
    }

    LOG_INF("sampler seed: %u\n",     common_sampler_get_seed(smpl));
    LOG_INF("sampler params: \n%s\n", sparams.print().c_str());
    LOG_INF("sampler chain: %s\n",    common_sampler_print(smpl).c_str());

    LOG_INF("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);

    // group-attention state
    // number of grouped KV tokens so far (used only if params.grp_attn_n > 1)
    int ga_i = 0;

    const int ga_n = params.grp_attn_n;
    const int ga_w = params.grp_attn_w;

    if (ga_n != 1) {
        GGML_ASSERT(ga_n > 0                    && "grp_attn_n must be positive");                     // NOLINT
        GGML_ASSERT(ga_w % ga_n == 0            && "grp_attn_w must be a multiple of grp_attn_n");     // NOLINT
      //GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of grp_attn_w");    // NOLINT
      //GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
        LOG_INF("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, ga_n, ga_w);
    }
    LOG_INF("\n");

    if (params.interactive) {
        const char * control_message;
        if (params.multiline_input) {
            control_message = " - To return control to the AI, end your input with '\\'.\n"
                              " - To return control without starting a new line, end your input with '/'.\n";
        } else {
            control_message = " - Press Return to return control to the AI.\n"
                              " - To return control without starting a new line, end your input with '/'.\n"
                              " - If you want to submit another line, end your input with '\\'.\n";
        }
        LOG_INF("== Running in interactive mode. ==\n");
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
        LOG_INF(       " - Press Ctrl+C to interject at any time.\n");
#endif
        LOG_INF(       "%s", control_message);
        if (params.conversation_mode && params.enable_chat_template && params.system_prompt.empty()) {
            LOG_INF(   " - Not using system message. To change it, set a different value via -sys PROMPT\n");
        }
        LOG_INF("\n");

        is_interacting = params.interactive_first;
    }

    bool is_antiprompt = false;
    bool input_echo    = true;
    bool display       = true;

    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;

    std::vector<int>   input_tokens;  g_input_tokens  = &input_tokens;
    std::vector<int>   output_tokens; g_output_tokens = &output_tokens;
    std::ostringstream output_ss;     g_output_ss     = &output_ss;
    std::ostringstream assistant_ss; // for storing current assistant message, used in conversation mode

    // the first thing we will do is to output the prompt, so set color accordingly
    console::set_display(DISPLAY_TYPE_PROMPT);
    display = params.display_prompt;

    std::vector<llama_token> embd;

    // single-token antiprompts
    std::vector<llama_token> antiprompt_token;

    for (const std::string & antiprompt : params.antiprompt) {
        auto ids = ::common_tokenize(ctx, antiprompt, false, true);
        if (ids.size() == 1) {
            antiprompt_token.push_back(ids[0]);
        }
    }

    if (llama_model_has_encoder(model)) {
        int enc_input_size = embd_inp.size();
        llama_token * enc_input_buf = embd_inp.data();

        if (llama_encode(ctx, llama_batch_get_one(enc_input_buf, enc_input_size))) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        embd_inp.clear();
        embd_inp.push_back(decoder_start_token_id);
    }

    while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
        // predict
        if (!embd.empty()) {
            // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            int max_embd_size = n_ctx - 4;

            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int) embd.size() > max_embd_size) {
                const int skipped_tokens = (int) embd.size() - max_embd_size;
                embd.resize(max_embd_size);

                console::set_display(DISPLAY_TYPE_ERROR);
                LOG_WRN("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                console::set_display(DISPLAY_TYPE_RESET);
            }

            if (ga_n == 1) {
                // infinite text generation via context shifting
                // if we run out of context:
                // - take the n_keep first tokens from the original prompt (via n_past)
                // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches

                if (n_past + (int) embd.size() >= n_ctx) {
                    if (!params.ctx_shift){
                        LOG_WRN("\n\n%s: context full and context shift is disabled => stopping\n", __func__);
                        break;
                    }

                    if (params.n_predict == -2) {
                        LOG_WRN("\n\n%s: context full and n_predict == %d => stopping\n", __func__, params.n_predict);
                        break;
                    }

                    const int n_left    = n_past - params.n_keep;
                    const int n_discard = n_left/2;

                    LOG_DBG("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                            n_past, n_left, n_ctx, params.n_keep, n_discard);

                    llama_memory_seq_rm (mem, 0, params.n_keep            , params.n_keep + n_discard);
                    llama_memory_seq_add(mem, 0, params.n_keep + n_discard, n_past, -n_discard);

                    n_past -= n_discard;

                    LOG_DBG("after swap: n_past = %d\n", n_past);

                    LOG_DBG("embd: %s\n", string_from(ctx, embd).c_str());

                    LOG_DBG("clear session path\n");
                    path_session.clear();
                }
            } else {
                // context extension via Self-Extend
                while (n_past >= ga_i + ga_w) {
                    const int ib = (ga_n*ga_i)/ga_w;
                    const int bd = (ga_w/ga_n)*(ga_n - 1);
                    const int dd = (ga_w/ga_n) - ib*bd - ga_w;

                    LOG_DBG("\n");
                    LOG_DBG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, n_past, ib*bd, ga_i + ib*bd, n_past + ib*bd);
                    LOG_DBG("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n, (ga_i + ib*bd)/ga_n, (ga_i + ib*bd + ga_w)/ga_n);
                    LOG_DBG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib*bd + ga_w, n_past + ib*bd, dd, ga_i + ib*bd + ga_w + dd, n_past + ib*bd + dd);

                    llama_memory_seq_add(mem, 0, ga_i,                n_past,              ib*bd);
                    llama_memory_seq_div(mem, 0, ga_i + ib*bd,        ga_i + ib*bd + ga_w, ga_n);
                    llama_memory_seq_add(mem, 0, ga_i + ib*bd + ga_w, n_past + ib*bd,      dd);

                    n_past -= bd;

                    ga_i += ga_w/ga_n;

                    LOG_DBG("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", n_past + bd, n_past, ga_i);
                }
            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            if (!embd.empty()) {
                const int n_eval = (int) embd.size();
                const int profile_ubatch_tokens =
                    std::min(n_eval, (int) llama_n_ubatch(ctx));

                // prefill/decode detector
                // Match llama_context::process_ubatch(): a physical ubatch with more
                // than one token is prefill. The one-token graph emitted before the
                // queued prompt must not consume the query's clock-profile sample.
                const bool is_prefill_eval =
                    !generation_started && profile_ubatch_tokens > 1;
                const bool entering_prefill = is_prefill_eval && !prefill_active;
                if (route_clock_producer_registered) {
                    route_clock_producer.prefill_active = is_prefill_eval;
                    route_clock_producer.input_tokens = (int32_t) embd_inp.size();
                    route_clock_producer.query_id = current_question_index;
                    if (entering_prefill) {
                        route_clock_producer.read_failure_streak = 0;
                        route_clock_producer.skip_boundaries = 0;
                    }
                }
                if (is_prefill_eval) {
                    // prefill phase
                    if (!prefill_active && ig->is_ignite_active) {
                        apply_dvfs(
                            ig->cpu_clk_idx_p,
                            params.cpu_gold_clk_idx_p,
                            params.cpu_prime_clk_idx_p,
                            ig->ram_clk_idx_p,
                            ig->gpu_clk_idx_p);
                    }
                    if (!prefill_active) {
                        prefill_active = true;
                        decode_active = false;
                    }
                } else if (generation_started) {
                    // decode phase
                    if (!decode_active && ig->is_ignite_active) {
                        apply_dvfs(
                            ig->cpu_clk_idx_d,
                            params.cpu_gold_clk_idx_d,
                            params.cpu_prime_clk_idx_d,
                            ig->ram_clk_idx_d,
                            ig->gpu_clk_idx_d);
                    }
                    if (!decode_active) {
                        prefill_active = false;
                        decode_active = true;
                    }
                }

                if (entering_prefill &&
                        (ffn_clock_switch_enabled || runtime_route_clock_enabled)) {
                    S25ClockSnapshot clocks;
                    llama_backend_policy_ffn_clock_result selection = {};
                    selection.distance = -1.0;
                    llama_backend_policy_runtime_route_result route_selection = {};
                    route_selection.distance = -1.0;
                    const bool clocks_valid = dvfs.read_s25_clock_snapshot(clocks);
                    bool thermal_throttled = false;

                    if (clocks_valid && requested_prefill_clocks_valid) {
                        thermal_throttled =
                            clocks.cpu_gold_khz < requested_prefill_clocks.cpu_gold_khz ||
                            clocks.cpu_prime_khz < requested_prefill_clocks.cpu_prime_khz ||
                            clocks.gpu_hz < requested_prefill_clocks.gpu_hz;

                        if (thermal_throttled && ffn_clock_switch_enabled) {
                            selection = llama_backend_policy_select_ffn_clock_profile(
                                    (int32_t) embd_inp.size(),
                                    (int32_t) profile_ubatch_tokens,
                                    clocks.cpu_gold_khz,
                                    clocks.cpu_prime_khz,
                                    clocks.gpu_hz);
                        } else if (ffn_clock_switch_enabled) {
                            selection.enabled = true;
                            selection.pending_changed =
                                llama_backend_policy_reset_ffn_clock_profile();
                        }

                        if (runtime_route_clock_enabled) {
                            // Always run the route selector. A normal-clock
                            // snapshot must walk hot -> warm -> cool through
                            // the configured direct transitions instead of
                            // attempting an illegal hot -> initial jump.
                            route_selection = llama_backend_policy_select_layer_route_profile(
                                    llama_runtime_route_active_profile(ctx),
                                    true,
                                    (int32_t) embd_inp.size(),
                                    (int32_t) profile_ubatch_tokens,
                                    clocks.cpu_gold_khz,
                                    clocks.cpu_prime_khz,
                                    clocks.gpu_hz);
                        }

                        if (ffn_clock_switch_enabled) {
                            if (thermal_throttled && selection.matched) {
                                LOG_INF(
                                        "ffn_clock_switch: query=%zu tokens=%zu/%d "
                                        "clock(gold=%lldkHz prime=%lldkHz gpu=%lldHz) "
                                        "profile=%s distance=%.6f pending_changed=%s\n",
                                        current_question_index,
                                        embd_inp.size(), profile_ubatch_tokens,
                                        clocks.cpu_gold_khz,
                                        clocks.cpu_prime_khz,
                                        clocks.gpu_hz,
                                        selection.profile,
                                        selection.distance,
                                        selection.pending_changed ? "yes" : "no");
                            } else if (thermal_throttled) {
                                LOG_WRN(
                                        "ffn_clock_switch: no profile matches query=%zu "
                                        "tokens=%zu/%d; keeping the base FFN policy\n",
                                        current_question_index,
                                        embd_inp.size(), profile_ubatch_tokens);
                            } else {
                                LOG_INF(
                                        "ffn_clock_switch: query=%zu has no clock drop "
                                        "(actual gold=%lld prime=%lld gpu=%lld, "
                                        "requested gold=%lld prime=%lld gpu=%lld); "
                                        "using the base FFN policy%s\n",
                                        current_question_index,
                                        clocks.cpu_gold_khz,
                                        clocks.cpu_prime_khz,
                                        clocks.gpu_hz,
                                        requested_prefill_clocks.cpu_gold_khz,
                                        requested_prefill_clocks.cpu_prime_khz,
                                        requested_prefill_clocks.gpu_hz,
                                        selection.pending_changed ? " (profile reset)" : "");
                            }
                        }

                        if (runtime_route_clock_enabled) {
                            if (route_selection.matched) {
                                const bool accepted = llama_runtime_route_request_profile(
                                        ctx, route_selection.profile);
                                LOG_INF(
                                        "runtime_routes: query=%zu clock-selected profile=%s "
                                        "distance=%.6f thermal_drop=%s request=%s\n",
                                        current_question_index,
                                        route_selection.profile,
                                        route_selection.distance,
                                        thermal_throttled ? "yes" : "no",
                                        accepted ? "accepted" : "rejected");
                            } else {
                                LOG_WRN(
                                        "runtime_routes: no layer route matches query=%zu "
                                        "tokens=%zu/%d; keeping the current route\n",
                                        current_question_index,
                                        embd_inp.size(), profile_ubatch_tokens);
                            }
                        }
                    } else if (!clocks_valid) {
                        LOG_WRN(
                                "clock_switch: failed to read the S25 Gold/Prime/GPU "
                                "clock snapshot for query=%zu; keeping current profiles\n",
                                current_question_index);
                    } else {
                        LOG_WRN(
                                "clock_switch: requested S25 prefill clocks are unavailable "
                                "for query=%zu; keeping current profiles\n",
                                current_question_index);
                    }

                    if (ffn_profile_trace.is_open()) {
                        const auto now = std::chrono::system_clock::now();
                        const auto sys_time_ms =
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                    now - start_sys_time).count();
                        ffn_profile_trace
                            << sys_time_ms << ","
                            << current_question_index << ","
                            << embd_inp.size() << ","
                            << profile_ubatch_tokens << ","
                            << params.cpu_gold_clk_idx_p << ","
                            << params.cpu_prime_clk_idx_p << ","
                            << ig->gpu_clk_idx_p << ","
                            << requested_prefill_clocks.cpu_gold_khz << ","
                            << requested_prefill_clocks.cpu_prime_khz << ","
                            << requested_prefill_clocks.gpu_hz << ","
                            << clocks.cpu_gold_khz << ","
                            << clocks.cpu_prime_khz << ","
                            << clocks.gpu_hz << ","
                            << (thermal_throttled ? 1 : 0) << ","
                            << (selection.matched ? 1 : 0) << ","
                            << (selection.pending_changed ? 1 : 0) << ","
                            << selection.profile << ","
                            << selection.distance << "\n";
                        ffn_profile_trace.flush();
                    }
                }

                LOG_DBG("eval: %s\n", string_from(ctx, embd).c_str());

                GGML_ASSERT(n_eval <= params.n_batch);

                if (llama_decode(ctx, llama_batch_get_one(embd.data(), n_eval))) {
                    LOG_ERR("%s : failed to eval\n", __func__);
                    return 1;
                }

                n_past += n_eval;

                // STRICT_LIMIT=0 is an explicit prefill-only JSON query. Finish
                // all asynchronous backend work now and enter the existing query
                // transition path before common_sampler_sample() can enqueue a
                // generated-token decode graph.
                const bool prefill_only_done =
                    is_prefill_eval &&
                    params.strict_limit &&
                    params.strict_limit_length == 0 &&
                    n_consumed == (int) embd_inp.size();
                if (prefill_only_done) {
                    llama_synchronize(ctx);
                    is_interacting = true;
                    LOG_DBG("strict prefill-only query complete\n");
                }

                LOG_DBG("n_past = %d\n", n_past);
                // Display total tokens alongside total time
                if (params.n_print > 0 && n_past % params.n_print == 0) {
                    LOG_DBG("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", n_past, n_ctx);
                }
                if (module_bench_done(params, embd, n_consumed, embd_inp)) {
                    if (output_path_infer != "/inference_stats.csv") {
                        llama_perf_context_print_custom(ctx, output_path_infer, start_sys_time, ig);
                    }
                    if (csv_load && output_path_load != "/inference_load.csv") {
                        llama_op_load_profile_print_custom(output_path_load, current_question_index, start_sys_time, csv_op_load, csv_op_load_breakdown);
                    }
                    if (any_sched_trace) {
                        ggml_backend_sched_trace_flush();
                    }
                    inference_started = false;
                    LOG_INF("module-bench complete; exiting after isolated module graph\n");
                    break;
                }
            }

            if (!embd.empty() && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }
        }

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            if (!generation_started) {
                if (ig->phase_pause > 0) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(ig->phase_pause));
                }
            }

// ------------------------------------------------
            // now, generation starts
            generation_started = true;
// ------------------------------------------------

            // optionally save the session on first sample (for faster prompt loading next time)
            if (session_do_save) {
                session_do_save = false;
                llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());

                LOG_DBG("saved session to %s\n", path_session.c_str());
            }

            const int64_t t_sample_us = ig->backend_compute_profile ? ggml_time_us() : 0;
            const llama_token id = common_sampler_sample(smpl, ctx, -1);

            common_sampler_accept(smpl, id, /* accept_grammar= */ true);
            if (ig->backend_compute_profile) {
                ggml_backend_sched_profile_add_sampling_ms((ggml_time_us() - t_sample_us) / 1000.0);
            }

            // LOG_DBG("last: %s\n", string_from(ctx, smpl->prev.to_vector()).c_str());

            embd.push_back(id);

            if (params.conversation_mode && !waiting_for_first_input && !llama_vocab_is_eog(vocab, id)) {
                assistant_ss << common_token_to_piece(ctx, id, false);
            }

            // echo this to console
            input_echo = true;

            // decrement remaining sampling budget
            --n_remain;

            LOG_DBG("n_remain: %d\n", n_remain);
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            LOG_DBG("embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                common_sampler_accept(smpl, embd_inp[n_consumed], /* accept_grammar= */ false);

                ++n_consumed;
                if ((int) embd.size() == params.n_batch) {
                    break;
                }
            }
        }

        // display text
        if (input_echo && display) {
            for (auto id : embd) {
                const std::string token_str = common_token_to_piece(ctx, id, params.special);

                // Console/Stream Output
                LOG("%s", token_str.c_str());

                // Record Displayed Tokens To Log
                // Note: Generated tokens are created one by one hence this check
                if (embd.size() > 1) {
                    // Incoming Requested Tokens
                    input_tokens.push_back(id);
                } else {
                    // Outgoing Generated Tokens
                    output_tokens.push_back(id);
                    output_ss << token_str;
                }
            }
        }

        // reset color to default if there is no pending user input
        if (input_echo && (int) embd_inp.size() == n_consumed) {
            console::set_display(DISPLAY_TYPE_RESET);
            display = true;
        }

        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
            // check for reverse prompt in the last n_prev tokens
            if (!params.antiprompt.empty()) {
                const int n_prev = 32;
                const std::string last_output = common_sampler_prev_str(smpl, ctx, n_prev);

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                // so we'll compensate for that by widening the search window a bit.
                for (std::string & antiprompt : params.antiprompt) {
                    size_t extra_padding = params.interactive ? 0 : 2;
                    size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                        ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                        : 0;

                    if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
                        if (params.interactive) {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                // check for reverse prompt using special tokens
                // avoid calling common_sampler_last() if last_output is empty
                if (!last_output.empty()) {
                    llama_token last_token = common_sampler_last(smpl);
                    for (auto token : antiprompt_token) {
                        if (token == last_token) {
                            if (params.interactive) {
                                is_interacting = true;
                            }
                            is_antiprompt = true;
                            break;
                        }
                    }
                }

                if (is_antiprompt) {
                    LOG_DBG("found antiprompt: %s\n", last_output.c_str());
                }
            }

            // deal with end of generation tokens in interactive mode
            if (!waiting_for_first_input && llama_vocab_is_eog(vocab, common_sampler_last(smpl))) {
                LOG_DBG("found an EOG token\n");

                if (params.interactive) {
                    if (!params.antiprompt.empty()) {
                        // tokenize and inject first reverse prompt
                        const auto first_antiprompt = common_tokenize(ctx, params.antiprompt.front(), false, true);
                        embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                        is_antiprompt = true;
                    }

                    if (params.enable_chat_template) {
                        chat_add_and_format("assistant", assistant_ss.str());
                    }
                    is_interacting = true;
                    LOG("\n");
                }
            }

            // in strict mode, n_decode controls the number of tokens generated per user input
            if (params.strict_limit_length < n_past - embd_inp.size() && generation_started && params.strict_limit) {
                LOG_DBG("reached generation limit of %d tokens\n", params.strict_limit_length);
                is_interacting = true;
            }

            if (params.conversation_mode && !waiting_for_first_input) {
                if (!prompt.empty()) {
                    prompt.clear();
                    is_interacting = false;
                }
            }

            if ((n_past > 0 || waiting_for_first_input) && is_interacting) {
// -------------------------------
                // Print inference time for previous question
                if (inference_started) {
                    // Pacing must observe device completion rather than only the
                    // host-side enqueue boundary. Keep this synchronization out
                    // of the default path because it is intentionally opt-in.
                    if (query_pacing_enabled) {
                        llama_synchronize(ctx);
                    }
                    auto inference_end_time = std::chrono::steady_clock::now();
                    auto inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end_time - inference_start_time).count();
                    // LOG_INF("Inference time for previous question: %lld ms\n", inference_duration);
                    common_perf_print(ctx, smpl);
                    if(output_path_infer!="/inference_stats.csv"){ // deprecated in future
                        llama_perf_context_print_custom(ctx, output_path_infer, start_sys_time, ig);
                    }
                    if (csv_load && output_path_load != "/inference_load.csv") {
                        llama_op_load_profile_print_custom(output_path_load, current_question_index, start_sys_time, csv_op_load, csv_op_load_breakdown);
                    }
                    if (any_sched_trace) {
                        ggml_backend_sched_trace_flush();
                    }
                    //check_hardware(device_name);
                    // common_sampler_free(smpl);
                    inference_started = false;

                    if (query_pacing_enabled && query_timing.valid) {
                        query_timing.inference_end = inference_end_time;
                        query_timing.postprocess_end = std::chrono::steady_clock::now();

                        const bool has_next_query = current_question_index < max_query_num;
                        const auto next_start = common_query_pacing_decide(
                                query_schedule_epoch,
                                current_question_index,
                                query_period,
                                query_timing.postprocess_end);

                        if (next_start.deadline_miss) {
                            queue_query_timing(
                                    query_timing,
                                    std::chrono::steady_clock::duration::zero(),
                                    std::chrono::steady_clock::duration::zero(),
                                    next_start.lateness,
                                    true,
                                    false,
                                    "deadline_miss");
                            LOG_ERR(
                                    "query_pacing: query %zu missed the next %lld ms "
                                    "start deadline by %.3f ms; stopping before another query\n",
                                    current_question_index,
                                    (long long) query_period.count(),
                                    query_duration_ms(next_start.lateness));
                            query_timing.valid = false;
                            query_pacing_exit_status = 2;
                            break;
                        }

                        if (!has_next_query) {
                            queue_query_timing(
                                    query_timing,
                                    std::chrono::steady_clock::duration::zero(),
                                    std::chrono::steady_clock::duration::zero(),
                                    std::chrono::steady_clock::duration::zero(),
                                    false,
                                    false,
                                    "final");
                            query_timing.valid = false;
                        }
                    }
                }
// -------------------------------
                LOG_DBG("waiting for user input\n");

                if (params.conversation_mode) {
                    LOG("\n> ");
                }

                if (params.input_prefix_bos) {
                    LOG_DBG("adding input prefix BOS token\n");
                    embd_inp.push_back(llama_vocab_bos(vocab));
                }

                std::string buffer;
                if (!params.input_prefix.empty() && !params.conversation_mode) {
                    LOG_DBG("appending input prefix: '%s'\n", params.input_prefix.c_str());
                    LOG("%s", params.input_prefix.c_str());
                }

                // color user input only
                console::set_display(DISPLAY_TYPE_USER_INPUT);
                display = params.display_prompt;

// ------------------------------------------------
                // seamless user-input/json-query mode 
                if (json_questions.size() == 0){
                    // 1. user-input mode
                    // !! not supported !!
                    std::string line;
                    bool another_line = true;
                    do {
                        another_line = console::readline(line, params.multiline_input);
                        buffer += line;
                    } while (another_line);
                } else if (current_question_index < max_query_num) {
                    // 2. json-query mode
                    // Use next question from JSON file
                    // TODO: apply seamless think mode only Qwen3.
                    std::chrono::steady_clock::time_point query_actual_start;
                    std::chrono::steady_clock::time_point query_scheduled_start;
                    std::chrono::steady_clock::duration query_actual_prewake_lead =
                        std::chrono::steady_clock::duration::zero();
                    bool query_prewake_issued = false;
                    if (query_pacing_enabled) {
                        const auto now = std::chrono::steady_clock::now();
                        if (!query_schedule_started) {
                            query_schedule_epoch = now;
                            query_schedule_started = true;
                            query_actual_start = now;
                            query_scheduled_start = now;
                        } else {
                            const auto start_decision = common_query_pacing_decide(
                                    query_schedule_epoch,
                                    current_question_index,
                                    query_period,
                                    now);
                            if (start_decision.wait > std::chrono::steady_clock::duration::zero()) {
                                if (query_prewake_enabled) {
                                    const auto prewake_time = start_decision.scheduled_start - query_prewake_lead;
                                    if (now < prewake_time) {
                                        std::this_thread::sleep_until(prewake_time);
                                    }

                                    // This only wakes the active prefill CPU pool's secondary
                                    // workers into their existing finite polling window. It
                                    // does not enqueue a graph or modify model state.
                                    ggml_threadpool_prewake_fn(prefill_threadpool);
                                    query_prewake_issued = true;
                                    const auto prewake_end = std::chrono::steady_clock::now();
                                    if (prewake_end < start_decision.scheduled_start) {
                                        query_actual_prewake_lead =
                                            start_decision.scheduled_start - prewake_end;
                                    }
                                    std::this_thread::sleep_until(start_decision.scheduled_start);
                                } else {
                                    // Preserve the original single-sleep path when pre-wake is off.
                                    std::this_thread::sleep_until(start_decision.scheduled_start);
                                }
                            }
                            query_scheduled_start = start_decision.scheduled_start;

                            const auto pacing_wake_time = std::chrono::steady_clock::now();
                            const auto wake_lateness = pacing_wake_time > query_scheduled_start ?
                                pacing_wake_time - query_scheduled_start :
                                std::chrono::steady_clock::duration::zero();
                            if (wake_lateness > query_wake_lateness_tolerance) {
                                queue_query_timing(
                                        query_timing,
                                        pacing_wake_time - query_timing.inference_end,
                                        pacing_wake_time - query_timing.postprocess_end,
                                        wake_lateness,
                                        true,
                                        false,
                                        "wake_deadline_miss");
                                LOG_ERR(
                                        "query_pacing: start of query %zu was %.3f ms late "
                                        "(tolerance=%lld ms); stopping\n",
                                        current_question_index + 1,
                                        query_duration_ms(wake_lateness),
                                        (long long) query_wake_lateness_tolerance.count());
                                query_timing.valid = false;
                                query_pacing_exit_status = 2;
                                break;
                            }

                            queue_query_timing(
                                    query_timing,
                                    pacing_wake_time - query_timing.inference_end,
                                    pacing_wake_time - query_timing.postprocess_end,
                                    std::chrono::steady_clock::duration::zero(),
                                    false,
                                    true,
                                    "paced");
                            query_timing.valid = false;
                            query_actual_start = pacing_wake_time;
                        }
                    }

                    current_question_index += 1;
                    if (query_pacing_enabled) {
                        query_timing.query_id = current_question_index;
                        query_timing.scheduled_start = query_scheduled_start;
                        query_timing.actual_start = query_actual_start;
                        query_timing.prewake_actual_lead = query_actual_prewake_lead;
                        query_timing.prewake_issued = query_prewake_issued;
                        query_timing.valid = true;
                    }
                    buffer = "/no_think "; // see `general.architecture`
                    auto tmp = json_questions[current_question_index-1]; // only json requires -1
                    buffer += tmp;

                    // Reset the context for an independent JSON question. A token sampled
                    // before entering this block is still pending in embd; carrying it
                    // across the reset would evaluate a separate one-token graph before
                    // the real prompt. Discard it, then queue a fresh BOS with the prompt
                    // so the whole input is evaluated as one physical ubatch.
                    ctx_kv_cache_clear(ctx);
                    embd.clear();
                    embd_inp.clear();
                    if (llama_vocab_get_add_bos(vocab)) {
                        const llama_token bos = llama_vocab_bos(vocab);
                        if (bos != LLAMA_TOKEN_NULL) {
                            embd_inp.push_back(bos);
                        }
                    }
                    llama_perf_context_reset(ctx);
                    if (ig->backend_compute_profile) {
                        ggml_backend_sched_profile_reset();
                    }
                    if (any_sched_trace) {
                        ggml_backend_sched_trace_reset();
                        ggml_backend_sched_trace_set_query_id((int) current_question_index);
                    }
                    if (csv_load) {
                        ggml_backend_op_load_profile_reset();
                    }
                    n_past = 0; n_consumed = 0; waiting_for_first_input = true;
                    common_sampler_reset(smpl);

                    // reset dvfs will be not called after query finished
                    prefill_active = false;
                    decode_active = false;
                    generation_started = false;

                    n_remain = params.n_predict;
                    ga_i = 0;
                    is_antiprompt = false;

                    // logger info
                    LOG_INF("[%zu/%zu] ", current_question_index, max_query_num);
                    // LOG_INF("Using question from file: %s\n", buffer.c_str());
                    LOG("%s\n", tmp.c_str());

                    // Record the begining time of inference for a new question
                    inference_start_time = std::chrono::steady_clock::now();
                    inference_started = true;
                } else if (current_question_index >= params.max_query_number) {
                    LOG_INF("Reached maximum query number (%d) from %s. Exiting interactive mode.\n", params.max_query_number, json_path.c_str());
                    break;
                } else {
                    LOG_INF("No more questions available in %s. Exiting interactive mode.\n", json_path.c_str());
                    break;
                }
// ------------------------------------------------

                // done taking input, reset color
                console::set_display(DISPLAY_TYPE_RESET);
                display = true;

                if (buffer.empty()) { // Ctrl+D on empty line exits
                    LOG("EOF by user\n");
                    break;
                }

                if (buffer.back() == '\n') {
                    // Implement #587:
                    // If the user wants the text to end in a newline,
                    // this should be accomplished by explicitly adding a newline by using \ followed by return,
                    // then returning control by pressing return again.
                    buffer.pop_back();
                }

                if (buffer.empty()) { // Enter key on empty line lets the user pass control back
                    LOG_DBG("empty line, passing control back\n");
                } else { // Add tokens to embd only if the input buffer is non-empty
                    // append input suffix if any
                    if (!params.input_suffix.empty() && !params.conversation_mode) {
                        LOG_DBG("appending input suffix: '%s'\n", params.input_suffix.c_str());
                        LOG("%s", params.input_suffix.c_str());
                    }

                    LOG_DBG("buffer: '%s'\n", buffer.c_str());

                    const size_t original_size = embd_inp.size();

                    if (params.escape) {
                        string_process_escapes(buffer);
                    }

                    bool format_chat = params.conversation_mode && params.enable_chat_template;
                    std::string user_inp = format_chat
                        ? chat_add_and_format("user", std::move(buffer))
                        : std::move(buffer);
                    // TODO: one inconvenient of current chat template implementation is that we can't distinguish between user input and special tokens (prefix/postfix)
                    const auto line_pfx = common_tokenize(ctx, params.input_prefix, false, true);
                    const auto line_inp = common_tokenize(ctx, user_inp,            false, format_chat);
                    const auto line_sfx = common_tokenize(ctx, params.input_suffix, false, true);

                    LOG_DBG("input tokens: %s\n", string_from(ctx, line_inp).c_str());

                    // if user stop generation mid-way, we must add EOT to finish model's last response
                    if (need_insert_eot && format_chat) {
                        llama_token eot = llama_vocab_eot(vocab);
                        embd_inp.push_back(eot == LLAMA_TOKEN_NULL ? llama_vocab_eos(vocab) : eot);
                        need_insert_eot = false;
                    }

                    embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
                    embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());

                    if (params.verbose_prompt) {
                        LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size() - original_size);
                    }

                    for (size_t i = original_size; i < embd_inp.size(); ++i) {
                        const llama_token token = embd_inp[i];
                        const std::string token_str = common_token_to_piece(ctx, token);
                        output_tokens.push_back(token);
                        output_ss << token_str;

                        if (params.verbose_prompt) {
                            LOG_INF("%6d -> '%s'\n", token, token_str.c_str());
                        }
                    }

                    // reset assistant message
                    assistant_ss.str("");

                    n_remain -= line_inp.size();
                    LOG_DBG("n_remain: %d\n", n_remain);
                }

                input_echo = false; // do not echo this again
            }

            if (n_past > 0 || waiting_for_first_input) {
                if (is_interacting) {
                    common_sampler_reset(smpl);
                }
                is_interacting = false;

                if (waiting_for_first_input && params.single_turn) {
                    params.interactive = false;
                    params.interactive_first = false;
                }
                waiting_for_first_input = false;
            }
        }

        // end of generation
        if (!embd.empty() && llama_vocab_is_eog(vocab, embd.back()) && !(params.interactive)) {
            LOG(" [end of text]\n");
            break;
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
        if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }

    if (route_clock_producer_registered) {
        (void) llama_runtime_route_set_layer_producer(ctx, nullptr, nullptr);
        route_clock_producer_registered = false;
        LOG_DBG(
                "runtime_routes: layer clock producer samples=%llu failures=%llu requests=%llu\n",
                (unsigned long long) route_clock_producer.samples,
                (unsigned long long) route_clock_producer.read_failures,
                (unsigned long long) route_clock_producer.route_requests);
    }
    
    #if IGNITE_USE_SYSTEM_DVFS
    if (record_thread.joinable()) {
        sigterm = true;
        record_thread.join();
    }
    #endif

    reset_dvfs();

    if (!path_session.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
        LOG("\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
        llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    }

    LOG("\n\n");
    common_perf_print(ctx, smpl);
    if (any_sched_trace) {
        ggml_backend_sched_trace_flush();
    }

    // Keep query-period logging out of the active schedule: rows are buffered
    // in memory and written only after the final query (or a deadline miss).
    if (query_timing_file && query_timing_file->is_open()) {
        *query_timing_file << std::fixed << std::setprecision(3);
        for (const auto & record : query_timing_records) {
            const auto & sample = record.sample;
            const auto start_lateness = sample.actual_start > sample.scheduled_start ?
                sample.actual_start - sample.scheduled_start :
                std::chrono::steady_clock::duration::zero();
            *query_timing_file
                << sample.query_id << ","
                << query_period.count() << ","
                << query_duration_ms(sample.scheduled_start - query_schedule_epoch) << ","
                << query_duration_ms(sample.actual_start - query_schedule_epoch) << ","
                << query_duration_ms(sample.inference_end - query_schedule_epoch) << ","
                << query_duration_ms(sample.postprocess_end - query_schedule_epoch) << ","
                << query_duration_ms(sample.inference_end - sample.actual_start) << ","
                << query_duration_ms(sample.postprocess_end - sample.inference_end) << ","
                << query_duration_ms(record.cooling) << ","
                << query_duration_ms(record.pacing_wait) << ","
                << query_duration_ms(start_lateness) << ","
                << query_duration_ms(record.overrun) << ","
                << (record.deadline_miss ? 1 : 0) << ","
                << (record.next_query_scheduled ? 1 : 0);
            if (query_prewake_enabled) {
                *query_timing_file
                    << "," << (sample.prewake_issued ? 1 : 0)
                    << "," << params.query_prewake_us
                    << "," << query_duration_us(sample.prewake_actual_lead);
            }
            *query_timing_file << "," << record.status << "\n";
        }
        query_timing_file->flush();
    }

    llama_backend_free();

    ggml_threadpool_free_fn(threadpool);
    ggml_threadpool_free_fn(threadpool_batch);

    return query_pacing_exit_status;
}
