#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {

struct callback_step {
    int checkpoint_index;
    uint64_t accepted_plan;
    bool publish_plan_one;
};

struct callback_observation {
    int checkpoint_index;
    uint64_t active_plan;
    uint64_t requested_plan;
};

struct callback_script {
    ggml_backend_sched_t sched = nullptr;
    std::vector<callback_step> steps;
    std::vector<callback_observation> observations;
    std::vector<int> producer_observations;
    int producer_publish_checkpoint = -1;
    uint64_t producer_plan = 0;
    size_t next_step = 0;
    bool failed = false;
};

static void route_request_producer(
        int checkpoint_index,
        const ggml_tensor *,
        int,
        uint64_t,
        void * user_data) {
    auto * script = static_cast<callback_script *>(user_data);
    script->producer_observations.push_back(checkpoint_index);
    if (checkpoint_index == script->producer_publish_checkpoint) {
        ggml_backend_sched_layer_checkpoint_set_plan_id(
                script->sched, script->producer_plan);
    }
}

static uint64_t route_checkpoint_callback(
        int checkpoint_index,
        const ggml_tensor *,
        int,
        uint64_t active_plan_id,
        uint64_t requested_plan_id,
        void * user_data) {
    auto * script = static_cast<callback_script *>(user_data);
    script->observations.push_back({ checkpoint_index, active_plan_id, requested_plan_id });

    if (script->next_step >= script->steps.size()) {
        script->failed = true;
        return active_plan_id;
    }

    const callback_step & step = script->steps[script->next_step++];
    if (step.checkpoint_index != checkpoint_index) {
        script->failed = true;
        return active_plan_id;
    }
    if (step.publish_plan_one) {
        // The scheduler has already taken requested_plan_id for this
        // checkpoint. This request must remain pending until the next one.
        ggml_backend_sched_layer_checkpoint_set_plan_id(script->sched, 1);
    }
    return step.accepted_plan;
}

struct route_fixture {
    ggml_context * ctx = nullptr;
    ggml_backend_t backends[2] = { nullptr, nullptr };
    ggml_backend_sched_t sched = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * x = nullptr;
    ggml_tensor * y = nullptr;
    ggml_tensor * out = nullptr;
    std::vector<ggml_tensor *> canonicals;
    std::vector<ggml_tensor *> alternates;
    std::vector<ggml_tensor *> boundaries;

    ~route_fixture() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backends[0]);
        ggml_backend_free(backends[1]);
        ggml_free(ctx);
    }

    bool build(
            int n_layers,
            callback_script * script,
            uint64_t initial_request,
            bool configure_routes = true,
            bool clear_after_registration = false,
            bool use_request_producer = false) {
        const size_t graph_size = 64;
        const ggml_init_params params = {
            /* .mem_size   = */ 128 * ggml_tensor_overhead() +
                                  ggml_graph_overhead_custom(graph_size, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            return false;
        }

        x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        ggml_set_input(x);
        ggml_set_input(y);
        ggml_set_name(x, "route-x");
        ggml_set_name(y, "route-y");

        graph = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_tensor * layer_input = nullptr;
        for (int layer = 0; layer < n_layers; ++layer) {
            ggml_tensor * canonical = nullptr;
            ggml_tensor * alternate = nullptr;
            if (layer == 0) {
                canonical = ggml_add(ctx, x, y);
                alternate = ggml_add(ctx, x, y);
            } else if (layer % 2 == 1) {
                canonical = ggml_mul(ctx, layer_input, y);
                alternate = ggml_mul(ctx, layer_input, y);
            } else {
                canonical = ggml_add(ctx, layer_input, y);
                alternate = ggml_add(ctx, layer_input, y);
            }

            const std::string canonical_name = "route-canonical-" + std::to_string(layer);
            const std::string alternate_name = "route-alternate-" + std::to_string(layer);
            ggml_set_name(canonical, canonical_name.c_str());
            ggml_set_name(alternate, alternate_name.c_str());
            canonicals.push_back(canonical);
            alternates.push_back(alternate);

            // Incremental graph expansion preserves the required order:
            // canonical, then its otherwise-unreferenced alternate clone.
            ggml_build_forward_expand(graph, canonical);
            ggml_build_forward_expand(graph, alternate);

            if (layer + 1 < n_layers) {
                ggml_tensor * boundary = ggml_scale(ctx, canonical, 1.0f);
                const std::string boundary_name = "route-boundary-" + std::to_string(layer);
                ggml_set_name(boundary, boundary_name.c_str());
                boundaries.push_back(boundary);
                ggml_build_forward_expand(graph, boundary);
                layer_input = boundary;
            } else {
                layer_input = canonical;
            }
        }

        out = ggml_scale(ctx, layer_input, 1.0f);
        ggml_set_name(out, "route-output");
        ggml_set_output(out);
        ggml_build_forward_expand(graph, out);

        backends[0] = ggml_backend_cpu_init();
        backends[1] = ggml_backend_cpu_init();
        if (backends[0] == nullptr || backends[1] == nullptr) {
            return false;
        }
        sched = ggml_backend_sched_new(backends, nullptr, 2, 128, false, true);
        if (sched == nullptr) {
            return false;
        }
        script->sched = sched;

        // All policy/candidate state is lazy. Exercise the null-safe APIs and
        // prove that a request before preparation is intentionally a no-op.
        ggml_backend_sched_layer_checkpoint_set_plan_id(sched, 77);
        ggml_backend_sched_layer_checkpoint_stats empty_checkpoint_stats = {};
        ggml_backend_sched_get_layer_checkpoint_stats(sched, &empty_checkpoint_stats);
        ggml_backend_sched_route_candidate_stats empty_route_stats = {};
        ggml_backend_sched_get_route_candidate_stats(sched, &empty_route_stats);
        if (ggml_backend_sched_layer_checkpoint_get_active_plan_id(sched) != 0 ||
                empty_checkpoint_stats.graph_runs != 0 ||
                empty_route_stats.graph_runs != 0) {
            return false;
        }
        ggml_backend_sched_clear_layer_checkpoints(sched);
        ggml_backend_sched_clear_route_candidates(sched);

        if (configure_routes) {
            if (!ggml_backend_sched_prepare_layer_checkpoints(
                        sched,
                        graph,
                        boundaries.data(),
                        (int) boundaries.size(),
                        use_request_producer ? route_request_producer : nullptr,
                        route_checkpoint_callback,
                        script)) {
                return false;
            }
            for (int layer = 0; layer < n_layers; ++layer) {
                if (!ggml_backend_sched_register_route_candidate(
                            sched, canonicals[layer], canonicals[layer], 0, layer) ||
                        !ggml_backend_sched_register_route_candidate(
                            sched, canonicals[layer], alternates[layer], 1, layer)) {
                    return false;
                }
            }
            if (clear_after_registration) {
                ggml_backend_sched_clear_route_candidates(sched);
                ggml_backend_sched_clear_layer_checkpoints(sched);
            }
        }

        for (int layer = 0; layer < n_layers; ++layer) {
            ggml_backend_sched_set_tensor_backend(sched, canonicals[layer], backends[0]);
            ggml_backend_sched_set_tensor_backend(sched, alternates[layer], backends[1]);
        }
        for (ggml_tensor * boundary : boundaries) {
            ggml_backend_sched_set_tensor_backend(sched, boundary, backends[0]);
        }
        ggml_backend_sched_set_tensor_backend(sched, out, backends[0]);
        if (configure_routes && !clear_after_registration) {
            ggml_backend_sched_layer_checkpoint_set_plan_id(sched, initial_request);
        }

        return ggml_backend_sched_alloc_graph(sched, graph);
    }
};

static bool check(bool condition, const char * scenario, const char * detail) {
    if (!condition) {
        std::fprintf(stderr, "%s: %s\n", scenario, detail);
        return false;
    }
    return true;
}

static bool check_output(route_fixture & fixture, int n_layers, const char * scenario) {
    const float x[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const float y[4] = { 2.0f, 3.0f, 4.0f, 5.0f };
    ggml_backend_tensor_set(fixture.x, x, 0, sizeof(x));
    ggml_backend_tensor_set(fixture.y, y, 0, sizeof(y));
    if (!check(
                ggml_backend_sched_graph_compute(fixture.sched, fixture.graph) ==
                    GGML_STATUS_SUCCESS,
                scenario,
                "graph compute failed")) {
        return false;
    }

    float output[4] = {};
    ggml_backend_tensor_get(fixture.out, output, 0, sizeof(output));
    for (int i = 0; i < 4; ++i) {
        float expected = x[i] + y[i];
        for (int layer = 1; layer < n_layers; ++layer) {
            expected = layer % 2 == 1 ? expected * y[i] : expected + y[i];
        }
        if (!check(std::fabs(output[i] - expected) < 1e-5f, scenario, "output mismatch")) {
            std::fprintf(stderr, "  index %d: got %.6f, expected %.6f\n", i, output[i], expected);
            return false;
        }
    }
    return true;
}

static bool run_two_layer_case(
        const char * scenario,
        uint64_t initial_request,
        uint64_t graph_start_plan,
        uint64_t next_layer_plan,
        uint64_t expected_canonical,
        uint64_t expected_alternate,
        uint64_t expected_commits,
        uint64_t expected_misses) {
    callback_script script;
    script.steps = {
        { -1, graph_start_plan, false },
        {  0, next_layer_plan, false },
    };
    route_fixture fixture;
    if (!check(fixture.build(2, &script, initial_request), scenario, "fixture setup failed") ||
            !check_output(fixture, 2, scenario)) {
        return false;
    }

    ggml_backend_sched_route_candidate_stats route_stats = {};
    ggml_backend_sched_get_route_candidate_stats(fixture.sched, &route_stats);
    ggml_backend_sched_layer_checkpoint_stats checkpoint_stats = {};
    ggml_backend_sched_get_layer_checkpoint_stats(fixture.sched, &checkpoint_stats);

    return
        check(!script.failed && script.next_step == script.steps.size(), scenario, "callback script mismatch") &&
        check(script.observations.size() == 2, scenario, "unexpected callback count") &&
        check(route_stats.graph_runs == 1, scenario, "unexpected route graph count") &&
        check(route_stats.groups_executed == 2, scenario, "unexpected route group count") &&
        check(route_stats.canonical_selected == expected_canonical, scenario, "canonical selection mismatch") &&
        check(route_stats.alternate_selected == expected_alternate, scenario, "alternate selection mismatch") &&
        check(route_stats.inactive_splits_skipped == 2, scenario, "inactive split count mismatch") &&
        check(route_stats.canonical_commits == expected_commits, scenario, "canonical commit count mismatch") &&
        check(route_stats.plan_misses == expected_misses, scenario, "plan miss count mismatch") &&
        check(route_stats.plan_latches == 2, scenario, "plan latch count mismatch") &&
        check(checkpoint_stats.graph_runs == 1, scenario, "unexpected checkpoint graph count") &&
        check(checkpoint_stats.checkpoint_hits == 1, scenario, "unexpected checkpoint hit count") &&
        check(checkpoint_stats.callback_calls == 2, scenario, "unexpected checkpoint callback count") &&
        check(
            ggml_backend_sched_layer_checkpoint_get_active_plan_id(fixture.sched) == next_layer_plan,
            scenario,
            "final active plan mismatch");
}

static bool run_deferred_request_case() {
    const char * scenario = "deferred-request";
    callback_script script;
    script.steps = {
        { -1, 0, false },
        {  0, 0, true  }, // publish 1 after snapshot, reject for this boundary
        {  1, 1, false }, // the same request must still be pending here
    };
    route_fixture fixture;
    if (!check(fixture.build(3, &script, 0), scenario, "fixture setup failed") ||
            !check_output(fixture, 3, scenario)) {
        return false;
    }

    ggml_backend_sched_route_candidate_stats route_stats = {};
    ggml_backend_sched_get_route_candidate_stats(fixture.sched, &route_stats);
    ggml_backend_sched_layer_checkpoint_stats checkpoint_stats = {};
    ggml_backend_sched_get_layer_checkpoint_stats(fixture.sched, &checkpoint_stats);

    return
        check(!script.failed && script.next_step == script.steps.size(), scenario, "callback script mismatch") &&
        check(script.observations.size() == 3, scenario, "unexpected callback count") &&
        check(script.observations[0].checkpoint_index == -1, scenario, "missing graph-start callback") &&
        check(script.observations[0].requested_plan == 0, scenario, "wrong graph-start request") &&
        check(script.observations[1].checkpoint_index == 0, scenario, "missing first checkpoint") &&
        check(script.observations[1].active_plan == 0, scenario, "first checkpoint active plan changed early") &&
        check(script.observations[1].requested_plan == 0, scenario, "first checkpoint snapshot was not stable") &&
        check(script.observations[2].checkpoint_index == 1, scenario, "missing second checkpoint") &&
        check(script.observations[2].active_plan == 0, scenario, "rejected request became active early") &&
        check(script.observations[2].requested_plan == 1, scenario, "deferred request was not retained") &&
        check(route_stats.groups_executed == 3, scenario, "unexpected route group count") &&
        check(route_stats.canonical_selected == 2, scenario, "canonical selection mismatch") &&
        check(route_stats.alternate_selected == 1, scenario, "alternate selection mismatch") &&
        check(route_stats.inactive_splits_skipped == 3, scenario, "inactive split count mismatch") &&
        check(route_stats.canonical_commits == 1, scenario, "canonical commit count mismatch") &&
        check(route_stats.plan_misses == 0, scenario, "unexpected plan miss") &&
        check(route_stats.plan_latches == 3, scenario, "plan latch count mismatch") &&
        check(checkpoint_stats.checkpoint_hits == 2, scenario, "unexpected checkpoint hit count") &&
        check(checkpoint_stats.callback_calls == 3, scenario, "unexpected checkpoint callback count") &&
        check(
            ggml_backend_sched_layer_checkpoint_get_active_plan_id(fixture.sched) == 1,
            scenario,
            "deferred plan was not accepted");
}

static bool run_boundary_producer_case() {
    const char * scenario = "boundary-producer";
    callback_script script;
    script.producer_publish_checkpoint = 0;
    script.producer_plan = 1;
    script.steps = {
        { -1, 0, false },
        {  0, 1, false },
    };
    route_fixture fixture;
    if (!check(
                fixture.build(2, &script, 0, true, false, true),
                scenario,
                "fixture setup failed") ||
            !check_output(fixture, 2, scenario)) {
        return false;
    }

    ggml_backend_sched_route_candidate_stats route_stats = {};
    ggml_backend_sched_get_route_candidate_stats(fixture.sched, &route_stats);
    ggml_backend_sched_layer_checkpoint_stats checkpoint_stats = {};
    ggml_backend_sched_get_layer_checkpoint_stats(fixture.sched, &checkpoint_stats);

    return
        check(!script.failed && script.next_step == script.steps.size(), scenario, "callback script mismatch") &&
        check(script.producer_observations == std::vector<int>({ 0 }), scenario, "producer ran at graph start or wrong checkpoint") &&
        check(script.observations.size() == 2, scenario, "unexpected callback count") &&
        check(script.observations[0].checkpoint_index == -1, scenario, "missing graph-start callback") &&
        check(script.observations[0].requested_plan == 0, scenario, "producer affected graph start") &&
        check(script.observations[1].checkpoint_index == 0, scenario, "missing layer checkpoint") &&
        check(script.observations[1].requested_plan == 1, scenario, "producer publication missed same checkpoint snapshot") &&
        check(route_stats.canonical_selected == 1, scenario, "first layer route mismatch") &&
        check(route_stats.alternate_selected == 1, scenario, "producer did not route next layer") &&
        check(checkpoint_stats.producer_calls == 1, scenario, "producer stats mismatch") &&
        check(checkpoint_stats.producer_total_us >= checkpoint_stats.producer_max_us, scenario, "invalid producer timing stats") &&
        check(
            ggml_backend_sched_layer_checkpoint_get_active_plan_id(fixture.sched) == 1,
            scenario,
            "producer plan was not accepted");
}

static bool run_boundary_producer_cancels_stale_request_case() {
    const char * scenario = "boundary-producer-cancels-stale-request";
    callback_script script;
    // Plan 1 is pending at graph start but rejected. The boundary producer
    // observes that plan 0 is still the correct active plan and republishes
    // it, cancelling the stale request before the checkpoint snapshot.
    script.producer_publish_checkpoint = 0;
    script.producer_plan = 0;
    script.steps = {
        { -1, 0, false },
        {  0, 0, false },
    };
    route_fixture fixture;
    if (!check(
                fixture.build(2, &script, 1, true, false, true),
                scenario,
                "fixture setup failed") ||
            !check_output(fixture, 2, scenario)) {
        return false;
    }

    ggml_backend_sched_route_candidate_stats route_stats = {};
    ggml_backend_sched_get_route_candidate_stats(fixture.sched, &route_stats);
    ggml_backend_sched_layer_checkpoint_stats checkpoint_stats = {};
    ggml_backend_sched_get_layer_checkpoint_stats(fixture.sched, &checkpoint_stats);

    return
        check(!script.failed && script.next_step == script.steps.size(), scenario, "callback script mismatch") &&
        check(script.producer_observations == std::vector<int>({ 0 }), scenario, "producer ran at wrong checkpoint") &&
        check(script.observations.size() == 2, scenario, "unexpected callback count") &&
        check(script.observations[0].requested_plan == 1, scenario, "stale request was not present at graph start") &&
        check(script.observations[1].active_plan == 0, scenario, "stale request became active early") &&
        check(script.observations[1].requested_plan == 0, scenario, "producer did not cancel stale request") &&
        check(route_stats.canonical_selected == 2, scenario, "cancelled request changed the route") &&
        check(route_stats.alternate_selected == 0, scenario, "stale alternate route was selected") &&
        check(checkpoint_stats.producer_calls == 1, scenario, "producer stats mismatch") &&
        check(
            ggml_backend_sched_layer_checkpoint_get_active_plan_id(fixture.sched) == 0,
            scenario,
            "stale plan remained active");
}

static bool run_subgraph_layer_switch_case() {
    const char * scenario = "subgraph-layer-switch";
    const size_t graph_size = 128;
    const ggml_init_params params = {
        /* .mem_size   = */ 256 * ggml_tensor_overhead() +
                              ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    ggml_backend_t backends[2] = {
        ggml_backend_cpu_init(),
        ggml_backend_cpu_init(),
    };
    ggml_backend_sched_t sched = backends[0] != nullptr && backends[1] != nullptr
        ? ggml_backend_sched_new(backends, nullptr, 2, 256, false, true)
        : nullptr;
    auto cleanup = [&]() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backends[0]);
        ggml_backend_free(backends[1]);
        ggml_free(ctx);
    };
    if (!check(ctx != nullptr && sched != nullptr, scenario, "backend setup failed")) {
        cleanup();
        return false;
    }

    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_input(x);
    ggml_set_input(y);
    ggml_set_name(x, "route-subgraph-x");
    ggml_set_name(y, "route-subgraph-y");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);
    std::vector<ggml_tensor *> canonical_firsts;
    std::vector<ggml_tensor *> canonical_outputs;
    std::vector<ggml_tensor *> alternate_firsts;
    std::vector<ggml_tensor *> alternate_outputs;
    ggml_tensor * layer_input = x;
    ggml_tensor * boundary = nullptr;

    for (int layer = 0; layer < 2; ++layer) {
        ggml_tensor * c1 = ggml_add(ctx, layer_input, y);
        ggml_tensor * c2 = ggml_mul(ctx, c1, y);
        ggml_tensor * c3 = ggml_add(ctx, c2, y);
        ggml_tensor * a1 = ggml_mul(ctx, layer_input, y);
        ggml_tensor * a2 = ggml_add(ctx, a1, y);
        ggml_tensor * a3 = ggml_mul(ctx, a2, y);
        // Reuse the scheduler's FFN branch-purity boundary to force two
        // splits per variant even though this host-only test has two CPU
        // instances with the same buffer type (which pass 3 coalesces).
        ggml_set_name(c1, ("ffn_up.route-subgraph-canonical-" + std::to_string(layer)).c_str());
        ggml_set_name(c2, ("ffn_down.route-subgraph-canonical-" + std::to_string(layer)).c_str());
        ggml_set_name(c3, ("route-subgraph-c3-" + std::to_string(layer)).c_str());
        ggml_set_name(a1, ("ffn_up.route-subgraph-alternate-" + std::to_string(layer)).c_str());
        ggml_set_name(a2, ("ffn_down.route-subgraph-alternate-" + std::to_string(layer)).c_str());
        ggml_set_name(a3, ("route-subgraph-a3-" + std::to_string(layer)).c_str());
        canonical_firsts.push_back(c1);
        canonical_outputs.push_back(c3);
        alternate_firsts.push_back(a1);
        alternate_outputs.push_back(a3);

        ggml_build_forward_expand(graph, c3);
        ggml_build_forward_expand(graph, a3);

        ggml_backend_sched_set_tensor_backend(sched, c1, backends[0]);
        ggml_backend_sched_set_tensor_backend(sched, c2, backends[0]);
        ggml_backend_sched_set_tensor_backend(sched, c3, backends[1]);
        ggml_backend_sched_set_tensor_backend(sched, a1, backends[1]);
        ggml_backend_sched_set_tensor_backend(sched, a2, backends[1]);
        ggml_backend_sched_set_tensor_backend(sched, a3, backends[0]);

        if (layer == 0) {
            boundary = ggml_scale(ctx, c3, 1.0f);
            ggml_set_name(boundary, "route-subgraph-boundary-0");
            ggml_build_forward_expand(graph, boundary);
            ggml_backend_sched_set_tensor_backend(sched, boundary, backends[0]);
            layer_input = boundary;
        }
    }

    ggml_tensor * out = ggml_scale(ctx, canonical_outputs.back(), 1.0f);
    ggml_set_name(out, "route-subgraph-output");
    ggml_set_output(out);
    ggml_build_forward_expand(graph, out);
    ggml_backend_sched_set_tensor_backend(sched, out, backends[0]);

    callback_script script;
    script.sched = sched;
    script.steps = {
        { -1, 0, false },
        {  0, 1, false },
    };
    ggml_tensor * boundaries[] = { boundary };
    bool setup_ok = ggml_backend_sched_prepare_layer_checkpoints(
            sched, graph, boundaries, 1, nullptr,
            route_checkpoint_callback, &script);
    for (int layer = 0; setup_ok && layer < 2; ++layer) {
        setup_ok = ggml_backend_sched_register_route_subgraph(
                sched,
                canonical_firsts[layer], canonical_outputs[layer],
                canonical_firsts[layer], canonical_outputs[layer],
                0, layer) &&
            ggml_backend_sched_register_route_subgraph(
                sched,
                canonical_firsts[layer], canonical_outputs[layer],
                alternate_firsts[layer], alternate_outputs[layer],
                1, layer);
    }
    setup_ok = setup_ok && ggml_backend_sched_alloc_graph(sched, graph);
    ggml_backend_sched_layer_checkpoint_set_plan_id(sched, 0);
    if (!check(setup_ok, scenario, "subgraph preparation failed")) {
        cleanup();
        return false;
    }

    const float x_data[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const float y_data[4] = { 2.0f, 3.0f, 4.0f, 5.0f };
    ggml_backend_tensor_set(x, x_data, 0, sizeof(x_data));
    ggml_backend_tensor_set(y, y_data, 0, sizeof(y_data));
    bool ok = check(
            ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
            scenario, "graph compute failed");

    const float expected[4] = { 36.0f, 171.0f, 528.0f, 1275.0f };
    float actual[4] = {};
    if (ok) {
        ggml_backend_tensor_get(out, actual, 0, sizeof(actual));
        for (int i = 0; i < 4; ++i) {
            ok = check(std::fabs(actual[i] - expected[i]) < 1e-5f,
                    scenario, "subgraph output mismatch") && ok;
        }
    }

    ggml_backend_sched_route_candidate_stats route_stats = {};
    ggml_backend_sched_get_route_candidate_stats(sched, &route_stats);
    ggml_backend_sched_layer_checkpoint_stats checkpoint_stats = {};
    ggml_backend_sched_get_layer_checkpoint_stats(sched, &checkpoint_stats);
    if (route_stats.prepared_candidate_splits != 8) {
        std::fprintf(stderr,
                "%s: got %d candidate splits (%d total), expected 8\n",
                scenario,
                route_stats.prepared_candidate_splits,
                ggml_backend_sched_get_n_splits(sched));
    }
    ok =
        check(!script.failed && script.next_step == script.steps.size(), scenario, "callback script mismatch") &&
        check(route_stats.registered_mappings == 4, scenario, "mapping count mismatch") &&
        check(route_stats.prepared_groups == 2, scenario, "prepared group count mismatch") &&
        check(route_stats.prepared_candidate_splits == 8, scenario, "candidate split coalescing mismatch") &&
        check(route_stats.groups_executed == 2, scenario, "executed group count mismatch") &&
        check(route_stats.canonical_selected == 1, scenario, "canonical selection mismatch") &&
        check(route_stats.alternate_selected == 1, scenario, "alternate selection mismatch") &&
        check(route_stats.inactive_splits_skipped == 4, scenario, "inactive subgraph split mismatch") &&
        check(route_stats.canonical_commits == 1, scenario, "terminal commit count mismatch") &&
        check(route_stats.plan_misses == 0, scenario, "unexpected plan miss") &&
        check(route_stats.plan_latches == 2, scenario, "plan latch count mismatch") &&
        check(checkpoint_stats.checkpoint_hits == 1, scenario, "checkpoint hit count mismatch") &&
        check(checkpoint_stats.callback_calls == 2, scenario, "checkpoint callback count mismatch") &&
        check(ggml_backend_sched_layer_checkpoint_get_active_plan_id(sched) == 1,
                scenario, "final active plan mismatch") && ok;

    cleanup();
    return ok;
}

static bool run_alternate_terminal_lifetime_case() {
    const char * scenario = "alternate-terminal-lifetime";
    constexpr int n_alternates = 24;
    constexpr int64_t n_elements = 256 * 1024; // 1 MiB per F32 terminal
    const size_t graph_size = 64;
    const ggml_init_params params = {
        /* .mem_size   = */ 128 * ggml_tensor_overhead() +
                              ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_t backends[] = { backend };
    ggml_backend_sched_t sched = backend != nullptr
        ? ggml_backend_sched_new(backends, nullptr, 1, 128, false, true)
        : nullptr;
    auto cleanup = [&]() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend);
        ggml_free(ctx);
    };
    if (!check(ctx != nullptr && sched != nullptr, scenario, "backend setup failed")) {
        cleanup();
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    ggml_set_input(input);
    ggml_set_name(input, "route-lifetime-input");
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);

    ggml_tensor * canonical = ggml_scale(ctx, input, 1.0f);
    ggml_set_name(canonical, "route-lifetime-canonical");
    ggml_build_forward_expand(graph, canonical);
    ggml_backend_sched_set_tensor_backend(sched, canonical, backend);

    std::vector<ggml_tensor *> alternates;
    alternates.reserve(n_alternates);
    for (int i = 0; i < n_alternates; ++i) {
        ggml_tensor * alternate = ggml_scale(ctx, input, (float) (i + 2));
        const std::string name = "route-lifetime-alternate-" + std::to_string(i);
        ggml_set_name(alternate, name.c_str());
        ggml_build_forward_expand(graph, alternate);
        ggml_backend_sched_set_tensor_backend(sched, alternate, backend);
        alternates.push_back(alternate);
    }

    ggml_tensor * output = ggml_scale(ctx, canonical, 1.0f);
    ggml_set_name(output, "route-lifetime-output");
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    ggml_backend_sched_set_tensor_backend(sched, output, backend);

    bool setup_ok = ggml_backend_sched_register_route_subgraph(
            sched, canonical, canonical, canonical, canonical, 0, 0);
    for (int i = 0; setup_ok && i < n_alternates; ++i) {
        setup_ok = ggml_backend_sched_register_route_subgraph(
                sched, canonical, canonical,
                alternates[i], alternates[i], (uint64_t) i + 1, 0);
    }

    size_t sizes[1] = {};
    if (setup_ok) {
        ggml_backend_sched_reserve_size(sched, graph, sizes);
    }

    // Input + canonical/output + one reusable alternate scratch terminal fit
    // comfortably below 8 MiB. Without scheduler lifetime dependencies, the
    // 24 alternate roots alone require 24 MiB and this assertion fails.
    const bool ok =
        check(setup_ok, scenario, "route registration failed") &&
        check(sizes[0] > 0, scenario, "reserve size was not reported") &&
        check(sizes[0] < 8 * 1024 * 1024,
                scenario, "alternate terminals remained live to graph end");

    cleanup();
    return ok;
}

static bool run_multi_consumer_commit_case() {
    const char * scenario = "multi-consumer-canonical-commit";
    const size_t graph_size = 32;
    const ggml_init_params params = {
        /* .mem_size   = */ 64 * ggml_tensor_overhead() +
                              ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    ggml_backend_t backends[2] = {
        ggml_backend_cpu_init(),
        ggml_backend_cpu_init(),
    };
    ggml_backend_sched_t sched = backends[0] != nullptr && backends[1] != nullptr
        ? ggml_backend_sched_new(backends, nullptr, 2, 64, false, true)
        : nullptr;
    auto cleanup = [&]() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backends[0]);
        ggml_backend_free(backends[1]);
        ggml_free(ctx);
    };
    if (!check(ctx != nullptr && sched != nullptr, scenario, "backend setup failed")) {
        cleanup();
        return false;
    }

    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_input(x);
    ggml_set_input(y);
    ggml_set_name(x, "route-fanout-x");
    ggml_set_name(y, "route-fanout-y");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_tensor * canonical_first = ggml_add(ctx, x, y);
    ggml_tensor * canonical = ggml_scale(ctx, canonical_first, 1.0f);
    ggml_tensor * alternate_first = ggml_mul(ctx, x, y);
    ggml_tensor * alternate = ggml_scale(ctx, alternate_first, 1.0f);
    ggml_set_name(canonical_first, "route-fanout-canonical-first");
    ggml_set_name(canonical, "route-fanout-canonical");
    ggml_set_name(alternate_first, "route-fanout-alternate-first");
    ggml_set_name(alternate, "route-fanout-alternate");
    ggml_build_forward_expand(graph, canonical);
    ggml_build_forward_expand(graph, alternate);

    // Model ffn_inp's fan-out: the committed canonical value feeds multiple
    // later consumers, which may themselves begin separate route subgraphs.
    ggml_tensor * consumer_a = ggml_scale(ctx, canonical, 2.0f);
    ggml_tensor * consumer_b = ggml_add(ctx, canonical, y);
    ggml_tensor * output = ggml_add(ctx, consumer_a, consumer_b);
    ggml_set_name(consumer_a, "route-fanout-consumer-a");
    ggml_set_name(consumer_b, "route-fanout-consumer-b");
    ggml_set_name(output, "route-fanout-output");
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);

    ggml_backend_sched_set_tensor_backend(sched, canonical_first, backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, canonical, backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, alternate_first, backends[1]);
    ggml_backend_sched_set_tensor_backend(sched, alternate, backends[1]);
    ggml_backend_sched_set_tensor_backend(sched, consumer_a, backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, consumer_b, backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, output, backends[0]);

    bool setup_ok =
        ggml_backend_sched_register_route_subgraph(
                sched, canonical_first, canonical,
                canonical_first, canonical, 0, 0) &&
        ggml_backend_sched_register_route_subgraph(
                sched, canonical_first, canonical,
                alternate_first, alternate, 1, 0);
    ggml_backend_sched_layer_checkpoint_set_plan_id(sched, 1);
    setup_ok = setup_ok && ggml_backend_sched_alloc_graph(sched, graph);
    if (!check(setup_ok, scenario, "route preparation failed")) {
        cleanup();
        return false;
    }

    const float x_data[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const float y_data[4] = { 2.0f, 3.0f, 4.0f, 5.0f };
    ggml_backend_tensor_set(x, x_data, 0, sizeof(x_data));
    ggml_backend_tensor_set(y, y_data, 0, sizeof(y_data));
    bool ok = check(
            ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
            scenario, "graph compute failed");

    float actual[4] = {};
    if (ok) {
        ggml_backend_tensor_get(output, actual, 0, sizeof(actual));
        for (int i = 0; i < 4; ++i) {
            // The selected alternate terminal is x*y. Both consumers must
            // observe the committed value:
            // 2*(x*y) + ((x*y)+y) = 3*x*y+y.
            const float expected = 3.0f * x_data[i] * y_data[i] + y_data[i];
            ok = check(std::fabs(actual[i] - expected) < 1e-5f,
                    scenario, "fan-out consumer observed stale canonical data") && ok;
        }
    }

    ggml_backend_sched_route_candidate_stats stats = {};
    ggml_backend_sched_get_route_candidate_stats(sched, &stats);
    ok =
        check(stats.groups_executed == 1, scenario, "unexpected route group count") &&
        check(stats.alternate_selected == 1, scenario, "alternate was not selected") &&
        check(stats.canonical_commits == 1, scenario, "alternate was not committed") && ok;

    cleanup();
    return ok;
}

static bool run_default_off_case() {
    const char * scenario = "default-off";
    callback_script baseline_script;
    route_fixture baseline;
    if (!check(
                baseline.build(2, &baseline_script, 0, false, false),
                scenario,
                "baseline setup failed") ||
            !check_output(baseline, 2, scenario)) {
        return false;
    }
    const int baseline_splits = ggml_backend_sched_get_n_splits(baseline.sched);

    callback_script cleared_script;
    route_fixture cleared;
    if (!check(
                cleared.build(2, &cleared_script, 0, true, true),
                scenario,
                "registered-then-cleared setup failed") ||
            !check_output(cleared, 2, scenario)) {
        return false;
    }
    const int cleared_splits = ggml_backend_sched_get_n_splits(cleared.sched);

    ggml_backend_sched_route_candidate_stats baseline_route_stats = {};
    ggml_backend_sched_get_route_candidate_stats(baseline.sched, &baseline_route_stats);
    ggml_backend_sched_route_candidate_stats cleared_route_stats = {};
    ggml_backend_sched_get_route_candidate_stats(cleared.sched, &cleared_route_stats);
    ggml_backend_sched_layer_checkpoint_stats baseline_checkpoint_stats = {};
    ggml_backend_sched_get_layer_checkpoint_stats(baseline.sched, &baseline_checkpoint_stats);
    ggml_backend_sched_layer_checkpoint_stats cleared_checkpoint_stats = {};
    ggml_backend_sched_get_layer_checkpoint_stats(cleared.sched, &cleared_checkpoint_stats);

    return
        check(baseline_splits == cleared_splits, scenario, "legacy split count changed after clear") &&
        check(baseline_route_stats.graph_runs == 0, scenario, "baseline entered route compute") &&
        check(cleared_route_stats.graph_runs == 0, scenario, "cleared graph entered route compute") &&
        check(cleared_route_stats.registered_mappings == 0, scenario, "route registrations survived clear") &&
        check(baseline_checkpoint_stats.graph_runs == 0, scenario, "baseline entered checkpoint compute") &&
        check(cleared_checkpoint_stats.graph_runs == 0, scenario, "checkpoint registration survived clear") &&
        check(baseline_checkpoint_stats.producer_calls == 0, scenario, "baseline invoked request producer") &&
        check(cleared_checkpoint_stats.producer_calls == 0, scenario, "cleared producer registration survived") &&
        check(
            ggml_backend_sched_get_n_layer_checkpoint_ranges(cleared.sched) == 0,
            scenario,
            "checkpoint ranges survived clear");
}

} // namespace

int main() {
    bool ok = true;
    ok = run_default_off_case() && ok;
    ok = run_two_layer_case(
        "canonical-to-alternate",
        0, 0, 1,
        1, 1, 1, 0) && ok;
    ok = run_two_layer_case(
        "alternate-to-canonical",
        1, 1, 0,
        1, 1, 1, 0) && ok;
    ok = run_two_layer_case(
        "unknown-plan-fallback",
        99, 99, 99,
        2, 0, 0, 2) && ok;
    ok = run_deferred_request_case() && ok;
    ok = run_boundary_producer_case() && ok;
    ok = run_boundary_producer_cancels_stale_request_case() && ok;
    ok = run_subgraph_layer_switch_case() && ok;
    ok = run_multi_consumer_commit_case() && ok;
    ok = run_alternate_terminal_lifetime_case() && ok;

    if (ok) {
        std::puts("scheduler route-candidate tests passed");
        return 0;
    }
    return 1;
}
