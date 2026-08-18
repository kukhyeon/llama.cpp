#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

static bool is_opencl_device(ggml_backend_dev_t dev) {
    return std::strcmp(ggml_backend_dev_name(dev), "GPUOpenCL") == 0;
}

static void set_environment(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value != nullptr ? value : "");
#else
    if (value != nullptr) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

struct scoped_environment {
    const char * name;
    bool had_value;
    std::string old_value;

    scoped_environment(const char * name, const char * value) : name(name) {
        const char * old = std::getenv(name);
        had_value = old != nullptr;
        if (old != nullptr) {
            old_value = old;
        }
        set_environment(name, value);
    }

    ~scoped_environment() {
        set_environment(name, had_value ? old_value.c_str() : nullptr);
    }
};

static bool test_opencl_event_protected_upload(ggml_backend_dev_t dev) {
    ggml_backend_dev_props props;
    {
        // Merely loading the backend must not opt the global scheduler into
        // async pipeline behavior or alter its host allocation selection.
        scoped_environment async_off("GGML_OPENCL_ASYNC_EVENTS", "0");
        ggml_backend_dev_get_props(dev, &props);
        if (!check(!props.caps.async && !props.caps.host_buffer && !props.caps.events,
                    "OpenCL async capabilities must remain opt-in") ||
                !check(ggml_backend_dev_host_buffer_type(dev) == nullptr,
                    "OpenCL host staging type must remain hidden while opt-in is disabled") ||
                !check(ggml_backend_event_new(dev) == nullptr,
                    "OpenCL event allocation must remain hidden while opt-in is disabled")) {
            return false;
        }
    }

    ggml_backend_t producer = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t consumer = ggml_backend_dev_init(dev, nullptr);
    if (!check(producer != nullptr && consumer != nullptr, "failed to initialize OpenCL backends")) {
        ggml_backend_free(producer);
        ggml_backend_free(consumer);
        return false;
    }

    ggml_init_params params = {
        /* .mem_size = */ 2 * ggml_tensor_overhead(),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!check(ctx != nullptr, "failed to create tensor context")) {
        ggml_backend_free(consumer);
        ggml_backend_free(producer);
        return false;
    }

    ggml_tensor * base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    ggml_tensor * view = ggml_view_1d(ctx, base, 8, 4 * sizeof(float));
    ggml_backend_buffer_t device_buffer = ggml_backend_alloc_ctx_tensors(ctx, producer);

    // With the feature disabled, a direct call to the async API retains its
    // historical blocking fallback, including safe immediate source reuse.
    bool ok = check(device_buffer != nullptr, "failed to allocate OpenCL tensor");
    if (ok) {
        float zeros[16] = {};
        ggml_backend_tensor_set(base, zeros, 0, sizeof(zeros));
        float source[8];
        for (int i = 0; i < 8; ++i) {
            source[i] = 1.0f + i;
        }
        {
            scoped_environment async_off("GGML_OPENCL_ASYNC_EVENTS", "0");
            ggml_backend_tensor_set_async(producer, view, source, 0, sizeof(source));
        }
        for (float & value : source) {
            value = -1.0f;
        }
        float actual[16];
        ggml_backend_tensor_get(base, actual, 0, sizeof(actual));
        for (int i = 0; i < 16; ++i) {
            const float expected = i >= 4 && i < 12 ? 1.0f + i - 4 : 0.0f;
            ok = check(actual[i] == expected,
                    "disabled async upload did not preserve blocking semantics") && ok;
        }
    }

    scoped_environment async_opt_in("GGML_OPENCL_ASYNC_EVENTS", "1");
    scoped_environment caps_hidden("GGML_OPENCL_ADVERTISE_ASYNC_CAPS", "0");
    scoped_environment host_hidden("GGML_OPENCL_EXPOSE_HOST_BUFFER", "0");
    ggml_backend_dev_get_props(dev, &props);
    ok = check(!props.caps.async && !props.caps.host_buffer && !props.caps.events,
            "targeted async events unexpectedly changed global capabilities") && ok;
    ok = check(ggml_backend_dev_host_buffer_type(dev) == nullptr,
            "targeted async events unexpectedly changed global host allocation") && ok;

    scoped_environment caps_opt_in("GGML_OPENCL_ADVERTISE_ASYNC_CAPS", "1");
    scoped_environment host_opt_in("GGML_OPENCL_EXPOSE_HOST_BUFFER", "1");
    ggml_backend_dev_get_props(dev, &props);
    ok = check(props.caps.async && props.caps.host_buffer && props.caps.events,
            "explicit global OpenCL capability advertisement failed") && ok;

    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
    ggml_backend_buffer_t staging = host_buft != nullptr
        ? ggml_backend_buft_alloc_buffer(host_buft, 8 * sizeof(float))
        : nullptr;
    ggml_backend_event_t event = ggml_backend_event_new(dev);

    ok =
        check(host_buft != nullptr && ggml_backend_buft_is_host(host_buft),
                "OpenCL host staging buffer type is unavailable") &&
        check(staging != nullptr && ggml_backend_buffer_get_base(staging) != nullptr,
                "failed to allocate OpenCL host staging buffer") &&
        check(event != nullptr, "failed to allocate OpenCL event") && ok;

    if (ok) {
        // A new event is a valid no-op checkpoint, allowing a staging ring to
        // synchronize every slot uniformly before its first use.
        ggml_backend_event_synchronize(event);

        float zeros[16] = {};
        ggml_backend_tensor_set(base, zeros, 0, sizeof(zeros));

        float * slot = static_cast<float *>(ggml_backend_buffer_get_base(staging));
        for (int i = 0; i < 8; ++i) {
            slot[i] = 10.0f + i;
        }

        ggml_backend_tensor_set_async(producer, view, slot, 0, 8 * sizeof(float));
        ggml_backend_event_record(event, producer);
        ggml_backend_event_wait(consumer, event);
        ggml_backend_synchronize(consumer);

        float actual[16];
        ggml_backend_tensor_get(base, actual, 0, sizeof(actual));
        for (int i = 0; i < 16; ++i) {
            const float expected = i >= 4 && i < 12 ? 10.0f + (i - 4) : 0.0f;
            ok = check(std::fabs(actual[i] - expected) == 0.0f,
                    "event-protected view upload produced incorrect data") && ok;
        }

        // Reusing the same event after it completes must protect the same
        // staging slot for a subsequent upload.
        ggml_backend_event_synchronize(event);
        for (int i = 0; i < 8; ++i) {
            slot[i] = 20.0f + i;
        }
        ggml_backend_tensor_set_async(producer, view, slot, 0, 8 * sizeof(float));
        ggml_backend_event_record(event, producer);
        ggml_backend_event_synchronize(event);

        ggml_backend_tensor_get(base, actual, 0, sizeof(actual));
        for (int i = 4; i < 12; ++i) {
            ok = check(std::fabs(actual[i] - (20.0f + i - 4)) == 0.0f,
                    "reused OpenCL event did not protect staging memory") && ok;
        }
    }

    ggml_backend_event_free(event);
    ggml_backend_buffer_free(staging);
    ggml_backend_buffer_free(device_buffer);
    ggml_free(ctx);
    ggml_backend_free(consumer);
    ggml_backend_free(producer);
    return ok;
}

int main() {
    ggml_backend_load_all();

    size_t tested = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!is_opencl_device(dev)) {
            continue;
        }
        ++tested;
        if (!test_opencl_event_protected_upload(dev)) {
            return 1;
        }
    }

    if (tested == 0) {
        std::printf("SKIP: no OpenCL device available\n");
    } else {
        std::printf("OK: tested %zu OpenCL device(s)\n", tested);
    }
    return 0;
}
