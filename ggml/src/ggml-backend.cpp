// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif


// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));
        return size;
    }
    return ggml_nbytes(tensor);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    bool is_ffn_parallel_reduce;
    struct ggml_tensor * inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_inputs;
    // graph view of this split
    struct ggml_cgraph graph;
};

struct ggml_backend_sched_ffn_executor;

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor * graph_inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_graph_inputs;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    int cpu_threads;
    int ffn_parallel_reduce_threads;
    int ffn_device_worker_cpu;
    int ffn_gpu_worker_cpu;
    int ffn_npu_worker_cpu;

    // Non-trivial worker state is owned separately because this scheduler is
    // allocated with calloc/free.
    ggml_backend_sched_ffn_executor * ffn_device_executor;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    int debug;

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;
};

using ggml_backend_sched_ffn_job_fn = void (*)(void *);

enum ggml_backend_sched_ffn_worker_mode {
    GGML_SCHED_FFN_WORKER_NONE = 0,
    GGML_SCHED_FFN_WORKER_CALLER,
    GGML_SCHED_FFN_WORKER_PERSISTENT,
    GGML_SCHED_FFN_WORKER_TRANSIENT,
    GGML_SCHED_FFN_WORKER_SERIAL,
};

enum ggml_backend_sched_ffn_collect_kind {
    GGML_SCHED_FFN_COLLECT_NONE = 0,
    GGML_SCHED_FFN_COLLECT_TICKET_WAIT,
    GGML_SCHED_FFN_COLLECT_THREAD_JOIN,
};

// A timeline is owned by one scheduler FFN group and written by at most one
// branch worker. The scheduler reads it only after the corresponding ticket
// wait or thread join has completed. Keeping these records branch-local avoids
// worker-side file I/O, allocation, and shared trace-buffer contention.
struct alignas(64) ggml_backend_sched_ffn_worker_timeline {
    enum ggml_backend_sched_ffn_worker_mode mode = GGML_SCHED_FFN_WORKER_NONE;
    enum ggml_backend_sched_ffn_collect_kind collect_kind = GGML_SCHED_FFN_COLLECT_NONE;

    uint64_t ticket = 0;
    int queue_depth_at_submit = -1;
    int64_t worker_tid = -1;
    int affinity_cpu = -1;
    int start_cpu = -1;
    int end_cpu = -1;
    int wait_blocked = -1;
    int collect_order = -1;

    int64_t dispatch_begin_us = -1;
    int64_t dispatch_end_us = -1;
    int64_t job_enqueued_us = -1;
    int64_t worker_dequeue_us = -1;
    int64_t job_start_us = -1;
    int64_t backend_begin_us = -1;
    int64_t graph_compute_return_us = -1;
    int64_t sync_begin_us = -1;
    int64_t sync_end_us = -1;
    int64_t job_end_us = -1;
    int64_t completion_publish_us = -1;
    int64_t collect_begin_us = -1;
    int64_t collect_end_us = -1;
};

static int64_t ggml_backend_sched_current_tid() {
#if defined(__linux__) && defined(SYS_gettid)
    return (int64_t) syscall(SYS_gettid);
#else
    return -1;
#endif
}

static int ggml_backend_sched_current_cpu() {
#if defined(__linux__) && defined(SYS_getcpu)
    unsigned cpu = 0;
    return syscall(SYS_getcpu, &cpu, nullptr, nullptr) == 0 ? (int) cpu : -1;
#else
    return -1;
#endif
}

static bool ggml_backend_sched_pin_current_thread_to_cpu(int cpu) {
    if (cpu < 0) {
        return true;
    }

#if defined(__linux__)
    if (cpu >= CPU_SETSIZE) {
        errno = EINVAL;
        return false;
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    return sched_setaffinity(0, sizeof(cpuset), &cpuset) == 0;
#else
    errno = EINVAL;
    return false;
#endif
}

struct ggml_backend_sched_ffn_executor {
    struct job {
        ggml_backend_sched_ffn_job_fn fn = nullptr;
        void * context = nullptr;
        enum ggml_status * status = nullptr;
        uint64_t ticket = 0;
        ggml_backend_sched_ffn_worker_timeline * timeline = nullptr;
    };

    struct worker {
        int backend_id;
        std::string backend_name;
        int affinity_cpu;

        std::mutex mutex;
        std::condition_variable work_cv;
        std::condition_variable state_cv;
        std::thread thread;

        job queue[GGML_SCHED_MAX_BACKENDS];
        int queue_head = 0;
        int queue_tail = 0;
        int queue_size = 0;
        uint64_t next_ticket = 1;
        uint64_t completed_ticket = 0;
        bool ready = false;
        bool stopping = false;

        worker(int backend_id, const char * backend_name, int affinity_cpu) :
            backend_id(backend_id),
            backend_name(backend_name ? backend_name : "unknown"),
            affinity_cpu(affinity_cpu) {
        }

        ~worker() {
            stop();
        }

        bool start() {
            try {
                thread = std::thread(&worker::run, this);
            } catch (const std::exception & error) {
                GGML_LOG_ERROR(
                        "%s: failed to start persistent FFN worker for backend %s: %s\n",
                        __func__, backend_name.c_str(), error.what());
                return false;
            } catch (...) {
                GGML_LOG_ERROR(
                        "%s: failed to start persistent FFN worker for backend %s\n",
                        __func__, backend_name.c_str());
                return false;
            }

            std::unique_lock<std::mutex> lock(mutex);
            state_cv.wait(lock, [&]() { return ready; });
            return true;
        }

        bool submit(
                ggml_backend_sched_ffn_job_fn fn,
                void * context,
                enum ggml_status * status,
                uint64_t * ticket,
                ggml_backend_sched_ffn_worker_timeline * timeline) {
            GGML_ASSERT(fn != nullptr);
            GGML_ASSERT(status != nullptr);
            GGML_ASSERT(ticket != nullptr);

            std::lock_guard<std::mutex> lock(mutex);
            if (stopping || queue_size >= GGML_SCHED_MAX_BACKENDS) {
                return false;
            }

            const uint64_t current_ticket = next_ticket++;
            if (timeline != nullptr) {
                timeline->ticket = current_ticket;
                timeline->queue_depth_at_submit = queue_size;
                timeline->affinity_cpu = affinity_cpu;
                timeline->job_enqueued_us = ggml_time_us();
            }
            queue[queue_tail] = { fn, context, status, current_ticket, timeline };
            queue_tail = (queue_tail + 1) % GGML_SCHED_MAX_BACKENDS;
            ++queue_size;
            *ticket = current_ticket;
            work_cv.notify_one();
            return true;
        }

        void wait(uint64_t ticket, ggml_backend_sched_ffn_worker_timeline * timeline) {
            if (timeline != nullptr) {
                timeline->collect_begin_us = ggml_time_us();
            }
            std::unique_lock<std::mutex> lock(mutex);
            if (timeline != nullptr) {
                timeline->wait_blocked = completed_ticket < ticket ? 1 : 0;
            }
            state_cv.wait(lock, [&]() { return completed_ticket >= ticket; });
            if (timeline != nullptr) {
                timeline->collect_end_us = ggml_time_us();
            }
        }

        void stop() {
            {
                std::lock_guard<std::mutex> lock(mutex);
                stopping = true;
            }
            work_cv.notify_one();
            if (thread.joinable()) {
                thread.join();
            }
        }

        void apply_affinity() {
            if (affinity_cpu < 0) {
                return;
            }

            if (!ggml_backend_sched_pin_current_thread_to_cpu(affinity_cpu)) {
                const int error = errno;
                GGML_LOG_WARN(
                        "%s: failed to pin persistent FFN worker backend %s to CPU %d: %s\n",
                        __func__, backend_name.c_str(), affinity_cpu, strerror(error));
            } else {
                GGML_LOG_INFO(
                        "%s: pinned persistent FFN worker backend %s to CPU %d\n",
                        __func__, backend_name.c_str(), affinity_cpu);
            }
        }

        void run() {
            apply_affinity();
            {
                std::lock_guard<std::mutex> lock(mutex);
                ready = true;
            }
            state_cv.notify_all();

            for (;;) {
                job current;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    work_cv.wait(lock, [&]() { return stopping || queue_size > 0; });
                    if (stopping && queue_size == 0) {
                        break;
                    }

                    current = queue[queue_head];
                    queue_head = (queue_head + 1) % GGML_SCHED_MAX_BACKENDS;
                    --queue_size;
                }

                if (current.timeline != nullptr) {
                    current.timeline->worker_dequeue_us = ggml_time_us();
                    current.timeline->worker_tid = ggml_backend_sched_current_tid();
                    current.timeline->start_cpu = ggml_backend_sched_current_cpu();
                    current.timeline->job_start_us = ggml_time_us();
                }
                try {
                    current.fn(current.context);
                } catch (const std::exception & error) {
                    *current.status = GGML_STATUS_FAILED;
                    GGML_LOG_ERROR(
                            "%s: persistent FFN worker backend %s caught an unexpected exception: %s\n",
                            __func__, backend_name.c_str(), error.what());
                } catch (...) {
                    *current.status = GGML_STATUS_FAILED;
                    GGML_LOG_ERROR(
                            "%s: persistent FFN worker backend %s caught an unexpected exception\n",
                            __func__, backend_name.c_str());
                }

                if (current.timeline != nullptr) {
                    current.timeline->end_cpu = ggml_backend_sched_current_cpu();
                    current.timeline->job_end_us = ggml_time_us();
                }

                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (current.timeline != nullptr) {
                        // Publish the completed timeline under the same mutex
                        // that releases worker::wait(). The caller never reads
                        // this record before the ticket predicate is satisfied.
                        current.timeline->completion_publish_us = ggml_time_us();
                    }
                    completed_ticket = current.ticket;
                }
                state_cv.notify_all();
            }
        }
    };

    int n_backends = 0;
    worker * workers[GGML_SCHED_MAX_BACKENDS] = {};

    ~ggml_backend_sched_ffn_executor() {
        shutdown();
    }

    bool initialize(ggml_backend_t * backends, int backend_count, const int * affinity_cpus) {
        GGML_ASSERT(backends != nullptr);
        GGML_ASSERT(backend_count > 0 && backend_count <= GGML_SCHED_MAX_BACKENDS);
        GGML_ASSERT(affinity_cpus != nullptr);
        n_backends = backend_count;

        for (int backend_id = 0; backend_id < n_backends; ++backend_id) {
            ggml_backend_t backend = backends[backend_id];
            ggml_backend_dev_t device = ggml_backend_get_device(backend);
            if (device != nullptr && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                continue;
            }

            worker * current = nullptr;
            try {
                current = new worker(backend_id, ggml_backend_name(backend), affinity_cpus[backend_id]);
            } catch (const std::exception & error) {
                GGML_LOG_ERROR(
                        "%s: failed to allocate persistent FFN worker for backend %s: %s\n",
                        __func__, ggml_backend_name(backend), error.what());
                shutdown();
                return false;
            } catch (...) {
                GGML_LOG_ERROR(
                        "%s: failed to allocate persistent FFN worker for backend %s\n",
                        __func__, ggml_backend_name(backend));
                shutdown();
                return false;
            }

            workers[backend_id] = current;
            if (!current->start()) {
                shutdown();
                return false;
            }
        }

        return true;
    }

    bool submit(
            int backend_id,
            ggml_backend_sched_ffn_job_fn fn,
            void * context,
            enum ggml_status * status,
            uint64_t * ticket,
            ggml_backend_sched_ffn_worker_timeline * timeline) {
        if (backend_id < 0 || backend_id >= n_backends || workers[backend_id] == nullptr) {
            return false;
        }
        return workers[backend_id]->submit(fn, context, status, ticket, timeline);
    }

    void wait(
            int backend_id,
            uint64_t ticket,
            ggml_backend_sched_ffn_worker_timeline * timeline) {
        GGML_ASSERT(backend_id >= 0 && backend_id < n_backends);
        GGML_ASSERT(workers[backend_id] != nullptr);
        workers[backend_id]->wait(ticket, timeline);
    }

    void shutdown() {
        for (int backend_id = 0; backend_id < n_backends; ++backend_id) {
            delete workers[backend_id];
            workers[backend_id] = nullptr;
        }
        n_backends = 0;
    }
};

static bool ggml_backend_sched_ffn_persistent_workers_enabled() {
    const char * value = getenv("GGML_FFN_PERSISTENT_DEVICE_WORKERS");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return strcmp(value, "1") == 0 || strcmp(value, "on") == 0 || strcmp(value, "ON") == 0 ||
        strcmp(value, "true") == 0 || strcmp(value, "TRUE") == 0 ||
        strcmp(value, "yes") == 0 || strcmp(value, "YES") == 0;
}

static int ggml_backend_sched_ffn_worker_cpu(const char * env_name, int fallback_cpu) {
    GGML_ASSERT(env_name != nullptr);

    const char * value = getenv(env_name);
    if (value == nullptr || value[0] == '\0') {
        return fallback_cpu;
    }

    errno = 0;
    char * end = nullptr;
    const long cpu = strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || cpu < -1 || cpu > INT_MAX) {
        GGML_LOG_WARN(
                "%s: invalid %s='%s'; using fallback CPU %d\n",
                __func__, env_name, value, fallback_cpu);
        return fallback_cpu;
    }
    return (int) cpu;
}

struct ggml_backend_sched_ffn_device_job {
    ggml_backend_t backend = nullptr;
    struct ggml_cgraph * graph = nullptr;
    enum ggml_status * status = nullptr;
    int64_t * dt_us = nullptr;
    ggml_backend_sched_ffn_worker_timeline * timeline = nullptr;
};

static void ggml_backend_sched_run_ffn_device_job(void * context) {
    auto * job = (ggml_backend_sched_ffn_device_job *) context;
    GGML_ASSERT(job != nullptr);
    GGML_ASSERT(job->backend != nullptr);
    GGML_ASSERT(job->graph != nullptr);
    GGML_ASSERT(job->status != nullptr);
    GGML_ASSERT(job->dt_us != nullptr);

    const int64_t t0_us = ggml_time_us();
    if (job->timeline != nullptr) {
        job->timeline->backend_begin_us = t0_us;
    }
    enum ggml_status status = GGML_STATUS_FAILED;
    try {
        status = ggml_backend_graph_compute_async(job->backend, job->graph);
        if (job->timeline != nullptr) {
            job->timeline->graph_compute_return_us = ggml_time_us();
        }
        if (status == GGML_STATUS_SUCCESS) {
            // Parallel FFN device branches must finish before the reduce split can
            // consume them. Keep this synchronization on the backend's dedicated
            // persistent host thread.
            if (job->timeline != nullptr) {
                job->timeline->sync_begin_us = ggml_time_us();
            }
            ggml_backend_synchronize(job->backend);
            if (job->timeline != nullptr) {
                job->timeline->sync_end_us = ggml_time_us();
            }
        }
    } catch (const std::exception & error) {
        status = GGML_STATUS_FAILED;
        GGML_LOG_ERROR(
                "%s: backend %s threw while executing a persistent FFN job: %s\n",
                __func__, ggml_backend_name(job->backend), error.what());
    } catch (...) {
        status = GGML_STATUS_FAILED;
        GGML_LOG_ERROR(
                "%s: backend %s threw while executing a persistent FFN job\n",
                __func__, ggml_backend_name(job->backend));
    }
    const int64_t t1_us = ggml_time_us();

    *job->status = status;
    *job->dt_us = t1_us - t0_us;
}

//
// Backend scheduler profiling (process-global, lightweight)
//
// This profiling path is intentionally coarse-grained:
// - phase split: prefill vs decode
// - backend split: CPU vs HTP vs GPU
// - attribution: per-graph wall-time buckets and per-op counters
//
// It is currently implemented as process-global mutable state, so it is
// intended for single-run instrumentation and is not thread-safe.

static ggml_backend_sched_profile_phase g_sched_profile_phase = GGML_BACKEND_SCHED_PROFILE_PREFILL;

enum ggml_sched_profile_backend_kind {
    GGML_SCHED_PROFILE_BACKEND_CPU,
    GGML_SCHED_PROFILE_BACKEND_HTP,
    GGML_SCHED_PROFILE_BACKEND_GPU,
};

struct ggml_backend_sched_profile_state {
    bool enabled = false;
    ggml_backend_sched_profile_data out = {};

    // Layer-id presence maps (index = layer id). These are used to count
    // unique layers touched in each phase/backend bucket without double
    // counting repeated ops from the same layer.
    std::vector<uint8_t> prefill_cpu_layers;
    std::vector<uint8_t> prefill_htp_layers;
    std::vector<uint8_t> prefill_gpu_layers;
    std::vector<uint8_t> decode_cpu_layers;
    std::vector<uint8_t> decode_htp_layers;
    std::vector<uint8_t> decode_gpu_layers;
};

static ggml_backend_sched_profile_state g_sched_profile;

static bool ggml_sched_profile_enabled(void) {
    return g_sched_profile.enabled;
}

struct ggml_backend_sched_trace_row {
    int query_id = -1;
    enum ggml_backend_sched_profile_phase phase = GGML_BACKEND_SCHED_PROFILE_PREFILL;
    int64_t graph_id = -1;
    int token_index = -1;
    int n_past = -1;
    int n_tokens = -1;

    int split_id = -1;
    int layer = -1;
    int node_start = -1;
    int node_end = -1;
    int node_count = 0;

    char backend[GGML_MAX_NAME] = {};
    char first_node[GGML_MAX_NAME] = {};
    char last_node[GGML_MAX_NAME] = {};
    enum ggml_op op_first = GGML_OP_NONE;
    enum ggml_op op_last = GGML_OP_NONE;

    int64_t copy_us = 0;
    int64_t wait_us = 0;
    int64_t compute_submit_us = 0;
    int64_t compute_wall_us = 0;
    bool is_ffn_group = false;
    char ffn_branch[GGML_MAX_NAME] = {};
    int group_id = -1;
    int64_t group_wall_us = 0;
    int64_t group_copy_us = 0;
    char timing_mode[32] = {};
};

struct ggml_backend_sched_trace_state {
    bool initialized = false;
    bool enabled = false;
    bool trace_prefill = true;
    bool trace_decode = true;
    bool warned_open = false;
    FILE * file = nullptr;
    std::string path = "output/scheduler_trace.csv";

    int64_t graph_id = 0;
    int query_id = -1;
    int token_index = -1;
    int n_past = -1;
    int n_tokens = -1;
    std::vector<ggml_backend_sched_trace_row> rows;
};

// llama-ignite-npu uses one context and one scheduler thread; rows are flushed
// after each query, outside the measured inference path.
static ggml_backend_sched_trace_state g_sched_trace;
static constexpr size_t GGML_SCHED_TRACE_ROW_RESERVE = 32768;

struct ggml_backend_sched_ffn_worker_trace_row {
    int schema_version = 1;
    int64_t trace_session_id = -1;
    int64_t group_seq = -1;
    int64_t job_id = -1;

    int query_id = -1;
    enum ggml_backend_sched_profile_phase phase = GGML_BACKEND_SCHED_PROFILE_PREFILL;
    int64_t graph_id = -1;
    int token_index = -1;
    int n_past = -1;
    int n_tokens = -1;
    int group_id = -1;
    int split_id = -1;
    int layer = -1;
    int node_count = 0;

    char backend[GGML_MAX_NAME] = {};
    char ffn_branch[GGML_MAX_NAME] = {};
    char first_node[GGML_MAX_NAME] = {};
    char last_node[GGML_MAX_NAME] = {};
    enum ggml_status status = GGML_STATUS_SUCCESS;

    int64_t group_begin_us = -1;
    int64_t copy_end_us = -1;
    int64_t compute_begin_us = -1;
    int64_t group_end_us = -1;
    ggml_backend_sched_ffn_worker_timeline timeline;

    // Additive partitions for common-path rows:
    //   group_copy + post_copy_setup + group_compute_wall = group_wall
    //   worker_setup + graph_call + post_graph_cleanup + sync_wait +
    //       job_finalize = branch_wall
    // sched_compute_wall mirrors scheduler_trace.csv and overlaps the branch
    // partition. Dispatch and queue metrics may also overlap each other.
    int64_t group_copy_us = -1;
    int64_t post_copy_setup_us = -1;
    int64_t dispatch_us = -1;
    int64_t queue_delay_us = -1;
    int64_t dispatch_to_start_us = -1;
    int64_t worker_setup_us = -1;
    int64_t graph_call_us = -1;
    int64_t post_graph_cleanup_us = -1;
    int64_t sync_wait_us = -1;
    int64_t sched_compute_wall_us = -1;
    int64_t job_finalize_us = -1;
    int64_t branch_wall_us = -1;
    int64_t publish_delay_us = -1;
    int64_t collect_wait_us = -1;
    int64_t start_offset_us = -1;
    int64_t finish_offset_us = -1;
    int64_t group_start_skew_us = -1;
    int64_t group_tail_after_last_backend_us = -1;
    int64_t group_tail_after_last_publish_us = -1;
    int64_t group_compute_wall_us = -1;
    int64_t group_wall_us = -1;
    int is_longest_duration = 0;
    int is_last_backend_finisher = 0;
    int is_last_completion_publisher = 0;
};

struct ggml_backend_sched_ffn_worker_trace_state {
    bool initialized = false;
    bool enabled = false;
    bool warned_open = false;
    FILE * file = nullptr;
    std::string path;

    int64_t trace_session_id = -1;
    int64_t graph_id = 0;
    int64_t next_group_seq = 0;
    int64_t next_job_id = 0;
    int query_id = -1;
    int token_index = -1;
    int n_past = -1;
    int n_tokens = -1;
    std::unique_ptr<ggml_backend_sched_ffn_worker_timeline[]> timelines;
    std::vector<ggml_backend_sched_ffn_worker_trace_row> rows;
};

static ggml_backend_sched_ffn_worker_trace_state g_ffn_worker_trace;
// Covers a typical 50-token, 3-branch, 32-layer run without hot-path growth
// while keeping the opt-in trace buffer substantially smaller than 32768 rows.
static constexpr size_t GGML_FFN_WORKER_TRACE_ROW_RESERVE = 8192;

static bool ggml_sched_trace_env_enabled(const char * name) {
    const char * env = getenv(name);
    if (env == nullptr) {
        return false;
    }

    return strcmp(env, "1") == 0 || strcmp(env, "on") == 0 || strcmp(env, "ON") == 0 ||
        strcmp(env, "yes") == 0 || strcmp(env, "YES") == 0 ||
        strcmp(env, "true") == 0 || strcmp(env, "TRUE") == 0;
}

static bool ggml_sched_trace_str_eq_ci(const char * lhs, const char * rhs) {
    if (lhs == nullptr || rhs == nullptr) {
        return false;
    }

    while (*lhs != '\0' && *rhs != '\0') {
        if (std::tolower((unsigned char) *lhs) != std::tolower((unsigned char) *rhs)) {
            return false;
        }
        ++lhs;
        ++rhs;
    }

    return *lhs == '\0' && *rhs == '\0';
}

static void ggml_sched_trace_init(void) {
    if (g_sched_trace.initialized) {
        return;
    }

    g_sched_trace.initialized = true;
    g_sched_trace.enabled = ggml_sched_trace_env_enabled("SCHED_TRACE") ||
        ggml_sched_trace_env_enabled("GGML_SCHED_TRACE");

    const char * phase = getenv("SCHED_TRACE_PHASE");
    if (phase == nullptr || phase[0] == '\0') {
        phase = getenv("GGML_SCHED_TRACE_PHASE");
    }
    if (phase != nullptr && phase[0] != '\0') {
        if (ggml_sched_trace_str_eq_ci(phase, "prefill")) {
            g_sched_trace.trace_prefill = true;
            g_sched_trace.trace_decode = false;
        } else if (ggml_sched_trace_str_eq_ci(phase, "decode")) {
            g_sched_trace.trace_prefill = false;
            g_sched_trace.trace_decode = true;
        } else if (!ggml_sched_trace_str_eq_ci(phase, "both") &&
                   !ggml_sched_trace_str_eq_ci(phase, "all")) {
            GGML_LOG_WARN(
                    "%s: invalid scheduler trace phase '%s'; expected prefill, decode, or both; tracing both phases\n",
                    __func__, phase);
        }
    }

    if (const char * path = getenv("SCHED_TRACE_PATH")) {
        if (path[0] != '\0') {
            g_sched_trace.path = path;
        }
    } else if (const char * path = getenv("GGML_SCHED_TRACE_PATH")) {
        if (path[0] != '\0') {
            g_sched_trace.path = path;
        }
    }

    if (g_sched_trace.enabled) {
        g_sched_trace.rows.reserve(GGML_SCHED_TRACE_ROW_RESERVE);
    }
}

static void ggml_ffn_worker_trace_init(void) {
    if (g_ffn_worker_trace.initialized) {
        return;
    }

    g_ffn_worker_trace.initialized = true;
    g_ffn_worker_trace.enabled = ggml_sched_trace_env_enabled("GGML_FFN_WORKER_TRACE") ||
        ggml_sched_trace_env_enabled("FFN_WORKER_TRACE");
    if (!g_ffn_worker_trace.enabled) {
        return;
    }

    const char * path = getenv("GGML_FFN_WORKER_TRACE_PATH");
    if (path == nullptr || path[0] == '\0') {
        path = getenv("FFN_WORKER_TRACE_PATH");
    }
    g_ffn_worker_trace.path = path != nullptr && path[0] != '\0'
        ? path
        : "output/ffn_worker_trace.csv";

    g_ffn_worker_trace.trace_session_id = ggml_time_us();
    g_ffn_worker_trace.timelines.reset(
            new ggml_backend_sched_ffn_worker_timeline[GGML_SCHED_MAX_BACKENDS]);
    g_ffn_worker_trace.rows.reserve(GGML_FFN_WORKER_TRACE_ROW_RESERVE);
}

static bool ggml_ffn_worker_trace_enabled(void) {
    ggml_ffn_worker_trace_init();
    return g_ffn_worker_trace.enabled;
}

static bool ggml_sched_trace_enabled(void) {
    ggml_sched_trace_init();
    return g_sched_trace.enabled;
}

static bool ggml_sched_trace_current_phase_selected(void) {
    return g_sched_profile_phase == GGML_BACKEND_SCHED_PROFILE_PREFILL
        ? g_sched_trace.trace_prefill
        : g_sched_trace.trace_decode;
}

static bool ggml_ffn_worker_trace_phase_enabled(void) {
    if (!ggml_ffn_worker_trace_enabled()) {
        return false;
    }

    ggml_sched_trace_init();
    return ggml_sched_trace_current_phase_selected();
}

static bool ggml_sched_trace_phase_enabled(void) {
    if (!ggml_sched_trace_enabled()) {
        return false;
    }

    return ggml_sched_trace_current_phase_selected();
}

static FILE * ggml_sched_trace_file(void) {
    ggml_sched_trace_init();
    if (!g_sched_trace.enabled) {
        return nullptr;
    }

    if (g_sched_trace.file != nullptr) {
        return g_sched_trace.file;
    }

    g_sched_trace.file = fopen(g_sched_trace.path.c_str(), "a");
    if (g_sched_trace.file == nullptr) {
        if (!g_sched_trace.warned_open) {
            GGML_LOG_WARN("%s: failed to open scheduler trace CSV '%s'\n", __func__, g_sched_trace.path.c_str());
            g_sched_trace.warned_open = true;
        }
        return nullptr;
    }

    fseek(g_sched_trace.file, 0, SEEK_END);
    const long pos = ftell(g_sched_trace.file);
    if (pos == 0) {
        fprintf(g_sched_trace.file,
                "query_id,phase,graph_id,token_index,n_past,n_tokens,"
                "split_id,backend,layer,"
                "node_start,node_end,node_count,"
                "first_node,last_node,op_first,op_last,"
                "copy_in_us,wait_before_copy_us,"
                "compute_submit_us,compute_wall_us,"
                "is_ffn_group,ffn_branch,group_id,group_wall_us,group_copy_us,"
                "timing_mode\n");
    }

    return g_sched_trace.file;
}

static void ggml_sched_trace_csv_cell(FILE * f, const char * value) {
    fputc('"', f);
    if (value != nullptr) {
        for (const char * p = value; *p != '\0'; ++p) {
            if (*p == '"') {
                fputc('"', f);
            }
            fputc(*p, f);
        }
    }
    fputc('"', f);
}

static const char * ggml_sched_profile_phase_name(enum ggml_backend_sched_profile_phase phase) {
    return phase == GGML_BACKEND_SCHED_PROFILE_PREFILL ? "prefill" : "decode";
}

static const char * ggml_ffn_worker_mode_name(enum ggml_backend_sched_ffn_worker_mode mode) {
    switch (mode) {
        case GGML_SCHED_FFN_WORKER_CALLER:     return "caller";
        case GGML_SCHED_FFN_WORKER_PERSISTENT: return "persistent";
        case GGML_SCHED_FFN_WORKER_TRANSIENT:  return "transient";
        case GGML_SCHED_FFN_WORKER_SERIAL:     return "serial_fallback";
        case GGML_SCHED_FFN_WORKER_NONE:       return "none";
    }
    return "unknown";
}

static const char * ggml_ffn_collect_kind_name(enum ggml_backend_sched_ffn_collect_kind kind) {
    switch (kind) {
        case GGML_SCHED_FFN_COLLECT_TICKET_WAIT: return "ticket_wait";
        case GGML_SCHED_FFN_COLLECT_THREAD_JOIN: return "thread_join";
        case GGML_SCHED_FFN_COLLECT_NONE:        return "none";
    }
    return "unknown";
}

static void ggml_ffn_worker_trace_optional_i64(FILE * f, int64_t value) {
    if (value >= 0) {
        fprintf(f, "%lld", (long long) value);
    }
}

static constexpr char GGML_FFN_WORKER_TRACE_CSV_HEADER[] =
        "schema_version,trace_session_id,group_seq,job_id,"
        "query_id,phase,graph_id,token_index,n_past,n_tokens,"
        "group_id,split_id,layer,backend,ffn_branch,node_count,first_node,last_node,"
        "worker_mode,collect_kind,status,ticket,queue_depth_at_submit,worker_tid,"
        "affinity_cpu,start_cpu,end_cpu,wait_blocked,collect_order,"
        "group_begin_us,copy_end_us,compute_begin_us,dispatch_begin_us,dispatch_end_us,"
        "job_enqueued_us,worker_dequeue_us,job_start_us,backend_begin_us,"
        "graph_compute_return_us,sync_begin_us,sync_end_us,job_end_us,"
        "completion_publish_us,collect_begin_us,collect_end_us,group_end_us,"
        "group_copy_us,post_copy_setup_us,dispatch_us,queue_delay_us,dispatch_to_start_us,"
        "worker_setup_us,graph_call_us,post_graph_cleanup_us,sync_wait_us,"
        "sched_compute_wall_us,job_finalize_us,"
        "branch_wall_us,publish_delay_us,collect_wait_us,"
        "start_offset_us,finish_offset_us,group_start_skew_us,"
        "group_tail_after_last_backend_us,group_tail_after_last_publish_us,"
        "group_compute_wall_us,group_wall_us,"
        "is_longest_duration,is_last_backend_finisher,is_last_completion_publisher\n";

static FILE * ggml_ffn_worker_trace_file(void) {
    ggml_ffn_worker_trace_init();
    if (!g_ffn_worker_trace.enabled) {
        return nullptr;
    }

    if (g_ffn_worker_trace.file != nullptr) {
        return g_ffn_worker_trace.file;
    }

    g_ffn_worker_trace.file = fopen(g_ffn_worker_trace.path.c_str(), "a+");
    if (g_ffn_worker_trace.file == nullptr) {
        if (!g_ffn_worker_trace.warned_open) {
            GGML_LOG_WARN(
                    "%s: failed to open FFN worker trace CSV '%s'\n",
                    __func__, g_ffn_worker_trace.path.c_str());
            g_ffn_worker_trace.warned_open = true;
        }
        return nullptr;
    }

    if (fseek(g_ffn_worker_trace.file, 0, SEEK_END) != 0) {
        if (!g_ffn_worker_trace.warned_open) {
            GGML_LOG_WARN(
                    "%s: failed to seek FFN worker trace CSV '%s'\n",
                    __func__, g_ffn_worker_trace.path.c_str());
        }
        fclose(g_ffn_worker_trace.file);
        g_ffn_worker_trace.file = nullptr;
        g_ffn_worker_trace.warned_open = true;
        return nullptr;
    }
    const long pos = ftell(g_ffn_worker_trace.file);
    if (pos < 0) {
        if (!g_ffn_worker_trace.warned_open) {
            GGML_LOG_WARN(
                    "%s: failed to inspect FFN worker trace CSV '%s'\n",
                    __func__, g_ffn_worker_trace.path.c_str());
        }
        fclose(g_ffn_worker_trace.file);
        g_ffn_worker_trace.file = nullptr;
        g_ffn_worker_trace.warned_open = true;
        return nullptr;
    }
    if (pos == 0) {
        if (fputs(GGML_FFN_WORKER_TRACE_CSV_HEADER, g_ffn_worker_trace.file) == EOF) {
            if (!g_ffn_worker_trace.warned_open) {
                GGML_LOG_WARN(
                        "%s: failed to write FFN worker trace CSV header '%s'\n",
                        __func__, g_ffn_worker_trace.path.c_str());
            }
            fclose(g_ffn_worker_trace.file);
            g_ffn_worker_trace.file = nullptr;
            g_ffn_worker_trace.warned_open = true;
            return nullptr;
        }
    } else {
        char existing_header[sizeof(GGML_FFN_WORKER_TRACE_CSV_HEADER)];
        if (fseek(g_ffn_worker_trace.file, 0, SEEK_SET) != 0 ||
                fgets(existing_header, sizeof(existing_header), g_ffn_worker_trace.file) == nullptr ||
                strcmp(existing_header, GGML_FFN_WORKER_TRACE_CSV_HEADER) != 0) {
            if (!g_ffn_worker_trace.warned_open) {
                GGML_LOG_WARN(
                        "%s: FFN worker trace CSV schema mismatch at '%s'; refusing to append\n",
                        __func__, g_ffn_worker_trace.path.c_str());
            }
            fclose(g_ffn_worker_trace.file);
            g_ffn_worker_trace.file = nullptr;
            g_ffn_worker_trace.warned_open = true;
            return nullptr;
        }
        if (fseek(g_ffn_worker_trace.file, 0, SEEK_END) != 0) {
            if (!g_ffn_worker_trace.warned_open) {
                GGML_LOG_WARN(
                        "%s: failed to restore FFN worker trace CSV position '%s'\n",
                        __func__, g_ffn_worker_trace.path.c_str());
            }
            fclose(g_ffn_worker_trace.file);
            g_ffn_worker_trace.file = nullptr;
            g_ffn_worker_trace.warned_open = true;
            return nullptr;
        }
    }

    return g_ffn_worker_trace.file;
}

static void ggml_ffn_worker_trace_flush_rows(void) {
    if (g_ffn_worker_trace.rows.empty()) {
        return;
    }

    FILE * f = ggml_ffn_worker_trace_file();
    if (f == nullptr) {
        g_ffn_worker_trace.rows.clear();
        return;
    }

    for (const auto & row : g_ffn_worker_trace.rows) {
        fprintf(f, "%d,%lld,%lld,%lld,%d,%s,%lld,%d,%d,%d,%d,%d,%d,",
                row.schema_version,
                (long long) row.trace_session_id,
                (long long) row.group_seq,
                (long long) row.job_id,
                row.query_id,
                ggml_sched_profile_phase_name(row.phase),
                (long long) row.graph_id,
                row.token_index,
                row.n_past,
                row.n_tokens,
                row.group_id,
                row.split_id,
                row.layer);
        ggml_sched_trace_csv_cell(f, row.backend);
        fputc(',', f);
        ggml_sched_trace_csv_cell(f, row.ffn_branch);
        fprintf(f, ",%d,", row.node_count);
        ggml_sched_trace_csv_cell(f, row.first_node);
        fputc(',', f);
        ggml_sched_trace_csv_cell(f, row.last_node);
        fprintf(f, ",%s,%s,%d,",
                ggml_ffn_worker_mode_name(row.timeline.mode),
                ggml_ffn_collect_kind_name(row.timeline.collect_kind),
                (int) row.status);

        const int64_t values[] = {
            row.timeline.ticket > 0 ? (int64_t) row.timeline.ticket : -1,
            row.timeline.queue_depth_at_submit,
            row.timeline.worker_tid,
            row.timeline.affinity_cpu,
            row.timeline.start_cpu,
            row.timeline.end_cpu,
            row.timeline.wait_blocked,
            row.timeline.collect_order,
            row.group_begin_us,
            row.copy_end_us,
            row.compute_begin_us,
            row.timeline.dispatch_begin_us,
            row.timeline.dispatch_end_us,
            row.timeline.job_enqueued_us,
            row.timeline.worker_dequeue_us,
            row.timeline.job_start_us,
            row.timeline.backend_begin_us,
            row.timeline.graph_compute_return_us,
            row.timeline.sync_begin_us,
            row.timeline.sync_end_us,
            row.timeline.job_end_us,
            row.timeline.completion_publish_us,
            row.timeline.collect_begin_us,
            row.timeline.collect_end_us,
            row.group_end_us,
            row.group_copy_us,
            row.post_copy_setup_us,
            row.dispatch_us,
            row.queue_delay_us,
            row.dispatch_to_start_us,
            row.worker_setup_us,
            row.graph_call_us,
            row.post_graph_cleanup_us,
            row.sync_wait_us,
            row.sched_compute_wall_us,
            row.job_finalize_us,
            row.branch_wall_us,
            row.publish_delay_us,
            row.collect_wait_us,
            row.start_offset_us,
            row.finish_offset_us,
            row.group_start_skew_us,
            row.group_tail_after_last_backend_us,
            row.group_tail_after_last_publish_us,
            row.group_compute_wall_us,
            row.group_wall_us,
            row.is_longest_duration,
            row.is_last_backend_finisher,
            row.is_last_completion_publisher,
        };
        for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
            ggml_ffn_worker_trace_optional_i64(f, values[i]);
            fputc(i + 1 == sizeof(values) / sizeof(values[0]) ? '\n' : ',', f);
        }
    }

    if (fflush(f) != 0 && !g_ffn_worker_trace.warned_open) {
        GGML_LOG_WARN(
                "%s: failed to flush FFN worker trace CSV '%s'\n",
                __func__, g_ffn_worker_trace.path.c_str());
        g_ffn_worker_trace.warned_open = true;
    }
    g_ffn_worker_trace.rows.clear();
}

static void ggml_sched_trace_flush_rows(void) {
    if (g_sched_trace.rows.empty()) {
        return;
    }

    FILE * f = ggml_sched_trace_file();
    if (f == nullptr) {
        g_sched_trace.rows.clear();
        return;
    }

    for (const auto & row : g_sched_trace.rows) {
        fprintf(f, "%d,%s,%lld,%d,%d,%d,",
                row.query_id,
                ggml_sched_profile_phase_name(row.phase),
                (long long) row.graph_id,
                row.token_index,
                row.n_past,
                row.n_tokens);

        fprintf(f, "%d,", row.split_id);
        ggml_sched_trace_csv_cell(f, row.backend);
        fprintf(f, ",%d,", row.layer);

        fprintf(f, "%d,%d,%d,", row.node_start, row.node_end, row.node_count);
        ggml_sched_trace_csv_cell(f, row.first_node);
        fputc(',', f);
        ggml_sched_trace_csv_cell(f, row.last_node);
        fputc(',', f);
        ggml_sched_trace_csv_cell(f, ggml_op_name(row.op_first));
        fputc(',', f);
        ggml_sched_trace_csv_cell(f, ggml_op_name(row.op_last));
        fputc(',', f);

        fprintf(f, "%lld,%lld,%lld,%lld,%d,",
                (long long) row.copy_us,
                (long long) row.wait_us,
                (long long) row.compute_submit_us,
                (long long) row.compute_wall_us,
                row.is_ffn_group ? 1 : 0);
        ggml_sched_trace_csv_cell(f, row.ffn_branch);
        fprintf(f, ",%d,%lld,%lld,",
                row.group_id,
                (long long) row.group_wall_us,
                (long long) row.group_copy_us);
        ggml_sched_trace_csv_cell(f, row.timing_mode);
        fputc('\n', f);
    }

    fflush(f);
    g_sched_trace.rows.clear();
}

void ggml_backend_sched_trace_set_enabled(bool enabled) {
    ggml_sched_trace_init();
    if (!enabled && g_sched_trace.enabled) {
        ggml_sched_trace_flush_rows();
    }

    g_sched_trace.enabled = enabled;
    if (enabled && g_sched_trace.rows.capacity() < GGML_SCHED_TRACE_ROW_RESERVE) {
        g_sched_trace.rows.reserve(GGML_SCHED_TRACE_ROW_RESERVE);
    } else if (!enabled && g_sched_trace.file != nullptr) {
        fclose(g_sched_trace.file);
        g_sched_trace.file = nullptr;
    }
}

void ggml_backend_sched_trace_set_path(const char * path) {
    if (path == nullptr || path[0] == '\0') {
        return;
    }

    ggml_sched_trace_init();
    if (g_sched_trace.path == path) {
        return;
    }

    ggml_sched_trace_flush_rows();
    if (g_sched_trace.file != nullptr) {
        fclose(g_sched_trace.file);
        g_sched_trace.file = nullptr;
    }
    g_sched_trace.path = path;
    g_sched_trace.warned_open = false;
}

void ggml_backend_sched_trace_reset(void) {
    const bool sched_trace_enabled = ggml_sched_trace_enabled();
    const bool worker_trace_enabled = ggml_ffn_worker_trace_enabled();
    if (!sched_trace_enabled && !worker_trace_enabled) {
        return;
    }

    if (sched_trace_enabled) {
        ggml_sched_trace_flush_rows();
        g_sched_trace.graph_id = 0;
        g_sched_trace.token_index = -1;
        g_sched_trace.n_past = -1;
        g_sched_trace.n_tokens = -1;
    }
    if (worker_trace_enabled) {
        ggml_ffn_worker_trace_flush_rows();
        g_ffn_worker_trace.graph_id = 0;
        g_ffn_worker_trace.token_index = -1;
        g_ffn_worker_trace.n_past = -1;
        g_ffn_worker_trace.n_tokens = -1;
    }
}

void ggml_backend_sched_trace_set_query_id(int query_id) {
    const bool sched_trace_enabled = ggml_sched_trace_enabled();
    const bool worker_trace_enabled = ggml_ffn_worker_trace_enabled();
    if (!sched_trace_enabled && !worker_trace_enabled) {
        return;
    }
    if (sched_trace_enabled) {
        g_sched_trace.query_id = query_id;
    }
    if (worker_trace_enabled) {
        g_ffn_worker_trace.query_id = query_id;
    }
}

void ggml_backend_sched_trace_set_ubatch(int token_index, int n_past, int n_tokens) {
    const bool sched_trace_enabled = ggml_sched_trace_enabled();
    const bool worker_trace_enabled = ggml_ffn_worker_trace_enabled();
    if (!sched_trace_enabled && !worker_trace_enabled) {
        return;
    }
    if (sched_trace_enabled) {
        g_sched_trace.token_index = token_index;
        g_sched_trace.n_past = n_past;
        g_sched_trace.n_tokens = n_tokens;
    }
    if (worker_trace_enabled) {
        g_ffn_worker_trace.token_index = token_index;
        g_ffn_worker_trace.n_past = n_past;
        g_ffn_worker_trace.n_tokens = n_tokens;
    }
}

void ggml_backend_sched_trace_flush(void) {
    if (g_sched_trace.initialized && !g_sched_trace.rows.empty()) {
        ggml_sched_trace_flush_rows();
    }
    if (g_ffn_worker_trace.initialized && !g_ffn_worker_trace.rows.empty()) {
        ggml_ffn_worker_trace_flush_rows();
    }
}

static inline int64_t ggml_sched_profile_time_us(void) {
    if (ggml_sched_profile_enabled()) {
        return ggml_time_us();
    }
    // compute_splits initializes trace state once per graph, so the fast path
    // can avoid taking the trace mutex for every copy/wait timestamp.
    return g_sched_trace.initialized && g_sched_trace.enabled &&
        ggml_sched_trace_current_phase_selected() ? ggml_time_us() : 0;
}

static ggml_sched_profile_backend_kind ggml_sched_profile_backend_bucket(ggml_backend_t backend) {
    if (backend != nullptr) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev != nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            return GGML_SCHED_PROFILE_BACKEND_CPU;
        }

        const char * name = ggml_backend_name(backend);
        if (name != nullptr && strncmp(name, "HTP", 3) == 0) {
            return GGML_SCHED_PROFILE_BACKEND_HTP;
        }
    }

    return GGML_SCHED_PROFILE_BACKEND_GPU;
}

static void ggml_sched_profile_mark_layer(std::vector<uint8_t> & seen, uint32_t & count, int layer_id) {
    if (layer_id < 0) {
        return;
    }

    const size_t idx = (size_t) layer_id;
    if (idx >= seen.size()) {
        seen.resize(idx + 1, 0);
    }

    if (!seen[idx]) {
        seen[idx] = 1;
        count++;
    }
}

static int ggml_sched_profile_extract_layer_id(const char * name) {
    if (!name || !name[0]) {
        return -1;
    }

    if (const char * p = strstr(name, "blk.")) {
        p += 4;
        if (!std::isdigit((unsigned char) *p)) {
            return -1;
        }

        int v = 0;
        while (std::isdigit((unsigned char) *p)) {
            v = v * 10 + (*p - '0');
            ++p;
        }

        if (*p == '.') {
            return v;
        }
    }

    if (const char * dash = strrchr(name, '-')) {
        const char * p = dash + 1;
        if (p[0] && std::isdigit((unsigned char) p[0])) {
            int v = 0;
            while (std::isdigit((unsigned char) *p)) {
                v = v * 10 + (*p - '0');
                ++p;
            }
            if (*p == '\0') {
                return v;
            }
        }
    }

    if (const char * p = strstr(name, "_l")) {
        const char * last = p;
        while ((p = strstr(p + 2, "_l"))) {
            last = p;
        }

        p = last + 2;
        if (p[0] && std::isdigit((unsigned char) p[0])) {
            int v = 0;
            while (std::isdigit((unsigned char) *p)) {
                v = v * 10 + (*p - '0');
                ++p;
            }
            if (*p == '\0') {
                return v;
            }
        }
    }

    return -1;
}

static void ggml_sched_profile_note_layers(const ggml_cgraph & graph, ggml_sched_profile_backend_kind bucket) {
    if (!ggml_sched_profile_enabled()) {
        return;
    }

    for (int i = 0; i < graph.n_nodes; ++i) {
        const ggml_tensor * t = graph.nodes[i];
        const int layer_id = ggml_sched_profile_extract_layer_id(t ? t->name : nullptr);
        if (layer_id < 0) {
            continue;
        }

        if (g_sched_profile_phase == GGML_BACKEND_SCHED_PROFILE_PREFILL) {
            switch (bucket) {
                case GGML_SCHED_PROFILE_BACKEND_CPU:
                    ggml_sched_profile_mark_layer(g_sched_profile.prefill_cpu_layers, g_sched_profile.out.prefill_cpu_layers, layer_id);
                    break;
                case GGML_SCHED_PROFILE_BACKEND_HTP:
                    ggml_sched_profile_mark_layer(g_sched_profile.prefill_htp_layers, g_sched_profile.out.prefill_htp_layers, layer_id);
                    break;
                case GGML_SCHED_PROFILE_BACKEND_GPU:
                    ggml_sched_profile_mark_layer(g_sched_profile.prefill_gpu_layers, g_sched_profile.out.prefill_gpu_layers, layer_id);
                    break;
            }
        } else {
            switch (bucket) {
                case GGML_SCHED_PROFILE_BACKEND_CPU:
                    ggml_sched_profile_mark_layer(g_sched_profile.decode_cpu_layers, g_sched_profile.out.decode_cpu_layers, layer_id);
                    break;
                case GGML_SCHED_PROFILE_BACKEND_HTP:
                    ggml_sched_profile_mark_layer(g_sched_profile.decode_htp_layers, g_sched_profile.out.decode_htp_layers, layer_id);
                    break;
                case GGML_SCHED_PROFILE_BACKEND_GPU:
                    ggml_sched_profile_mark_layer(g_sched_profile.decode_gpu_layers, g_sched_profile.out.decode_gpu_layers, layer_id);
                    break;
            }
        }
    }
}

static inline void ggml_sched_profile_add_op_type_count(enum ggml_op op, ggml_sched_profile_backend_kind bucket) {
    if (!ggml_sched_profile_enabled()) {
        return;
    }

    if (op < 0 || op >= GGML_OP_COUNT) {
        return;
    }

    if (g_sched_profile_phase == GGML_BACKEND_SCHED_PROFILE_PREFILL) {
        switch (bucket) {
            case GGML_SCHED_PROFILE_BACKEND_CPU:
                g_sched_profile.out.prefill_cpu_ops_by_type[op]++;
                break;
            case GGML_SCHED_PROFILE_BACKEND_HTP:
                g_sched_profile.out.prefill_htp_ops_by_type[op]++;
                break;
            case GGML_SCHED_PROFILE_BACKEND_GPU:
                g_sched_profile.out.prefill_gpu_ops_by_type[op]++;
                break;
        }
    } else {
        switch (bucket) {
            case GGML_SCHED_PROFILE_BACKEND_CPU:
                g_sched_profile.out.decode_cpu_ops_by_type[op]++;
                break;
            case GGML_SCHED_PROFILE_BACKEND_HTP:
                g_sched_profile.out.decode_htp_ops_by_type[op]++;
                break;
            case GGML_SCHED_PROFILE_BACKEND_GPU:
                g_sched_profile.out.decode_gpu_ops_by_type[op]++;
                break;
        }
    }
}

static inline void ggml_sched_profile_add_graph_op_type_counts(const ggml_cgraph & graph, ggml_sched_profile_backend_kind bucket) {
    if (!ggml_sched_profile_enabled()) {
        return;
    }

    for (int i = 0; i < graph.n_nodes; ++i) {
        const ggml_tensor * t = graph.nodes[i];
        if (t == nullptr) {
            continue;
        }
        ggml_sched_profile_add_op_type_count(t->op, bucket);
    }
}

void ggml_backend_sched_profile_set_enabled(bool enabled) {
    g_sched_profile.enabled = enabled;
    if (!enabled) {
        ggml_backend_sched_profile_reset();
    }
}

void ggml_backend_sched_profile_reset(void) {
    g_sched_profile_phase = GGML_BACKEND_SCHED_PROFILE_PREFILL;
    g_sched_profile.out = {};
    g_sched_profile.prefill_cpu_layers.clear();
    g_sched_profile.prefill_htp_layers.clear();
    g_sched_profile.prefill_gpu_layers.clear();
    g_sched_profile.decode_cpu_layers.clear();
    g_sched_profile.decode_htp_layers.clear();
    g_sched_profile.decode_gpu_layers.clear();
}

void ggml_backend_sched_profile_set_phase(enum ggml_backend_sched_profile_phase phase) {
    if (!ggml_sched_profile_enabled() && !ggml_sched_trace_enabled()) {
        return;
    }

    g_sched_profile_phase = phase;
}

static inline void ggml_sched_profile_add_copy_us(const int64_t dt_us) {
    if (!ggml_sched_profile_enabled()) {
        return;
    }

    const double dt_ms = (double) dt_us / 1000.0;
    if (g_sched_profile_phase == GGML_BACKEND_SCHED_PROFILE_PREFILL) {
        g_sched_profile.out.prefill_copy_ms += dt_ms;
    } else {
        g_sched_profile.out.decode_copy_ms += dt_ms;
    }
}

static inline void ggml_sched_profile_add_wait_us(const int64_t dt_us) {
    if (!ggml_sched_profile_enabled()) {
        return;
    }

    const double dt_ms = (double) dt_us / 1000.0;
    if (g_sched_profile_phase == GGML_BACKEND_SCHED_PROFILE_PREFILL) {
        g_sched_profile.out.prefill_wait_ms += dt_ms;
    } else {
        g_sched_profile.out.decode_wait_ms += dt_ms;
    }
}

void ggml_backend_sched_profile_add_build_ms(double build_ms) {
    if (!ggml_sched_profile_enabled()) {
        return;
    }

    if (g_sched_profile_phase == GGML_BACKEND_SCHED_PROFILE_PREFILL) {
        g_sched_profile.out.prefill_build_ms += build_ms;
    } else {
        g_sched_profile.out.decode_build_ms += build_ms;
    }
}

void ggml_backend_sched_profile_add_sampling_ms(double sampling_ms) {
    if (!ggml_sched_profile_enabled()) {
        return;
    }

    if (g_sched_profile_phase == GGML_BACKEND_SCHED_PROFILE_PREFILL) {
        g_sched_profile.out.prefill_sampling_ms += sampling_ms;
    } else {
        g_sched_profile.out.decode_sampling_ms += sampling_ms;
    }
}

struct ggml_backend_sched_profile_data ggml_backend_sched_profile_get(void) {
    return g_sched_profile.out;
}

struct ggml_backend_op_load_profile_state {
    bool enabled = false;
    bool breakdown_enabled = false;
    ggml_backend_sched_profile_phase phase = GGML_BACKEND_SCHED_PROFILE_PREFILL;
    ggml_backend_op_load_profile_data out = {};
    std::vector<std::string> name_patterns;
    std::vector<std::string> name_labels;
    std::mutex mutex;
};

static ggml_backend_op_load_profile_state g_op_load_profile;

static void ggml_op_load_profile_copy_name(char * dst, const std::string & src) {
    snprintf(dst, GGML_BACKEND_OP_LOAD_PROFILE_NAME_LEN, "%s", src.c_str());
}

static void ggml_op_load_profile_sync_name_groups_locked(void) {
    g_op_load_profile.out.n_name_groups = (uint32_t) std::min(
            g_op_load_profile.name_labels.size(),
            (size_t) GGML_BACKEND_OP_LOAD_PROFILE_MAX_GROUPS);

    for (uint32_t i = 0; i < g_op_load_profile.out.n_name_groups; ++i) {
        ggml_op_load_profile_copy_name(g_op_load_profile.out.name_groups[i], g_op_load_profile.name_labels[i]);
    }
}

static void ggml_op_load_profile_sync_ops_enabled_locked(void) {
    bool any = false;
    for (int op = 0; op < GGML_OP_COUNT; ++op) {
        if (g_op_load_profile.out.ops_enabled[op]) {
            any = true;
            break;
        }
    }

    if (!any) {
        g_op_load_profile.out.ops_enabled[GGML_OP_MUL_MAT]             = 1;
        g_op_load_profile.out.ops_enabled[GGML_OP_MUL_MAT_ID]          = 1;
        g_op_load_profile.out.ops_enabled[GGML_OP_MUL_MAT_SRC0_REGION] = 1;
        g_op_load_profile.out.ops_enabled[GGML_OP_CONT]                = 1;
        g_op_load_profile.out.ops_enabled[GGML_OP_SOFT_MAX]            = 1;
        g_op_load_profile.out.ops_enabled[GGML_OP_SET_ROWS]            = 1;
    }
}

static void ggml_op_load_profile_clear_accum_locked(void) {
    uint8_t ops_enabled[GGML_OP_COUNT];
    const bool breakdown_enabled = g_op_load_profile.breakdown_enabled;
    memcpy(ops_enabled, g_op_load_profile.out.ops_enabled, sizeof(ops_enabled));
    g_op_load_profile.out = {};
    memcpy(g_op_load_profile.out.ops_enabled, ops_enabled, sizeof(ops_enabled));
    g_op_load_profile.out.breakdown_enabled = breakdown_enabled ? 1 : 0;
    ggml_op_load_profile_sync_ops_enabled_locked();
    ggml_op_load_profile_sync_name_groups_locked();
}

static bool ggml_op_load_profile_op_from_name(const std::string & name, enum ggml_op & op_out) {
    if (name == "MAT_MUL") {
        op_out = GGML_OP_MUL_MAT;
        return true;
    }

    for (int op = 0; op < GGML_OP_COUNT; ++op) {
        const char * op_name = ggml_op_name((enum ggml_op) op);
        if (op_name != nullptr && name == op_name) {
            op_out = (enum ggml_op) op;
            return true;
        }
    }

    return false;
}

static std::string ggml_op_load_profile_trim_token(const char * begin, const char * end) {
    while (begin < end && std::isspace((unsigned char) *begin)) {
        ++begin;
    }
    while (end > begin && std::isspace((unsigned char) *(end - 1))) {
        --end;
    }

    return std::string(begin, end);
}

static bool ggml_op_load_profile_wildcard_match(const char * pattern, const char * text) {
    const char * star = nullptr;
    const char * retry = nullptr;

    while (*text != '\0') {
        if (*pattern == '?' || *pattern == *text) {
            ++pattern;
            ++text;
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
        ++pattern;
    }

    return *pattern == '\0';
}

static std::string ggml_op_load_profile_label_from_pattern(const std::string & pattern) {
    std::string label = pattern;
    const size_t eq = label.find('=');
    if (eq != std::string::npos) {
        label = ggml_op_load_profile_trim_token(label.c_str(), label.c_str() + eq);
    } else {
        while (!label.empty() && (label.back() == '*' || label.back() == '-' || label.back() == '_' || label.back() == '.')) {
            label.pop_back();
        }
    }

    for (char & c : label) {
        if (!std::isalnum((unsigned char) c) && c != '_') {
            c = '_';
        }
    }

    if (label.empty()) {
        label = "node";
    }

    if (label.size() >= GGML_BACKEND_OP_LOAD_PROFILE_NAME_LEN) {
        label.resize(GGML_BACKEND_OP_LOAD_PROFILE_NAME_LEN - 1);
    }

    return label;
}

static int ggml_op_load_profile_match_name(const char * name) {
    if (name == nullptr || name[0] == '\0') {
        return -1;
    }

    for (size_t i = 0; i < g_op_load_profile.name_patterns.size(); ++i) {
        if (ggml_op_load_profile_wildcard_match(g_op_load_profile.name_patterns[i].c_str(), name)) {
            return (int) i;
        }
    }

    return -1;
}

void ggml_backend_op_load_profile_set_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(g_op_load_profile.mutex);
    g_op_load_profile.enabled = enabled;
    ggml_op_load_profile_clear_accum_locked();
}

bool ggml_backend_op_load_profile_enabled(void) {
    return g_op_load_profile.enabled;
}

void ggml_backend_op_load_profile_set_breakdown_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(g_op_load_profile.mutex);
    g_op_load_profile.breakdown_enabled = enabled;
    ggml_op_load_profile_clear_accum_locked();
}

bool ggml_backend_op_load_profile_breakdown_enabled(void) {
    return g_op_load_profile.breakdown_enabled;
}

void ggml_backend_op_load_profile_set_ops_from_env(const char * ops_csv) {
    std::lock_guard<std::mutex> lock(g_op_load_profile.mutex);

    uint8_t ops_enabled[GGML_OP_COUNT] = {};
    bool all = false;

    if (ops_csv != nullptr && ops_csv[0] != '\0') {
        const char * token_begin = ops_csv;
        const char * p = ops_csv;
        while (true) {
            if (*p == ',' || *p == '\0') {
                std::string token = ggml_op_load_profile_trim_token(token_begin, p);
                if (!token.empty()) {
                    if (token == "ALL" || token == "all") {
                        all = true;
                    } else {
                        enum ggml_op op;
                        if (ggml_op_load_profile_op_from_name(token, op)) {
                            ops_enabled[op] = 1;
                        } else {
                            GGML_LOG_WARN("%s: unknown op '%s' in CSV_OP_LOAD_OPS\n", __func__, token.c_str());
                        }
                    }
                }

                if (*p == '\0') {
                    break;
                }
                token_begin = p + 1;
            }
            ++p;
        }
    }

    if (all) {
        for (int op = 0; op < GGML_OP_COUNT; ++op) {
            ops_enabled[op] = 1;
        }
    }

    memcpy(g_op_load_profile.out.ops_enabled, ops_enabled, sizeof(ops_enabled));
    ggml_op_load_profile_clear_accum_locked();
}

void ggml_backend_op_load_profile_set_patterns_from_env(const char * patterns_csv) {
    std::lock_guard<std::mutex> lock(g_op_load_profile.mutex);

    g_op_load_profile.name_patterns.clear();
    g_op_load_profile.name_labels.clear();

    if (patterns_csv != nullptr && patterns_csv[0] != '\0') {
        const char * token_begin = patterns_csv;
        const char * p = patterns_csv;
        while (true) {
            if (*p == ',' || *p == '\0') {
                std::string token = ggml_op_load_profile_trim_token(token_begin, p);
                if (!token.empty()) {
                    if (g_op_load_profile.name_patterns.size() >= GGML_BACKEND_OP_LOAD_PROFILE_MAX_GROUPS) {
                        GGML_LOG_WARN("%s: ignoring pattern '%s'; max groups is %d\n",
                                __func__, token.c_str(), GGML_BACKEND_OP_LOAD_PROFILE_MAX_GROUPS);
                    } else {
                        std::string label = ggml_op_load_profile_label_from_pattern(token);
                        std::string pattern = token;
                        const size_t eq = token.find('=');
                        if (eq != std::string::npos) {
                            pattern = ggml_op_load_profile_trim_token(token.c_str() + eq + 1, token.c_str() + token.size());
                        }

                        if (!pattern.empty()) {
                            g_op_load_profile.name_patterns.push_back(pattern);
                            g_op_load_profile.name_labels.push_back(label);
                        }
                    }
                }

                if (*p == '\0') {
                    break;
                }
                token_begin = p + 1;
            }
            ++p;
        }
    }

    ggml_op_load_profile_clear_accum_locked();
}

bool ggml_backend_op_load_profile_is_op_enabled(enum ggml_op op) {
    if (!g_op_load_profile.enabled || op < 0 || op >= GGML_OP_COUNT) {
        return false;
    }

    return g_op_load_profile.out.ops_enabled[op] != 0;
}

bool ggml_backend_op_load_profile_should_measure(enum ggml_op op, const char * name) {
    if (!g_op_load_profile.enabled) {
        return false;
    }
    if (op >= 0 && op < GGML_OP_COUNT && g_op_load_profile.out.ops_enabled[op]) {
        return true;
    }

    return ggml_op_load_profile_match_name(name) >= 0;
}

void ggml_backend_op_load_profile_reset(void) {
    std::lock_guard<std::mutex> lock(g_op_load_profile.mutex);
    ggml_op_load_profile_clear_accum_locked();
}

void ggml_backend_op_load_profile_set_phase(enum ggml_backend_sched_profile_phase phase) {
    if (!g_op_load_profile.enabled && !g_op_load_profile.breakdown_enabled) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_op_load_profile.mutex);
    g_op_load_profile.phase = phase;
}

void ggml_backend_op_load_profile_add_ms(enum ggml_backend_op_load_profile_backend backend, enum ggml_op op, double ms) {
    if (!g_op_load_profile.enabled || op < 0 || op >= GGML_OP_COUNT ||
            backend < 0 || backend >= GGML_BACKEND_OP_LOAD_PROFILE_COUNT ||
            !g_op_load_profile.out.ops_enabled[op]) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_op_load_profile.mutex);

    double * ms_by_type = nullptr;
    uint64_t * count_by_type = nullptr;

    if (g_op_load_profile.phase == GGML_BACKEND_SCHED_PROFILE_PREFILL) {
        switch (backend) {
            case GGML_BACKEND_OP_LOAD_PROFILE_CPU:
                ms_by_type    = g_op_load_profile.out.prefill_cpu_ms_by_type;
                count_by_type = g_op_load_profile.out.prefill_cpu_count_by_type;
                break;
            case GGML_BACKEND_OP_LOAD_PROFILE_HTP:
                ms_by_type    = g_op_load_profile.out.prefill_htp_ms_by_type;
                count_by_type = g_op_load_profile.out.prefill_htp_count_by_type;
                break;
            case GGML_BACKEND_OP_LOAD_PROFILE_GPU:
                ms_by_type    = g_op_load_profile.out.prefill_gpu_ms_by_type;
                count_by_type = g_op_load_profile.out.prefill_gpu_count_by_type;
                break;
            case GGML_BACKEND_OP_LOAD_PROFILE_COUNT:
                break;
        }
    } else {
        switch (backend) {
            case GGML_BACKEND_OP_LOAD_PROFILE_CPU:
                ms_by_type    = g_op_load_profile.out.decode_cpu_ms_by_type;
                count_by_type = g_op_load_profile.out.decode_cpu_count_by_type;
                break;
            case GGML_BACKEND_OP_LOAD_PROFILE_HTP:
                ms_by_type    = g_op_load_profile.out.decode_htp_ms_by_type;
                count_by_type = g_op_load_profile.out.decode_htp_count_by_type;
                break;
            case GGML_BACKEND_OP_LOAD_PROFILE_GPU:
                ms_by_type    = g_op_load_profile.out.decode_gpu_ms_by_type;
                count_by_type = g_op_load_profile.out.decode_gpu_count_by_type;
                break;
            case GGML_BACKEND_OP_LOAD_PROFILE_COUNT:
                break;
        }
    }

    if (ms_by_type != nullptr && count_by_type != nullptr) {
        ms_by_type[op] += ms;
        count_by_type[op]++;
    }
}

void ggml_backend_op_load_profile_add_us(enum ggml_backend_op_load_profile_backend backend, enum ggml_op op, double usec) {
    ggml_backend_op_load_profile_add_ms(backend, op, usec / 1000.0);
}

void ggml_backend_op_load_profile_add_tensor_ms(
        enum ggml_backend_op_load_profile_backend backend,
        enum ggml_op op,
        const char * name,
        double ms) {
    ggml_backend_op_load_profile_add_ms(backend, op, ms);

    if (!g_op_load_profile.enabled ||
            backend < 0 || backend >= GGML_BACKEND_OP_LOAD_PROFILE_COUNT) {
        return;
    }

    const int group = ggml_op_load_profile_match_name(name);
    if (group < 0 || group >= (int) GGML_BACKEND_OP_LOAD_PROFILE_MAX_GROUPS) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_op_load_profile.mutex);

    double * ms_by_name = nullptr;
    uint64_t * count_by_name = nullptr;

    if (g_op_load_profile.phase == GGML_BACKEND_SCHED_PROFILE_PREFILL) {
        switch (backend) {
            case GGML_BACKEND_OP_LOAD_PROFILE_CPU:
                ms_by_name    = g_op_load_profile.out.prefill_cpu_ms_by_name;
                count_by_name = g_op_load_profile.out.prefill_cpu_count_by_name;
                break;
            case GGML_BACKEND_OP_LOAD_PROFILE_HTP:
                ms_by_name    = g_op_load_profile.out.prefill_htp_ms_by_name;
                count_by_name = g_op_load_profile.out.prefill_htp_count_by_name;
                break;
            case GGML_BACKEND_OP_LOAD_PROFILE_GPU:
                ms_by_name    = g_op_load_profile.out.prefill_gpu_ms_by_name;
                count_by_name = g_op_load_profile.out.prefill_gpu_count_by_name;
                break;
            case GGML_BACKEND_OP_LOAD_PROFILE_COUNT:
                break;
        }
    } else {
        switch (backend) {
            case GGML_BACKEND_OP_LOAD_PROFILE_CPU:
                ms_by_name    = g_op_load_profile.out.decode_cpu_ms_by_name;
                count_by_name = g_op_load_profile.out.decode_cpu_count_by_name;
                break;
            case GGML_BACKEND_OP_LOAD_PROFILE_HTP:
                ms_by_name    = g_op_load_profile.out.decode_htp_ms_by_name;
                count_by_name = g_op_load_profile.out.decode_htp_count_by_name;
                break;
            case GGML_BACKEND_OP_LOAD_PROFILE_GPU:
                ms_by_name    = g_op_load_profile.out.decode_gpu_ms_by_name;
                count_by_name = g_op_load_profile.out.decode_gpu_count_by_name;
                break;
            case GGML_BACKEND_OP_LOAD_PROFILE_COUNT:
                break;
        }
    }

    if (ms_by_name != nullptr && count_by_name != nullptr) {
        ms_by_name[group] += ms;
        count_by_name[group]++;
    }
}

void ggml_backend_op_load_profile_add_tensor_us(
        enum ggml_backend_op_load_profile_backend backend,
        enum ggml_op op,
        const char * name,
        double usec) {
    ggml_backend_op_load_profile_add_tensor_ms(backend, op, name, usec / 1000.0);
}

void ggml_backend_op_load_profile_add_backend_timer_ms(
        enum ggml_backend_op_load_profile_backend backend,
        enum ggml_backend_op_load_profile_timer timer,
        double ms) {
    if (!g_op_load_profile.breakdown_enabled ||
            backend < 0 || backend >= GGML_BACKEND_OP_LOAD_PROFILE_COUNT ||
            timer < 0 || timer >= GGML_BACKEND_OP_LOAD_PROFILE_TIMER_COUNT) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_op_load_profile.mutex);

    double * timer_ms = nullptr;
    if (g_op_load_profile.phase == GGML_BACKEND_SCHED_PROFILE_PREFILL) {
        switch (backend) {
            case GGML_BACKEND_OP_LOAD_PROFILE_HTP:
                timer_ms = g_op_load_profile.out.prefill_htp_timer_ms;
                break;
            case GGML_BACKEND_OP_LOAD_PROFILE_GPU:
                timer_ms = g_op_load_profile.out.prefill_gpu_timer_ms;
                break;
            case GGML_BACKEND_OP_LOAD_PROFILE_CPU:
            case GGML_BACKEND_OP_LOAD_PROFILE_COUNT:
                break;
        }
    } else {
        switch (backend) {
            case GGML_BACKEND_OP_LOAD_PROFILE_HTP:
                timer_ms = g_op_load_profile.out.decode_htp_timer_ms;
                break;
            case GGML_BACKEND_OP_LOAD_PROFILE_GPU:
                timer_ms = g_op_load_profile.out.decode_gpu_timer_ms;
                break;
            case GGML_BACKEND_OP_LOAD_PROFILE_CPU:
            case GGML_BACKEND_OP_LOAD_PROFILE_COUNT:
                break;
        }
    }

    if (timer_ms != nullptr) {
        timer_ms[timer] += ms;
    }
}

void ggml_backend_op_load_profile_add_backend_timer_us(
        enum ggml_backend_op_load_profile_backend backend,
        enum ggml_backend_op_load_profile_timer timer,
        double usec) {
    ggml_backend_op_load_profile_add_backend_timer_ms(backend, timer, usec / 1000.0);
}

const char * ggml_backend_op_load_profile_timer_name(enum ggml_backend_op_load_profile_timer timer) {
    switch (timer) {
        case GGML_BACKEND_OP_LOAD_PROFILE_TIMER_GRAPH:         return "graph";
        case GGML_BACKEND_OP_LOAD_PROFILE_TIMER_SYNC:          return "sync";
        case GGML_BACKEND_OP_LOAD_PROFILE_TIMER_DISPATCH:      return "dispatch";
        case GGML_BACKEND_OP_LOAD_PROFILE_TIMER_FUSED:         return "fused";
        case GGML_BACKEND_OP_LOAD_PROFILE_TIMER_ENQUEUE:       return "enqueue";
        case GGML_BACKEND_OP_LOAD_PROFILE_TIMER_PROFILE_WAIT:  return "profile_wait";
        case GGML_BACKEND_OP_LOAD_PROFILE_TIMER_PROFILE_QUERY: return "profile_query";
        case GGML_BACKEND_OP_LOAD_PROFILE_TIMER_BATCH_PUSH:    return "batch_push";
        case GGML_BACKEND_OP_LOAD_PROFILE_TIMER_FLUSH_BATCH:   return "flush_batch";
        case GGML_BACKEND_OP_LOAD_PROFILE_TIMER_DSPQUEUE_WRITE:return "dspqueue_write";
        case GGML_BACKEND_OP_LOAD_PROFILE_TIMER_FLUSH_PENDING: return "flush_pending";
        case GGML_BACKEND_OP_LOAD_PROFILE_TIMER_DSPQUEUE_READ: return "dspqueue_read";
        case GGML_BACKEND_OP_LOAD_PROFILE_TIMER_RESPONSE_POP:  return "response_pop";
        case GGML_BACKEND_OP_LOAD_PROFILE_TIMER_ADD_OP:        return "add_op";
        case GGML_BACKEND_OP_LOAD_PROFILE_TIMER_ADD_TENSOR:    return "add_tensor";
        case GGML_BACKEND_OP_LOAD_PROFILE_TIMER_COUNT:         break;
    }
    return "unknown";
}

struct ggml_backend_op_load_profile_data ggml_backend_op_load_profile_get(void) {
    std::lock_guard<std::mutex> lock(g_op_load_profile.mutex);
    return g_op_load_profile.out;
}

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = tensor->src[i];
        if (src == NULL) {
            continue;
        }
        // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
        // not an ideal solution
        if (tensor->op != GGML_OP_ROPE && src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
            // check if a backend with higher prio wants to offload the op
            if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                for (int b = 0; b < src_backend_id; b++) {
                    if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                        SET_CAUSE(tensor, "1.off");
                        return b;
                    }
                }
            }
            SET_CAUSE(tensor, "1.wgt%d", i);
            return src_backend_id;
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_name(node->op), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
static bool ggml_backend_sched_parse_ffn_branch_name(
        const char * name,
        std::string * branch,
        int * layer) {
    static const char * prefixes[] = {
        "ffn_up.",
        "ffn_gate.",
        "ffn_swiglu.",
        "ffn_down.",
    };

    if (name == nullptr) {
        return false;
    }

    for (const char * prefix : prefixes) {
        const size_t prefix_len = strlen(prefix);
        if (strncmp(name, prefix, prefix_len) != 0) {
            continue;
        }

        const char * branch_start = name + prefix_len;
        const char * layer_start = strrchr(branch_start, '-');
        if (layer_start == nullptr || layer_start == branch_start || layer_start[1] == '\0') {
            return false;
        }

        char * end = nullptr;
        const long parsed = strtol(layer_start + 1, &end, 10);
        if (end == layer_start + 1 || *end != '\0') {
            return false;
        }

        if (branch != nullptr) {
            branch->assign(branch_start, layer_start - branch_start);
        }
        if (layer != nullptr) {
            *layer = (int) parsed;
        }
        return true;
    }

    return false;
}

static bool ggml_backend_sched_parse_ffn_parallel_branch_node(
        const struct ggml_tensor * node, std::string * branch, int * layer) {
    if (node == nullptr) {
        return false;
    }

    if (ggml_backend_sched_parse_ffn_branch_name(node->name, branch, layer)) {
        return true;
    }

    // In a single-shard FFN, the down projection can become the final ffn_out-<layer>
    // tensor after the graph builder names the returned FFN result. Recover the branch
    // identity from its swiglu input so debug-only split isolation can still see it.
    if ((node->op == GGML_OP_MUL_MAT ||
         node->op == GGML_OP_MUL_MAT_SRC0_REGION) &&
            node->name != nullptr && strncmp(node->name, "ffn_out-", 8) == 0) {
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            if (node->src[i] != nullptr &&
                    ggml_backend_sched_parse_ffn_branch_name(node->src[i]->name, branch, layer)) {
                return true;
            }
        }
    }

    return false;
}

static bool ggml_backend_sched_is_ffn_parallel_branch_node(const struct ggml_tensor * node) {
    return ggml_backend_sched_parse_ffn_parallel_branch_node(node, nullptr, nullptr);
}

struct ggml_backend_sched_ffn_reduce_info {
    int layer = -1;
    std::vector<std::string> branches;
};

static bool ggml_backend_sched_collect_ffn_parallel_reduce_info(
        const struct ggml_tensor * node, ggml_backend_sched_ffn_reduce_info * info) {
    std::string branch;
    int layer = -1;
    if (ggml_backend_sched_parse_ffn_parallel_branch_node(node, &branch, &layer)) {
        info->layer = layer;
        info->branches = { std::move(branch) };
        return true;
    }
    if (node == nullptr || node->op != GGML_OP_ADD || node->src[0] == nullptr || node->src[1] == nullptr) {
        return false;
    }

    ggml_backend_sched_ffn_reduce_info lhs;
    ggml_backend_sched_ffn_reduce_info rhs;
    if (!ggml_backend_sched_collect_ffn_parallel_reduce_info(node->src[0], &lhs) ||
            !ggml_backend_sched_collect_ffn_parallel_reduce_info(node->src[1], &rhs) ||
            lhs.layer != rhs.layer) {
        return false;
    }

    info->layer = lhs.layer;
    info->branches = std::move(lhs.branches);
    for (std::string & rhs_branch : rhs.branches) {
        if (std::find(info->branches.begin(), info->branches.end(), rhs_branch) != info->branches.end()) {
            return false;
        }
        info->branches.push_back(std::move(rhs_branch));
    }
    return true;
}

static bool ggml_backend_sched_is_ffn_parallel_reduce_node(const struct ggml_tensor * node) {
    ggml_backend_sched_ffn_reduce_info info;
    return node != nullptr && node->op == GGML_OP_ADD &&
        ggml_backend_sched_collect_ffn_parallel_reduce_info(node, &info) && info.branches.size() >= 2;
}

void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        split->is_ffn_parallel_reduce = false;
        int cur_backend_id = split->backend_id;
        bool cur_split_has_ffn_parallel_branch = false;
        bool cur_split_needs_ffn_branch_isolation = false;
        bool cur_split_has_ffn_parallel_reduce = false;
        std::vector<std::pair<int, std::string>> ffn_parallel_branch_keys;
        std::vector<bool> ffn_parallel_reduce_nodes(graph->n_nodes, false);
        for (int j = 0; j < graph->n_nodes; ++j) {
            const struct ggml_tensor * node = graph->nodes[j];
            ffn_parallel_reduce_nodes[j] = ggml_backend_sched_is_ffn_parallel_reduce_node(node);
            std::string branch;
            int layer = -1;
            if (!ggml_backend_sched_parse_ffn_branch_name(node->name, &branch, &layer)) {
                continue;
            }

            const auto duplicate = std::find_if(ffn_parallel_branch_keys.begin(), ffn_parallel_branch_keys.end(),
                    [&](const std::pair<int, std::string> & prev) {
                        return prev.first == layer && prev.second == branch;
                    });
            if (duplicate == ffn_parallel_branch_keys.end()) {
                ffn_parallel_branch_keys.emplace_back(layer, std::move(branch));
            }
        }
        auto ffn_layer_has_multiple_branches = [&](const struct ggml_tensor * node) {
            std::string branch;
            int layer = -1;
            if (!ggml_backend_sched_parse_ffn_branch_name(node->name, &branch, &layer)) {
                return false;
            }

            int n_layer_branches = 0;
            for (const auto & key : ffn_parallel_branch_keys) {
                if (key.first == layer && ++n_layer_branches >= 2) {
                    return true;
                }
            }
            return false;
        };
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);
            const bool node_is_ffn_parallel_branch = ggml_backend_sched_is_ffn_parallel_branch_node(node);
            const bool node_is_ffn_parallel_reduce = ffn_parallel_reduce_nodes[i];
            const bool node_needs_ffn_branch_isolation =
                node_is_ffn_parallel_branch &&
                (sched->debug >= 2 || ffn_layer_has_multiple_branches(node));

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id &&
                    cur_split_has_ffn_parallel_branch != node_is_ffn_parallel_branch &&
                    (cur_split_needs_ffn_branch_isolation ||
                     (node_needs_ffn_branch_isolation && split->i_start < i))) {
                // Keep FFN shard branch splits pure. If a branch is fused with
                // the surrounding layer prefix/suffix on the same backend, the
                // scheduler cannot collect consecutive branch splits into one
                // parallel FFN group in normal (non-debug) multi-shard runs.
                need_new_split = true;
            }
            if (node_backend_id == cur_backend_id &&
                    cur_split_has_ffn_parallel_branch &&
                    node_is_ffn_parallel_reduce) {
                // Keep each FFN shard branch split pure. When the reduce backend
                // equals the backend of the last branch, the default scheduler
                // would fuse that branch and the reduce ADDs into one split. That
                // prevents the branch from joining the parallel FFN group.
                need_new_split = true;
            }
            if (node_backend_id == cur_backend_id &&
                    sched->ffn_parallel_reduce_threads > 0 &&
                    cur_split_has_ffn_parallel_reduce &&
                    !node_is_ffn_parallel_reduce &&
                    ggml_backend_dev_type(ggml_backend_get_device(sched->backends[cur_backend_id])) ==
                        GGML_BACKEND_DEVICE_TYPE_CPU) {
                // Keep the reduce split pure so its CPU thread override does not
                // leak into the following residual/norm nodes on the same backend.
                need_new_split = true;
            }
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                    // check if the split has too many inputs
                    // FIXME: count the number of inputs instead of only checking when full
                    if (split->n_inputs == GGML_SCHED_MAX_SPLIT_INPUTS) {
                        const size_t id = hash_id(src);
                        int src_backend_id = sched->hv_tensor_backend_ids[id];
                        bool supported = ggml_backend_sched_buffer_supported(sched, src, cur_backend_id);
                        if (src_backend_id != cur_backend_id && tensor_id_copy(id, cur_backend_id, 0) == NULL && !supported) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                split->is_ffn_parallel_reduce = false;
                cur_backend_id = node_backend_id;
                cur_split_has_ffn_parallel_branch = false;
                cur_split_needs_ffn_branch_isolation = false;
                cur_split_has_ffn_parallel_reduce = false;
            }

            if (node_is_ffn_parallel_branch) {
                cur_split_has_ffn_parallel_branch = true;
                cur_split_needs_ffn_branch_isolation =
                    cur_split_needs_ffn_branch_isolation || node_needs_ffn_branch_isolation;
            }
            if (node_is_ffn_parallel_reduce) {
                cur_split_has_ffn_parallel_reduce = true;
                split->is_ffn_parallel_reduce = true;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src->flags & GGML_TENSOR_FLAG_INPUT && sched->n_copies > 1) {
                    if (tensor_id_copy(src_id, src_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[src_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy;
                            if (c == sched->cur_copy) {
                                tensor_copy = src; // use the original tensor as the current copy
                            } else {
                                tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                                ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            }
                            ggml_set_input(tensor_copy);
                            ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            tensor_id_copy(src_id, src_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_graph_inputs = sched->n_graph_inputs++;
                        GGML_ASSERT(n_graph_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        sched->graph_inputs[n_graph_inputs] = src;
                    }
                }

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + sched->n_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sched->n_copies;

    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
    sched->debug_prev_graph_size = sched->debug_graph_size;
    sched->debug_graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        // Optimize this split of the graph. This needs to happen before we make graph_copy,
        // so they are in sync.
        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph);

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
    }

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids);
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

struct ggml_backend_sched_ffn_branch_info {
    std::string branch;
    int layer = -1;
};

static bool ggml_backend_sched_parse_ffn_branch_node(
        const struct ggml_tensor * node,
        ggml_backend_sched_ffn_branch_info * info) {
    return ggml_backend_sched_parse_ffn_parallel_branch_node(node, &info->branch, &info->layer);
}

static bool ggml_backend_sched_split_ffn_branch_info(
        const struct ggml_backend_sched_split * split,
        ggml_backend_sched_ffn_branch_info * info) {
    bool found = false;
    ggml_backend_sched_ffn_branch_info split_info;

    for (int i = 0; i < split->graph.n_nodes; ++i) {
        ggml_backend_sched_ffn_branch_info node_info;
        if (!ggml_backend_sched_parse_ffn_branch_node(split->graph.nodes[i], &node_info)) {
            return false;
        }

        if (found && (split_info.layer != node_info.layer || split_info.branch != node_info.branch)) {
            return false;
        }

        found = true;
        split_info = std::move(node_info);
    }

    if (!found) {
        return false;
    }

    *info = std::move(split_info);
    return true;
}

static int ggml_backend_sched_collect_parallel_ffn_group(
        const struct ggml_backend_sched_split * splits,
        int n_splits,
        int split_id,
        bool allow_single,
        std::vector<ggml_backend_sched_ffn_branch_info> * infos) {
    GGML_ASSERT(infos != nullptr);
    infos->clear();

    ggml_backend_sched_ffn_branch_info first;
    if (!ggml_backend_sched_split_ffn_branch_info(&splits[split_id], &first)) {
        return 0;
    }

    infos->push_back(first);

    for (int i = split_id + 1; i < n_splits; ++i) {
        ggml_backend_sched_ffn_branch_info cur;
        if (!ggml_backend_sched_split_ffn_branch_info(&splits[i], &cur)) {
            break;
        }
        if (cur.layer != first.layer) {
            break;
        }
        const auto duplicate = std::find_if(infos->begin(), infos->end(), [&](const ggml_backend_sched_ffn_branch_info & prev) {
            return prev.branch == cur.branch;
        });
        if (duplicate != infos->end()) {
            break;
        }
        infos->push_back(std::move(cur));
    }

    return infos->size() >= 2 || allow_single ? (int) infos->size() : 0;
}

struct ggml_backend_sched_trace_io_times {
    int64_t copy_us = 0;
    int64_t wait_us = 0;
};

static int ggml_backend_sched_trace_graph_layer(const ggml_cgraph & graph) {
    for (int i = 0; i < graph.n_nodes; ++i) {
        const ggml_tensor * node = graph.nodes[i];
        const int layer = ggml_sched_profile_extract_layer_id(node ? node->name : nullptr);
        if (layer >= 0) {
            return layer;
        }
    }
    return -1;
}

static void ggml_sched_trace_copy_string(char * dst, size_t dst_size, const char * src) {
    GGML_ASSERT(dst != nullptr);
    GGML_ASSERT(dst_size > 0);
    snprintf(dst, dst_size, "%s", src ? src : "");
}

static void ggml_backend_sched_trace_append_split(
        ggml_backend_sched_t sched,
        int64_t graph_id,
        int split_id,
        const struct ggml_backend_sched_split * split,
        const ggml_backend_sched_trace_io_times & io_times,
        int64_t compute_submit_us,
        int64_t compute_wall_us,
        bool is_ffn_group,
        const char * ffn_branch,
        int group_id,
        int64_t group_wall_us,
        int64_t group_copy_us,
        const char * timing_mode) {
    GGML_ASSERT(sched != nullptr);
    GGML_ASSERT(split != nullptr);

    const ggml_cgraph & graph = split->graph;
    const ggml_tensor * first = graph.n_nodes > 0 ? graph.nodes[0] : nullptr;
    const ggml_tensor * last  = graph.n_nodes > 0 ? graph.nodes[graph.n_nodes - 1] : nullptr;

    g_sched_trace.rows.emplace_back();
    ggml_backend_sched_trace_row & row = g_sched_trace.rows.back();
    row.query_id = g_sched_trace.query_id;
    row.phase = g_sched_profile_phase;
    row.graph_id = graph_id;
    row.token_index = g_sched_trace.token_index;
    row.n_past = g_sched_trace.n_past;
    row.n_tokens = g_sched_trace.n_tokens;
    row.split_id = split_id;
    row.layer = ggml_backend_sched_trace_graph_layer(graph);
    row.node_start = split->i_start;
    row.node_end = split->i_end > split->i_start ? split->i_end - 1 : split->i_start;
    row.node_count = graph.n_nodes;
    row.op_first = first ? first->op : GGML_OP_NONE;
    row.op_last = last ? last->op : GGML_OP_NONE;
    row.copy_us = io_times.copy_us;
    row.wait_us = io_times.wait_us;
    row.compute_submit_us = compute_submit_us;
    row.compute_wall_us = compute_wall_us;
    row.is_ffn_group = is_ffn_group;
    row.group_id = group_id;
    row.group_wall_us = group_wall_us;
    row.group_copy_us = group_copy_us;

    ggml_sched_trace_copy_string(row.backend, sizeof(row.backend), ggml_backend_name(sched->backends[split->backend_id]));
    ggml_sched_trace_copy_string(row.first_node, sizeof(row.first_node), first ? first->name : "");
    ggml_sched_trace_copy_string(row.last_node, sizeof(row.last_node), last ? last->name : "");
    ggml_sched_trace_copy_string(row.ffn_branch, sizeof(row.ffn_branch), ffn_branch);
    ggml_sched_trace_copy_string(row.timing_mode, sizeof(row.timing_mode), timing_mode);
}

static int64_t ggml_ffn_worker_trace_delta(int64_t end_us, int64_t begin_us) {
    return end_us >= 0 && begin_us >= 0 ? end_us - begin_us : -1;
}

static void ggml_backend_sched_ffn_worker_trace_append_group(
        ggml_backend_sched_t sched,
        int64_t graph_id,
        int split_id,
        int ffn_group_size,
        const std::vector<ggml_backend_sched_ffn_branch_info> & ffn_group_infos,
        const ggml_backend_sched_ffn_worker_timeline * timelines,
        const std::vector<enum ggml_status> & statuses,
        const std::vector<int64_t> & sched_compute_wall_us,
        int64_t group_begin_us,
        int64_t copy_end_us,
        int64_t compute_begin_us,
        int64_t group_end_us) {
    GGML_ASSERT(sched != nullptr);
    GGML_ASSERT(graph_id >= 0);
    GGML_ASSERT(ffn_group_size > 0);
    GGML_ASSERT((int) ffn_group_infos.size() == ffn_group_size);
    GGML_ASSERT(timelines != nullptr);
    GGML_ASSERT((int) statuses.size() == ffn_group_size);
    GGML_ASSERT((int) sched_compute_wall_us.size() == ffn_group_size);

    int64_t first_backend_begin_us = -1;
    int64_t last_backend_begin_us = -1;
    int64_t last_backend_finish_us = -1;
    int64_t last_completion_publish_us = -1;
    int64_t longest_branch_wall_us = -1;

    for (int i = 0; i < ffn_group_size; ++i) {
        const ggml_backend_sched_ffn_worker_timeline & timeline = timelines[i];
        if (timeline.backend_begin_us >= 0) {
            first_backend_begin_us = first_backend_begin_us < 0
                ? timeline.backend_begin_us
                : std::min(first_backend_begin_us, timeline.backend_begin_us);
            last_backend_begin_us = std::max(last_backend_begin_us, timeline.backend_begin_us);
        }
        if (timeline.sync_end_us >= 0) {
            last_backend_finish_us = std::max(last_backend_finish_us, timeline.sync_end_us);
        }
        if (timeline.completion_publish_us >= 0) {
            last_completion_publish_us = std::max(
                    last_completion_publish_us, timeline.completion_publish_us);
        }
        longest_branch_wall_us = std::max(
                longest_branch_wall_us,
                ggml_ffn_worker_trace_delta(timeline.job_end_us, timeline.job_start_us));
    }

    const int64_t group_seq = g_ffn_worker_trace.next_group_seq++;
    const int64_t group_start_skew_us = ggml_ffn_worker_trace_delta(
            last_backend_begin_us, first_backend_begin_us);
    const int64_t group_tail_after_last_backend_us = ggml_ffn_worker_trace_delta(
            group_end_us, last_backend_finish_us);
    const int64_t group_tail_after_last_publish_us = ggml_ffn_worker_trace_delta(
            group_end_us, last_completion_publish_us);

    for (int i = 0; i < ffn_group_size; ++i) {
        const struct ggml_backend_sched_split * split = &sched->splits[split_id + i];
        const ggml_cgraph & graph = split->graph;
        const ggml_tensor * first = graph.n_nodes > 0 ? graph.nodes[0] : nullptr;
        const ggml_tensor * last = graph.n_nodes > 0 ? graph.nodes[graph.n_nodes - 1] : nullptr;

        g_ffn_worker_trace.rows.emplace_back();
        ggml_backend_sched_ffn_worker_trace_row & row = g_ffn_worker_trace.rows.back();
        row.trace_session_id = g_ffn_worker_trace.trace_session_id;
        row.group_seq = group_seq;
        row.job_id = g_ffn_worker_trace.next_job_id++;
        row.query_id = g_ffn_worker_trace.query_id;
        row.phase = g_sched_profile_phase;
        row.graph_id = graph_id;
        row.token_index = g_ffn_worker_trace.token_index;
        row.n_past = g_ffn_worker_trace.n_past;
        row.n_tokens = g_ffn_worker_trace.n_tokens;
        row.group_id = split_id;
        row.split_id = split_id + i;
        row.layer = ffn_group_infos[i].layer;
        row.node_count = graph.n_nodes;
        row.status = statuses[i];
        row.sched_compute_wall_us = sched_compute_wall_us[i] > 0
            ? sched_compute_wall_us[i]
            : -1;
        row.group_begin_us = group_begin_us;
        row.copy_end_us = copy_end_us;
        row.compute_begin_us = compute_begin_us;
        row.group_end_us = group_end_us;
        row.timeline = timelines[i];

        ggml_sched_trace_copy_string(
                row.backend, sizeof(row.backend),
                ggml_backend_name(sched->backends[split->backend_id]));
        ggml_sched_trace_copy_string(
                row.ffn_branch, sizeof(row.ffn_branch), ffn_group_infos[i].branch.c_str());
        ggml_sched_trace_copy_string(
                row.first_node, sizeof(row.first_node), first ? first->name : "");
        ggml_sched_trace_copy_string(
                row.last_node, sizeof(row.last_node), last ? last->name : "");

        const ggml_backend_sched_ffn_worker_timeline & timeline = row.timeline;
        row.group_copy_us = ggml_ffn_worker_trace_delta(copy_end_us, group_begin_us);
        row.post_copy_setup_us = ggml_ffn_worker_trace_delta(compute_begin_us, copy_end_us);
        row.dispatch_us = ggml_ffn_worker_trace_delta(
                timeline.dispatch_end_us, timeline.dispatch_begin_us);
        row.queue_delay_us = ggml_ffn_worker_trace_delta(
                timeline.worker_dequeue_us, timeline.job_enqueued_us);
        row.dispatch_to_start_us = ggml_ffn_worker_trace_delta(
                timeline.job_start_us, timeline.dispatch_begin_us);
        row.worker_setup_us = ggml_ffn_worker_trace_delta(
                timeline.backend_begin_us, timeline.job_start_us);
        row.graph_call_us = ggml_ffn_worker_trace_delta(
                timeline.graph_compute_return_us, timeline.backend_begin_us);
        row.post_graph_cleanup_us = ggml_ffn_worker_trace_delta(
                timeline.sync_begin_us, timeline.graph_compute_return_us);
        row.sync_wait_us = ggml_ffn_worker_trace_delta(
                timeline.sync_end_us, timeline.sync_begin_us);
        row.job_finalize_us = ggml_ffn_worker_trace_delta(
                timeline.job_end_us, timeline.sync_end_us);
        row.branch_wall_us = ggml_ffn_worker_trace_delta(
                timeline.job_end_us, timeline.job_start_us);
        row.publish_delay_us = ggml_ffn_worker_trace_delta(
                timeline.completion_publish_us, timeline.job_end_us);
        row.collect_wait_us = ggml_ffn_worker_trace_delta(
                timeline.collect_end_us, timeline.collect_begin_us);
        row.start_offset_us = ggml_ffn_worker_trace_delta(
                timeline.backend_begin_us, compute_begin_us);
        row.finish_offset_us = ggml_ffn_worker_trace_delta(
                timeline.sync_end_us, compute_begin_us);
        row.group_start_skew_us = group_start_skew_us;
        row.group_tail_after_last_backend_us = group_tail_after_last_backend_us;
        row.group_tail_after_last_publish_us = group_tail_after_last_publish_us;
        row.group_compute_wall_us = ggml_ffn_worker_trace_delta(group_end_us, compute_begin_us);
        row.group_wall_us = ggml_ffn_worker_trace_delta(group_end_us, group_begin_us);
        row.is_longest_duration =
            row.branch_wall_us >= 0 && row.branch_wall_us == longest_branch_wall_us ? 1 : 0;
        row.is_last_backend_finisher =
            timeline.sync_end_us >= 0 && timeline.sync_end_us == last_backend_finish_us ? 1 : 0;
        row.is_last_completion_publisher =
            timeline.completion_publish_us >= 0 &&
            timeline.completion_publish_us == last_completion_publish_us ? 1 : 0;
    }
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;

    const bool trace_enabled = ggml_sched_trace_phase_enabled();
    const int64_t trace_graph_id = trace_enabled ? g_sched_trace.graph_id++ : -1;
    const bool worker_trace_enabled = ggml_ffn_worker_trace_phase_enabled();
    const int64_t worker_trace_graph_id = worker_trace_enabled ? g_ffn_worker_trace.graph_id++ : -1;

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;

    auto prof_add = [&](const ggml_cgraph & graph, ggml_sched_profile_backend_kind split_bucket, int64_t dt_us) {
        if (!ggml_sched_profile_enabled()) {
            return;
        }

        const double dt_ms = (double) dt_us / 1000.0;
        g_sched_profile.out.total_ops += (uint64_t) graph.n_nodes;

        ggml_sched_profile_add_graph_op_type_counts(graph, split_bucket);

        if (g_sched_profile_phase == GGML_BACKEND_SCHED_PROFILE_PREFILL) {
            switch (split_bucket) {
                case GGML_SCHED_PROFILE_BACKEND_CPU:
                    g_sched_profile.out.prefill_cpu_ops += (uint64_t) graph.n_nodes;
                    g_sched_profile.out.prefill_cpu_ms  += dt_ms;
                    break;
                case GGML_SCHED_PROFILE_BACKEND_HTP:
                    g_sched_profile.out.prefill_htp_ops += (uint64_t) graph.n_nodes;
                    g_sched_profile.out.prefill_htp_ms  += dt_ms;
                    break;
                case GGML_SCHED_PROFILE_BACKEND_GPU:
                    g_sched_profile.out.prefill_gpu_ops += (uint64_t) graph.n_nodes;
                    g_sched_profile.out.prefill_gpu_ms  += dt_ms;
                    break;
            }
        } else {
            switch (split_bucket) {
                case GGML_SCHED_PROFILE_BACKEND_CPU:
                    g_sched_profile.out.decode_cpu_ops += (uint64_t) graph.n_nodes;
                    g_sched_profile.out.decode_cpu_ms  += dt_ms;
                    break;
                case GGML_SCHED_PROFILE_BACKEND_HTP:
                    g_sched_profile.out.decode_htp_ops += (uint64_t) graph.n_nodes;
                    g_sched_profile.out.decode_htp_ms  += dt_ms;
                    break;
                case GGML_SCHED_PROFILE_BACKEND_GPU:
                    g_sched_profile.out.decode_gpu_ops += (uint64_t) graph.n_nodes;
                    g_sched_profile.out.decode_gpu_ms  += dt_ms;
                    break;
            }
        }
    };

    auto copy_split_inputs = [&](struct ggml_backend_sched_split * split, ggml_backend_sched_trace_io_times * trace_io_times) {
        auto add_copy_us = [&](int64_t dt_us) {
            if (trace_io_times != nullptr) {
                trace_io_times->copy_us += dt_us;
            }
            ggml_sched_profile_add_copy_us(dt_us);
        };
        auto add_wait_us = [&](int64_t dt_us) {
            if (trace_io_times != nullptr) {
                trace_io_times->wait_us += dt_us;
            }
            ggml_sched_profile_add_wait_us(dt_us);
        };

        const int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    const int64_t t0_us = ggml_sched_profile_time_us();
                    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                    const int64_t t1_us = ggml_sched_profile_time_us();
                    add_wait_us(t1_us - t0_us);
                } else {
                    const int64_t t0_us = ggml_sched_profile_time_us();
                    ggml_backend_synchronize(split_backend);
                    const int64_t t1_us = ggml_sched_profile_time_us();
                    add_wait_us(t1_us - t0_us);
                }
                {
                    const int64_t t0_us = ggml_sched_profile_time_us();
                    ggml_backend_tensor_copy(input, input_cpy);
                    const int64_t t1_us = ggml_sched_profile_time_us();
                    add_copy_us(t1_us - t0_us);
                }
            } else {
                // wait for the split backend to finish using the input before overwriting it
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    const int64_t t0_us = ggml_sched_profile_time_us();
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                    const int64_t t1_us = ggml_sched_profile_time_us();
                    add_wait_us(t1_us - t0_us);
                } else {
                    const int64_t t0_us = ggml_sched_profile_time_us();
                    ggml_backend_synchronize(split_backend);
                    const int64_t t1_us = ggml_sched_profile_time_us();
                    add_wait_us(t1_us - t0_us);
                }

                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                ggml_tensor * node = split->graph.nodes[0];
                if (split->graph.n_nodes > 0 &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) && (
                    (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)
                    //|| (node->src[1] == input_cpy && node->op == GGML_OP_ADD_ID) /* GGML_OP_ADD_ID weights are small and not worth splitting */
                    )) {

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    {
                        const int64_t t0_us = ggml_sched_profile_time_us();
                        ggml_backend_synchronize(input_backend);
                        const int64_t t1_us = ggml_sched_profile_time_us();
                        add_wait_us(t1_us - t0_us);
                    }

                    // get the ids
                    ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;

                    // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
                    // in that case, we use the original ids tensor
                    for (int i = input_id + 1; i < split->n_inputs; i++) {
                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
                            ids_tensor = split->inputs[i];
                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
                            break;
                        }
                    }

                    if (ids_tensor != prev_ids_tensor) {
                        ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                        {
                            const int64_t t0_us = ggml_sched_profile_time_us();
                            ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
                            const int64_t t1_us = ggml_sched_profile_time_us();
                            add_copy_us(t1_us - t0_us);
                        }
                        {
                            const int64_t t0_us = ggml_sched_profile_time_us();
                            ggml_backend_synchronize(ids_backend);
                            const int64_t t1_us = ggml_sched_profile_time_us();
                            add_wait_us(t1_us - t0_us);
                        }

                        // find the used experts
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                                GGML_ASSERT(id >= 0 && id < n_expert);
                                ggml_bitset_set(used_ids.data(), id);
                            }
                        }

                        prev_ids_tensor = ids_tensor;
                    }

                    // group consecutive experts and copy them together
                    auto copy_experts = [&](int32_t first_id, int32_t last_id) {
                        const size_t expert_offset = first_id * expert_size;
                        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                        const size_t padding = std::min<size_t>(expert_size, 512);
                        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

                        const int64_t t0_us = ggml_sched_profile_time_us();
                        ggml_backend_tensor_set_async(split_backend,
                                input_cpy,
                                (const uint8_t *)input->data + expert_offset, expert_offset,
                                // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
                                // this is necessary for MMQ in the CUDA backend
                                expert_size_copy + padding_end);
                        const int64_t t1_us = ggml_sched_profile_time_us();
                        add_copy_us(t1_us - t0_us);
                    };

                    int id = 0;
                    while (!ggml_bitset_get(used_ids.data(), id)) {
                        id++;
                    }
                    int32_t first_id = id;
                    int32_t last_id = first_id;

                    for (++id; id < n_expert; ++id) {
                        if (!ggml_bitset_get(used_ids.data(), id)) {
                            continue;
                        }

                        if (id == last_id + 1) {
                            last_id = id;
                            continue;
                        }

                        copy_experts(first_id, last_id);

                        first_id = id;
                        last_id = id;
                    }
                    copy_experts(first_id, last_id);
                } else {
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    bool async_ok = false;
                    if (split_backend->iface.cpy_tensor_async) {
                        const int64_t t0_us = ggml_sched_profile_time_us();
                        async_ok = split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy);
                        const int64_t t1_us = ggml_sched_profile_time_us();
                        add_copy_us(t1_us - t0_us);
                    }

                    if (!async_ok) {
                        {
                            const int64_t t0_us = ggml_sched_profile_time_us();
                            ggml_backend_synchronize(input_backend);
                            const int64_t t1_us = ggml_sched_profile_time_us();
                            add_wait_us(t1_us - t0_us);
                        }
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            const int64_t t0_us = ggml_sched_profile_time_us();
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                            const int64_t t1_us = ggml_sched_profile_time_us();
                            add_wait_us(t1_us - t0_us);
                        } else {
                            const int64_t t0_us = ggml_sched_profile_time_us();
                            ggml_backend_synchronize(split_backend);
                            const int64_t t1_us = ggml_sched_profile_time_us();
                            add_wait_us(t1_us - t0_us);
                        }
                        {
                            const int64_t t0_us = ggml_sched_profile_time_us();
                            ggml_backend_tensor_copy(input, input_cpy);
                            const int64_t t1_us = ggml_sched_profile_time_us();
                            add_copy_us(t1_us - t0_us);
                        }
                    }
                }
            }
        }

    };

    auto ffn_reduce_set_n_threads = [&](struct ggml_backend_sched_split * split) -> ggml_backend_set_n_threads_t {
        if (sched->ffn_parallel_reduce_threads <= 0 ||
                sched->cpu_threads <= 0 ||
                sched->ffn_parallel_reduce_threads >= sched->cpu_threads ||
                !split->is_ffn_parallel_reduce) {
            return nullptr;
        }

        ggml_backend_t backend = sched->backends[split->backend_id];
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev == nullptr || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            return nullptr;
        }

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        return reg != nullptr
            ? (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads")
            : nullptr;
    };

    struct ffn_reduce_thread_guard {
        ggml_backend_t backend;
        ggml_backend_set_n_threads_t set_n_threads;
        int restore_threads;

        ffn_reduce_thread_guard(
                ggml_backend_t backend,
                ggml_backend_set_n_threads_t set_n_threads,
                int reduce_threads,
                int restore_threads) :
            backend(backend), set_n_threads(set_n_threads), restore_threads(restore_threads) {
            if (set_n_threads != nullptr) {
                set_n_threads(backend, reduce_threads);
            }
        }

        ~ffn_reduce_thread_guard() {
            if (set_n_threads != nullptr) {
                ggml_backend_synchronize(backend);
                set_n_threads(backend, restore_threads);
            }
        }
    };

    auto compute_split_no_callback = [&](
            struct ggml_backend_sched_split * split,
            int64_t * dt_us,
            bool sync_after,
            ggml_backend_sched_ffn_worker_timeline * timeline) {
        ggml_backend_t split_backend = sched->backends[split->backend_id];
        ggml_backend_set_n_threads_t set_n_threads = ffn_reduce_set_n_threads(split);
        const int64_t t0_us = ggml_time_us();
        if (timeline != nullptr) {
            timeline->backend_begin_us = t0_us;
        }
        enum ggml_status ec;
        {
            ffn_reduce_thread_guard guard(
                    split_backend,
                    set_n_threads,
                    sched->ffn_parallel_reduce_threads,
                    sched->cpu_threads);
            ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (timeline != nullptr) {
                timeline->graph_compute_return_us = ggml_time_us();
            }
        }
        if (ec == GGML_STATUS_SUCCESS && sync_after) {
            // FFN parallel groups need backend completion inside the parallel section.
            // Otherwise async backends can defer the real wait to the following split.
            if (timeline != nullptr) {
                timeline->sync_begin_us = ggml_time_us();
            }
            ggml_backend_synchronize(split_backend);
            if (timeline != nullptr) {
                timeline->sync_end_us = ggml_time_us();
            }
        }
        const int64_t t1_us = ggml_time_us();
        *dt_us = t1_us - t0_us;
        return ec;
    };

    auto compute_split_with_callback = [&](struct ggml_backend_sched_split * split) {
        ggml_backend_t split_backend = sched->backends[split->backend_id];
        const ggml_sched_profile_backend_kind split_bucket = ggml_sched_profile_backend_bucket(split_backend);
        ggml_backend_set_n_threads_t set_n_threads = ffn_reduce_set_n_threads(split);
        ffn_reduce_thread_guard guard(
                split_backend,
                set_n_threads,
                sched->ffn_parallel_reduce_threads,
                sched->cpu_threads);

        // similar to ggml_backend_compare_graph_backend
        for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
            struct ggml_tensor * t = split->graph.nodes[j0];

            // check if the user needs data from this node
            bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

            int j1 = j0;

            // determine the range [j0, j1] of nodes that can be computed together
            while (!need && j1 < split->graph.n_nodes - 1) {
                t = split->graph.nodes[++j1];
                need = sched->callback_eval(t, true, sched->callback_eval_user_data);
            }

            struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

            const bool profile_enabled = ggml_sched_profile_enabled();
            const int64_t t0_us = profile_enabled ? ggml_time_us() : 0;
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }

            // TODO: pass backend to the callback, then the user can decide if they want to synchronize
            ggml_backend_synchronize(split_backend);
            const int64_t t1_us = profile_enabled ? ggml_time_us() : 0;

            prof_add(gv, split_bucket, t1_us - t0_us);

            if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                break;
            }

            j0 = j1;
        }

        return GGML_STATUS_SUCCESS;
    };

    auto record_split_event = [&](struct ggml_backend_sched_split * split) {
        const int split_backend_id = split->backend_id;
        if (split->n_inputs > 0 && sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], sched->backends[split_backend_id]);
        }
    };

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        ggml_backend_t split_backend = sched->backends[split->backend_id];
        const ggml_sched_profile_backend_kind split_bucket = ggml_sched_profile_backend_bucket(split_backend);

        ggml_sched_profile_note_layers(split->graph, split_bucket);

        std::vector<ggml_backend_sched_ffn_branch_info> ffn_group_infos;
        const bool allow_single_ffn_group = sched->debug >= 2;
        const int ffn_group_size = !sched->callback_eval
            ? ggml_backend_sched_collect_parallel_ffn_group(
                    splits, sched->n_splits, split_id, allow_single_ffn_group, &ffn_group_infos)
            : 0;
        if (ffn_group_size > 0) {
            ggml_backend_sched_ffn_worker_timeline * group_timelines = nullptr;
            if (worker_trace_enabled) {
                group_timelines = g_ffn_worker_trace.timelines.get();
                for (int i = 0; i < ffn_group_size; ++i) {
                    group_timelines[i] = {};
                }
            }

            std::vector<ggml_sched_profile_backend_kind> group_buckets(ffn_group_size);
            bool group_backends_unique = true;
            int cpu_branch_count = 0;
            for (int i = 0; i < ffn_group_size; ++i) {
                struct ggml_backend_sched_split * group_split = &splits[split_id + i];
                ggml_backend_t group_backend = sched->backends[group_split->backend_id];
                group_buckets[i] = ggml_sched_profile_backend_bucket(group_backend);
                cpu_branch_count += group_buckets[i] == GGML_SCHED_PROFILE_BACKEND_CPU ? 1 : 0;
                for (int j = 0; j < i; ++j) {
                    if (splits[split_id + j].backend_id == group_split->backend_id) {
                        group_backends_unique = false;
                        break;
                    }
                }
                if (i > 0) {
                    ggml_sched_profile_note_layers(group_split->graph, group_buckets[i]);
                }
            }
            const bool run_group_serially = !group_backends_unique || cpu_branch_count > 1;

            const int64_t group_t0_us = ggml_time_us();
            const int64_t copy_t0_us = group_t0_us;
            GGML_ASSERT(ffn_group_size <= GGML_SCHED_MAX_BACKENDS);
            ggml_backend_sched_trace_io_times io_times[GGML_SCHED_MAX_BACKENDS];
            for (int i = 0; i < ffn_group_size; ++i) {
                ggml_backend_sched_trace_io_times * trace_io_times = nullptr;
                if (trace_enabled) {
                    io_times[i] = {};
                    trace_io_times = &io_times[i];
                }
                copy_split_inputs(&splits[split_id + i], trace_io_times);
            }
            const int64_t copy_t1_us = ggml_time_us();

            std::vector<int64_t> dt_us(ffn_group_size, 0);
            std::vector<enum ggml_status> status(ffn_group_size, GGML_STATUS_SUCCESS);
            const bool use_persistent_device_workers = sched->ffn_device_executor != nullptr;
            std::vector<std::thread> workers;
            if (!run_group_serially && !use_persistent_device_workers) {
                workers.reserve(ffn_group_size > 0 ? ffn_group_size - 1 : 0);
            }
            ggml_backend_sched_ffn_device_job device_jobs[GGML_SCHED_MAX_BACKENDS];
            uint64_t device_tickets[GGML_SCHED_MAX_BACKENDS] = {};
            bool device_jobs_submitted[GGML_SCHED_MAX_BACKENDS] = {};

            // The thread that calls a CPU graph also participates in the CPU
            // thread pool as worker 0. Keep that role on the stable scheduler
            // caller instead of assigning it to a new std::thread for every
            // FFN group. Device backends only use their host threads to submit
            // work and wait for completion, so run those branches in workers.
            // Preserve the previous last-branch behavior when no CPU backend
            // is present in the group.
            int main_i = ffn_group_size - 1;
            for (int i = 0; i < ffn_group_size; ++i) {
                if (group_buckets[i] == GGML_SCHED_PROFILE_BACKEND_CPU) {
                    main_i = i;
                    break;
                }
            }

            const int64_t compute_t0_us = ggml_time_us();
            if (run_group_serially) {
                // A backend context must not be entered concurrently from the
                // caller and its persistent worker. Duplicate backend IDs (or
                // multiple CPU branches sharing the CPU thread pool) therefore
                // fall back to deterministic serial execution.
                for (int i = 0; i < ffn_group_size; ++i) {
                    ggml_backend_sched_ffn_worker_timeline * timeline =
                        group_timelines != nullptr ? &group_timelines[i] : nullptr;
                    if (timeline != nullptr) {
                        timeline->mode = GGML_SCHED_FFN_WORKER_SERIAL;
                        timeline->worker_tid = ggml_backend_sched_current_tid();
                        timeline->start_cpu = ggml_backend_sched_current_cpu();
                        timeline->job_start_us = ggml_time_us();
                    }
                    status[i] = compute_split_no_callback(
                            &splits[split_id + i], &dt_us[i], true, timeline);
                    if (timeline != nullptr) {
                        timeline->end_cpu = ggml_backend_sched_current_cpu();
                        timeline->job_end_us = ggml_time_us();
                        timeline->completion_publish_us = timeline->job_end_us;
                    }
                }
            } else {
                for (int i = 0; i < ffn_group_size; ++i) {
                    if (i == main_i) {
                        continue;
                    }

                    if (use_persistent_device_workers &&
                            group_buckets[i] != GGML_SCHED_PROFILE_BACKEND_CPU) {
                        struct ggml_backend_sched_split * device_split = &splits[split_id + i];
                        ggml_backend_sched_ffn_worker_timeline * timeline =
                            group_timelines != nullptr ? &group_timelines[i] : nullptr;
                        if (timeline != nullptr) {
                            timeline->mode = GGML_SCHED_FFN_WORKER_PERSISTENT;
                            timeline->collect_kind = GGML_SCHED_FFN_COLLECT_TICKET_WAIT;
                            timeline->dispatch_begin_us = ggml_time_us();
                        }
                        device_jobs[i].backend = sched->backends[device_split->backend_id];
                        device_jobs[i].graph = &device_split->graph;
                        device_jobs[i].status = &status[i];
                        device_jobs[i].dt_us = &dt_us[i];
                        device_jobs[i].timeline = timeline;
                        const bool submitted = sched->ffn_device_executor->submit(
                                device_split->backend_id,
                                ggml_backend_sched_run_ffn_device_job,
                                &device_jobs[i],
                                &status[i],
                                &device_tickets[i],
                                timeline);
                        if (timeline != nullptr) {
                            timeline->dispatch_end_us = ggml_time_us();
                        }
                        if (submitted) {
                            device_jobs_submitted[i] = true;
                        } else {
                            if (timeline != nullptr) {
                                timeline->collect_kind = GGML_SCHED_FFN_COLLECT_NONE;
                            }
                            GGML_LOG_ERROR(
                                    "%s: failed to submit persistent FFN job for backend %s, layer %d\n",
                                    __func__,
                                    ggml_backend_name(device_jobs[i].backend),
                                    ffn_group_infos[i].layer);
                            status[i] = GGML_STATUS_FAILED;
                        }
                    } else {
                        ggml_backend_sched_ffn_worker_timeline * timeline =
                            group_timelines != nullptr ? &group_timelines[i] : nullptr;
                        if (timeline != nullptr) {
                            timeline->mode = GGML_SCHED_FFN_WORKER_TRANSIENT;
                            timeline->collect_kind = GGML_SCHED_FFN_COLLECT_THREAD_JOIN;
                            timeline->dispatch_begin_us = ggml_time_us();
                        }
                        workers.emplace_back([&, i]() {
                            ggml_backend_sched_ffn_worker_timeline * worker_timeline =
                                group_timelines != nullptr ? &group_timelines[i] : nullptr;
                            if (worker_timeline != nullptr) {
                                worker_timeline->worker_dequeue_us = ggml_time_us();
                                worker_timeline->worker_tid = ggml_backend_sched_current_tid();
                            }
                            // A transient fallback thread would otherwise
                            // inherit the strict CPU worker-0 mask (CPU 7 in
                            // the S25 layout) and contend with the CPU branch.
                            const int worker_cpu = group_buckets[i] == GGML_SCHED_PROFILE_BACKEND_HTP
                                ? sched->ffn_npu_worker_cpu
                                : group_buckets[i] == GGML_SCHED_PROFILE_BACKEND_GPU
                                    ? sched->ffn_gpu_worker_cpu
                                    : sched->ffn_device_worker_cpu;
                            if (worker_timeline != nullptr) {
                                worker_timeline->affinity_cpu = worker_cpu;
                            }
                            ggml_backend_sched_pin_current_thread_to_cpu(
                                    worker_cpu);
                            if (worker_timeline != nullptr) {
                                worker_timeline->start_cpu = ggml_backend_sched_current_cpu();
                                worker_timeline->job_start_us = ggml_time_us();
                            }
                            try {
                                status[i] = compute_split_no_callback(
                                        &splits[split_id + i], &dt_us[i], true, worker_timeline);
                            } catch (const std::exception & error) {
                                status[i] = GGML_STATUS_FAILED;
                                GGML_LOG_ERROR(
                                        "%s: transient FFN worker backend %s threw an exception: %s\n",
                                        __func__,
                                        ggml_backend_name(sched->backends[splits[split_id + i].backend_id]),
                                        error.what());
                            } catch (...) {
                                status[i] = GGML_STATUS_FAILED;
                                GGML_LOG_ERROR(
                                        "%s: transient FFN worker backend %s threw an exception\n",
                                        __func__,
                                        ggml_backend_name(sched->backends[splits[split_id + i].backend_id]));
                            }
                            if (worker_timeline != nullptr) {
                                worker_timeline->end_cpu = ggml_backend_sched_current_cpu();
                                worker_timeline->job_end_us = ggml_time_us();
                                worker_timeline->completion_publish_us = ggml_time_us();
                            }
                        });
                        if (timeline != nullptr) {
                            timeline->dispatch_end_us = ggml_time_us();
                        }
                    }
                }

                ggml_backend_sched_ffn_worker_timeline * main_timeline =
                    group_timelines != nullptr ? &group_timelines[main_i] : nullptr;
                if (main_timeline != nullptr) {
                    main_timeline->mode = GGML_SCHED_FFN_WORKER_CALLER;
                    main_timeline->worker_tid = ggml_backend_sched_current_tid();
                    main_timeline->start_cpu = ggml_backend_sched_current_cpu();
                    main_timeline->job_start_us = ggml_time_us();
                }
                try {
                    status[main_i] = compute_split_no_callback(
                            &splits[split_id + main_i], &dt_us[main_i], true, main_timeline);
                } catch (const std::exception & error) {
                    status[main_i] = GGML_STATUS_FAILED;
                    GGML_LOG_ERROR(
                            "%s: caller FFN branch backend %s threw an exception: %s\n",
                            __func__,
                            ggml_backend_name(sched->backends[splits[split_id + main_i].backend_id]),
                            error.what());
                } catch (...) {
                    status[main_i] = GGML_STATUS_FAILED;
                    GGML_LOG_ERROR(
                            "%s: caller FFN branch backend %s threw an exception\n",
                            __func__,
                            ggml_backend_name(sched->backends[splits[split_id + main_i].backend_id]));
                }
                if (main_timeline != nullptr) {
                    main_timeline->end_cpu = ggml_backend_sched_current_cpu();
                    main_timeline->job_end_us = ggml_time_us();
                    main_timeline->completion_publish_us = main_timeline->job_end_us;
                }
                int collect_order = 0;
                for (int i = 0; i < ffn_group_size; ++i) {
                    if (device_jobs_submitted[i]) {
                        ggml_backend_sched_ffn_worker_timeline * timeline =
                            group_timelines != nullptr ? &group_timelines[i] : nullptr;
                        if (timeline != nullptr) {
                            timeline->collect_order = collect_order++;
                        }
                        sched->ffn_device_executor->wait(
                                splits[split_id + i].backend_id,
                                device_tickets[i],
                                timeline);
                    }
                }
                for (size_t worker_index = 0; worker_index < workers.size(); ++worker_index) {
                    const int branch_index = (int) worker_index < main_i
                        ? (int) worker_index
                        : (int) worker_index + 1;
                    ggml_backend_sched_ffn_worker_timeline * timeline = group_timelines != nullptr
                        ? &group_timelines[branch_index]
                        : nullptr;
                    if (timeline != nullptr) {
                        timeline->collect_order = collect_order++;
                        timeline->collect_begin_us = ggml_time_us();
                    }
                    workers[worker_index].join();
                    if (timeline != nullptr) {
                        timeline->collect_end_us = ggml_time_us();
                    }
                }
            }
            const int64_t compute_t1_us = ggml_time_us();
            const int64_t group_t1_us = compute_t1_us;
            const char * group_timing_mode = run_group_serially
                ? "serial_fallback"
                : use_persistent_device_workers ? "persistent_synced_wall" : "synced_wall";

            if (worker_trace_graph_id >= 0) {
                ggml_backend_sched_ffn_worker_trace_append_group(
                        sched,
                        worker_trace_graph_id,
                        split_id,
                        ffn_group_size,
                        ffn_group_infos,
                        group_timelines,
                        status,
                        dt_us,
                        group_t0_us,
                        copy_t1_us,
                        compute_t0_us,
                        group_t1_us);
            }

            for (int i = 0; i < ffn_group_size; ++i) {
                if (status[i] != GGML_STATUS_SUCCESS) {
                    return status[i];
                }
            }

            for (int i = 0; i < ffn_group_size; ++i) {
                struct ggml_backend_sched_split * group_split = &splits[split_id + i];
                prof_add(group_split->graph, group_buckets[i], dt_us[i]);
                record_split_event(group_split);
            }

            const int64_t copy_us = copy_t1_us - copy_t0_us;
            const int64_t compute_wall_us = compute_t1_us - compute_t0_us;
            const int64_t group_wall_us = group_t1_us - group_t0_us;
            int64_t serial_compute_us = 0;
            int64_t min_branch_us = LLONG_MAX;
            int64_t max_branch_us = 0;
            std::string split_list;
            std::string branch_list;
            for (int i = 0; i < ffn_group_size; ++i) {
                ggml_backend_t group_backend = sched->backends[splits[split_id + i].backend_id];
                serial_compute_us += dt_us[i];
                min_branch_us = std::min(min_branch_us, dt_us[i]);
                max_branch_us = std::max(max_branch_us, dt_us[i]);

                if (!split_list.empty()) {
                    split_list += "/";
                }
                split_list += std::to_string(split_id + i);

                if (!branch_list.empty()) {
                    branch_list += " + ";
                }
                branch_list += ggml_backend_name(group_backend);
                branch_list += "[";
                branch_list += ffn_group_infos[i].branch;
                branch_list += "] ";
                char branch_ms[32];
                snprintf(branch_ms, sizeof(branch_ms), "%.3f", (double) dt_us[i] / 1000.0);
                branch_list += branch_ms;
                branch_list += " ms";
            }
            const int64_t overlap_us = std::max<int64_t>(0, serial_compute_us - compute_wall_us);
            const double speedup = compute_wall_us > 0 ? (double) serial_compute_us / (double) compute_wall_us : 0.0;
            const double balance = max_branch_us > 0 ? 100.0 * (double) min_branch_us / (double) max_branch_us : 0.0;
            double overlap = min_branch_us > 0 && min_branch_us != LLONG_MAX ? 100.0 * (double) overlap_us / (double) min_branch_us : 0.0;
            overlap = std::min(100.0, std::max(0.0, overlap));

            GGML_LOG_DEBUG(
                    "sched: parallel ffn group layer %d splits %s: %s, compute_wall %.3f ms, group_wall %.3f ms, copy %.3f ms, speedup %.2fx, overlap %.1f%%, balance %.1f%%\n",
                    ffn_group_infos[0].layer,
                    split_list.c_str(),
                    branch_list.c_str(),
                    (double) compute_wall_us / 1000.0,
                    (double) group_wall_us / 1000.0,
                    (double) copy_us / 1000.0,
                    speedup,
                    overlap,
                    balance);

            if (trace_graph_id >= 0) {
                for (int i = 0; i < ffn_group_size; ++i) {
                    ggml_backend_sched_trace_append_split(
                            sched,
                            trace_graph_id,
                            split_id + i,
                            &splits[split_id + i],
                            io_times[i],
                            dt_us[i],
                            dt_us[i],
                            true,
                            ffn_group_infos[i].branch.c_str(),
                            split_id,
                            group_wall_us,
                            copy_us,
                            group_timing_mode);
                }
            }

            split_id += ffn_group_size - 1;
            continue;
        }

        ggml_backend_sched_trace_io_times io_times;
        if (trace_enabled) {
            io_times = {};
        }
        copy_split_inputs(split, trace_enabled ? &io_times : nullptr);

        if (!sched->callback_eval) {
            int64_t dt_us = 0;
            enum ggml_status ec = compute_split_no_callback(split, &dt_us, false, nullptr);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
            prof_add(split->graph, split_bucket, dt_us);
            record_split_event(split);

            if (trace_graph_id >= 0) {
                const bool split_is_cpu = split_bucket == GGML_SCHED_PROFILE_BACKEND_CPU;
                ggml_backend_sched_trace_append_split(
                        sched,
                        trace_graph_id,
                        split_id,
                        split,
                        io_times,
                        dt_us,
                        split_is_cpu ? dt_us : -1,
                        false,
                        "",
                        -1,
                        -1,
                        io_times.copy_us,
                        split_is_cpu ? "sync_wall" : "async_submit");
            }
        } else {
            enum ggml_status ec = compute_split_with_callback(split);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
            record_split_event(split);
            if (trace_graph_id >= 0) {
                ggml_backend_sched_trace_append_split(
                        sched,
                        trace_graph_id,
                        split_id,
                        split,
                        io_times,
                        -1,
                        -1,
                        false,
                        "",
                        -1,
                        -1,
                        io_times.copy_us,
                        "callback");
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;
    sched->ffn_device_worker_cpu = ggml_backend_sched_ffn_worker_cpu(
            "GGML_FFN_DEVICE_WORKER_CPU", -1);
    sched->ffn_gpu_worker_cpu = ggml_backend_sched_ffn_worker_cpu(
            "GGML_FFN_GPU_WORKER_CPU", sched->ffn_device_worker_cpu);
    sched->ffn_npu_worker_cpu = ggml_backend_sched_ffn_worker_cpu(
            "GGML_FFN_NPU_WORKER_CPU", sched->ffn_device_worker_cpu);

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    if (ggml_backend_sched_ffn_persistent_workers_enabled()) {
        int affinity_cpus[GGML_SCHED_MAX_BACKENDS];
        for (int backend_id = 0; backend_id < n_backends; ++backend_id) {
            switch (ggml_sched_profile_backend_bucket(sched->backends[backend_id])) {
                case GGML_SCHED_PROFILE_BACKEND_CPU:
                    affinity_cpus[backend_id] = -1;
                    break;
                case GGML_SCHED_PROFILE_BACKEND_HTP:
                    affinity_cpus[backend_id] = sched->ffn_npu_worker_cpu;
                    break;
                case GGML_SCHED_PROFILE_BACKEND_GPU:
                    affinity_cpus[backend_id] = sched->ffn_gpu_worker_cpu;
                    break;
            }
        }

        ggml_backend_sched_ffn_executor * executor = nullptr;
        try {
            executor = new ggml_backend_sched_ffn_executor();
        } catch (const std::exception & error) {
            GGML_LOG_ERROR(
                    "%s: failed to allocate persistent FFN device executor: %s; using per-group threads\n",
                    __func__, error.what());
        } catch (...) {
            GGML_LOG_ERROR(
                    "%s: failed to allocate persistent FFN device executor; using per-group threads\n",
                    __func__);
        }

        if (executor != nullptr) {
            if (executor->initialize(sched->backends, n_backends, affinity_cpus)) {
                sched->ffn_device_executor = executor;
                GGML_LOG_INFO(
                        "%s: persistent FFN device workers enabled "
                        "(GPU CPU %d, NPU CPU %d, legacy CPU %d)\n",
                        __func__,
                        sched->ffn_gpu_worker_cpu,
                        sched->ffn_npu_worker_cpu,
                        sched->ffn_device_worker_cpu);
            } else {
                delete executor;
                GGML_LOG_ERROR(
                        "%s: persistent FFN device worker initialization failed; using per-group threads\n",
                        __func__);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }

    // Stop persistent workers before releasing backend events, graphs, or
    // buffers they may reference.
    delete sched->ffn_device_executor;
    sched->ffn_device_executor = nullptr;

    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    free(sched->splits);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;

    return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    return ggml_backend_sched_compute_splits(sched);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

void ggml_backend_sched_set_ffn_parallel_reduce_threads(
        ggml_backend_sched_t sched, int cpu_threads, int reduce_threads) {
    GGML_ASSERT(sched);
    sched->cpu_threads = cpu_threads;
    sched->ffn_parallel_reduce_threads = reduce_threads;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
