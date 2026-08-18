#ifndef GGML_OPENCL_H
#define GGML_OPENCL_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

//
// backend API
//
GGML_BACKEND_API ggml_backend_t ggml_backend_opencl_init(void);
GGML_BACKEND_API bool ggml_backend_is_opencl(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl_buffer_type(void);

// Available when GGML_OPENCL_ASYNC_EVENTS=1. Memory returned by this buffer
// type may be used as the source of ggml_backend_tensor_set_async(); the caller
// must record an OpenCL backend event and wait for it before reusing or freeing
// the source allocation. This explicit getter does not expose the type through
// ggml_backend_dev_host_buffer_type(); global host-buffer selection requires
// the separate GGML_OPENCL_EXPOSE_HOST_BUFFER=1 opt-in.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_opencl_reg(void);

#ifdef  __cplusplus
}
#endif

#endif // GGML_OPENCL_H
