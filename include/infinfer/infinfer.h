// Conan::ImportStart
#pragma once
#include <infinfer/kernel/kernel.h>
// Conan::ImportEnd

// C linkage: see the note in kernel.h. 见 kernel.h 的说明。
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file infinfer.h
 *
 * @brief [en] The single public include for the infinfer C API. It pulls in every
 *            public contract header; today that is the kernel contract only.
 * @brief [zh] infinfer C API 的唯一聚合入口：包含全部公开契约头；当前只有内核契约。
 * @note [en] Runtime headers (graph, tensor, backend, postprocess) join this file
 *            when they land; consumers should include this header and never the
 *            individual pieces.
 * @note [zh] 运行时头（graph/tensor/backend/postprocess）落地后会并入本文件；
 *            消费方应只包含本头，而不是各个分件。
 * @since 0.1.0
 */

#ifdef __cplusplus
}
#endif
