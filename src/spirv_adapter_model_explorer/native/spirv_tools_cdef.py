# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

CDEF = """
typedef unsigned int uint32_t;
typedef unsigned long long size_t;
typedef _Bool bool;

typedef enum {
  SPV_SUCCESS = 0,
  SPV_UNSUPPORTED = 1,
  SPV_END_OF_STREAM = 2,
  SPV_WARNING = 3,
  SPV_FAILED_MATCH = 4,
  SPV_REQUESTED_TERMINATION = 5,
  SPV_ERROR_INTERNAL = -1,
  SPV_ERROR_OUT_OF_MEMORY = -2,
  SPV_ERROR_INVALID_POINTER = -3,
  SPV_ERROR_INVALID_BINARY = -4,
  SPV_ERROR_INVALID_TEXT = -5,
  SPV_ERROR_INVALID_TABLE = -6,
  SPV_ERROR_INVALID_VALUE = -7,
  SPV_ERROR_INVALID_DIAGNOSTIC = -8,
  SPV_ERROR_INVALID_LOOKUP = -9,
  SPV_ERROR_INVALID_ID = -10,
  SPV_ERROR_INVALID_CFG = -11,
  SPV_ERROR_INVALID_LAYOUT = -12,
  SPV_ERROR_INVALID_CAPABILITY = -13,
  SPV_ERROR_INVALID_DATA = -14,
  SPV_ERROR_INVALID_TEXT_ENCODING = -15
} spv_result_t;

typedef enum {
  SPV_ENV_UNIVERSAL_1_0,
  SPV_ENV_VULKAN_1_0,
  SPV_ENV_UNIVERSAL_1_1,
  SPV_ENV_OPENCL_2_1,
  SPV_ENV_OPENCL_2_2,
  SPV_ENV_OPENGL_4_0,
  SPV_ENV_OPENGL_4_1,
  SPV_ENV_OPENGL_4_2,
  SPV_ENV_OPENGL_4_3,
  SPV_ENV_OPENGL_4_5,
  SPV_ENV_UNIVERSAL_1_2,
  SPV_ENV_OPENCL_1_2,
  SPV_ENV_OPENCL_EMBEDDED_1_2,
  SPV_ENV_OPENCL_2_0,
  SPV_ENV_OPENCL_EMBEDDED_2_0,
  SPV_ENV_OPENCL_EMBEDDED_2_1,
  SPV_ENV_OPENCL_EMBEDDED_2_2,
  SPV_ENV_UNIVERSAL_1_3,
  SPV_ENV_VULKAN_1_1,
  SPV_ENV_WEBGPU_0,
  SPV_ENV_UNIVERSAL_1_4,
  SPV_ENV_VULKAN_1_1_SPIRV_1_4,
  SPV_ENV_UNIVERSAL_1_5,
  SPV_ENV_VULKAN_1_2,
  SPV_ENV_UNIVERSAL_1_6,
  SPV_ENV_VULKAN_1_3,
  SPV_ENV_VULKAN_1_4,
  SPV_ENV_MAX
} spv_target_env;

typedef struct spv_const_binary_t {
  const uint32_t* code;
  const size_t wordCount;
} spv_const_binary_t;

typedef struct spv_binary_t {
  uint32_t* code;
  size_t wordCount;
} spv_binary_t;

typedef struct spv_text_t {
  const char* str;
  size_t length;
} spv_text_t;

typedef struct spv_position_t {
  size_t line;
  size_t column;
  size_t index;
} spv_position_t;

typedef struct spv_diagnostic_t {
  spv_position_t position;
  char* error;
  bool isTextSource;
} spv_diagnostic_t;

typedef struct spv_context_t spv_context_t;
typedef struct spv_validator_options_t spv_validator_options_t;

typedef spv_const_binary_t* spv_const_binary;
typedef spv_binary_t* spv_binary;
typedef spv_text_t* spv_text;
typedef spv_diagnostic_t* spv_diagnostic;
typedef const spv_context_t* spv_const_context;
typedef spv_context_t* spv_context;
typedef spv_validator_options_t* spv_validator_options;
typedef const spv_validator_options_t* spv_const_validator_options;

spv_context spvContextCreate(spv_target_env env);
void spvContextDestroy(spv_context context);
spv_validator_options spvValidatorOptionsCreate(void);
void spvValidatorOptionsDestroy(spv_validator_options options);
void spvValidatorOptionsSetFriendlyNames(
    spv_validator_options options, bool val);
spv_result_t spvTextToBinaryWithOptions(
    const spv_const_context context,
    const char* text,
    const size_t length,
    const uint32_t options,
    spv_binary* binary,
    spv_diagnostic* diagnostic);
spv_result_t spvBinaryToText(
    const spv_const_context context,
    const uint32_t* binary,
    const size_t word_count,
    const uint32_t options,
    spv_text* text,
    spv_diagnostic* diagnostic);
spv_result_t spvValidateWithOptions(
    const spv_const_context context,
    const spv_const_validator_options options,
    const spv_const_binary binary,
    spv_diagnostic* diagnostic);
void spvBinaryDestroy(spv_binary binary);
void spvTextDestroy(spv_text text);
void spvDiagnosticDestroy(spv_diagnostic diagnostic);
"""
