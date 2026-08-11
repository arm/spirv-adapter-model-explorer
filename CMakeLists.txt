# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(spirv_adapter_model_explorer_native LANGUAGES C CXX)

include(FetchContent)

set(SPIRV_HEADERS_TAG "vulkan-sdk-1.4.350.0" CACHE STRING "Pinned SPIRV-Headers tag")
set(SPIRV_TOOLS_TAG "vulkan-sdk-1.4.350.0" CACHE STRING "Pinned SPIRV-Tools tag")

FetchContent_Declare(
  SPIRV-Headers
  GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
  GIT_TAG ${SPIRV_HEADERS_TAG}
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(SPIRV-Headers)
FetchContent_GetProperties(SPIRV-Headers)

# SPIRV-Tools' CMake build consumes this cache variable when it is built as a
# subproject and the headers are not checked out under its external/ directory.
set(SPIRV-Headers_SOURCE_DIR "${spirv-headers_SOURCE_DIR}" CACHE PATH "SPIRV-Headers source directory" FORCE)

set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
set(SPIRV_WERROR OFF CACHE BOOL "" FORCE)
set(SPIRV_TOOLS_BUILD_STATIC OFF CACHE BOOL "" FORCE)
set(SPIRV_TOOLS_BUILD_SHARED ON CACHE BOOL "" FORCE)
set(SPIRV_TOOLS_BUILD_OPT OFF CACHE BOOL "" FORCE)
set(SPIRV_TOOLS_BUILD_REDUCE OFF CACHE BOOL "" FORCE)
set(SPIRV_TOOLS_BUILD_FUZZER OFF CACHE BOOL "" FORCE)
set(SPIRV_TOOLS_BUILD_LINKER OFF CACHE BOOL "" FORCE)
set(SPIRV_TOOLS_BUILD_LINT OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  SPIRV-Tools
  GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Tools.git
  GIT_TAG ${SPIRV_TOOLS_TAG}
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(SPIRV-Tools)

if(WIN32)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(SPIRV_ADAPTER_VENDOR_PLATFORM "win_arm64")
  else()
    set(SPIRV_ADAPTER_VENDOR_PLATFORM "win_amd64")
  endif()
elseif(APPLE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(SPIRV_ADAPTER_VENDOR_PLATFORM "macos_arm64")
  else()
    set(SPIRV_ADAPTER_VENDOR_PLATFORM "macos_x86_64")
  endif()
else()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(SPIRV_ADAPTER_VENDOR_PLATFORM "linux_aarch64")
  else()
    set(SPIRV_ADAPTER_VENDOR_PLATFORM "linux_x86_64")
  endif()
endif()

install(
  FILES $<TARGET_FILE:SPIRV-Tools-shared>
  DESTINATION spirv_adapter_model_explorer/vendor/${SPIRV_ADAPTER_VENDOR_PLATFORM}
  COMPONENT spirv_adapter_runtime
)
