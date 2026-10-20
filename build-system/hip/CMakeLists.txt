# SPDX-FileCopyrightText: 2026 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

# Dependency-free standalone project used to pre-compile the HIP kernels
# (see the LUXRAYS_PRECOMPILE_HIP option in src/slg/CMakeLists.txt and
# build-system/cmake/HIPBake.cmake, which holds the bake rules shared with
# the main build).
#
# It is used by the CI (see .github/workflows/wheel-builder.yml), where
# building LuxCore (i.e. the LuxCoreDeps dependencies) just in order to bake
# the kernels would be a waste of time, and where the wheel legs have no HIP
# SDK or ROCm. The code objects are device code, the same on every platform:
# the main build embeds them on any platform through LUXRAYS_HIP_BAKED_DIR.
#
# It compiles the hip-kernel-baker tool with the system compiler, the
# generated kernel sources and hipew alone, then the kernels with the clang of
# ROCm (Linux) or of the HIP SDK (Windows). The hiprtc library of the same
# ROCm / SDK must be loadable: the bake extracts its runtime header (see
# build-system/cmake/Scripts/ExtractHiprtcHeader.cmake). No GPU is needed.
#
# Usage:
#
#   cmake -S build-system/hip -B bake-build -G Ninja \
#     [-DHIPRT_INCLUDE_DIR=<HIPRT source tree>] \
#     [-DLUXRAYS_HIP_BAKE_FAMILIES=gfx10-3-generic] [-DLUXRAYS_HIP_BAKE_JOBS=4]
#   cmake --build bake-build
#
# HIPRT_INCLUDE_DIR is the directory with hiprt/impl/hiprt_device_impl.h: the
# source tree of the HIPRT version shipped with pyluxcore (the HIPRT trace
# kernel includes its device code). Without it, only the path kernels are
# baked. The baked kernels are written in bake-build/baked/.

cmake_minimum_required(VERSION 3.20)

project(HipBaker CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  # As the main build (the LuxCore headers test WIN32)
  add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS _SCL_SECURE_NO_WARNINGS)
  add_compile_options("/DWIN32" "/D_WINDOWS" "/EHsc" "/bigobj")
endif()

set(SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
get_filename_component(SOURCE_ROOT "${SOURCE_ROOT}" ABSOLUTE)

# The kernel source files and the bake variants
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/KernelFiles.cmake")
list(TRANSFORM LUXRAYS_KERNELS PREPEND "${SOURCE_ROOT}/")
list(TRANSFORM SLG_KERNELS PREPEND "${SOURCE_ROOT}/")

set(LUXRAYS_HIP_BAKE_FAMILIES "${HIP_BAKE_FAMILIES}" CACHE STRING
  "The families of architectures the HIP kernels are baked for (a subset of ${HIP_BAKE_FAMILIES})")
foreach(hip_family ${LUXRAYS_HIP_BAKE_FAMILIES})
  if(NOT hip_family IN_LIST HIP_BAKE_FAMILIES)
    message(FATAL_ERROR "Unknown family in LUXRAYS_HIP_BAKE_FAMILIES: ${hip_family} (valid: ${HIP_BAKE_FAMILIES})")
  endif()
endforeach()
set(LUXRAYS_HIP_BAKE_JOBS 4 CACHE STRING "The maximum number of HIP kernels compiled at the same time")
set(HIPRT_INCLUDE_DIR "" CACHE PATH
  "The HIPRT source tree (with hiprt/impl/hiprt_device_impl.h), required to bake the HIPRT trace kernel")

# The tool used to write the kernel sources and parameters
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/KernelPreprocess.cmake")

set(generated_kernels_dir "${CMAKE_BINARY_DIR}/generated/kernels")
file(MAKE_DIRECTORY ${generated_kernels_dir})

PreprocessOCLKernels(${generated_kernels_dir} luxrays ${LUXRAYS_KERNELS})
PreprocessOCLKernels(${generated_kernels_dir} slg ${SLG_KERNELS})

# The hip-kernel-baker tool: it links the generated kernel sources and hipew
# alone (i.e. no LuxCoreDeps dependency)
set(HIP_BAKER_SRCS
  "${SOURCE_ROOT}/src/slg/kernels/hipkernelbaker.cpp"
  "${SOURCE_ROOT}/src/slg/kernels/kernelsources.cpp"
  "${SOURCE_ROOT}/deps/hipew/src/hipew.cpp"
)
foreach(hip_kernel ${LUXRAYS_KERNELS} ${SLG_KERNELS})
  get_filename_component(hip_kernel_name ${hip_kernel} NAME_WE)
  list(APPEND HIP_BAKER_SRCS "${generated_kernels_dir}/${hip_kernel_name}_kernel.cpp")
endforeach()

add_executable(hip-kernel-baker ${HIP_BAKER_SRCS})
# luxrays/cfg.h is a generated header (see src/luxrays/CMakeLists.txt).
# The version value doesn't matter here: the tool doesn't use it.
set(generated_header_dir "${CMAKE_BINARY_DIR}/generated/include/luxrays")
file(MAKE_DIRECTORY ${generated_header_dir})
configure_file(
  "${SOURCE_ROOT}/include/luxrays/cfg.h.in"
  "${generated_header_dir}/cfg.h")
target_include_directories(hip-kernel-baker PRIVATE
  "${SOURCE_ROOT}/include"
  "${SOURCE_ROOT}/deps/hipew/include"
  "${CMAKE_BINARY_DIR}/generated/include")
# hipew loads the HIP libraries at run time
target_link_libraries(hip-kernel-baker PRIVATE ${CMAKE_DL_LIBS})

# The clang used to compile the kernels
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/HIPCompiler.cmake")
find_hip_clang()
if(NOT HIP_CLANG_EXECUTABLE)
  message(FATAL_ERROR "No ROCm or HIP SDK clang able to compile the HIP kernels found (LUXRAYS_HIP_CLANG, HIP_PATH, ROCM_PATH, /opt/rocm)")
endif()
message(STATUS "Baking the HIP kernels with ${HIP_CLANG_EXECUTABLE} (${LUXRAYS_HIP_BAKE_FAMILIES})")

include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/HIPBake.cmake")
hip_bake_modules("${LUXRAYS_HIP_BAKE_FAMILIES}" hip_modules hip_hiprt_modules hip_hiprt_variants)

if(HIPRT_INCLUDE_DIR)
  if(NOT EXISTS "${HIPRT_INCLUDE_DIR}/hiprt/impl/hiprt_device_impl.h")
    message(FATAL_ERROR "${HIPRT_INCLUDE_DIR}/hiprt/impl/hiprt_device_impl.h not found (HIPRT_INCLUDE_DIR)")
  endif()
  message(STATUS "Baking the HIPRT trace kernel with the HIPRT headers of ${HIPRT_INCLUDE_DIR}")
  list(APPEND hip_modules ${hip_hiprt_modules})
else()
  message(WARNING "No HIPRT_INCLUDE_DIR: the HIPRT trace kernel is not baked")
endif()

hip_bake_kernels(
  BAKER hip-kernel-baker
  BAKED_DIR "${CMAKE_BINARY_DIR}/baked"
  CLANG ${HIP_CLANG_EXECUTABLE}
  MODULES ${hip_modules}
  HIPRT_VARIANTS ${hip_hiprt_variants}
  HIPRT_INCLUDE_DIR ${HIPRT_INCLUDE_DIR}
  JOBS ${LUXRAYS_HIP_BAKE_JOBS}
  SOURCE_ROOT ${SOURCE_ROOT}
  CO_FILES hip_co_files
)

add_custom_target(hip-bake ALL DEPENDS ${hip_co_files})
