# SPDX-FileCopyrightText: 2026 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

# Detect the offline clang of the HIP SDK (Windows) or ROCm (Linux), used to
# bake the HIP kernels in code objects for the generic architectures (hiprtc
# can't target them).
#
# The clang is looked for, in order, in LUXRAYS_HIP_CLANG (a cache variable),
# HIP_PATH, ROCM_PATH and /opt/rocm. A system clang is not used: it usually
# lacks the HIP headers and the device libraries. The clang found is verified
# with a compile probe for the first family of HIP_BAKE_FAMILIES (the generic
# targets require code object v6, i.e. LLVM 19 / ROCm 6.2 or later).
#
# Sets in the caller scope:
# - HIP_CLANG_EXECUTABLE: the clang to use, empty if none works

macro(find_hip_clang)
  set(HIP_CLANG_EXECUTABLE "")

  set(LUXRAYS_HIP_CLANG "" CACHE FILEPATH
    "The clang used to bake the HIP kernels. Leave empty to look for it in HIP_PATH, ROCM_PATH and /opt/rocm.")

  set(hip_clang_candidates)
  if(LUXRAYS_HIP_CLANG)
    list(APPEND hip_clang_candidates "${LUXRAYS_HIP_CLANG}")
  endif()
  foreach(hip_root_var HIP_PATH ROCM_PATH)
    if(DEFINED ENV{${hip_root_var}})
      file(TO_CMAKE_PATH "$ENV{${hip_root_var}}" hip_root)
      # The HIP SDK for Windows has the compiler in bin, ROCm in llvm/bin
      list(APPEND hip_clang_candidates
        "${hip_root}/bin/clang++${CMAKE_EXECUTABLE_SUFFIX}"
        "${hip_root}/llvm/bin/clang++${CMAKE_EXECUTABLE_SUFFIX}")
    endif()
  endforeach()
  if(NOT WIN32)
    list(APPEND hip_clang_candidates "/opt/rocm/llvm/bin/clang++")
  endif()

  list(GET HIP_BAKE_FAMILIES 0 hip_probe_family)
  set(hip_probe_src "${CMAKE_BINARY_DIR}/hip-probe.hip")
  set(hip_probe_co "${CMAKE_BINARY_DIR}/hip-probe.co")
  file(WRITE ${hip_probe_src} "extern \"C\" __global__ void Probe(float *a) { a[threadIdx.x] = sqrtf(a[threadIdx.x]); }\n")

  foreach(hip_clang ${hip_clang_candidates})
    if(EXISTS "${hip_clang}")
      execute_process(
        COMMAND "${hip_clang}" -x hip
          --offload-device-only --no-gpu-bundle-output
          --offload-arch=${hip_probe_family} -mcode-object-version=6
          -include hip/hip_runtime.h -O3
          ${hip_probe_src} -o ${hip_probe_co}
        RESULT_VARIABLE hip_probe_result
        OUTPUT_VARIABLE hip_probe_output
        ERROR_VARIABLE hip_probe_output
      )
      if(hip_probe_result EQUAL 0 AND EXISTS ${hip_probe_co})
        set(HIP_CLANG_EXECUTABLE "${hip_clang}")
        break()
      else()
        message(STATUS "${hip_clang} can't compile HIP code objects for ${hip_probe_family}:\n${hip_probe_output}")
      endif()
    endif()
  endforeach()

  file(REMOVE ${hip_probe_src} ${hip_probe_co})
endmacro()
