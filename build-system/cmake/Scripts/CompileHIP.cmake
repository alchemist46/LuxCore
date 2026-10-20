################################################################################
# Copyright 1998-2026 by authors (see AUTHORS.txt)
#
#   This file is part of LuxCoreRender.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

# Compile a HIP source written by the hip-kernel-baker in a code object for a
# family of architectures (a generic target, e.g. gfx10-3-generic) with the
# offline clang of the HIP SDK or ROCm (hiprtc can't target the generic
# architectures).
#
# The options are read from the PARAMS file (one per line): they are the ones
# hiprtc gets at run time, i.e. the ones of the kernel key. The others are the
# ones hiprtc adds itself (as shown by AMD_COMGR_EMIT_VERBOSE_LOGS=1):
# - its runtime header (HIPRTC_HEADER, see ExtractHiprtcHeader.cmake) instead
#   of the HIP headers of the SDK;
# - -fshort-wchar and the symbol internalization.
# Added here:
# - the target and the code object version (v6, required by the generic
#   targets);
# - the device only compilation, as a single plain code object (no offload
#   bundle: the run time loads it with hipModuleLoadData());
# - no compilation unit ID (a __hip_cuid_<hash> symbol, only used to link the
#   device static variables with the host code): its default hash includes the
#   source path, so without it the code objects are the same from any build
#   tree (e.g. a local build and the CI bake).

if(NOT DEFINED HIP_CLANG_EXECUTABLE OR NOT DEFINED FAMILY OR NOT DEFINED SRC OR NOT DEFINED PARAMS OR NOT DEFINED HIPRTC_HEADER OR NOT DEFINED DST)
  message(FATAL_ERROR "Usage: cmake -DHIP_CLANG_EXECUTABLE=... -DFAMILY=... -DSRC=... -DPARAMS=... -DHIPRTC_HEADER=... -DDST=... -P CompileHIP.cmake")
endif()

file(STRINGS ${PARAMS} HIP_PARAMETERS)

# INCLUDE_DIR (optional): the HIPRT headers of the trace kernel. Not part of
# the parameters: they are not part of the kernel key.
set(HIP_INCLUDE_OPTIONS)
if(INCLUDE_DIR)
  set(HIP_INCLUDE_OPTIONS -I${INCLUDE_DIR})
endif()

execute_process(
  COMMAND ${HIP_CLANG_EXECUTABLE} -x hip
  --offload-device-only --no-gpu-bundle-output
  --offload-arch=${FAMILY} -mcode-object-version=6 -fuse-cuid=none
  -D__HIPCC_RTC__ -nogpuinc -include ${HIPRTC_HEADER}
  -fshort-wchar -mllvm -amdgpu-internalize-symbols -Wno-gnu-line-marker
  ${HIP_INCLUDE_OPTIONS}
  ${HIP_PARAMETERS}
  ${SRC} -o ${DST}
  RESULT_VARIABLE HIP_RESULT
  OUTPUT_VARIABLE HIP_OUTPUT
  ERROR_VARIABLE HIP_OUTPUT
)
if(NOT HIP_RESULT EQUAL 0)
  file(REMOVE ${DST})
  message(FATAL_ERROR "Unable to compile ${SRC} for ${FAMILY}:\n${HIP_OUTPUT}")
endif()
