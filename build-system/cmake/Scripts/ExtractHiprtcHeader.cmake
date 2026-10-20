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

# Extract the header hiprtc includes implicitly in every program
# (hiprtc_runtime.h), so the offline clang compiles the baked kernels with
# the same declarations as hiprtc at run time. The header exists only inside
# the hiprtc library: comgr writes it in its temporary directory at each
# compilation and keeps the directory with AMD_COMGR_SAVE_TEMPS=1. The
# hip-kernel-baker --hiprtc-probe compiles a small program with the temporary
# directory pointed to a directory of the build.
#
# The file is written only if its content changed, so the kernels are not
# compiled again for nothing.

if(NOT DEFINED BAKER OR NOT DEFINED DST)
  message(FATAL_ERROR "Usage: cmake -DBAKER=... -DDST=... -P ExtractHiprtcHeader.cmake")
endif()

get_filename_component(DST_DIR ${DST} DIRECTORY)
set(TMP_DIR "${DST_DIR}/hiprtc-temps")
file(REMOVE_RECURSE ${TMP_DIR})
file(MAKE_DIRECTORY ${TMP_DIR})

# The temporary directory: TMP/TEMP on Windows, TMPDIR elsewhere
execute_process(
  COMMAND ${CMAKE_COMMAND} -E env
    AMD_COMGR_SAVE_TEMPS=1
    TMP=${TMP_DIR} TEMP=${TMP_DIR} TMPDIR=${TMP_DIR}
    ${BAKER} --hiprtc-probe
  RESULT_VARIABLE PROBE_RESULT
  OUTPUT_VARIABLE PROBE_OUTPUT
  ERROR_VARIABLE PROBE_OUTPUT
)
if(NOT PROBE_RESULT EQUAL 0)
  file(REMOVE_RECURSE ${TMP_DIR})
  message(FATAL_ERROR "Unable to compile a program with hiprtc:\n${PROBE_OUTPUT}")
endif()

file(GLOB_RECURSE HEADERS "${TMP_DIR}/*/hiprtc_runtime.h")
list(LENGTH HEADERS HEADERS_COUNT)
if(HEADERS_COUNT EQUAL 0)
  file(REMOVE_RECURSE ${TMP_DIR})
  message(FATAL_ERROR "hiprtc_runtime.h not found in the comgr temporary files (${PROBE_OUTPUT})")
endif()
list(GET HEADERS 0 HEADER)

file(COPY_FILE ${HEADER} ${DST} ONLY_IF_DIFFERENT)
file(REMOVE_RECURSE ${TMP_DIR})

message(STATUS "Extracted hiprtc_runtime.h (${PROBE_OUTPUT})")
