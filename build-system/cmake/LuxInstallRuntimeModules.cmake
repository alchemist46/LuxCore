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

# The libraries a standalone executable loads at run time with dlopen(), so
# they are not in its RUNTIME_DEPENDENCY_SET: the OIDN CPU device module
# (OIDN looks for it next to its core library) and, when one is shipped, the
# HIPRT library (hiprtew finds it through the executable's $ORIGIN/../lib
# rpath). Installed in the executable's component, next to its other
# libraries.
function(lux_install_runtime_modules component)
  if(WIN32)
    set(module_type BIN)
  else()
    set(module_type LIB)
  endif()

  if(WIN32 OR LINUX)
    install(
      PROGRAMS "${LUX_OIDN_DEVICE_CPU}"
      TYPE ${module_type}
      COMPONENT ${component}
      PERMISSIONS
        OWNER_WRITE OWNER_READ OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
    )
  endif()

  # The same setting as the pyluxcore wheel (see src/pyluxcore/CMakeLists.txt)
  set(hiprt_library "${LUXRAYS_HIPRT_LIBRARY}")
  if(NOT hiprt_library AND DEFINED ENV{LUXRAYS_HIPRT_LIBRARY})
    file(TO_CMAKE_PATH "$ENV{LUXRAYS_HIPRT_LIBRARY}" hiprt_library)
  endif()
  if(hiprt_library AND NOT LUXRAYS_DISABLE_OPENCL AND NOT LUXRAYS_DISABLE_HIP)
    install(
      PROGRAMS "${hiprt_library}"
      TYPE ${module_type}
      COMPONENT ${component}
      PERMISSIONS
        OWNER_WRITE OWNER_READ OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
    )
  endif()
endfunction()
