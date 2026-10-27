################################################################################
# Copyright 1998-2025 by Authors (see AUTHORS.txt)
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

################################################################################
#
# PyLuxCore (Python bindings)
#
################################################################################

set(PYLUXCORE_SRCS)
include_directories(${PROJECT_SOURCE_DIR}/deps/bcd-1.1/include)

pybind11_add_module(pyluxcore MODULE
  ${PROJECT_SOURCE_DIR}/src/pyluxcore/pyluxcore.cpp
  ${PROJECT_SOURCE_DIR}/src/pyluxcore/pyluxcoreforblender.cpp
)

target_link_libraries(pyluxcore PRIVATE
    onetbb::onetbb
    oidn::oidn
    luxcore_static
    OpenVDB::openvdb
    pybind11::pybind11
    Python::Module
    robin_hood::robin_hood
    openimageio::openimageio
    embree::embree
)

if(APPLE)
  target_link_libraries(pyluxcore PRIVATE OpenMP::OpenMP)
else()
  target_link_libraries(pyluxcore PRIVATE OpenMP::OpenMP_CXX)
endif(APPLE)

set_target_properties(pyluxcore PROPERTIES PREFIX "")
if(MSVC)
  set_target_properties(pyluxcore PROPERTIES SUFFIX ".pyd")
endif(MSVC)

# Install
INSTALL(
  TARGETS pyluxcore
  DESTINATION pyluxcore
  COMPONENT pyluxcore
  PERMISSIONS
    OWNER_WRITE OWNER_READ OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

# Install pdb file (if msvc and debug)
if(MSVC)
  install(FILES "$<$<CONFIG:Debug>:$<TARGET_PDB_FILE:pyluxcore>>"
    DESTINATION pyluxcore
    COMPONENT pyluxcore
  )
endif()

# MacOS debug info
if(APPLE)
  find_program(DSYMUTIL_PROGRAM dsymutil)
  if (DSYMUTIL_PROGRAM)
    MESSAGE(STATUS "Found dsymutil: ${DSYMUTIL_PROGRAM}")
    foreach(lang C CXX)
      foreach(var LINK_EXECUTABLE CREATE_SHARED_LIBRARY CREATE_SHARED_MODULE)
        set(CMAKE_${lang}_${var} "${CMAKE_${lang}_${var}}" "${DSYMUTIL_PROGRAM} $<TARGET_FILE>")
      endforeach()
    endforeach()
    install(FILES "$<$<CONFIG:Debug,RelWithDebInfo>:$<TARGET_FILE:pyluxcore>.dsym>"
      DESTINATION pyluxcore
      COMPONENT pyluxcore
    )
  endif()
endif()

# The OIDN CPU device module: OIDN's core loads it at run time, by name, from
# its own folder, so it goes where the wheel repair puts the core library
# (pyluxcore.libs). The repair (delvewheel on Windows, see
# build-system/luxmake/wheel.py) patches its imports to the vendored
# libraries without renaming it.
if(WIN32 OR LINUX)
  install(
    PROGRAMS "${LUX_OIDN_DEVICE_CPU}"
    DESTINATION "pyluxcore.libs"
    COMPONENT pyluxcore
  )
endif()


# HIPRT (optional): hiprtew loads the HIPRT library at run time, also from the
# pyluxcore folder, so it can be shipped in the wheel. LUXRAYS_HIPRT_LIBRARY is
# the path of the library built from
# https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT (i.e.
# dist/bin/Release/libhiprt0300164.so). When empty, HIPRT isn't shipped and
# it is looked for in the system.
# It can also be set through the environment.
if(NOT LUXRAYS_DISABLE_OPENCL AND NOT LUXRAYS_DISABLE_HIP)
  set(LUXRAYS_HIPRT_LIBRARY "" CACHE FILEPATH
    "HIPRT library to ship next to pyluxcore. Leave empty to not ship HIPRT.")
  if(NOT LUXRAYS_HIPRT_LIBRARY AND DEFINED ENV{LUXRAYS_HIPRT_LIBRARY})
    file(TO_CMAKE_PATH "$ENV{LUXRAYS_HIPRT_LIBRARY}" LUXRAYS_HIPRT_LIBRARY)
  endif()

  if(LUXRAYS_HIPRT_LIBRARY)
    if(NOT EXISTS "${LUXRAYS_HIPRT_LIBRARY}")
      message(FATAL_ERROR "HIPRT library not found: ${LUXRAYS_HIPRT_LIBRARY}")
    endif()

    # hiprtew only looks for the library of the HIPRT version it implements
    # (see HIPRTEW_LIBRARY_NAME): any other one would never be loaded
    file(STRINGS "${PROJECT_SOURCE_DIR}/deps/hipew/include/hiprtew.h"
      hiprtew_version_line REGEX "#define HIPRTEW_VERSION_STR ")
    string(REGEX REPLACE ".*\"([0-9]+)\".*" "\\1"
      hiprtew_version "${hiprtew_version_line}")
    if(WIN32)
      set(hiprt_library_name "hiprt${hiprtew_version}64.dll")
    else()
      set(hiprt_library_name "libhiprt${hiprtew_version}64.so")
    endif()
    get_filename_component(hiprt_file_name "${LUXRAYS_HIPRT_LIBRARY}" NAME)
    if(NOT hiprt_file_name STREQUAL hiprt_library_name)
      message(FATAL_ERROR
        "HIPRT library '${LUXRAYS_HIPRT_LIBRARY}' doesn't match the HIPRT "
        "version used by LuxCore: '${hiprt_library_name}' is required")
    endif()

    message(STATUS "Shipping HIPRT library: ${LUXRAYS_HIPRT_LIBRARY}")
    install(
      PROGRAMS "${LUXRAYS_HIPRT_LIBRARY}"
      DESTINATION pyluxcore
      COMPONENT pyluxcore
    )
    install(
      FILES "${PROJECT_SOURCE_DIR}/deps/hipew/HIPRT-LICENSE.txt"
      DESTINATION pyluxcore
      COMPONENT pyluxcore
    )

    # HIPRT compiles its kernels at run time and reads headers of its own
    # from HIPRT_PATH, which hiprtew sets to the folder of the library when
    # they are there (see HiprtInit()). They are in the HIPRT source tree,
    # 3 levels above the library (dist/bin/Release).
    get_filename_component(hiprt_root "${LUXRAYS_HIPRT_LIBRARY}" DIRECTORY)
    get_filename_component(hiprt_root "${hiprt_root}/../../.." ABSOLUTE)
    if(EXISTS "${hiprt_root}/hiprt/hiprt_types.h")
      message(STATUS "Shipping HIPRT headers: ${hiprt_root}/hiprt")
      install(
        DIRECTORY "${hiprt_root}/hiprt"
        DESTINATION pyluxcore
        COMPONENT pyluxcore
        FILES_MATCHING PATTERN "*.h"
        PATTERN "cache" EXCLUDE
      )
    else()
      message(WARNING "HIPRT headers not found in ${hiprt_root}/hiprt: "
        "HIPRT_PATH must point to them at run time")
    endif()
  endif()
endif()
