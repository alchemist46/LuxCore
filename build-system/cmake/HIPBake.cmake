# SPDX-FileCopyrightText: 2026 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

# The rules baking the HIP kernels, shared by the main build
# (src/slg/CMakeLists.txt) and the standalone bake project used by the CI
# (build-system/hip), so the two produce the same code objects.
#
# HIP_VARIANTS, HIP_HIPRT_VARIANTS and HIP_BAKE_FAMILIES are defined in
# build-system/cmake/KernelFiles.cmake.

# The modules to bake, as <variant>:<family> pairs, for the given families:
# - <out_modules>: the path kernel variants of each family;
# - <out_hiprt_modules>: the HIPRT trace kernel variants of the families;
# - <out_hiprt_variants>: all the HIPRT trace kernel variant names (the baker
#   writes the sources of all of them).
function(hip_bake_modules families out_modules out_hiprt_modules out_hiprt_variants)
  set(modules)
  foreach(variant ${HIP_VARIANTS})
    foreach(family ${families})
      list(APPEND modules ${variant}:${family})
    endforeach()
  endforeach()

  set(hiprt_modules)
  set(hiprt_variants)
  foreach(module ${HIP_HIPRT_VARIANTS})
    string(REPLACE ":" ";" module_fields ${module})
    list(GET module_fields 0 variant)
    list(GET module_fields 1 family)
    list(APPEND hiprt_variants ${variant})
    if(family IN_LIST families)
      list(APPEND hiprt_modules ${module})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES hiprt_variants)

  set(${out_modules} ${modules} PARENT_SCOPE)
  set(${out_hiprt_modules} ${hiprt_modules} PARENT_SCOPE)
  set(${out_hiprt_variants} ${hiprt_variants} PARENT_SCOPE)
endfunction()

# Add the commands baking the kernels:
#
#   hip_bake_kernels(
#     BAKER <target>              the hip-kernel-baker executable target
#     BAKED_DIR <dir>             where the sources, the parameters, the
#                                 manifest and the code objects are written
#     CLANG <path>                the HIP SDK or ROCm clang
#     MODULES <variant:family>... the code objects to compile
#     HIPRT_VARIANTS <variant>... the variants including the HIPRT headers
#     HIPRT_INCLUDE_DIR <dir>     the HIPRT headers (for the HIPRT variants)
#     JOBS <n>                    the maximum number of concurrent compilations
#                                 (Ninja generators only)
#     SOURCE_ROOT <dir>           the LuxCore source tree
#     CO_FILES <var>)             output: the code object files
function(hip_bake_kernels)
  cmake_parse_arguments(ARG ""
    "BAKER;BAKED_DIR;CLANG;HIPRT_INCLUDE_DIR;JOBS;SOURCE_ROOT;CO_FILES"
    "MODULES;HIPRT_VARIANTS" ${ARGN})

  set(scripts_dir "${ARG_SOURCE_ROOT}/build-system/cmake/Scripts")
  file(MAKE_DIRECTORY ${ARG_BAKED_DIR})

  # Write the kernel sources and parameters for each variant. The tool
  # rewrites a file only if its content changed, so a rebuild of the tool
  # (e.g. any change in luxrays) doesn't compile the kernels again.
  set(baker_outputs ${ARG_BAKED_DIR}/hipmanifest.txt)
  foreach(variant ${HIP_VARIANTS} ${ARG_HIPRT_VARIANTS})
    list(APPEND baker_outputs
      ${ARG_BAKED_DIR}/${variant}.hip
      ${ARG_BAKED_DIR}/${variant}.params
    )
  endforeach()

  add_custom_command(
    OUTPUT ${baker_outputs}
    COMMAND $<TARGET_FILE:${ARG_BAKER}> ${ARG_BAKED_DIR}
    DEPENDS ${ARG_BAKER}
    COMMENT "Baking the HIP kernel sources for the pre-compilation"
  )

  # The header hiprtc includes implicitly, extracted from the hiprtc library
  # of the same SDK (written only if its content changed)
  set(hiprtc_header ${ARG_BAKED_DIR}/hiprtc_runtime.h)
  add_custom_command(
    OUTPUT ${hiprtc_header}
    COMMAND ${CMAKE_COMMAND}
      -DBAKER=$<TARGET_FILE:${ARG_BAKER}>
      -DDST=${hiprtc_header}
      -P "${scripts_dir}/ExtractHiprtcHeader.cmake"
    DEPENDS
      ${ARG_BAKER}
      "${scripts_dir}/ExtractHiprtcHeader.cmake"
    COMMENT "Extracting the hiprtc runtime header"
  )

  # Compile each module. A compilation of the path kernel takes minutes and
  # GBs of memory: the number of the concurrent ones is limited.
  set_property(GLOBAL APPEND PROPERTY JOB_POOLS hip_bake=${ARG_JOBS})

  set(co_files)
  foreach(module ${ARG_MODULES})
    string(REPLACE ":" ";" module_fields ${module})
    list(GET module_fields 0 variant)
    list(GET module_fields 1 family)
    set(co_file ${ARG_BAKED_DIR}/${variant}_${family}.co)
    list(APPEND co_files ${co_file})

    # The HIPRT trace kernel includes the HIPRT headers
    set(include_arg)
    if(variant IN_LIST ARG_HIPRT_VARIANTS)
      set(include_arg -DINCLUDE_DIR=${ARG_HIPRT_INCLUDE_DIR})
    endif()

    add_custom_command(
      OUTPUT ${co_file}
      COMMAND ${CMAKE_COMMAND}
        -DHIP_CLANG_EXECUTABLE=${ARG_CLANG}
        -DFAMILY=${family}
        -DSRC=${ARG_BAKED_DIR}/${variant}.hip
        -DPARAMS=${ARG_BAKED_DIR}/${variant}.params
        -DHIPRTC_HEADER=${hiprtc_header}
        ${include_arg}
        -DDST=${co_file}
        -P "${scripts_dir}/CompileHIP.cmake"
      DEPENDS
        ${ARG_BAKED_DIR}/${variant}.hip
        ${ARG_BAKED_DIR}/${variant}.params
        ${hiprtc_header}
        "${scripts_dir}/CompileHIP.cmake"
      COMMENT "Compiling the ${variant} HIP kernels for ${family}"
      JOB_POOL hip_bake
    )
  endforeach()

  set(${ARG_CO_FILES} ${co_files} PARENT_SCOPE)
endfunction()
