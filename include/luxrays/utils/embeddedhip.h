/***************************************************************************
 * Copyright 1998-2026 by authors (see AUTHORS.txt)
 *
 *   This file is part of LuxCoreRender.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/

#ifndef _LUXRAYS_EMBEDDEDHIP_H
#define	_LUXRAYS_EMBEDDEDHIP_H

#include <cstddef>
#include <string>

namespace luxrays {

// Return the embedded pre-compiled HIP code object with the given kernel key
// (GetHIPBakedKernelKey() in luxrays/utils/hipkernel.h) for the given family
// of architectures (GetHIPBakedKernelFamily()), or nullptr if there isn't one.
// The returned data is valid for the entire life of the process.
//
// The kernels are pre-compiled at build time by the hip-kernel-baker tool
// (see src/slg/kernels/hipkernelbaker.cpp) and embedded in the binaries. The
// definition of this function is provided either by the generated registry
// (src/slg/CMakeLists.txt) or, when the HIP pre-compilation is disabled, by a
// stub.
const unsigned char *GetEmbeddedHIPKernel(const std::string &key, const std::string &family,
		std::size_t *size);

// Return true if pre-compiled HIP kernels are embedded in the binaries (i.e.
// if the HIP pre-compilation was enabled at build time)
bool HasEmbeddedHIPKernels();

}

#endif
