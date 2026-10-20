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

#include "luxrays/utils/embeddedhip.h"

// Stub used when the HIP pre-compilation of the kernels is disabled
// (LUXRAYS_PRECOMPILE_HIP CMake option) or not possible (no HIP clang and no
// LUXRAYS_HIP_BAKED_DIR)

namespace luxrays {

const unsigned char *GetEmbeddedHIPKernel(const std::string &key, const std::string &family,
		std::size_t *size) {
	if (size)
		*size = 0;
	return nullptr;
}

bool HasEmbeddedHIPKernels() {
	return false;
}

}
