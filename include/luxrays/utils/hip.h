/***************************************************************************
 * Copyright 1998-2026 by authors (see AUTHORS.txt)                        *
 *                                                                         *
 *   This file is part of LuxCoreRender.                                   *
 *                                                                         *
 * Licensed under the Apache License, Version 2.0 (the "License");         *
 * you may not use this file except in compliance with the License.        *
 * You may obtain a copy of the License at                                 *
 *                                                                         *
 *     http://www.apache.org/licenses/LICENSE-2.0                          *
 *                                                                         *
 * Unless required by applicable law or agreed to in writing, software     *
 * distributed under the License is distributed on an "AS IS" BASIS,       *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.*
 * See the License for the specific language governing permissions and     *
 * limitations under the License.                                          *
 ***************************************************************************/

#ifndef _LUXRAYS_HIP_H
#define	_LUXRAYS_HIP_H

#if !defined(LUXRAYS_DISABLE_HIP)

#include <stdexcept>

#include <hipew.h>
#include <hiprtew.h>

#include "luxrays/utils/strutils.h"

namespace luxrays {

#define CHECK_HIP_ERROR(err) CheckHIPError(err, __FILE__, __LINE__)

inline void CheckHIPError(const hipError_t err, const char *file, const int line) {
	if (err != hipSuccess) {
		const char *errorNameStr = hipGetErrorName ? hipGetErrorName(err) : nullptr;
		if (!errorNameStr)
			errorNameStr = "hipGetErrorName(ERROR)";

		const char *errorStr = hipGetErrorString ? hipGetErrorString(err) : nullptr;
		if (!errorStr)
			errorStr = "hipGetErrorString(ERROR)";

		throw std::runtime_error("HIP runtime API error " + std::string(errorNameStr) + " "
				"(code: " + ToString(err) + ", file:" + std::string(file) + ", line: " + ToString(line) + ")"
				": " + std::string(errorStr) + "\n");
	}
}

#define CHECK_HIPRT_ERROR(err) CheckHIPRTError(err, __FILE__, __LINE__)

inline void CheckHIPRTError(const hiprtError err, const char *file, const int line) {
	if (err != hiprtSuccess) {
		throw std::runtime_error("HIPRT error "
				"(code: " + ToString(err) + ", file:" + std::string(file) + ", line: " + ToString(line) + ")\n");
	}
}

#define CHECK_HIPRTC_ERROR(err) CheckHIPRTCError(err, __FILE__, __LINE__)

inline void CheckHIPRTCError(const hiprtcResult err, const char *file, const int line) {
	if (err != HIPRTC_SUCCESS) {
		throw std::runtime_error("HIP HIPRTC error "
				"(code: " + ToString(err) + ", file:" + std::string(file) + ", line: " + ToString(line) + ")"
				": " + std::string(hiprtcGetErrorString(err)) + "\n");
	}
}

}

#endif

#endif	/* _LUXRAYS_HIP_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
