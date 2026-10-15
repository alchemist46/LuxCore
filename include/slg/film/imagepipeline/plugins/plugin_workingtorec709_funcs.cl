#line 2 "plugin_workingtorec709_funcs.cl"

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

//------------------------------------------------------------------------------
// WorkingToRec709Plugin_Apply
//------------------------------------------------------------------------------

__kernel void WorkingToRec709Plugin_Apply(
		const uint filmWidth, const uint filmHeight,
		__global float *channel_IMAGEPIPELINE,
		const float m00, const float m01, const float m02,
		const float m10, const float m11, const float m12,
		const float m20, const float m21, const float m22) {
	const size_t gid = get_global_id(0);
	if (gid >= filmWidth * filmHeight)
		return;

	// Check if the pixel has received any sample
	if (!isinf(channel_IMAGEPIPELINE[gid * 3])) {
		__global float *pixel = &channel_IMAGEPIPELINE[gid * 3];
		const float r = pixel[0];
		const float g = pixel[1];
		const float b = pixel[2];

		pixel[0] = m00 * r + m01 * g + m02 * b;
		pixel[1] = m10 * r + m11 * g + m12 * b;
		pixel[2] = m20 * r + m21 * g + m22 * b;
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
