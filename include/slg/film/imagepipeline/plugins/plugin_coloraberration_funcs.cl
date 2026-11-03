#line 2 "plugin_coloraberration_funcs.cl"

/***************************************************************************
 * Copyright 1998-2020 by authors (see AUTHORS.txt)                        *
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
// ColorAberrationPlugin_Apply
//------------------------------------------------------------------------------

float3 ColorAberrationPlugin_BilinearSampleImage(
		__global float *channel_IMAGEPIPELINE,
		const uint width, const uint height,
		const float x, const float y) {
	const uint x1 = clamp(Floor2UInt(x), 0u, width - 1);
	const uint y1 = clamp(Floor2UInt(y), 0u, height - 1);
	const uint x2 = clamp(x1 + 1, 0u, width - 1);
	const uint y2 = clamp(y1 + 1, 0u, height - 1);
	const float tx = clamp(x - x1, 0.f, 1.f);
	const float ty = clamp(y - y1, 0.f, 1.f);

	float3 c;
	c = ((1.f - tx) * (1.f - ty)) * VLOAD3F(&channel_IMAGEPIPELINE[(y1 * width + x1) * 3]);
	c += (tx * (1.f - ty)) * VLOAD3F(&channel_IMAGEPIPELINE[(y1 * width + x2) * 3]);
	c += ((1.f - tx) * ty) * VLOAD3F(&channel_IMAGEPIPELINE[(y2 * width + x1) * 3]);
	c += (tx * ty) * VLOAD3F(&channel_IMAGEPIPELINE[(y2 * width + x2) * 3]);

	return c;
}

__kernel void ColorAberrationPlugin_Apply(
		const uint filmWidth, const uint filmHeight,
		__global float *channel_IMAGEPIPELINE,
		__global float *tmpBuffer,
		const float amountX, const float amountY) {
	const size_t gid = get_global_id(0);
	if (gid >= filmWidth * filmHeight)
		return;

	// Check if the pixel has received any sample
	if (!isinf(channel_IMAGEPIPELINE[gid * 3])) {
		const uint x = gid % filmWidth;
		const uint y = gid / filmWidth;
		const float nx = x / (float)filmWidth;
		const float ny = y / (float)filmHeight;
		const float xOffset = nx - .5f;
		const float yOffset = ny - .5f;
		const float tOffset = sqrt(xOffset * xOffset + yOffset * yOffset);

		const float rbX = (.5f + xOffset * (1.f + tOffset * amountX)) * filmWidth;
		const float rbY = (.5f + yOffset * (1.f + tOffset * amountY)) * filmHeight;
		const float gX =  (.5f + xOffset * (1.f - tOffset * amountX)) * filmWidth;
		const float gY =  (.5f + yOffset * (1.f - tOffset * amountY)) * filmHeight;

		const float3 redblue = MAKE_FLOAT3(1.f, 0.f, 1.f);
		const float3 green = MAKE_FLOAT3(0.f, 1.f, 0.f);

		__global float *pixel = &channel_IMAGEPIPELINE[gid * 3];
		float3 newValue = VLOAD3F(pixel);
		newValue += redblue * ColorAberrationPlugin_BilinearSampleImage(channel_IMAGEPIPELINE, filmWidth, filmHeight, rbX, rbY);
		newValue += green * ColorAberrationPlugin_BilinearSampleImage(channel_IMAGEPIPELINE, filmWidth, filmHeight, gX, gY);
		// I added redblue+green luminance so I divide by 2.0 to go back
		// to original luminance
		newValue *= .5f;

		VSTORE3F(newValue, &tmpBuffer[gid * 3]);
	}
}

//------------------------------------------------------------------------------
// ColorAberrationPlugin_ApplySpectral
//------------------------------------------------------------------------------

// The same as ColorAberrationPlugin::ApplySpectral(): each channel is the sum of
// its own channel at the scaled positions of the wavelengths, with the weights
// of the table (4 floats per wavelength: the magnification minus 1 and the
// weights of the 3 channels)
__kernel void ColorAberrationPlugin_ApplySpectral(
		const uint filmWidth, const uint filmHeight,
		__global float *channel_IMAGEPIPELINE,
		__global float *tmpBuffer,
		const float centerX, const float centerY,
		const uint samples, __global const float *spectralTable) {
	const size_t gid = get_global_id(0);
	if (gid >= filmWidth * filmHeight)
		return;

	// Check if the pixel has received any sample
	if (!isinf(channel_IMAGEPIPELINE[gid * 3])) {
		const uint x = gid % filmWidth;
		const uint y = gid / filmWidth;
		const float dx = x + .5f - centerX;
		const float dy = y + .5f - centerY;

		float3 sum = MAKE_FLOAT3(0.f, 0.f, 0.f);
		for (uint k = 0; k < samples; ++k) {
			__global const float *entry = &spectralTable[k * 4];
			const float s = entry[0];
			const float3 weight = VLOAD3F(&entry[1]);

			sum += weight * ColorAberrationPlugin_BilinearSampleImage(channel_IMAGEPIPELINE,
					filmWidth, filmHeight, x + s * dx, y + s * dy);
		}

		VSTORE3F(sum, &tmpBuffer[gid * 3]);
	}
}

//------------------------------------------------------------------------------
// ColorAberrationPlugin_Copy
//------------------------------------------------------------------------------

__kernel void ColorAberrationPlugin_Copy(
		const uint filmWidth, const uint filmHeight,
		__global float *channel_IMAGEPIPELINE,
		__global float *tmpBuffer) {
	const size_t gid = get_global_id(0);
	if (gid >= filmWidth * filmHeight)
		return;

	// Check if the pixel has received any sample
	if (!isinf(channel_IMAGEPIPELINE[gid * 3])) {
		const uint index = gid * 3;

		channel_IMAGEPIPELINE[index] = tmpBuffer[index];
		channel_IMAGEPIPELINE[index + 1] = tmpBuffer[index + 1];
		channel_IMAGEPIPELINE[index + 2] = tmpBuffer[index + 2];
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
