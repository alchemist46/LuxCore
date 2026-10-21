#line 2 "sampleresult_funcs.cl"

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

// Only the radiance groups of the film: the others are never read (the film
// has a buffer for each of its groups only)
OPENCL_FORCE_INLINE void SampleResult_ClearRadiance(__constant const Film* restrict film,
		__global SampleResult *sampleResult) {
	for (uint i = 0; i < film->radianceGroupCount; ++i)
		VSTORE3F(BLACK, sampleResult->radiancePerPixelNormalized[i].c);
}

//------------------------------------------------------------------------------
// The members of the AOV channels are computed only when the film has a channel
// that reads them (Film_AddSample(), VarianceClamping_Clamp()): on the GPU each
// update is a read-modify-write of a large per-task structure.
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE bool SampleResult_HasDirectChannels(__constant const Film* restrict film) {
	return film->hasChannelDirectDiffuse || film->hasChannelDirectDiffuseReflect || film->hasChannelDirectDiffuseTransmit ||
			film->hasChannelDirectGlossy || film->hasChannelDirectGlossyReflect || film->hasChannelDirectGlossyTransmit;
}

OPENCL_FORCE_INLINE bool SampleResult_HasIndirectChannels(__constant const Film* restrict film) {
	return film->hasChannelIndirectDiffuse || film->hasChannelIndirectDiffuseReflect || film->hasChannelIndirectDiffuseTransmit ||
			film->hasChannelIndirectGlossy || film->hasChannelIndirectGlossyReflect || film->hasChannelIndirectGlossyTransmit ||
			film->hasChannelIndirectSpecular || film->hasChannelIndirectSpecularReflect || film->hasChannelIndirectSpecularTransmit;
}

// The shading normal is set with the albedo (see AdvancePaths_HitObject())
OPENCL_FORCE_INLINE bool SampleResult_HasAlbedoChannels(__constant const Film* restrict film) {
	return film->hasChannelAlbedo || film->hasChannelAvgShadingNormal || film->hasChannelShadingNormal;
}

OPENCL_FORCE_INLINE void SampleResult_Init(__constant const Film* restrict film,
		__global SampleResult *sampleResult) {
	// Initialize only Spectrum fields

	SampleResult_ClearRadiance(film, sampleResult);

	if (SampleResult_HasDirectChannels(film)) {
		VSTORE3F(BLACK, sampleResult->directDiffuse.c);
		VSTORE3F(BLACK, sampleResult->directDiffuseReflect.c);
		VSTORE3F(BLACK, sampleResult->directDiffuseTransmit.c);
		VSTORE3F(BLACK, sampleResult->directGlossy.c);
		VSTORE3F(BLACK, sampleResult->directGlossyReflect.c);
		VSTORE3F(BLACK, sampleResult->directGlossyTransmit.c);
	}
	if (film->hasChannelEmission)
		VSTORE3F(BLACK, sampleResult->emission.c);
	if (SampleResult_HasIndirectChannels(film)) {
		VSTORE3F(BLACK, sampleResult->indirectDiffuse.c);
		VSTORE3F(BLACK, sampleResult->indirectDiffuseReflect.c);
		VSTORE3F(BLACK, sampleResult->indirectDiffuseTransmit.c);
		VSTORE3F(BLACK, sampleResult->indirectGlossy.c);
		VSTORE3F(BLACK, sampleResult->indirectGlossyReflect.c);
		VSTORE3F(BLACK, sampleResult->indirectGlossyTransmit.c);
		VSTORE3F(BLACK, sampleResult->indirectSpecular.c);
		VSTORE3F(BLACK, sampleResult->indirectSpecularReflect.c);
		VSTORE3F(BLACK, sampleResult->indirectSpecularTransmit.c);
	}
	sampleResult->rayCount = 0.f;
	if (film->hasChannelIrradiance)
		VSTORE3F(BLACK, sampleResult->irradiance.c);

	sampleResult->firstPathVertexEvent = NONE;
	sampleResult->firstPathVertex = true;
	// sampleResult->lastPathVertex can not be really initialized here without knowing
	// the max. path depth.
	sampleResult->lastPathVertex = true;
}

OPENCL_FORCE_INLINE void SampleResult_AddEmission(__constant const Film* restrict film,
		__global SampleResult *sampleResult, const uint lightID,
		const float3 pathThroughput, const float3 incomingRadiance) {
	const float3 radiance = pathThroughput * incomingRadiance;

	// Avoid out of bound access if the light group doesn't exist. This can happen
	// with RT modes.
	const uint id = min(lightID, film->radianceGroupCount - 1u);
	VADD3F(sampleResult->radiancePerPixelNormalized[id].c, radiance);

	if (sampleResult->firstPathVertex) {
		if (film->hasChannelEmission)
			VADD3F(sampleResult->emission.c, radiance);
	} else {
		if (film->hasChannelIndirectShadowMask)
			sampleResult->indirectShadowMask = 0.f;

		const BSDFEvent firstPathVertexEvent = SampleResult_HasIndirectChannels(film) ?
			sampleResult->firstPathVertexEvent : NONE;
		if ((firstPathVertexEvent & (DIFFUSE | REFLECT)) == (DIFFUSE | REFLECT)) {
			VADD3F(sampleResult->indirectDiffuseReflect.c, radiance);
		} else if ((firstPathVertexEvent & (DIFFUSE | TRANSMIT)) == (DIFFUSE | TRANSMIT)) {
			VADD3F(sampleResult->indirectDiffuseTransmit.c, radiance);
		} else if ((firstPathVertexEvent & (GLOSSY | REFLECT)) == (GLOSSY | REFLECT)) {
			VADD3F(sampleResult->indirectGlossyReflect.c, radiance);
		} else if ((firstPathVertexEvent & (GLOSSY | TRANSMIT)) == (GLOSSY | TRANSMIT)) {
			VADD3F(sampleResult->indirectGlossyTransmit.c, radiance);
		} else if ((firstPathVertexEvent & (SPECULAR | REFLECT)) == (SPECULAR | REFLECT)) {
			VADD3F(sampleResult->indirectSpecularReflect.c, radiance);
		} else if ((firstPathVertexEvent & (SPECULAR | TRANSMIT)) == (SPECULAR | TRANSMIT)) {
			VADD3F(sampleResult->indirectSpecularTransmit.c, radiance);
		}
	}
}

// firstPathVertex and irradiancePathThroughput are the values of the vertex of
// the shadow ray: the sample may already be at the next vertex
OPENCL_FORCE_INLINE void SampleResult_AddDirectLight(__constant const Film* restrict film,
		__global SampleResult *sampleResult, const uint lightID,
		const BSDFEvent bsdfEvent, const float3 pathThroughput, const float3 incomingRadiance,
		const float lightScale, const bool firstPathVertex,
		const float3 irradiancePathThroughput) {
	const float3 radiance = pathThroughput * incomingRadiance;

	// Avoid out of bound access if the light group doesn't exist. This can happen
	// with RT modes.
	const uint id = min(lightID, film->radianceGroupCount - 1u);
	VADD3F(sampleResult->radiancePerPixelNormalized[id].c, radiance);

	if (firstPathVertex) {
		if (film->hasChannelDirectShadowMask)
			sampleResult->directShadowMask = fmax(0.f, sampleResult->directShadowMask - lightScale);

		if (!SampleResult_HasDirectChannels(film)) {
		} else if ((bsdfEvent & (DIFFUSE | REFLECT)) == (DIFFUSE | REFLECT)) {
			VADD3F(sampleResult->directDiffuseReflect.c, radiance);
		} else if ((bsdfEvent & (DIFFUSE | TRANSMIT)) == (DIFFUSE | TRANSMIT)) {
			VADD3F(sampleResult->directDiffuseTransmit.c, radiance);
		} else if ((bsdfEvent & (GLOSSY | REFLECT)) == (GLOSSY | REFLECT)) {
			VADD3F(sampleResult->directGlossyReflect.c, radiance);
		} else if ((bsdfEvent & (GLOSSY | TRANSMIT)) == (GLOSSY | TRANSMIT)) {
			VADD3F(sampleResult->directGlossyTransmit.c, radiance);
		}
	} else {
		if (film->hasChannelIndirectShadowMask)
			sampleResult->indirectShadowMask = fmax(0.f, sampleResult->indirectShadowMask - lightScale);

		const BSDFEvent firstPathVertexEvent = SampleResult_HasIndirectChannels(film) ?
			sampleResult->firstPathVertexEvent : NONE;
		if ((firstPathVertexEvent & (DIFFUSE | REFLECT)) == (DIFFUSE | REFLECT)) {
			VADD3F(sampleResult->indirectDiffuseReflect.c, radiance);
		} else if ((firstPathVertexEvent & (DIFFUSE | TRANSMIT)) == (DIFFUSE | TRANSMIT)) {
			VADD3F(sampleResult->indirectDiffuseTransmit.c, radiance);
		} else if ((firstPathVertexEvent & (GLOSSY | REFLECT)) == (GLOSSY | REFLECT)) {
			VADD3F(sampleResult->indirectGlossyReflect.c, radiance);
		} else if ((firstPathVertexEvent & (GLOSSY | TRANSMIT)) == (GLOSSY | TRANSMIT)) {
			VADD3F(sampleResult->indirectGlossyTransmit.c, radiance);
		} else if ((firstPathVertexEvent & (SPECULAR | REFLECT)) == (SPECULAR | REFLECT)) {
			VADD3F(sampleResult->indirectSpecularReflect.c, radiance);
		} else if ((firstPathVertexEvent & (SPECULAR | TRANSMIT)) == (SPECULAR | TRANSMIT)) {
			VADD3F(sampleResult->indirectSpecularTransmit.c, radiance);
		}

		if (film->hasChannelIrradiance)
			VADD3F(sampleResult->irradiance.c, irradiancePathThroughput * incomingRadiance);
	}
}

OPENCL_FORCE_INLINE float3 SampleResult_GetSpectrum(__constant const Film* restrict film,
		__global SampleResult *sampleResult,
		float3 filmRadianceGroupScale[FILM_MAX_RADIANCE_GROUP_COUNT]) {
	float3 c = BLACK;
	for (uint i = 0; i < film->radianceGroupCount; ++i)
		c += VLOAD3F(sampleResult->radiancePerPixelNormalized[i].c) * filmRadianceGroupScale[i];

	return c;
}

OPENCL_FORCE_INLINE float SampleResult_GetRadianceY(__constant const Film* restrict film,
		__global SampleResult *sampleResult) {
	float y = 0.f;
	
	for (uint i = 0; i < film->radianceGroupCount; ++i)
		y += Spectrum_Y(VLOAD3F(sampleResult->radiancePerPixelNormalized[i].c));

	return y;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
