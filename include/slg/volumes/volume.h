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

#ifndef _SLG_VOLUME_H
#define	_SLG_VOLUME_H

#include "luxrays/luxrays.h"
#include "luxrays/usings.h"
#include "slg/materials/material.h"
#include "slg/textures/texture.h"
#include "slg/usings.h"
#include <functional>
#include <typeinfo>

namespace slg {

// OpenCL data types
namespace ocl {
#include "slg/volumes/volume_types.cl"
}

class BSDF;

class Volume : public Material {
public:
	Volume(TextureConstRef ior, TextureConstPtr emission) :
		Material(nullptr, nullptr, nullptr, nullptr),
		iorTex(ior),
		volumeEmissionTex(emission),
		volumeLightID(0),
		priority(0)
	{ }
	virtual ~Volume() { }

	void SetVolumeLightID(const u_int id) { volumeLightID = id; }
	u_int GetVolumeLightID() const { return volumeLightID; }
	void SetPriority(const int p) { priority = p; }
	int GetPriority() const { return priority; }

	TextureConstRef GetIORTexture() const { return iorTex; }
	bool HasIORTexture() const { return true; }
	TextureConstRef GetVolumeEmissionTexture() const { return *volumeEmissionTex; }
	bool HasVolumeEmissionTexture() const { return bool(volumeEmissionTex); }

	float GetIOR(const HitPoint &hitPoint) const { return GetIORTexture().GetFloatValue(hitPoint); }

	// Returns the ray t value of the scatter event. If (t <= 0.0) there was
	// no scattering. In any case, it applies transmittance to connectionThroughput
	// too. With transmittanceOnly (shadow rays), there is no scattering: the
	// transmittance of the whole segment is applied (exact for the homogeneous
	// volumes, per step for the heterogeneous ones). pathThroughput is the
	// throughput of the path before the ray (it steers the choice of the
	// colour channel the scatter distance is sampled from).
	// surfacePassThroughEvent: when the segment had no scatter event, the
	// pass-through event for the surface at its end. A homogeneous volume
	// decides with u, so u is confined to the no-scatter part of [0, 1): it
	// is remapped to [0, 1) again, otherwise the surface's own choices (the
	// glass lobe, the Mix component) would be biased. The others leave it.
	virtual float Scatter(const luxrays::Ray &ray, const float u, const bool scatteredStart,
		const bool transmittanceOnly, const luxrays::Spectrum &pathThroughput,
		luxrays::Spectrum *connectionThroughput, luxrays::Spectrum *connectionEmission,
		float *surfacePassThroughEvent) const = 0;

	virtual void AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexsreferencedTexs) const;
	virtual void UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex);

	virtual luxrays::PropertiesUPtr ToProperties() const;

	friend class SchlickScatter;
	friend class BSDF;

protected:
	virtual luxrays::Spectrum SigmaA(const HitPoint &hitPoint) const = 0;
	virtual luxrays::Spectrum SigmaS(const HitPoint &hitPoint) const = 0;
	luxrays::Spectrum SigmaT(const HitPoint &hitPoint) const {
		return SigmaA(hitPoint) + SigmaS(hitPoint);
	}
	luxrays::Spectrum Emission(const HitPoint &hitPoint) const {
		return (volumeEmissionTex) ? volumeEmissionTex->GetSpectrumValue(hitPoint).Clamp() : luxrays::Spectrum(0.f);
	}

	std::reference_wrapper<const Texture> iorTex;
	// This is a different kind of emission texture from the one in
	// Material class (i.e. is not sampled by direct light).
	TextureConstPtr volumeEmissionTex;
	u_int volumeLightID;
	int priority;
};

// An utility class
class SchlickScatter {
public:
	SchlickScatter(VolumeConstRef volume, TextureConstRef g);

	luxrays::Spectrum Albedo(const HitPoint &hitPoint) const;

	luxrays::Spectrum Evaluate(const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir, const luxrays::Vector &localEyeDir, BSDFEvent *event,
		float *directPdfW = NULL, float *reversePdfW = NULL) const;
	luxrays::Spectrum Sample(const HitPoint &hitPoint,
		const luxrays::Vector &localFixedDir, luxrays::Vector *localSampledDir,
		const float u0, const float u1, const float passThroughEvent,
		float *pdfW, BSDFEvent *event) const;
	void Pdf(const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir, const luxrays::Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const;

	TextureConstRef GetG() const { return g; }
	void SetG(TextureConstRef tex) { g = tex; }

	VolumeConstRef volume;

private:
	luxrays::Spectrum GetColor(const HitPoint &hitPoint) const;
	std::reference_wrapper<const Texture> g;
};

// Some utilities
template <typename T>
void updvol(T volume, const MaterialConstRef oldMat, MaterialRef newMat);

template<>
inline void updvol(VolumeConstPtr volume, MaterialConstRef oldMat, MaterialRef newMat) {

	// Check
	try {
		auto& oldVol = dynamic_cast<const Volume &>(oldMat);
		if (volume != &oldVol) return;  // Old material was not this volume
	} catch (std::bad_cast&) {
		return;  // Old material was not a volume
	}

	// Replace
	auto& newVol = dynamic_cast<Volume &>(newMat);
	volume = &newVol;
}

template<>
inline void updvol(
	std::reference_wrapper<const Volume> volume,
	MaterialConstRef oldMat,
	MaterialRef newMat
) {
	auto& oldVol = dynamic_cast<const Volume &>(oldMat);
	auto& newVol = dynamic_cast<const Volume &>(newMat);

	if (std::addressof(volume.get()) == std::addressof(oldVol)) {
		volume = std::reference_wrapper(newVol);
	}
}

}  // namespace slg

#endif	/* _SLG_VOLUME_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
