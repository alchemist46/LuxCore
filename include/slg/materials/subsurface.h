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

#ifndef _SLG_SUBSURFACEMAT_H
#define	_SLG_SUBSURFACEMAT_H

#include "slg/materials/material.h"

namespace slg {

//------------------------------------------------------------------------------
// Random-walk subsurface scattering material
//
// A GGX dielectric reflection (the Schlick Fresnel of ior, energy compensated)
// over a scattering medium. On an eye path the part that enters the surface
// (1 - Ec(mu_eye)) goes into the object along a cosine-weighted direction and
// the path tracer walks it through the medium (subsurface_funcs.cl): color is
// the multiple-scattering albedo, radius * scale the mean free path per
// channel and anisotropy the Henyey-Greenstein g. The walk leaves the object
// at a new point with a Lambertian transmission. Elsewhere (light paths,
// BiDir, inside a Mix) the medium is approximated by a Lambertian base of
// albedo color under the same reflection.
//------------------------------------------------------------------------------

class SubsurfaceMaterial : public Material {
public:
	SubsurfaceMaterial(TextureConstPtr frontTransp, TextureConstPtr backTransp,
			TextureConstPtr emitted, TextureConstPtr bump,
			TextureConstPtr col, TextureConstPtr rad, TextureConstPtr sc,
			TextureConstPtr aniso, TextureConstPtr ior, TextureConstPtr rough);

	virtual MaterialType GetType() const { return SUBSURFACEMAT; }
	virtual BSDFEvent GetEventTypes() const { return DIFFUSE | GLOSSY | REFLECT | TRANSMIT; };

	virtual luxrays::Spectrum Albedo(const HitPoint &hitPoint) const;
	virtual luxrays::Spectrum EvaluateTotal(const HitPoint &hitPoint) const;

	virtual luxrays::Spectrum Evaluate(const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir, const luxrays::Vector &localEyeDir, BSDFEvent *event,
		float *directPdfW = NULL, float *reversePdfW = NULL) const;
	virtual luxrays::Spectrum Sample(const HitPoint &hitPoint,
		const luxrays::Vector &localFixedDir, luxrays::Vector *localSampledDir,
		const float u0, const float u1, const float passThroughEvent,
		float *pdfW, BSDFEvent *event) const;
	virtual void Pdf(const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir, const luxrays::Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const;

	// The medium of the walk at the entry point
	void GetWalkParams(const HitPoint &hitPoint, luxrays::Spectrum *color,
		luxrays::Spectrum *radius, float *anisotropy) const;

	virtual void AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const;
	virtual void UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex);

	virtual luxrays::PropertiesUPtr ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const;

	TextureConstPtr GetColor() const { return color; }
	TextureConstPtr GetRadius() const { return radius; }
	TextureConstPtr GetScale() const { return scale; }
	TextureConstPtr GetAnisotropy() const { return anisotropy; }
	TextureConstPtr GetIOR() const { return ior; }
	TextureConstPtr GetRoughness() const { return roughness; }

private:
	// The GGX alpha and the Schlick F0 of the reflection (F0 < 0: none)
	void GetReflection(const HitPoint &hitPoint, float *alpha, float *F0) const;

	TextureConstPtr color, radius, scale, anisotropy, ior, roughness;
};

}

#endif	/* _SLG_SUBSURFACEMAT_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
