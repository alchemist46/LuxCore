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

#ifndef _SLG_PATHINFO_H
#define	_SLG_PATHINFO_H

#include <ostream>

#include "slg/slg.h"
#include "slg/utils/pathdepthinfo.h"
#include "slg/utils/pathvolumeinfo.h"

namespace slg {

// OpenCL data types
namespace ocl {
#include "slg/utils/pathinfo_types.cl"
}

//------------------------------------------------------------------------------
// PathInfo
//------------------------------------------------------------------------------

class PathInfo {
public:
	PathInfo();
	~PathInfo() { }

	bool IsSpecularPath() const { return isNearlyS; }
	bool IsSpecularPath(const BSDFEvent event, const float glossiness, const float glossinessThreshold) const {
		return isNearlyS && IsNearlySpecular(event, glossiness, glossinessThreshold);
	}

	bool IsSDPath() const { return isNearlySD; }
	bool IsSDSPath() const { return isNearlySDS; }

	bool UseRR(const u_int rrDepth) const;

	// The wavelength of the dispersion (see BSDFEvent DISPERSED) to the hit
	// point of a new vertex
	void SetDispersion(HitPoint &hitPoint) const {
		hitPoint.wavelength = wavelength;
		hitPoint.dispersionTinted = dispersionTinted;
	}

	PathDepthInfo depth;
	PathVolumeInfo volume;

	// Last path vertex information
	BSDFEvent lastBSDFEvent;

	// The wavelength (nm) of the dispersion of all the vertices of the path
	// (0: none, the materials pick one per vertex) and whether the path has
	// its colour (after the first dispersive refraction)
	float wavelength;
	bool dispersionTinted;

	// The wavelength of the random number u (uniform in the visible range,
	// as the colours of the dispersive materials)
	static float Wavelength(const float u) { return 380.f + 400.f * u; }

	static bool IsNearlySpecular(const BSDFEvent event, const float glossiness, const float glossinessThreshold);
	static bool CanBeNearlySpecular(const BSDF &bsdf, const float glossinessThreshold);

protected:
	// Specular, Specular+ Diffuse and Specular+ Diffuse Specular+ paths
	bool isNearlyS, isNearlySD, isNearlySDS;
};

//------------------------------------------------------------------------------
// EyePathInfo
//------------------------------------------------------------------------------

class EyePathInfo : public PathInfo {
public:
	EyePathInfo();
	~EyePathInfo() { }

	// trackEtaScale: update etaScale (the Russian roulette on the path
	// throughput is in use). regularizationRoughness: the floor of the path
	// regularization, a glossy bounce sharper than it doesn't start the
	// regularization (anyNonSpecularBounce)
	void AddVertex(const BSDF &bsdf, const BSDFEvent event, const float pdfW,
			const float glossinessThreshold, const bool trackEtaScale = false,
			const float regularizationRoughness = 0.f);
	
	bool IsCausticPath() const { return isNearlyCaustic && (depth.depth > 1); }
	bool IsCausticPath(const BSDFEvent event, const float glossiness, const float glossinessThreshold) const;

	bool isPassThroughPath;

	// Last path vertex information
	float lastBSDFPdfW;
	float lastGlossiness;
	luxrays::Normal lastShadeN;
	bool lastFromVolume, isTransmittedPath;
	// The product of the eta^2 of the refractions of the path, for the
	// Russian roulette on the path throughput (see
	// PathTracer_RussianRouletteProb()); 1 when the rule is not in use
	float etaScale;
	// A non-specular bounce happened (the path regularization starts after it)
	// A rough bounce (diffuse, or glossy at least as rough as the path
	// regularization floor) happened: the path regularization starts
	bool anyNonSpecularBounce;

private:
	bool isNearlyCaustic;
};

inline std::ostream &operator<<(std::ostream &os, const EyePathInfo &epi) {
	os << "EyePathInfo[" <<
			epi.depth << ", " <<
			epi.volume << ", " <<
			epi.isPassThroughPath << ", " <<
			epi.lastBSDFEvent << ", " <<
			epi.lastBSDFPdfW << ", " <<
			epi.isPassThroughPath << ", " <<
			epi.lastGlossiness << ", " <<
			epi.lastShadeN << ", " <<
			epi.isTransmittedPath << ", " <<
			epi.lastFromVolume << ", " <<
			epi.IsCausticPath() << ", " <<
			epi.IsSpecularPath() << ", " <<
			epi.IsSDPath() << ", " <<
			epi.IsSDSPath() <<
			"]";

	return os;
}

//------------------------------------------------------------------------------
// LightPathInfo
//------------------------------------------------------------------------------

class LightPathInfo : public PathInfo {
public:
	LightPathInfo();
	~LightPathInfo() { }

	void AddVertex(const BSDF &bsdf, const BSDFEvent event, const float glossinessThreshold);

	bool IsCausticPath(const BSDFEvent event, const float glossiness, const float glossinessThreshold) const;

	luxrays::Point lensPoint;
};

inline std::ostream &operator<<(std::ostream &os, const LightPathInfo &lpi) {
	os << "LightPathInfo[" <<
			lpi.depth << ", " <<
			lpi.volume << ", " <<
			lpi.lensPoint << ", " <<
			lpi.lastBSDFEvent << ", " <<
			lpi.IsSpecularPath() << ", " <<
			lpi.IsSDPath() << ", " <<
			lpi.IsSDSPath() <<
			"]";

	return os;
}

}

#endif	/* _SLG_PATHINFO_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
