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

#ifndef _SLG_THINFILMCOATING_H
#define _SLG_THINFILMCOATING_H

#include "luxrays/core/geometry/vector.h"
#include "luxrays/core/color/color.h"
#include "slg/core/workingcolorspace.h"

namespace slg {

// The thin film interference factor (see materialdefs_funcs_thinfilmcoating.cl):
// the Airy reflectance of a film of thickness filmThickness (nm) and IOR
// filmIOR over the substrate, divided by the Fresnel reflectance of the bare
// interface, in the working colour space. iorIncident is the IOR on the side
// of localFixedDir.
extern luxrays::Spectrum CalcFilmColor(const luxrays::Vector &localFixedDir, const float filmThickness, const float filmIOR,
		const float iorIncident, const float iorSubstrate, const WorkingColorSpace &workingColorSpace);
// The same over a conductor of complex IOR (eta, k), Linear Rec.709 per channel
extern luxrays::Spectrum CalcFilmColorConductor(const luxrays::Vector &localFixedDir, const float filmThickness, const float filmIOR,
		const float iorIncident, const luxrays::Spectrum &eta, const luxrays::Spectrum &k,
		const WorkingColorSpace &workingColorSpace);
// The factor of the transmission of an interface of bare Fresnel reflectance
// bareR with the film factor filmColor: a lossless film transmits what it does
// not reflect, (1 - bareR filmColor) / (1 - bareR)
inline luxrays::Spectrum FilmTransmissionScale(const float bareR, const luxrays::Spectrum &filmColor) {
	if (!(bareR < 1.f))
		return luxrays::Spectrum(1.f);
	return ((luxrays::Spectrum(1.f) - filmColor * bareR) / (1.f - bareR)).Clamp(0.f);
}

// The complex IOR of a conductor of normal reflectance f0 and edge tint
// (Gulbrandsen 2014), Linear Rec.709 per channel
extern void FilmConductorIOR(const luxrays::Spectrum &f0, const luxrays::Spectrum &edgeTint,
		luxrays::Spectrum *eta, luxrays::Spectrum *k);

}

#endif /* _SLG_THINFILMCOATING_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
