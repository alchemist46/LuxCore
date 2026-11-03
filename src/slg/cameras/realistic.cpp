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

#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>

#include <boost/algorithm/string.hpp>

#include "luxrays/core/epsilon.h"
#include "luxrays/utils/utils.h"
#include "slg/cameras/realistic.h"
#include "slg/core/sdl.h"
#include "slg/scene/scene.h"

// The lens model shared with the GPU (camera_funcs.cl), compiled through the
// OpenCL emulation (see luxrays/devices/cpudevice_oclemul.h)
namespace slg { namespace ocl { namespace {
#include "luxrays/devices/cpudevice_oclemul.h"
#include "slg/cameras/lens_funcs.cl"
} } }

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// The built-in lenses
//
// They are lens designs: realistic.focallength scales them to any focal
// length, which keeps their f-number, their field of view in degrees, their
// distortion and their vignetting. The tables are at a usual focal length of
// each design, the one used with a focal length of 0. A design covers a frame
// from the focal length that puts the corners within its field (the
// triplet's 20 degrees cover a 36x24 mm frame from 60 mm): below it, the
// corners are softer and darker, black for the narrow designs.
//------------------------------------------------------------------------------

// A double-Gauss 50 mm f/2: US patent 2,673,491 (Tronnier, expired), as in W.
// Smith, "Modern Lens Design", p. 312, scaled to 50 mm. The source gives only
// the nd of the glasses: their Abbe numbers are estimates from their types
// (high index crowns and flints). The focus sets the distance to the film.
static const RealisticCamera::LensSurface DGAUSS50[] = {
	{ 29.475f, 3.76f, 1.67f, 47.2f, 12.6f },
	{ 84.83f, .12f, 1.f, 0.f, 12.6f },
	{ 19.275f, 4.025f, 1.67f, 47.2f, 11.5f },
	{ 40.77f, 3.275f, 1.699f, 30.1f, 11.5f },
	{ 12.75f, 5.705f, 1.f, 0.f, 9.f },
	{ 0.f, 4.5f, 1.f, 0.f, 8.55f },
	{ -14.495f, 1.18f, 1.603f, 38.f, 8.5f },
	{ 40.77f, 6.065f, 1.658f, 50.9f, 10.f },
	{ -20.385f, .19f, 1.f, 0.f, 10.f },
	{ 437.065f, 3.22f, 1.717f, 47.9f, 10.f },
	{ -39.73f, 36.f, 1.f, 0.f, 10.f }
};

// The other lenses are from Optiland's optiland/samples/objectives.py (MIT
// license), designs of the book of M. Laikin, "Lens Design" (4th ed., CRC
// Press, 2007), with the nd and Vd of their catalog glasses (Schott, Hikari),
// scaled to their focal length. The books give no clear apertures: they pass
// the whole beam of the f-number over the field of the design (+1%), the
// lens vignettes beyond. Checked with a trace of each prescription: the focal
// length, the image height of the field, the spot sizes and the color
// correction of the design.

// A Cooke triplet 50 mm f/5, for a field of 20 degrees: Zemax's "Cooke 40
// degree field" sample (SK16, F2, SK16), the stop 1 mm behind the flint
static const RealisticCamera::LensSurface TRIPLET50[] = {
	{ 22.0041f, 3.25755f, 1.62041f, 60.2809f, 9.72338f },
	{ -435.572f, 6.00496f, 1.f, 0.f, 9.33334f },
	{ -22.2037f, .999538f, 1.62004f, 36.3665f, 4.70458f },
	{ 20.2832f, .999568f, 1.f, 0.f, 4.12948f },
	{ 0.f, 3.74879f, 1.f, 0.f, 3.88008f },
	{ 79.6492f, 2.95081f, 1.62041f, 60.2809f, 6.55369f },
	{ -18.3874f, 0.f, 1.f, 0.f, 6.97734f }
};

// A Tessar 50 mm f/4.5, for a field of 20.5 degrees (N-SK15, F2, K10 and
// N-SK15): M. Laikin, "Lens Design", 4th ed., p. 63
static const RealisticCamera::LensSurface TESSAR50[] = {
	{ 16.6705f, 3.49069f, 1.62296f, 58.0192f, 8.19631f },
	{ -124.762f, 2.56893f, 1.f, 0.f, 7.68282f },
	{ -26.1608f, 1.12563f, 1.62004f, 36.3665f, 5.41917f },
	{ 15.1622f, .886743f, 1.f, 0.f, 4.5864f },
	{ 0.f, 1.91857f, 1.f, 0.f, 4.47092f },
	{ -94.0585f, 1.12563f, 1.50137f, 56.4092f, 5.61728f },
	{ 16.2715f, 4.2386f, 1.62296f, 58.0192f, 6.71387f },
	{ -19.0331f, 0.f, 1.f, 0.f, 7.1745f }
};

// A Heliar 120 mm f/5, for a field of 10 degrees (N-SK16 and LLF6): Laikin
// p. 63
static const RealisticCamera::LensSurface HELIAR120[] = {
	{ 50.4906f, 10.7977f, 1.62041f, 60.3236f, 16.6322f },
	{ -43.4212f, 3.59645f, 1.53172f, 48.8629f, 15.3496f },
	{ 350.014f, 9.09846f, 1.f, 0.f, 13.4712f },
	{ -38.0332f, 2.39843f, 1.53172f, 48.8629f, 10.2208f },
	{ 38.4745f, 1.51581f, 1.f, 0.f, 9.46394f },
	{ 0.f, 3.15274f, 1.f, 0.f, 9.33425f },
	{ 516.515f, 2.99804f, 1.53172f, 48.8629f, 10.63f },
	{ 29.3736f, 9.96309f, 1.62041f, 60.3236f, 11.8152f },
	{ -39.0657f, 0.f, 1.f, 0.f, 12.8019f }
};

// A Petzval 85 mm f/1.4 (N-LAK12, SF4, N-LAK12, LF5): Laikin p. 75, a
// design for a field of 7 degrees, here twice that in the corners of the frame:
// the curved field and the vignetting of the swirly bokeh
static const RealisticCamera::LensSurface PETZVAL85[] = {
	{ 56.6164f, 17.0724f, 1.6779f, 55.1996f, 31.6292f },
	{ -114.893f, 2.56086f, 1.f, 0.f, 30.4337f },
	{ -85.4816f, 6.82897f, 1.7552f, 27.5795f, 27.6459f },
	{ 232.607f, 4.2681f, 1.f, 0.f, 24.5512f },
	{ 0.f, 38.4087f, 1.f, 0.f, 23.2366f },
	{ 50.0392f, 12.8043f, 1.6779f, 55.1996f, 20.3882f },
	{ -105.004f, 9.47946f, 1.f, 0.f, 18.8793f },
	{ -36.7697f, 3.41448f, 1.58144f, 40.8513f, 13.9927f },
	{ 128.21f, 0.f, 1.f, 0.f, 13.073f }
};

// A telephoto 200 mm f/5.6, for a field of 10 degrees (N-BK7, N-BASF2,
// N-ZK7, N-LAF33, SF1): Laikin p. 91
static const RealisticCamera::LensSurface TELEPHOTO200[] = {
	{ 34.3515f, 9.56275f, 1.5168f, 64.1673f, 18.4076f },
	{ -107.594f, 3.59953f, 1.66446f, 36.0006f, 17.4632f },
	{ 121.256f, 1.92375f, 1.f, 0.f, 15.8151f },
	{ 0.f, 41.3826f, 1.f, 0.f, 15.321f },
	{ -22.857f, 3.59953f, 1.50847f, 61.1878f, 15.6194f },
	{ -29.6881f, 4.01947f, 1.78582f, 44.0519f, 16.91f },
	{ -45.726f, .623918f, 1.f, 0.f, 18.7304f },
	{ -681.463f, 3.17159f, 1.71736f, 29.5128f, 19.8976f },
	{ -110.766f, 0.f, 1.f, 0.f, 20.1718f }
};

// A retrofocus wide angle 27 mm f/4 for a field of 50 degrees, the corners of
// a 36x24 mm frame, with a barrel distortion of 33% there (N-SK4 and SF1):
// Laikin p. 108
static const RealisticCamera::LensSurface RETROFOCUS27[] = {
	{ 329.565f, 10.2852f, 1.61272f, 58.6343f, 59.4533f },
	{ 48.7763f, 39.5485f, 1.f, 0.f, 41.2343f },
	{ 68.8982f, 14.7201f, 1.71736f, 29.5128f, 33.0155f },
	{ -309.201f, 8.78764f, 1.61272f, 58.6343f, 31.0064f },
	{ 38.7174f, 17.0569f, 1.f, 0.f, 23.0263f },
	{ -72.7695f, 13.7122f, 1.71736f, 29.5128f, 21.0808f },
	{ -63.8955f, 15.8926f, 1.61272f, 58.6343f, 20.6645f },
	{ -91.6654f, 40.4906f, 1.f, 0.f, 19.7048f },
	{ 0.f, 4.11406f, 1.f, 0.f, 9.44638f },
	{ 562.816f, 5.75969f, 1.61272f, 58.6343f, 11.0689f },
	{ -141.026f, 2.48901f, 1.f, 0.f, 12.1367f },
	{ 823.87f, 13.7081f, 1.61272f, 58.6343f, 13.0233f },
	{ -38.088f, 17.9949f, 1.71736f, 29.5128f, 14.8047f },
	{ -132.609f, 11.0216f, 1.f, 0.f, 18.3712f },
	{ 427.233f, 5.75969f, 1.71736f, 29.5128f, 21.0695f },
	{ 99.8565f, 10.7336f, 1.61272f, 58.6343f, 21.8347f },
	{ -159.741f, .617109f, 1.f, 0.f, 22.6585f },
	{ 138.438f, 13.0292f, 1.61272f, 58.6343f, 23.1026f },
	{ -607.018f, 0.f, 1.f, 0.f, 23.1259f }
};

// A fisheye 10 mm f/1.8 (N-BK7, N-SK5, SF1, N-LAK9, N-PSK3): Laikin p. 114, a
// field of 170 degrees in the book, 164 here (the chief ray of a wider field
// misses a surface). Its round image, 23.5 mm across, fills the height of a
// 36x24 mm frame; it vignettes beyond 55 degrees (half the beam at 82)
static const RealisticCamera::LensSurface FISHEYE10[] = {
	{ 295.118f, 8.75869f, 1.5168f, 64.1673f, 97.9816f },
	{ 58.9456f, 44.7786f, 1.f, 0.f, 55.3053f },
	{ -461.816f, 9.45607f, 1.58913f, 61.2668f, 49.6546f },
	{ 49.0263f, 31.2211f, 1.f, 0.f, 36.6615f },
	{ 81.1203f, 22.0333f, 1.71736f, 29.5128f, 33.1316f },
	{ -81.5007f, 8.90012f, 1.691f, 54.7084f, 30.5751f },
	{ 54.6492f, 123.246f, 1.f, 0.f, 24.663f },
	{ 67.7628f, 12.1919f, 1.691f, 54.7084f, 22.5275f },
	{ 26.9296f, 29.4899f, 1.55232f, 63.4563f, 20.0075f },
	{ -43.1302f, .955849f, 1.f, 0.f, 18.8286f },
	{ -42.9352f, 3.90142f, 1.71736f, 29.5128f, 18.2827f },
	{ -105.07f, 5.85213f, 1.f, 0.f, 18.125f },
	{ 0.f, 5.85213f, 1.f, 0.f, 16.6898f },
	{ 50.4942f, 23.5548f, 1.55232f, 63.4563f, 17.8094f },
	{ -41.1746f, 2.66272f, 1.f, 0.f, 16.6585f },
	{ -36.0199f, 5.24741f, 1.71736f, 29.5128f, 15.5868f },
	{ -74.5416f, 0.f, 1.f, 0.f, 15.577f }
};

namespace {

struct LensPreset {
	const char *name;
	const RealisticCamera::LensSurface *surfaces;
	u_int count;
};

#define LENS_PRESET(name, table) { name, table, sizeof(table) / sizeof(table[0]) }
const LensPreset LENS_PRESETS[] = {
	LENS_PRESET("dgauss", DGAUSS50),
	LENS_PRESET("triplet", TRIPLET50),
	LENS_PRESET("tessar", TESSAR50),
	LENS_PRESET("heliar", HELIAR120),
	LENS_PRESET("petzval", PETZVAL85),
	LENS_PRESET("telephoto", TELEPHOTO200),
	LENS_PRESET("retrofocus", RETROFOCUS27),
	LENS_PRESET("fisheye", FISHEYE10)
};
#undef LENS_PRESET

}

vector<RealisticCamera::LensSurface> RealisticCamera::GetLensPreset(const string &name) {
	for (const LensPreset &preset : LENS_PRESETS) {
		if (name == preset.name)
			return vector<LensSurface>(preset.surfaces, preset.surfaces + preset.count);
	}

	throw runtime_error("Unknown RealisticCamera lens preset: " + name);
}

vector<RealisticCamera::LensSurface> RealisticCamera::ReadLensFile(const string &fileName) {
	ifstream file(fileName);
	if (!file.is_open())
		throw runtime_error("Unable to open the RealisticCamera lens file: " + fileName);

	vector<LensSurface> lens;
	string line;
	u_int lineNumber = 0;
	while (getline(file, line)) {
		++lineNumber;

		// Without the comment
		const size_t comment = line.find('#');
		if (comment != string::npos)
			line.erase(comment);
		boost::trim(line);
		if (line.empty())
			continue;

		istringstream ss(line);
		ss.imbue(locale::classic());
		vector<float> values;
		float v;
		while (ss >> v)
			values.push_back(v);
		if (!ss.eof())
			throw runtime_error("Syntax error in the RealisticCamera lens file " + fileName + " at line " + ToString(lineNumber));

		LensSurface surface;
		if (values.size() == 5) {
			// radius thickness nd vd semi-aperture
			surface.radius = values[0];
			surface.thickness = values[1];
			surface.ior = values[2];
			surface.abbe = values[3];
			surface.semiAperture = values[4];
		} else if (values.size() == 4) {
			// pbrt: radius thickness eta aperture-diameter (eta 0 for the stop)
			surface.radius = values[0];
			surface.thickness = values[1];
			surface.ior = values[2];
			surface.abbe = 0.f;
			surface.semiAperture = values[3] * .5f;
		} else
			throw runtime_error("The RealisticCamera lens file " + fileName + " needs 5 values (or 4) at line " + ToString(lineNumber));

		if ((surface.semiAperture <= 0.f) || (surface.thickness < 0.f))
			throw runtime_error("Wrong semi-aperture or thickness in the RealisticCamera lens file " + fileName + " at line " + ToString(lineNumber));
		lens.push_back(surface);
	}

	return lens;
}

//------------------------------------------------------------------------------
// RealisticCamera
//------------------------------------------------------------------------------

RealisticCamera::RealisticCamera(const Point &o, const Point &t,
		const Vector &u, const float *region) :
		ProjectiveCamera(REALISTIC, region, o, t, u),
		focalLength(0.f), sensorWidth(36.f), fStop(0.f),
		stopBlades(0), stopRotation(0.f), stopRoundness(0.f), stopObstruction(0.f),
		dispersion(false), filmZ(0.f), effectiveFocalLength(0.f), effectiveFStop(0.f) {
	memset(&lensCamera, 0, sizeof(lensCamera));
}

RealisticCamera::~RealisticCamera() {
}

namespace {

// Traces through count surfaces from the first one (see Lens_Trace()), at the
// d line, with a round stop
bool TraceLens(const vector<float> &surfaces, const u_int first, const u_int count,
		const bool toFront, const float stopRadius, Point &o, Vector &d) {
	slg::ocl::float3 rayOrig = slg::ocl::make_float3(o.x, o.y, o.z);
	slg::ocl::float3 rayDir = slg::ocl::make_float3(d.x, d.y, d.z);
	if (!slg::ocl::Lens_Trace(&surfaces[first * LENS_SURFACE_SIZE], count, toFront,
			stopRadius, 0, 0.f, 0.f, 0.f, LENS_WAVELENGTH_D, 0, &rayOrig, &rayDir))
		return false;

	o = Point(rayOrig.x, rayOrig.y, rayOrig.z);
	d = Vector(rayDir.x, rayDir.y, rayDir.z);

	return true;
}

}

// The last lens set-ups, by the values they are for: a camera edit parses a
// new camera (moving the camera in the viewport does too), which would set up
// the same lens again
namespace {
struct LensSetUp {
	string key;
	slg::ocl::RealisticCamera lensCamera;
	vector<float> lensData;
	float filmZ, effectiveFocalLength, effectiveFStop;
};

const size_t LENS_SETUP_CACHE_SIZE = 8;
mutex lensSetUpCacheMutex;
deque<LensSetUp> lensSetUpCache;
}

void RealisticCamera::UpdateLens() {
	// The values the lens is set up for
	stringstream key;
	key.imbue(locale::classic());
	key.precision(9);
	key << lensFileName << "|" << lensPreset << "|";
	for (const LensSurface &s : lens)
		key << s.radius << " " << s.thickness << " " << s.ior << " " << s.abbe << " " << s.semiAperture << ";";
	key << "|" << focalLength << "|" << sensorWidth << "|" << fStop << "|" << focalDistance <<
			"|" << stopBlades << "|" << stopRotation << "|" << stopRoundness << "|" << stopObstruction <<
			"|" << dispersion << "|" << screenWindow[0] << " " << screenWindow[1] << " " <<
			screenWindow[2] << " " << screenWindow[3];
	if (key.str() == lensKey)
		return;

	{
		lock_guard<mutex> lock(lensSetUpCacheMutex);
		for (const LensSetUp &setUp : lensSetUpCache) {
			if (setUp.key == key.str()) {
				lensCamera = setUp.lensCamera;
				lensData = setUp.lensData;
				filmZ = setUp.filmZ;
				effectiveFocalLength = setUp.effectiveFocalLength;
				effectiveFStop = setUp.effectiveFStop;
				lensKey = setUp.key;
				return;
			}
		}
	}

	const double tStart = WallClockTime();

	const u_int n = lens.size();
	if (n == 0)
		throw runtime_error("RealisticCamera has no lens");
	u_int stopCount = 0;
	u_int stopRow = 0;
	for (u_int j = 0; j < n; ++j) {
		if (lens[j].radius == 0.f) {
			++stopCount;
			stopRow = j;
		}
	}
	if (stopCount != 1)
		throw runtime_error("The lens of RealisticCamera needs one aperture stop (a surface of radius 0), not " + ToString(stopCount));
	// The surfaces are from the film to the front
	const u_int stopIndex = n - 1 - stopRow;

	// The surfaces in meters from the film to the front, scaled by scale, the
	// front vertex at z = 0
	auto buildSurfaces = [&](const float scale) {
		vector<float> surfaces(n * LENS_SURFACE_SIZE);
		vector<float> zs(n);
		float z = 0.f;
		for (u_int j = 0; j < n; ++j) {
			zs[j] = z;
			z -= lens[j].thickness * 1e-3f * scale;
		}

		const float lD = LENS_WAVELENGTH_D * 1e-3f;
		const float lF = LENS_WAVELENGTH_F * 1e-3f;
		const float lC = LENS_WAVELENGTH_C * 1e-3f;
		for (u_int i = 0; i < n; ++i) {
			const u_int j = n - 1 - i;

			// The medium on the object side is the one after the previous
			// surface (1 or 0, pbrt's stop, for air)
			float ior = (j > 0) ? lens[j - 1].ior : 1.f;
			float abbe = (j > 0) ? lens[j - 1].abbe : 0.f;
			if (ior <= 1.f) {
				ior = 1.f;
				abbe = 0.f;
			}

			// Cauchy's n = A + B / lambda^2 with n_d = ior and (n_d - 1) /
			// (n_F - n_C) = abbe
			float cauchyA = ior;
			float cauchyB = 0.f;
			if (abbe > 0.f) {
				cauchyB = (ior - 1.f) / (abbe * (1.f / (lF * lF) - 1.f / (lC * lC)));
				cauchyA = ior - cauchyB / (lD * lD);
			}

			float *surface = &surfaces[i * LENS_SURFACE_SIZE];
			surface[0] = lens[j].radius * 1e-3f * scale;
			surface[1] = zs[j];
			surface[2] = lens[j].semiAperture * 1e-3f * scale;
			surface[3] = ior;
			surface[4] = cauchyA;
			surface[5] = cauchyB;
		}

		return surfaces;
	};

	//--------------------------------------------------------------------------
	// The focal length: a ray parallel to the axis at 1% of the front
	// semi-aperture
	//--------------------------------------------------------------------------

	vector<float> surfaces = buildSurfaces(1.f);
	float frontZ = surfaces[(n - 1) * LENS_SURFACE_SIZE + 1];
	float maxStopRadius = surfaces[stopIndex * LENS_SURFACE_SIZE + 2];

	float h = .01f * surfaces[(n - 1) * LENS_SURFACE_SIZE + 2];
	Point o(0.f, h, frontZ + 1.f);
	Vector d(0.f, 0.f, -1.f);
	if (!TraceLens(surfaces, 0, n, false, maxStopRadius, o, d) || (d.y >= 0.f))
		throw runtime_error("The lens of RealisticCamera does not bring a ray parallel to its axis to a focus");
	const float nativeFocalLength = h * d.z / d.y;

	const float scale = (focalLength > 0.f) ? (focalLength * 1e-3f / nativeFocalLength) : 1.f;
	if (scale != 1.f) {
		surfaces = buildSurfaces(scale);
		frontZ = surfaces[(n - 1) * LENS_SURFACE_SIZE + 1];
		maxStopRadius = surfaces[stopIndex * LENS_SURFACE_SIZE + 2];
	}
	const float f = nativeFocalLength * scale;
	effectiveFocalLength = f * 1e3f;

	//--------------------------------------------------------------------------
	// The entrance pupil: the image of the stop by the surfaces in front of
	// it, its position and its magnification
	//--------------------------------------------------------------------------

	const float stopZ = surfaces[stopIndex * LENS_SURFACE_SIZE + 1];
	const u_int frontFirst = stopIndex + 1;
	const u_int frontCount = n - frontFirst;
	float pupilZ = stopZ;
	float pupilMagnification = 1.f;
	if (frontCount > 0) {
		o = Point(0.f, 0.f, stopZ);
		d = Normalize(Vector(0.f, .01f, 1.f));
		if (!TraceLens(surfaces, frontFirst, frontCount, true, maxStopRadius, o, d) || (d.y == 0.f))
			throw runtime_error("RealisticCamera can not find the entrance pupil of the lens");
		pupilZ = o.z - o.y * d.z / d.y;

		h = .01f * maxStopRadius;
		o = Point(0.f, h, frontZ + 1.f);
		d = Vector(0.f, 0.f, -1.f);
		if (!TraceLens(surfaces, frontFirst, frontCount, false, maxStopRadius, o, d) || (d.z >= 0.f))
			throw runtime_error("RealisticCamera can not find the entrance pupil of the lens");
		const float stopHeight = o.y + d.y * (stopZ - o.z) / d.z;
		if (stopHeight <= 0.f)
			throw runtime_error("RealisticCamera can not find the entrance pupil of the lens");
		pupilMagnification = h / stopHeight;
	}

	// The entrance pupil at the camera origin
	for (u_int i = 0; i < n; ++i)
		surfaces[i * LENS_SURFACE_SIZE + 1] -= pupilZ;
	frontZ -= pupilZ;
	const float rearZ = surfaces[1];

	//--------------------------------------------------------------------------
	// The stop radius of the f-number
	//--------------------------------------------------------------------------

	const float minFStop = f / (2.f * maxStopRadius * pupilMagnification);
	float stopRadius = maxStopRadius;
	if (fStop > 0.f) {
		if (fStop < minFStop * (1.f - 1e-4f)) {
			SLG_LOG("WARNING: the RealisticCamera f-number " << fStop << " is below the widest of the lens, f/" << minFStop);
		} else
			stopRadius = f / (2.f * fStop) / pupilMagnification;
	}
	effectiveFStop = f / (2.f * stopRadius * pupilMagnification);

	//--------------------------------------------------------------------------
	// The focus: the film where a ray through 0.7 of the entrance pupil from
	// the point of the axis at the focus distance crosses the axis (the zone
	// of the best focus with the spherical aberration)
	//--------------------------------------------------------------------------

	const float pupilRadius = stopRadius * pupilMagnification;
	const bool focusAtInfinity = (focalDistance <= 0.f) || (focalDistance > 1e4f * f);
	if (!focusAtInfinity && (focalDistance <= frontZ))
		throw runtime_error("RealisticCamera can not focus inside its lens: the focus distance " + ToString(focalDistance) +
				" m is closer than the front of the lens, " + ToString(frontZ) + " m");
	bool focused = false;
	for (float zone = .7f; (zone > .01f) && !focused; zone *= .5f) {
		const float y = zone * pupilRadius;
		if (focusAtInfinity) {
			o = Point(0.f, y, frontZ + 1e-3f);
			d = Vector(0.f, 0.f, -1.f);
		} else {
			// The ray from the point of the axis at the focus distance to the
			// height y of the entrance pupil (at z = 0), started just in front
			// of the lens: from far away, the float intersections with the
			// spheres would lose their precision
			const float startZ = Min(frontZ + 1e-3f, focalDistance);
			o = Point(0.f, y * (focalDistance - startZ) / focalDistance, startZ);
			d = Normalize(Vector(0.f, y, -focalDistance));
		}

		if (TraceLens(surfaces, 0, n, false, stopRadius, o, d) && (d.y < 0.f) && (d.z < 0.f)) {
			filmZ = o.z - o.y * d.z / d.y;
			focused = true;
		}
	}
	if (!focused)
		throw runtime_error("RealisticCamera can not focus at " + ToString(focalDistance) + " m");
	if (filmZ >= rearZ)
		throw runtime_error("RealisticCamera can not focus at " + ToString(focalDistance) + " m: the film would be inside the lens");

	//--------------------------------------------------------------------------
	// The sensor
	//--------------------------------------------------------------------------

	// The screen window units 1 at half the sensor width, as the field of view
	// of the perspective camera (the longer side of the frame by default)
	const float sensorScale = sensorWidth * 1e-3f * .5f;
	const float halfWidth = Max(fabsf(screenWindow[0]), fabsf(screenWindow[1]));
	const float halfHeight = Max(fabsf(screenWindow[2]), fabsf(screenWindow[3]));
	const float maxFilmRadius = sensorScale * sqrtf(Sqr(halfWidth) + Sqr(halfHeight));

	//--------------------------------------------------------------------------
	// The camera
	//--------------------------------------------------------------------------

	lensData = surfaces;
	lensData.resize(n * LENS_SURFACE_SIZE + LENS_PUPIL_BUCKETS * 4, 0.f);

	memset(&lensCamera, 0, sizeof(lensCamera));
	lensCamera.rearZ = rearZ;
	lensCamera.maxFilmRadius = maxFilmRadius;
	lensCamera.focalLength = f;
	lensCamera.weightScale = 1.f;
	lensCamera.stopRadius = stopRadius;
	lensCamera.stopRotation = Radians(stopRotation);
	lensCamera.stopRoundness = stopRoundness;
	lensCamera.stopObstruction = stopObstruction;
	lensCamera.stopBlades = stopBlades;
	lensCamera.surfacesOffset = 0;
	lensCamera.surfaceCount = n;
	lensCamera.pupilBoundsOffset = n * LENS_SURFACE_SIZE;
	lensCamera.dispersion = dispersion;

	//--------------------------------------------------------------------------
	// The exit pupil bounds: the bounds of the points of the plane of the rear
	// vertex the rays of the film points of each ring go through the lens
	// from, on a grid over 1.5 times the rear semi-aperture
	//--------------------------------------------------------------------------

	const float rearRadius = 1.5f * surfaces[2];
	const int grid = 96;
	const float step = 2.f * rearRadius / grid;
	vector<float> wavelengths;
	if (dispersion) {
		wavelengths.push_back(400.f);
		wavelengths.push_back(LENS_WAVELENGTH_D);
		wavelengths.push_back(700.f);
	} else
		wavelengths.push_back(LENS_WAVELENGTH_D);

	const float *surfacesData = &lensData[0];
	#pragma omp parallel for
	for (int bucket = 0; bucket < LENS_PUPIL_BUCKETS; ++bucket) {
		float xMin = numeric_limits<float>::infinity();
		float xMax = -numeric_limits<float>::infinity();
		float yMin = numeric_limits<float>::infinity();
		float yMax = -numeric_limits<float>::infinity();

		for (u_int k = 0; k < 3; ++k) {
			const float r = maxFilmRadius * (bucket + .5f * k) / LENS_PUPIL_BUCKETS;
			const ocl::float3 filmPoint = ocl::make_float3(r, 0.f, filmZ);

			for (int gy = 0; gy < grid; ++gy) {
				const float py = -rearRadius + (gy + .5f) * step;
				for (int gx = 0; gx < grid; ++gx) {
					const float px = -rearRadius + (gx + .5f) * step;

					for (const float wavelength : wavelengths) {
						ocl::float3 rayOrig = filmPoint;
						ocl::float3 rayDir = ocl::normalize(ocl::make_float3(px - r, py, rearZ - filmZ));
						if (ocl::Lens_Trace(surfacesData, n, true, stopRadius,
								stopBlades, lensCamera.stopRotation, stopRoundness, stopObstruction,
								wavelength, dispersion, &rayOrig, &rayDir)) {
							xMin = Min(xMin, px);
							xMax = Max(xMax, px);
							yMin = Min(yMin, py);
							yMax = Max(yMax, py);
							break;
						}
					}
				}
			}
		}

		float *bounds = &lensData[n * LENS_SURFACE_SIZE + bucket * 4];
		if (xMin <= xMax) {
			// Wider by a grid step for the points between the grid ones
			bounds[0] = Max(xMin - step, -rearRadius);
			bounds[1] = Min(xMax + step, rearRadius);
			bounds[2] = Max(yMin - step, -rearRadius);
			bounds[3] = Min(yMax + step, rearRadius);
		} else {
			// No ray: the weight 0 (RealisticCamera_LensRay())
			bounds[0] = bounds[1] = bounds[2] = bounds[3] = 0.f;
		}
	}

	//--------------------------------------------------------------------------
	// The exposure: the weight 1 on average at the center of the film
	//--------------------------------------------------------------------------

	const u_int centerSamples = 64;
	double centerWeight = 0.0;
	for (u_int j = 0; j < centerSamples; ++j) {
		for (u_int i = 0; i < centerSamples; ++i) {
			ocl::float3 rayOrig, rayDir;
			centerWeight += ocl::RealisticCamera_LensRay(&lensCamera, &lensData[0],
					ocl::make_float3(0.f, 0.f, filmZ),
					(i + .5f) / centerSamples, (j + .5f) / centerSamples,
					LENS_WAVELENGTH_D, &rayOrig, &rayDir);
		}
	}
	if (centerWeight <= 0.0)
		throw runtime_error("No light goes through the lens of RealisticCamera to the center of the film");
	lensCamera.weightScale = (float)(Sqr(centerSamples) / centerWeight);

	lensKey = key.str();

	{
		lock_guard<mutex> lock(lensSetUpCacheMutex);
		lensSetUpCache.push_front({ lensKey, lensCamera, lensData, filmZ, effectiveFocalLength, effectiveFStop });
		if (lensSetUpCache.size() > LENS_SETUP_CACHE_SIZE)
			lensSetUpCache.pop_back();
	}

	const double tEnd = WallClockTime();
	SDL_LOG("RealisticCamera lens set up in " << int((tEnd - tStart) * 1000.0) << "ms: focal length " <<
			effectiveFocalLength << " mm, f/" << effectiveFStop << " (widest f/" << minFStop <<
			"), film " << (rearZ - filmZ) * 1e3f << " mm behind the rear vertex");
}

void RealisticCamera::InitCameraTransforms(CameraTransforms *trans) {
	UpdateLens();

	// This is a trick I use from LuxCoreRenderer to set cameraToWorld to
	// identity matrix.
	if (orig == target)
		trans->cameraToWorld = Transform();
	else {
		const Transform worldToCamera = LookAt(orig, target, up);
		trans->cameraToWorld = Inverse(worldToCamera);
	}

	// The screen to the sensor (see UpdateLens()): the lens inverts the image
	const float sensorScale = sensorWidth * 1e-3f * .5f;
	trans->screenToCamera = luxrays::Translate(Vector(0.f, 0.f, filmZ)) *
			Scale(-sensorScale, -sensorScale, 1.f);
	trans->screenToWorld = trans->cameraToWorld * trans->screenToCamera;
	trans->rasterToScreen = luxrays::Translate(Vector(screenWindow[0], screenWindow[3], 0.f)) *
		Scale(screenWindow[1] - screenWindow[0], screenWindow[2] - screenWindow[3], 1.f) *
		Scale(1.f / filmWidth, 1.f / filmHeight, 1.f);
	trans->rasterToCamera = trans->screenToCamera * trans->rasterToScreen;
	trans->rasterToWorld = trans->screenToWorld * trans->rasterToScreen;
}

void RealisticCamera::InitCameraData() {
}

void RealisticCamera::InitRay(Ray *ray, const float filmX, const float filmY) const {
	// The ray of the pixel through a pinhole at the entrance pupil
	const Point filmPoint = camTrans.rasterToCamera * Point(filmX, filmHeight - filmY, 0.f);
	ray->Update(Point(0.f, 0.f, 0.f), Normalize(Vector(-filmPoint.x, -filmPoint.y, lensCamera.focalLength)));
}

void RealisticCamera::UpdateAuto(SceneConstRef scene, IntersectionDevice *device) {
	if (autoFocus) {
		// The distance along the axis from the entrance pupil
		const Ray ray(camTrans.cameraToWorld * Point(0.f, 0.f, 0.f),
				Normalize(camTrans.cameraToWorld * Vector(0.f, 0.f, 1.f)));
		RayHit rayHit;
		if (TraceAutoRay(scene, device, ray, &rayHit)) {
			focalDistance = rayHit.t;
			InitCameraTransforms(&camTrans);
		}
	}

	Camera::UpdateAuto(scene, device);
}

Spectrum RealisticCamera::GenerateRay(const float time,
		const float filmX, const float filmY,
		Ray *ray, PathVolumeInfo *volInfo,
		const float u0, const float u1, const float wavelength) const {
	if (HasVolume())
		volInfo->AddVolume(GetVolume());

	// The point of the sensor
	const Point filmPoint = camTrans.rasterToCamera * Point(filmX, filmHeight - filmY, 0.f);

	ocl::float3 rayOrig, rayDir;
	const float weight = ocl::RealisticCamera_LensRay(&lensCamera, &lensData[0],
			ocl::make_float3(filmPoint.x, filmPoint.y, filmPoint.z),
			u0, u1, wavelength, &rayOrig, &rayDir);

	ray->o = Point(rayOrig.x, rayOrig.y, rayOrig.z);
	ray->d = Vector(rayDir.x, rayDir.y, rayDir.z);
	// The clipping planes across the axis at their distances from the
	// entrance pupil, as the perspective camera's (a scene can clip a wall in
	// front of the camera)
	float tHither = 0.f;
	if (ray->d.z > 0.f) {
		tHither = Max((clipHither - ray->o.z) / ray->d.z, 0.f);
		ray->maxt = Max((clipYon - ray->o.z) / ray->d.z, tHither);
	} else
		ray->maxt = clipYon;
	ray->time = time;

	if (motionSystem) {
		*ray = motionSystem->Sample(ray->time) * (camTrans.cameraToWorld * (*ray));
		// I need to normalize the direction vector again because the motion
		// system could include some kind of scale
		ray->d = Normalize(ray->d);
	} else
		*ray = camTrans.cameraToWorld * (*ray);
	ray->mint = Max(MachineEpsilon::E(ray->o), tHither);

	// World arbitrary clipping plane support
	if (enableClippingPlane)
		ApplyArbitraryClippingPlane(ray);

	return Spectrum(weight);
}

bool RealisticCamera::GetSamplePosition(Ray *eyeRay, float *filmX, float *filmY) const {
	// The lens mapping has no inverse (see SupportsLightTracing())
	return false;
}

bool RealisticCamera::LocalSampleLens(const float time, const float u1, const float u2,
		Point *lensPoint) const {
	*lensPoint = Point();

	return false;
}

bool RealisticCamera::SampleLens(const float time, const float u1, const float u2,
		Point *lensPoint) const {
	*lensPoint = Point();

	return false;
}

void RealisticCamera::GetPDF(const Ray &eyeRay, const float eyeDistance,
		const float filmX, const float filmY,
		float *pdfW, float *fluxToRadianceFactor) const {
	if (pdfW)
		*pdfW = 0.f;
	if (fluxToRadianceFactor)
		*fluxToRadianceFactor = 0.f;
}

PropertiesUPtr RealisticCamera::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const {
	PropertiesUPtr props = ProjectiveCamera::ToProperties(imgMapCache, useRealFileName);

	props->Set(Property("scene.camera.type")("realistic"));
	if (!lensFileName.empty())
		props->Set(Property("scene.camera.realistic.lens.file")(lensFileName));
	else
		props->Set(Property("scene.camera.realistic.lens.preset")(lensPreset));
	props->Set(Property("scene.camera.realistic.focallength")(focalLength));
	props->Set(Property("scene.camera.realistic.sensorwidth")(sensorWidth));
	props->Set(Property("scene.camera.realistic.fstop")(fStop));
	props->Set(Property("scene.camera.realistic.dispersion")(dispersion));
	props->Set(Property("scene.camera.bokeh.blades")(stopBlades));
	props->Set(Property("scene.camera.bokeh.rotation")(stopRotation));
	props->Set(Property("scene.camera.bokeh.roundness")(stopRoundness));
	props->Set(Property("scene.camera.bokeh.obstruction")(stopObstruction));

	return props;
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
