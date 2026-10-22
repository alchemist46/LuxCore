#line 2 "materialdefs_funcs_thinfilmcoating.cl"

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

//---------------------------------------------------------------------------------
// Thin film coating (utility functions, to be used in materials where appropriate)
//---------------------------------------------------------------------------------

// A reduced set of the CIE tables from luxrays/core/color/spds/data/xyzbasis.h, 
// in the range specified by MIN_WAVELENGTH and MAX_WAVELENGTH, with WAVELENGTH_STEP 
// intervals instead of 1 nm intervals.
// Pre-computed with the function at the bottom of this file.
__constant float CIE_X_reduced[] = {
	0.001231153961271f,
	0.003753236029297f,
	0.012869009748101f,
	0.038468241691589f,
	0.120767198503017f,
	0.271807909011841f,
	0.346129596233368f,
	0.339094102382660f,
	0.297257900238037f,
	0.206811502575874f,
	0.104297898709774f,
	0.036412831395864f,
	0.006296345964074f,
	0.006982080172747f,
	0.055378608405590f,
	0.153854206204414f,
	0.277101695537567f,
	0.418311506509781f,
	0.577821493148804f,
	0.745518803596497f,
	0.902318120002747f,
	1.018006443977356f,
	1.062909603118896f,
	1.013047695159912f,
	0.872919976711273f,
	0.663810372352600f,
	0.466193914413452f,
	0.298001110553741f,
	0.174832299351692f,
	0.093330606818199f,
	0.049818608909845f,
	0.024452749639750f,
	0.012129199691117f,
	0.006195407826453f,
};

__constant float CIE_Y_reduced[] = {
	0.000035215209209f,
	0.000106136103568f,
	0.000357266690116f,
	0.001069879974239f,
	0.003526400076225f,
	0.010664430446923f,
	0.021718239411712f,
	0.036225710064173f,
	0.057458721101284f,
	0.087232798337936f,
	0.133452802896500f,
	0.199417993426323f,
	0.308578014373779f,
	0.482939511537552f,
	0.690842390060425f,
	0.849491596221924f,
	0.947225213050842f,
	0.992811620235443f,
	0.996898710727692f,
	0.958134889602661f,
	0.879781603813171f,
	0.769154727458954f,
	0.643844783306122f,
	0.515632271766663f,
	0.393032014369965f,
	0.275624513626099f,
	0.182974398136139f,
	0.112769097089767f,
	0.064769759774208f,
	0.034200478345156f,
	0.018120689317584f,
	0.008846157230437f,
	0.004380074795336f,
	0.002237275009975f,
};

__constant float CIE_Z_reduced[] = {
	0.005802907049656f,
	0.017730150371790f,
	0.060998030006886f,
	0.183256804943085f,
	0.579530298709869f,
	1.323929548263550f,
	1.730382084846497f,
	1.775867104530334f,
	1.688737154006958f,
	1.338736176490784f,
	0.856619298458099f,
	0.491967290639877f,
	0.286168605089188f,
	0.168560802936554f,
	0.083845309913158f,
	0.044898588210344f,
	0.021989250555634f,
	0.009529305621982f,
	0.004202399868518f,
	0.002196799963713f,
	0.001683066948317f,
	0.001146667054854f,
	0.000842560024466f,
	0.000383466685889f,
	0.000202186696697f,
	0.000056933331507f,
	0.000021813330022f,
	0.000001333332989f,
	0.000000000000000f,
	0.000000000000000f,
	0.000000000000000f,
	0.000000000000000f,
	0.000000000000000f,
	0.000000000000000f,
};

//---------------------------------------------------------------------------------
// Spectrum to RGB
//---------------------------------------------------------------------------------

// XYZ to Linear Rec.709 with the equal-energy white: a flat reflectance
// spectrum gives a neutral colour of the same value
__constant float ThinFilmCoating_XYZToRGB[3][3] = {
	{ 2.689655172413794f, -1.275862068965517f, -0.413793103448276f },
	{ -1.022108172127912f, 1.978286616660087f, 0.043821555467825f },
	{ 0.061224489795918f, -0.224489795918367f, 1.163265306122448f },
};

// The sums of the reduced CIE tables: X, Y and Z divided by them are 1 for a
// flat spectrum of value 1
#define THINFILM_CIE_X_SUM 10.678131428547200f
#define THINFILM_CIE_Y_SUM 10.683555383588102f
#define THINFILM_CIE_Z_SUM 10.679285937881216f

OPENCL_FORCE_INLINE float3 ThinFilmCoating_ConvertXYZToRGB(const float3 xyzColor) {
	return MAKE_FLOAT3(
		ThinFilmCoating_XYZToRGB[0][0] * xyzColor.x + ThinFilmCoating_XYZToRGB[0][1] * xyzColor.y + ThinFilmCoating_XYZToRGB[0][2] * xyzColor.z,
		ThinFilmCoating_XYZToRGB[1][0] * xyzColor.x + ThinFilmCoating_XYZToRGB[1][1] * xyzColor.y + ThinFilmCoating_XYZToRGB[1][2] * xyzColor.z,
		ThinFilmCoating_XYZToRGB[2][0] * xyzColor.x + ThinFilmCoating_XYZToRGB[2][1] * xyzColor.y + ThinFilmCoating_XYZToRGB[2][2] * xyzColor.z
	);
}

//---------------------------------------------------------------------------------
// Complex numbers (x the real part, y the imaginary one)
//---------------------------------------------------------------------------------

OPENCL_FORCE_INLINE float2 ThinFilm_CMul(const float2 a, const float2 b) {
	return MAKE_FLOAT2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

OPENCL_FORCE_INLINE float2 ThinFilm_CDiv(const float2 a, const float2 b) {
	const float d = b.x * b.x + b.y * b.y;
	return MAKE_FLOAT2((a.x * b.x + a.y * b.y) / d, (a.y * b.x - a.x * b.y) / d);
}

// The square root with a non-negative imaginary part: the wave that decays
// into an absorbing medium, or the evanescent one past the critical angle
OPENCL_FORCE_INLINE float2 ThinFilm_CSqrt(const float2 z) {
	const float r = sqrt(z.x * z.x + z.y * z.y);
	const float re = sqrt(fmax(0.f, .5f * (r + z.x)));
	const float im = sqrt(fmax(0.f, .5f * (r - z.x)));
	// The principal root has the sign of z.y in its imaginary part: with
	// z.y < 0 the root with Im >= 0 is its negation
	return (z.y < 0.f) ? MAKE_FLOAT2(-re, im) : MAKE_FLOAT2(re, im);
}

OPENCL_FORCE_INLINE float ThinFilm_CNorm(const float2 z) {
	return z.x * z.x + z.y * z.y;
}

//---------------------------------------------------------------------------------
// Thin film interference
//
// A film of thickness d (nm) and real IOR n1 between the medium of the
// incident light (real IOR n0) and a substrate of complex IOR N2 = n2 + i k2:
// the amplitudes r = (r01 + r12 e^(i phi)) / (1 + r01 r12 e^(i phi)) of the s
// and p polarizations, with phi = 4 pi n1 d cos(theta1) / lambda, and the
// Airy reflectance R = (|r_s|^2 + |r_p|^2) / 2 over the visible spectrum
// (Belcour and Barla 2017, without their Fourier approximation). The
// substrate IOR does not change with the wavelength.
//
// CalcFilmColor() is that reflectance divided by the Fresnel reflectance of
// the bare interface: the factor the materials multiply their Fresnel term
// with, in the working colour space. A film with the IOR of the incident
// medium, or of zero thickness, gives 1.
//---------------------------------------------------------------------------------

// Past the optical path where 34 wavelengths resolve the fringes, the film
// is incoherent: the sum of the two interfaces' intensities
#define THINFILM_MAX_COHERENT_OPD 4000.f

OPENCL_FORCE_INLINE float ThinFilm_IncoherentR(const float R01, const float R12) {
	const float d = 1.f - R01 * R12;
	return (d > 0.f) ? ((R01 + R12 - 2.f * R01 * R12) / d) : 1.f;
}

// The Fresnel amplitude of the s or p polarization between a real IOR side
// (q1 = n1 cos1, a1 = n1) and a complex one (q2 = N2 cos2, N2^2)
OPENCL_FORCE_INLINE float2 ThinFilm_Rs(const float q1, const float2 q2) {
	return ThinFilm_CDiv(MAKE_FLOAT2(q1 - q2.x, -q2.y), MAKE_FLOAT2(q1 + q2.x, q2.y));
}

OPENCL_FORCE_INLINE float2 ThinFilm_Rp(const float n1, const float cos1, const float2 N2Sqr, const float2 q2) {
	// (N2^2 cos1 - n1 q2) / (N2^2 cos1 + n1 q2)
	const float2 a = MAKE_FLOAT2(N2Sqr.x * cos1, N2Sqr.y * cos1);
	const float2 b = MAKE_FLOAT2(n1 * q2.x, n1 * q2.y);
	return ThinFilm_CDiv(MAKE_FLOAT2(a.x - b.x, a.y - b.y), MAKE_FLOAT2(a.x + b.x, a.y + b.y));
}

// The film reflectance of a substrate of complex IOR (eta2, k2) over the
// wavelengths, as XYZ, and the bare interface's Fresnel reflectance. False at
// the total internal reflection into the film.
OPENCL_FORCE_NOT_INLINE bool ThinFilm_Reflectance(const float cos0, const float filmThickness,
		const float n0, const float n1, const float eta2, const float k2,
		float3 *xyz, float *bareR) {
	const float sin0Sqr = fmax(0.f, 1.f - cos0 * cos0);

	// N2^2 and q2 = N2 cos(theta2) = sqrt(N2^2 - n0^2 sin0^2)
	const float2 N2Sqr = MAKE_FLOAT2(eta2 * eta2 - k2 * k2, 2.f * eta2 * k2);
	const float2 q2 = ThinFilm_CSqrt(MAKE_FLOAT2(N2Sqr.x - n0 * n0 * sin0Sqr, N2Sqr.y));

	// The bare interface n0 | N2
	*bareR = .5f * (ThinFilm_CNorm(ThinFilm_Rs(n0 * cos0, q2)) +
			ThinFilm_CNorm(ThinFilm_Rp(n0, cos0, N2Sqr, q2)));

	const float cos1Sqr = 1.f - Sqr(n0 / n1) * sin0Sqr;
	if (cos1Sqr <= 0.f)
		return false;
	const float cos1 = sqrt(cos1Sqr);

	// The interfaces n0 | n1 and n1 | N2
	const float rs01 = (n0 * cos0 - n1 * cos1) / (n0 * cos0 + n1 * cos1);
	const float rp01 = (n1 * cos0 - n0 * cos1) / (n1 * cos0 + n0 * cos1);
	const float2 rs12 = ThinFilm_Rs(n1 * cos1, q2);
	const float2 rp12 = ThinFilm_Rp(n1, cos1, N2Sqr, q2);

	const float opd = 2.f * n1 * filmThickness * cos1;
	if (opd > THINFILM_MAX_COHERENT_OPD) {
		const float R = .5f * (ThinFilm_IncoherentR(rs01 * rs01, ThinFilm_CNorm(rs12)) +
				ThinFilm_IncoherentR(rp01 * rp01, ThinFilm_CNorm(rp12)));
		*xyz = TO_FLOAT3(R);
		return true;
	}

	const uint NUM_WAVELENGTHS = 34;
	const uint MIN_WAVELENGTH = 380;
	const uint WAVELENGTH_STEP = 10;

	float3 xyzSum = MAKE_FLOAT3(0.f, 0.f, 0.f);
	for (uint i = 0; i < NUM_WAVELENGTHS; ++i) {
		const float waveLength = (float)(WAVELENGTH_STEP * i + MIN_WAVELENGTH);
		const float phi = 2.f * M_PI_F * opd / waveLength;
		const float2 e = MAKE_FLOAT2(cos(phi), sin(phi));

		// s polarization
		const float2 se = ThinFilm_CMul(rs12, e);
		const float Rs = ThinFilm_CNorm(MAKE_FLOAT2(rs01 + se.x, se.y)) /
				ThinFilm_CNorm(MAKE_FLOAT2(1.f + rs01 * se.x, rs01 * se.y));
		// p polarization
		const float2 pe = ThinFilm_CMul(rp12, e);
		const float Rp = ThinFilm_CNorm(MAKE_FLOAT2(rp01 + pe.x, pe.y)) /
				ThinFilm_CNorm(MAKE_FLOAT2(1.f + rp01 * pe.x, rp01 * pe.y));

		const float R = .5f * (Rs + Rp);
		xyzSum.x += R * CIE_X_reduced[i];
		xyzSum.y += R * CIE_Y_reduced[i];
		xyzSum.z += R * CIE_Z_reduced[i];
	}

	*xyz = MAKE_FLOAT3(xyzSum.x / THINFILM_CIE_X_SUM, xyzSum.y / THINFILM_CIE_Y_SUM,
			xyzSum.z / THINFILM_CIE_Z_SUM);
	return true;
}

// The film factor (Linear Rec.709) of one substrate IOR
OPENCL_FORCE_INLINE float3 ThinFilm_Factor(const float cos0, const float filmThickness,
		const float n0, const float n1, const float eta2, const float k2) {
	float3 xyz;
	float bareR;
	const bool filmReached = ThinFilm_Reflectance(cos0, filmThickness, n0, n1, eta2, k2, &xyz, &bareR);
	if (!(bareR > 1e-6f))
		return WHITE;
	// At the total internal reflection into the film the interface reflects
	// everything
	if (!filmReached)
		return TO_FLOAT3(1.f / bareR);

	return fmax(ThinFilmCoating_ConvertXYZToRGB(xyz), BLACK) / bareR;
}

// A dielectric substrate: iorIncident the IOR on the side of localFixedDir,
// iorSubstrate the one of the other side
OPENCL_FORCE_NOT_INLINE float3 CalcFilmColor(const float3 localFixedDir, const float filmThickness,
		const float filmIOR, const float iorIncident, const float iorSubstrate) {
	if (!(filmThickness > 0.f))
		return WHITE;

	return WorkingColorSpace_FromRec709(ThinFilm_Factor(fabs(localFixedDir.z), filmThickness,
			iorIncident, filmIOR, iorSubstrate, 0.f));
}

// A conductor substrate of complex IOR (eta, k) per Linear Rec.709 channel
OPENCL_FORCE_NOT_INLINE float3 CalcFilmColorConductor(const float3 localFixedDir, const float filmThickness,
		const float filmIOR, const float iorIncident, const float3 eta, const float3 k) {
	if (!(filmThickness > 0.f))
		return WHITE;

	const float cos0 = fabs(localFixedDir.z);
	const float r = ThinFilm_Factor(cos0, filmThickness, iorIncident, filmIOR, eta.x, k.x).x;
	const float g = ThinFilm_Factor(cos0, filmThickness, iorIncident, filmIOR, eta.y, k.y).y;
	const float b = ThinFilm_Factor(cos0, filmThickness, iorIncident, filmIOR, eta.z, k.z).z;

	return WorkingColorSpace_FromRec709(MAKE_FLOAT3(r, g, b));
}

// The factor of the transmission of an interface of bare Fresnel reflectance
// bareR with the film factor filmColor: a lossless film transmits what it does
// not reflect, (1 - bareR filmColor) / (1 - bareR)
OPENCL_FORCE_INLINE float3 ThinFilm_TransmissionScale(const float bareR, const float3 filmColor) {
	if (!(bareR < 1.f))
		return WHITE;
	return fmax((WHITE - bareR * filmColor) / (1.f - bareR), BLACK);
}

// The complex IOR of a conductor of normal reflectance f0 and edge tint g per
// channel (Gulbrandsen 2014), for the film over the metals of the artistic
// models
OPENCL_FORCE_INLINE void ThinFilm_ConductorIORChannel(const float f0, const float edgeTint,
		float *eta, float *k) {
	const float r = clamp(f0, 0.f, .99f);
	const float g = clamp(edgeTint, 0.f, 1.f);
	const float sqrtR = sqrt(r);
	const float n = g * (1.f - r) / (1.f + r) + (1.f - g) * (1.f + sqrtR) / (1.f - sqrtR);
	*eta = n;
	*k = sqrt(fmax(0.f, (r * Sqr(n + 1.f) - Sqr(n - 1.f)) / (1.f - r)));
}

OPENCL_FORCE_INLINE void ThinFilm_ConductorIOR(const float3 f0, const float3 edgeTint,
		float3 *eta, float3 *k) {
	// No address of a vector component in OpenCL C
	float er, eg, eb, kr, kg, kb;
	ThinFilm_ConductorIORChannel(f0.x, edgeTint.x, &er, &kr);
	ThinFilm_ConductorIORChannel(f0.y, edgeTint.y, &eg, &kg);
	ThinFilm_ConductorIORChannel(f0.z, edgeTint.z, &eb, &kb);
	*eta = MAKE_FLOAT3(er, eg, eb);
	*k = MAKE_FLOAT3(kr, kg, kb);
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
