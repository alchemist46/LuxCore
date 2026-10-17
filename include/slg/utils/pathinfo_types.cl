#line 2 "pathinfo_types.cl"

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

typedef struct {
	PathDepthInfo depth;
	PathVolumeInfo volume;

	int isPassThroughPath;

	// Last path vertex information
	BSDFEvent lastBSDFEvent;
	float lastBSDFPdfW;
	float lastGlossiness;
	Normal lastShadeN;
	bool lastFromVolume, isTransmittedPath;
	// The product of the eta^2 of the refractions of the path, for the
	// Russian roulette on the path throughput (see
	// PathTracer_RussianRouletteProb()); 1 when the rule is not in use
	float etaScale;
	// A non-specular bounce happened (the path regularization starts after it)
	int anyNonSpecularBounce;

	int isNearlyCaustic;
	// Specular, Specular+ Diffuse and Specular+ Diffuse Specular+ paths
	int isNearlyS, isNearlySD, isNearlySDS;
} EyePathInfo;
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
