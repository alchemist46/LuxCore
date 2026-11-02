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

#ifndef _SLG_OUTLINE_PLUGIN_H
#define	_SLG_OUTLINE_PLUGIN_H

#include <vector>
#include <memory>
#include <typeinfo>

#include "luxrays/luxrays.h"
#include "luxrays/core/color/color.h"
#include "luxrays/utils/serializationutils.h"
#include "slg/film/imagepipeline/imagepipeline.h"

namespace slg {

//------------------------------------------------------------------------------
// Outline plugin
//
// Lines over the image with the parameters of Redshift's Contour shader:
// external lines (the silhouettes) where the OBJECT_ID changes, and internal
// lines inside one object where the DEPTH jumps by more than depthThreshold
// of the nearer depth (self-occlusion) or the AVG_SHADING_NORMAL turns by more
// than normalAngle (the creases). Each kind has its own colour, opacity and
// thickness, the external lines cover the internal ones. Each test needs its
// channel (Film::Parse() adds them when the film is not initialized yet) and
// is skipped without it; without the OBJECT_ID, the depth jumps are external
// lines. The edges are found between neighbour pixels (a crease also when the
// pixels beyond them turn by more than normalAngle and the boundary takes
// more than 0.4 of it, as the film's filter spreads a crease over two
// boundaries). A line is drawn on the near side of its edge, centred on a
// crease and where two objects touch (no depth jump by depthThreshold, or
// 0.05 without the depth test: SEPARATE_MESHES_INTERSECTING), also on the far
// object with SEPARATE_MESHES_OVERLAPPING, along a straight segment fitted to
// the edges around it, and each pixel takes the fraction of its area that the
// lines cover (anti-aliased lines). The lines are drawn only over the objects
// on the two sides of their edge. The thickness is in pixels at 1920x1080,
// scaled by sqrt(width height / (1920 1080)) with resolutionRelative, times
// scale.
//------------------------------------------------------------------------------

class OutlinePlugin : public ImagePipelinePlugin {
public:
	// The style of a kind of line
	struct LineStyle {
		bool enable = true;
		luxrays::Spectrum color = luxrays::Spectrum(0.f);
		float opacity = 1.f;
		float thickness = 1.f;

		template<class Archive> void serialize(Archive &ar, const u_int version) {
			ar & enable;
			ar & color;
			ar & opacity;
			ar & thickness;
		}
	};

	// Redshift's Detect Separate Meshes: the lines where two objects touch
	// (intersecting) and on the far object of a silhouette (overlapping)
	typedef enum {
		SEPARATE_MESHES_OFF = 0,
		SEPARATE_MESHES_INTERSECTING = 1,
		SEPARATE_MESHES_OVERLAPPING = 2,
		SEPARATE_MESHES_BOTH = 3
	} SeparateMeshes;

	OutlinePlugin(const LineStyle &external, const LineStyle &internal,
			const SeparateMeshes separateMeshes, const float depthThreshold,
			const float normalAngle, const float scale, const bool resolutionRelative);
	virtual ~OutlinePlugin() {};

	virtual ImagePipelinePlugin *Copy() const;

	virtual void Apply(Film &film, const u_int index);

	// The channels the lines need
	bool UsesObjectID() const { return external.enable || internal.enable; }
	bool UsesDepth() const { return external.enable || (internal.enable && (depthThreshold > 0.f)); }
	bool UsesNormal() const { return internal.enable && (normalAngle > 0.f); }

	friend class boost::serialization::access;

private:
	OutlinePlugin();

	template<class Archive> void serialize(Archive &ar, const u_int version) {
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ImagePipelinePlugin);
		ar & external;
		ar & internal;
		ar & separateMeshes;
		ar & depthThreshold;
		ar & normalAngle;
		ar & scale;
		ar & resolutionRelative;
	}

	LineStyle external, internal;
	SeparateMeshes separateMeshes;
	// 0 turns the test off
	float depthThreshold;
	// Degrees, 0 turns the test off
	float normalAngle;
	float scale;
	bool resolutionRelative;
};

}

BOOST_CLASS_VERSION(slg::OutlinePlugin, 2)

BOOST_CLASS_EXPORT_KEY(slg::OutlinePlugin)

#endif	/*  _SLG_OUTLINE_PLUGIN_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
