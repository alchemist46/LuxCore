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
// Lines over the image where the OBJECT_ID changes (the silhouettes), where
// the DEPTH jumps by more than depthThreshold of the nearer depth, and where
// the AVG_SHADING_NORMAL turns by more than normalAngle (the creases). Each
// test needs its channel (Film::Parse() adds them when the film is not
// initialized yet) and is skipped without it. The edges are found between
// neighbour pixels (a crease also when the pixels beyond them turn by more
// than normalAngle and the boundary takes more than 0.4 of it, as the film's
// filter spreads a crease over two boundaries). The line is drawn width
// pixels wide on the near side of each edge, centred on a crease, along a
// straight segment fitted to the edges around it, and each pixel takes the
// fraction of its area that the lines cover (anti-aliased lines). The lines
// are drawn only over the objects on the two sides of their edge.
//------------------------------------------------------------------------------

class OutlinePlugin : public ImagePipelinePlugin {
public:
	OutlinePlugin(const luxrays::Spectrum &color, const float amount, const float width,
			const bool objectIDEnabled, const float depthThreshold, const float normalAngle);
	virtual ~OutlinePlugin() {};

	virtual ImagePipelinePlugin *Copy() const;

	virtual void Apply(Film &film, const u_int index);

	friend class boost::serialization::access;

private:
	OutlinePlugin();

	template<class Archive> void serialize(Archive &ar, const u_int version) {
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ImagePipelinePlugin);
		ar & color;
		ar & amount;
		ar & width;
		ar & objectIDEnabled;
		ar & depthThreshold;
		ar & normalAngle;
	}

	luxrays::Spectrum color;
	float amount, width;
	bool objectIDEnabled;
	// 0 turns the test off
	float depthThreshold;
	// Degrees, 0 turns the test off
	float normalAngle;
};

}

BOOST_CLASS_VERSION(slg::OutlinePlugin, 1)

BOOST_CLASS_EXPORT_KEY(slg::OutlinePlugin)

#endif	/*  _SLG_OUTLINE_PLUGIN_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
