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

#include "luxrays/core/dataset.h"
#include "luxrays/core/intersectiondevice.h"
#include "slg/core/sdl.h"
#include "slg/scene/scene.h"
#include "slg/cameras/camera.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Scene preprocess
//------------------------------------------------------------------------------

void Scene::PreprocessCamera(const u_int filmWidth, const u_int filmHeight, const u_int *filmSubRegion) {
	camera->Update(filmWidth, filmHeight, filmSubRegion);
}

void Scene::Preprocess(Context& ctx, const u_int filmWidth, const u_int filmHeight,
		const u_int *filmSubRegion, const bool useRTMode) {
	// Before anything that reads the materials (the light sources, the
	// compiled scene)
	if (editActions.Has(MATERIALS_EDIT) || editActions.Has(MATERIAL_TYPES_EDIT))
		matDefs.UpdateMixVertexWeights();

	//--------------------------------------------------------------------------
	// Check if I have to update geometry
	//--------------------------------------------------------------------------

	if (!dataSet || editActions.Has(GEOMETRY_EDIT) ||
			(editActions.Has(GEOMETRY_TRANS_EDIT) &&
				!dataSet->DoesAllAcceleratorsSupportUpdate())) {
		if (ctx.IsRunning()) {
			// Stop all intersection devices
			ctx.Stop();
		}

		// Rebuild the data set
		dataSet = std::make_unique<DataSet>(ctx);

		// Add all objects
		for (u_int i = 0; i < objDefs.GetSize(); ++i)
			dataSet->Add(objDefs.GetSceneObject(i).GetExtMesh());

		dataSet->Preprocess();

		// Set the LuxRays DataSet
		ctx.SetDataSet(dataSet);

		// Restart all intersection devices
		ctx.Start();
	} else if(editActions.Has(GEOMETRY_TRANS_EDIT)) {
		// I have only to update the DataSet bounding boxes
		dataSet->UpdateBBoxes();
		ctx.UpdateDataSet();
	}
	
	// Only at this point I can safely trace rays

	//--------------------------------------------------------------------------
	// Check if I have to update the camera
	//--------------------------------------------------------------------------
	
	if (editActions.Has(CAMERA_EDIT))
		PreprocessCamera(filmWidth, filmHeight, filmSubRegion);

	// Update auto-focus and auto-volume. Their ray is traced by a native
	// device if there is one (Embree is built for it anyway), else by a
	// hardware device: the CPU accelerator isn't built only for one ray (0.2 s
	// for 1.7M triangles, 0.65 s for 5.5M) when the GPU renders without native
	// threads.
	IntersectionDevice *autoDevice = nullptr;
	if (ctx.IsRunning()) {
		for (IntersectionDevice &device : ctx.GetIntersectionDevices()) {
			if (!device.HasHWSupport()) {
				autoDevice = &device;
				break;
			} else if (!autoDevice)
				autoDevice = &device;
		}
	}
	camera->UpdateAuto(*this, autoDevice);

	// At this point, both the data set and the camera are updated
	const BBox sceneBBox = Union(dataSet->GetBBox(), camera->GetBBox());
	sceneBSphere = sceneBBox.BoundingSphere();		

	// Before the light sources preprocessing: it can trace shadow rays (i.e.
	// the direct light sampling cache)
	UpdateShadowBlockers();

	//--------------------------------------------------------------------------
	// Check if something has changed in light sources
	//--------------------------------------------------------------------------

	const bool lightsEdited = editActions.Has(GEOMETRY_EDIT) ||
			editActions.Has(GEOMETRY_TRANS_EDIT) ||
			editActions.Has(MATERIALS_EDIT) ||
			editActions.Has(MATERIAL_TYPES_EDIT) ||
			editActions.Has(LIGHTS_EDIT) ||
			editActions.Has(LIGHT_TYPES_EDIT) ||
			editActions.Has(IMAGEMAPS_EDIT);
	if (lightsEdited)
		lightDefs.Preprocess(*this, useRTMode);

	// And for visibility maps: built again only after the same edits (a
	// camera edit keeps them, the points the old camera did not see use
	// the light's own distribution)
	lightDefs.UpdateVisibilityMaps(*this, useRTMode, lightsEdited);

	//--------------------------------------------------------------------------
	// Preprocess image maps according resize policy
	//--------------------------------------------------------------------------

	imgMapCache.Preprocess(*this, useRTMode);

	//--------------------------------------------------------------------------
	// Reset the edit actions
	//--------------------------------------------------------------------------

	editActions.Reset();
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
