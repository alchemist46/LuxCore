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

#include <thread>

#include <boost/algorithm/string/predicate.hpp>

#include "slg/lights/light.h"
#include "slg/scene/scene.h"
#include "slg/lights/trianglelight.h"
#include "slg/lights/strategies/logpower.h"
#include "slg/usings.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// LightSourceDefinitions
//------------------------------------------------------------------------------

LightSourceDefinitions::LightSourceDefinitions() :
	lightTypeCount(LIGHT_SOURCE_TYPE_COUNT, 0)
{
	emitLightStrategy = std::make_unique<LightStrategyLogPower>();
	illuminateLightStrategy = std::make_unique<LightStrategyLogPower>();
	infiniteLightStrategy = std::make_unique<LightStrategyLogPower>();
	triangleLightSolidAngle = false;
	lightGroupCount = 1;
}

LightSourceDefinitions::~LightSourceDefinitions() {
}

/// Insert a new light source in the container
///
/// If a light source with the same name already exists in the container,
/// it is extracted and returned.
LightSourceUPtr LightSourceDefinitions::DefineLightSource(LightSourceUPtr&& newLight) {
	const string &name = newLight->GetName();

	auto e = lightsByName.find(name);

	if (e != lightsByName.end()) {
		// Light source already exists
		auto oldLight = std::move(e->second);

		// Update name/LightSource definition
		lightsByName.erase(e);
		lightsByName[name] = std::move(newLight);
		return oldLight;
	} else {
		// Add the new LightSource
		lightsByName[name] = std::move(newLight);
		return LightSourceUPtr();
	}
}

bool LightSourceDefinitions::IsLightSourceDefined(const std::string &name) const {
	return (lightsByName.count(name) > 0);
}

LightSourceConstRef LightSourceDefinitions::GetLightSource(const string &name) const {
	// Check if the LightSource has been already defined
	auto e = lightsByName.find(name);

	if (e == lightsByName.end())
		throw runtime_error("Reference to an undefined LightSource in LightSourceDefinitions::GetLightSource(): " + name);

	return *e->second;
}

LightSourceRef LightSourceDefinitions::GetLightSource(const string &name) {
	// Check if the LightSource has been already defined
	auto e = lightsByName.find(name);

	if (e == lightsByName.end())
		throw runtime_error("Reference to an undefined LightSource in LightSourceDefinitions::GetLightSource(): " + name);

	return *e->second;
}

LightSourceConstPtr LightSourceDefinitions::GetLightSourcePtr(const string &name) const {
	// Check if the LightSource has been already defined
	auto e = lightsByName.find(name);

	if (e == lightsByName.end())
		throw runtime_error("Reference to an undefined LightSource in LightSourceDefinitions::GetLightSource(): " + name);

	return LightSourceConstPtr(e->second.get());
}

LightSourcePtr LightSourceDefinitions::GetLightSourcePtr(const string &name) {
	// Check if the LightSource has been already defined
	auto e = lightsByName.find(name);

	if (e == lightsByName.end())
		throw runtime_error("Reference to an undefined LightSource in LightSourceDefinitions::GetLightSource(): " + name);

	return LightSourcePtr(e->second.get());
}

TriangleLightConstRef LightSourceDefinitions::GetLightSourceByMeshAndTriIndex(const u_int meshIndex, const u_int triIndex) const {
	const u_int offset = lightIndexOffsetByMeshIndex[meshIndex];
	const u_int lightIndex = lightIndexByTriIndex[offset + triIndex];

	return static_cast<TriangleLightConstRef>(lights[lightIndex].get());
}

vector<string> LightSourceDefinitions::GetLightSourceNames() const {
	vector<string> names;
	names.reserve(lights.size());
	for (auto const &e : lightsByName)
		names.push_back(e.first);

	return names;
}

LightSourceUPtr LightSourceDefinitions::DeleteLightSource(const string &name) {
	auto e = lightsByName.find(name);

	if (e == lightsByName.end())
		throw runtime_error("Reference to an undefined LightSource in LightSourceDefinitions::DeleteLightSource(): " + name);


	auto oldLight = std::move(e->second);
	lightsByName.erase(e);
	return oldLight;
}


std::vector<LightSourceUPtr> LightSourceDefinitions::DeleteLightSourceStartWith(
	const string &namePrefix
) {
	std::vector<LightSourceUPtr> oldLights;

	// Build the list of lights to delete
	std::vector<string> nameList;
	for (auto const &e : lightsByName) {
		const string &name = e.first;

		if (name.starts_with(namePrefix))
			nameList.push_back(name);
	}

	// Remove the lights from the container and return the removed lights
	for (auto const &name : nameList) {
		auto oldLight = DeleteLightSource(name);
		oldLights.push_back(std::move(oldLight));
	}
	return oldLights;
}


std::vector<LightSourceUPtr> LightSourceDefinitions::DeleteLightSourceByMaterial(
	MaterialConstRef mat
) {
	std::vector<LightSourceUPtr> oldLights;

	// Build the list of lights to delete
	std::vector<string> nameList;
	for (auto const &e : lightsByName) {
		const string &name = e.first;
		auto& l = *e.second;

		if (
			l.GetType() == TYPE_TRIANGLE
			&& (static_cast<const TriangleLight&>(l)).lightMaterial == &mat
		) {
			nameList.push_back(name);
		}
	}

	// Remove the lights from the container and return the removed lights
	for (auto const &name : nameList) {
		auto oldLight = DeleteLightSource(name);
		oldLights.push_back(std::move(oldLight));
	}
	return oldLights;
}

void LightSourceDefinitions::SetLightStrategy(const luxrays::Properties &props) {
	if (LightStrategy::GetType(props) != emitLightStrategy->GetType()) {
		emitLightStrategy = LightStrategy::FromProperties(props);
	}

	if (LightStrategy::GetType(props) != illuminateLightStrategy->GetType()) {
		illuminateLightStrategy = LightStrategy::FromProperties(props);
	}

	if (LightStrategy::GetType(props) != infiniteLightStrategy->GetType()) {
		infiniteLightStrategy = LightStrategy::FromProperties(props);
	}

	// The solid angle sampling of the triangle lights. Not for the BiDir
	// engines: their MIS stores the area pdf of the light sampling on the
	// light vertex, before the eye vertex it connects to is known.
	const std::string engineType = props.Get(Property("renderengine.type")("PATHOCL")).Get<std::string>();
	triangleLightSolidAngle = props.Get(Property("lightstrategy.trianglelight.solidangle")(false)).Get<bool>() &&
			(engineType != "BIDIRCPU") && (engineType != "BIDIRVMCPU");
}

void LightSourceDefinitions::UpdateVolumeReferences(
	VolumeConstRef oldVol, VolumeRef newVol
) {
	for (auto const &e : lightsByName) {
		auto& l = *e.second;

		l.UpdateVolumeReferences(oldVol, newVol);
	}

}

void LightSourceDefinitions::Preprocess(SceneConstRef scene, const bool useRTMode) {
	// Update lightGroupCount, envLightSources, intersectableLightSources,
	// lightIndexOffsetByMeshIndex, lightsDistribution, etc.

//	const double start = WallClockTime();
	SLG_LOG("Preprocessing light source definitions");

	lightGroupCount = 0;
	lights.clear();
	intersectableLightSources.clear();
	envLightSources.clear();
	fill(lightTypeCount.begin(), lightTypeCount.end(), 0);

	// Build 2 tables to go from mesh index and triangle index to light index:
	// the offset of each mesh in lightIndexByTriIndex and its size here, the
	// light indices below
	const u_int meshCount = scene.GetObjects().GetSize();
	lightIndexOffsetByMeshIndex.resize(meshCount);
	u_int lightTriCount = 0;
	for (u_int meshIndex = 0; meshIndex < meshCount; ++meshIndex) {
		auto& so = scene.GetObjects().GetSceneObject(meshIndex);

		if (so.GetMaterial().IsLightSource()) {
			lightIndexOffsetByMeshIndex[meshIndex] = lightTriCount;
			lightTriCount += so.GetExtMesh().GetTotalTriangleCount();
		} else
			lightIndexOffsetByMeshIndex[meshIndex] = NULL_INDEX;
	}
	lightIndexByTriIndex.assign(lightTriCount, NULL_INDEX);

	// The mesh index of the last triangle light: the triangle lights of a mesh
	// often follow each other
	SceneObjectConstPtr lastSceneObject = nullptr;
	u_int lastMeshIndex = NULL_INDEX;

	u_int i = 0;
	for (auto const &e : lightsByName) {
		auto& l = *e.second;

		// Initialize the light source index
		l.lightSceneIndex = i;

		// Update the light group count
		lightGroupCount = Max(lightGroupCount, l.GetID() + 1);

		// Update the light type count
		++lightTypeCount[l.GetType()];

		// Update the list of all lights
		lights.push_back(l);

		// Update the list of env. lights
		if (l.IsEnvironmental())
			envLightSources.push_back(static_cast<EnvLightSourceRef>(l));

		// Update the list of triangle lights and the light index of their
		// triangle (the triangle light named
		// EncodeTriangleLightNamePrefix(objectName) + triangleIndex)
		if (l.GetType() == TYPE_TRIANGLE) {
			auto& tl = static_cast<TriangleLightRef>(l);
			intersectableLightSources.push_back(tl);

			if (tl.sceneObject != lastSceneObject) {
				lastSceneObject = tl.sceneObject;
				lastMeshIndex = scene.GetObjects().GetSceneObjectIndex(*tl.sceneObject);
			}
			tl.meshIndex = lastMeshIndex;
			tl.solidAngleSampling = triangleLightSolidAngle;

			const u_int offset = lightIndexOffsetByMeshIndex[tl.meshIndex];
			if (offset != NULL_INDEX)
				lightIndexByTriIndex[offset + tl.triangleIndex] = i;
		}

		++i;
	}

//	const double end1 = WallClockTime();
//	SLG_LOG("Light step #1 preprocessing time: " << (end1 - start) << "secs");

	// I need to check all volume definitions for radiance group usage too
	for (u_int i = 0; i < scene.GetMaterials().GetSize(); ++i) {
		auto& mat = scene.GetMaterials().GetMaterial(i);

		try {
			auto& vol = dynamic_cast<VolumeConstRef>(mat);
			if (vol.HasVolumeEmissionTexture()) {
				// Update the light group count
				lightGroupCount = Max(lightGroupCount, vol.GetVolumeLightID() + 1);
			}
		}
		catch(std::bad_cast&) { /* Not a volume: pass */}

	}
	//SLG_LOG("Radiance group count: " << lightGroupCount);

//	const double end2 = WallClockTime();
//	SLG_LOG("Light step #2 preprocessing time: " << (end2 - end1) << "secs");

	// Build the light strategy
	if (lights.size() > 1024) {
		// Use a multi-thread initialization for large number of light sources
		std::thread emitLightStrategyThread([this, &scene, useRTMode]() {
			emitLightStrategy->Preprocess(scene, TASK_EMIT, useRTMode);
		});
		std::thread illuminateLightStrategyThread([this, &scene, useRTMode]() {
			illuminateLightStrategy->Preprocess(scene, TASK_ILLUMINATE, useRTMode);
		});
		std::thread infiniteLightStrategyThread([this, &scene, useRTMode]() {
			infiniteLightStrategy->Preprocess(scene, TASK_INFINITE_ONLY, useRTMode);
		});

		emitLightStrategyThread.join();
		illuminateLightStrategyThread.join();
		infiniteLightStrategyThread.join();
	} else {
		emitLightStrategy->Preprocess(scene, TASK_EMIT, useRTMode);
		illuminateLightStrategy->Preprocess(scene, TASK_ILLUMINATE, useRTMode);
		infiniteLightStrategy->Preprocess(scene, TASK_INFINITE_ONLY, useRTMode);
	}

//	const double end3 = WallClockTime();
//	SLG_LOG("Light step #3 preprocessing time: " << (end3 - end2) << "secs");

//	const double endTotal = WallClockTime();
//	SLG_LOG("Light total preprocessing time: " << (endTotal - start) << "secs");
}

void LightSourceDefinitions::UpdateVisibilityMaps(SceneConstRef scene, const bool useRTMode) {
	// Build visibility maps for Env. lights
	for (EnvLightSource& envLight : GetEnvLightSources())
		envLight.UpdateVisibilityMap(scene, useRTMode);
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
