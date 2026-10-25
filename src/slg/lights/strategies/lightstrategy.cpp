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

#include "slg/lights/strategies/lightstrategyregistry.h"
#include "slg/scene/scene.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// LightStrategy
//------------------------------------------------------------------------------

// The type tag of a configuration. The default is LIGHT_TREE (LOG_POWER until
// 2026-10-17): at equal error it was 1.3-1.55x as fast on the GPU and
// 1.0-1.08x on the CPU in the G0 scenes. DLS_CACHE, the direct light sampling
// cache removed on 2026-10-25, is the light tree too: its learned cells pick
// the lights that reach each region of the scene, as the cache did, without
// the cache's pre-pass.
static string GetTypeTag(const Properties &cfg) {
	const string type = cfg.Get(
		Property("lightstrategy.type")(LightStrategyLightTree::GetObjectTag())
	).Get<string>();

	return (type == "DLS_CACHE") ? LightStrategyLightTree::GetObjectTag() : type;
}

LightStrategyType LightStrategy::GetType(const luxrays::Properties &cfg) {
	return String2LightStrategyType(GetTypeTag(cfg));
}

//------------------------------------------------------------------------------
// Static methods used by LightStrategyRegistry
//------------------------------------------------------------------------------

PropertiesUPtr LightStrategy::ToProperties(const Properties &cfg) {
	const string type = GetTypeTag(cfg);
	if (cfg.IsDefined("lightstrategy.type") && (cfg.Get("lightstrategy.type").Get<string>() == "DLS_CACHE"))
		SLG_LOG("The DLS_CACHE light strategy is no longer available: using " << type);

	LightStrategyRegistry::ToProperties func;

	if (LightStrategyRegistry::STATICTABLE_NAME(ToProperties).Get(type, func)) {
		PropertiesUPtr props = func(cfg);
		*props << Property("lightstrategy.type")(type);
		return props;
	} else
		throw runtime_error(
			"Unknown light strategy type in LightStrategy::ToProperties(): " + type
		);
}

LightStrategyUPtr LightStrategy::FromProperties(const Properties &cfg) {
	const string type = GetTypeTag(cfg);

	LightStrategyRegistry::FromProperties func;
	if (LightStrategyRegistry::STATICTABLE_NAME(FromProperties).Get(type, func))
		return func(cfg);
	else
		throw runtime_error(
			"Unknown filter type in LightStrategy::FromProperties(): " + type
		);
}

string LightStrategy::FromPropertiesOCL(const Properties &cfg) {
	throw runtime_error("Called LightStrategy::FromPropertiesOCL()");
}

LightStrategyType LightStrategy::String2LightStrategyType(const string &type) {
	LightStrategyRegistry::GetObjectType func;
	if (LightStrategyRegistry::STATICTABLE_NAME(GetObjectType).Get(type, func))
		return func();
	else
		throw runtime_error(
			std::string("Unknown light strategy type in ")
			+ std::string("LightStrategy::String2LightStrategyType(): ")
			+ type
		);
}

string LightStrategy::LightStrategyType2String(const LightStrategyType type) {
	LightStrategyRegistry::GetObjectTag func;
	if (LightStrategyRegistry::STATICTABLE_NAME(GetObjectTag).Get(type, func))
		return func();
	else
		throw runtime_error(
			std::string("Unknown light strategy type in ")
			+ std::string("LightStrategy::LightStrategyType2String(): ")
			+ ToString(type)
		);
}

PropertiesUPtr LightStrategy::GetDefaultProps() {

	auto props = std::make_unique<Properties>();

	return props;
}

//------------------------------------------------------------------------------
// LightStrategyRegistry
//
// For the registration of each LightStrategy sub-class with LightStrategy StaticTables
//
// NOTE: you have to place all STATICTABLE_REGISTER() in the same .cpp file of the
// main base class (i.e. the one holding the StaticTable) because otherwise
// static members initialization order is not defined.
//------------------------------------------------------------------------------

OBJECTSTATICREGISTRY_STATICFIELDS(LightStrategyRegistry);

//------------------------------------------------------------------------------

OBJECTSTATICREGISTRY_REGISTER(LightStrategyRegistry, LightStrategyUniform);
OBJECTSTATICREGISTRY_REGISTER(LightStrategyRegistry, LightStrategyPower);
OBJECTSTATICREGISTRY_REGISTER(LightStrategyRegistry, LightStrategyLogPower);
OBJECTSTATICREGISTRY_REGISTER(LightStrategyRegistry, LightStrategyLightTree);
// Just add here any new LightStrategy (don't forget in the .h too)
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
