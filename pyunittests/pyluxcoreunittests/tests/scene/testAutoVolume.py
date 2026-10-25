# -*- coding: utf-8 -*-
################################################################################
# Copyright 1998-2026 by authors (see AUTHORS.txt)
#
#   This file is part of LuxCoreRender.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import unittest
import pyluxcore

from pyluxcoreunittests.tests.utils import *

# The camera volume found by the auto volume ray: the ray is traced by Embree
# or, when the GPU renders without native threads, by the GPU (the CPU
# accelerator isn't built only for one ray)

# The centre of simple-mat-cube1.ply and a point outside of it
CUBE_CENTER = "-1.49 -0.11 1.48"
OUTSIDE = "-1.49 -8.0 1.48"

def CameraProps(cameraOrig):
	props = pyluxcore.Properties()
	props.Set(pyluxcore.Property("scene.camera.lookat.orig", cameraOrig))
	props.Set(pyluxcore.Property("scene.camera.lookat.target", [cameraOrig[0], cameraOrig[1] + 1.0, cameraOrig[2]]))
	props.Set(pyluxcore.Property("scene.camera.autovolume.enable", True))
	return props

def SessionCameraVolume(session):
	props = session.GetRenderConfig().GetScene().ToProperties()
	return props.Get("scene.camera.volume").GetString() if props.IsDefined("scene.camera.volume") else None

def GetCameraVolume(engineType, cameraOrig, cameraTarget, edits = []):
	cfgProps = pyluxcore.Properties()
	cfgProps.SetFromFile("resources/scenes/simple/simple.cfg")
	cfgProps.Set(pyluxcore.Property("renderengine.type", engineType))
	cfgProps.Set(pyluxcore.Property("sampler.type", "SOBOL"))
	cfgProps.Set(GetDefaultEngineProperties(engineType))
	cfgProps.Set(LuxCoreTest.customConfigProps)
	cfgProps.Set(pyluxcore.Property("film.width", 16))
	cfgProps.Set(pyluxcore.Property("film.height", 16))
	cfgProps.Set(pyluxcore.Property("batch.haltspp", 1))
	cfgProps.Delete("batch.haltthreshold")
	# No CPU render thread: the GPU engine has no reason to build Embree
	if engineType.endswith("OCL"):
		cfgProps.Set(pyluxcore.Property("opencl.native.threads.count", 0))

	sceneProps = pyluxcore.Properties()
	sceneProps.SetFromString("""
		scene.volumes.fog.type = clear
		scene.volumes.fog.absorption = 0.5 0.5 0.5
		scene.volumes.fog.ior = 1.0
		scene.materials.foggy.type = matte
		scene.materials.foggy.kd = 0.5 0.5 0.5
		scene.materials.foggy.volume.interior = fog
		scene.objects.box1.ply = resources/scenes/simple/simple-mat-cube1.ply
		scene.objects.box1.material = foggy
		scene.camera.autovolume.enable = 1
		""")
	sceneProps.Set(pyluxcore.Property("scene.camera.lookat.orig", cameraOrig))
	sceneProps.Set(pyluxcore.Property("scene.camera.lookat.target", cameraTarget))

	scene = pyluxcore.Scene("resources/scenes/simple/simple.scn")
	scene.Parse(sceneProps)
	config = pyluxcore.RenderConfig(cfgProps, scene)
	session = pyluxcore.RenderSession(config)
	session.Start()
	try:
		volumes = [SessionCameraVolume(session)]
		# The camera moves in a scene edit: the volume is found again
		for edit in edits:
			session.BeginSceneEdit()
			session.GetRenderConfig().GetScene().Parse(CameraProps(edit))
			session.EndSceneEdit()
			volumes.append(SessionCameraVolume(session))
		return volumes
	finally:
		session.Stop()

def TestAutoVolume(cls, params):
	engineType = params[0]

	inside = [float(x) for x in CUBE_CENTER.split()]
	outside = [float(x) for x in OUTSIDE.split()]

	# Inside the cube: its interior volume. Outside: the exterior volume of the
	# cube, none. Then moved in and out of the cube by scene edits.
	cls.assertEqual(GetCameraVolume(engineType, inside, [inside[0], inside[1] + 1.0, inside[2]]), ["fog"])
	cls.assertEqual(GetCameraVolume(engineType, outside, [outside[0], outside[1] + 1.0, outside[2]]), [None])
	cls.assertEqual(GetCameraVolume(engineType, outside, [outside[0], outside[1] + 1.0, outside[2]],
			[inside, outside, inside]), [None, "fog", None, "fog"])

class AutoVolume(unittest.TestCase):
	pass

AutoVolume = AddTests(AutoVolume, TestAutoVolume,
		[e for e in GetTestCases() if (e[0] in ("PATHCPU", "PATHOCL")) and (e[1] == "SOBOL")])
