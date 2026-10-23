# -*- coding: utf-8 -*-
################################################################################
# Copyright 1998-2018 by authors (see AUTHORS.txt)
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
from pyluxcoreunittests.tests.render import *
from pyluxcoreunittests.tests.imagetest import *

def TestSceneEditRendering(cls, params):
	engineType = params[0]
	samplerType = params[1]
	renderConfigAdditionalProps = params[2]

	# Create the rendering configuration
	cfgProps = pyluxcore.Properties()
	cfgProps.SetFromFile("resources/scenes/simple/simple.cfg")

	# Set the rendering engine
	cfgProps.Set(pyluxcore.Property("renderengine.type", engineType))
	# Set the sampler
	cfgProps.Set(pyluxcore.Property("sampler.type", samplerType))

	cfgProps.Set(renderConfigAdditionalProps)
	cfgProps.Set(LuxCoreTest.customConfigProps)

	config = pyluxcore.RenderConfig(cfgProps)

	# Run the rendering
	StandardAnimTest(cls, "SceneEditRendering_" + engineType + ("" if not samplerType else ("_" + samplerType)), config, 5)

def SceneEdit(self, session, frame):
	config = session.GetRenderConfig()
	scene = config.GetScene()

	props = pyluxcore.Properties()
	props.SetFromString("""
		scene.objects.box2.ply = resources/scenes/simple/simple-mat-cube2.ply
		scene.objects.box2.material = greenmatte
		scene.objects.box2.transformation = 1 0 0 0  0 1 0 0  0 0 1 0  %f 0 0 1
		""" % ((frame + 1) / 5.0))

	scene.Parse(props)

class SceneEditRendering(ImageTest):
	def SceneEdit(self, session, frame):
		SceneEdit(self, session, frame)

SceneEditRendering = AddTests(SceneEditRendering, TestSceneEditRendering, GetTestCases())

################################################################################
# Scene edit with the PhotonGI indirect cache
################################################################################

# The light colour of each frame: the cache must be computed again after each
# edit, a stale cache keeps the indirect light of the previous colour
PGIC_LIGHT_COLORS = ["120.0 120.0 120.0", "120.0 0.0 0.0", "0.0 0.0 120.0"]

PGIC_PROPS = """
	path.photongi.indirect.enabled = 1
	path.photongi.photon.maxcount = 2000000
	"""

def PhotonGILightEdit(frame):
	# The whole material: a scene edit defines it again
	return pyluxcore.Properties().SetFromString("""
		scene.materials.whitelight.type = matte
		scene.materials.whitelight.kd = 0.0 0.0 0.0
		scene.materials.whitelight.emission = %s
		""" % PGIC_LIGHT_COLORS[frame])

def PhotonGIConfig(params, frame):
	engineType = params[0]
	samplerType = params[1]
	renderConfigAdditionalProps = params[2]

	cfgProps = pyluxcore.Properties()
	cfgProps.SetFromFile("resources/scenes/simple/simple.cfg")
	cfgProps.Set(pyluxcore.Property("renderengine.type", engineType))
	cfgProps.Set(pyluxcore.Property("sampler.type", samplerType))
	cfgProps.Set(renderConfigAdditionalProps)
	cfgProps.Set(pyluxcore.Properties().SetFromString(PGIC_PROPS))
	cfgProps.Set(LuxCoreTest.customConfigProps)

	sceneProps = pyluxcore.Properties()
	sceneProps.SetFromFile("resources/scenes/simple/simple.scn")
	sceneProps.Set(PhotonGILightEdit(frame))
	scene = pyluxcore.Scene()
	scene.Parse(sceneProps)

	return pyluxcore.RenderConfig(cfgProps, scene)

# Each frame of the edited session is compared with a new session of the same
# scene (no reference images)
def TestSceneEditRenderingPhotonGI(cls, params):
	name = "SceneEditRenderingPhotonGI_" + params[0] + "_" + params[1]

	session = pyluxcore.RenderSession(PhotonGIConfig(params, 0))
	session.Start()
	for frame in range(len(PGIC_LIGHT_COLORS)):
		if frame > 0:
			session.BeginSceneEdit()
			session.GetRenderConfig().GetScene().Parse(PhotonGILightEdit(frame))
			session.EndSceneEdit()

		DoRenderSessionWaitForDone(session)
		session.UpdateStats()
		edited = GetImagePipelineImage(session.GetFilm())
		edited.save(IMAGES_DIR + "/" + name + "-%04d.png" % frame)

		freshSession = DoRenderSession(PhotonGIConfig(params, frame))
		fresh = GetImagePipelineImage(freshSession.GetFilm())
		freshSession = None
		fresh.save(IMAGES_DIR + "/" + name + "-fresh-%04d.png" % frame)

		# Two caches differ by up to ~1.5% in the mean (each traces its own
		# photons); a stale cache moves it far more (the light changes colour)
		(sameImage, diffFraction, meanDiff) = CompareImageNoisy(edited, fresh, maxMeanDifference = 0.04)
		if not sameImage:
			cls.fail("%s frame %d differs from a new session (%.2f%% of the pixels differ, "
					"the mean brightness differs %.2f%%)" % (name, frame, 100.0 * diffFraction, 100.0 * meanDiff))
	session.Stop()

class SceneEditRenderingPhotonGI(ImageTest):
	pass

SceneEditRenderingPhotonGI = AddTests(SceneEditRenderingPhotonGI, TestSceneEditRenderingPhotonGI,
		[p for p in GetTestCases() if p[0] in ("PATHCPU", "PATHOCL") and p[1] == "SOBOL"])
