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

import array as pyarray
import time
import unittest
import pyluxcore

from pyluxcoreunittests.tests.utils import *

# The working space to Linear Rec.709 (include/slg/core/workingcolorspace_matrices.h)
TO_REC709 = {
	"rec2020": (
		(1.660490870e+00, -5.876411796e-01, -7.284992188e-02),
		(-1.245504022e-01, 1.132899880e+00, -8.349306881e-03),
		(-1.815076545e-02, -1.005789414e-01, 1.118729711e+00)),
	"acescg": (
		(1.705050945e+00, -6.217921376e-01, -8.325892687e-02),
		(-1.302563995e-01, 1.140804768e+00, -1.054826658e-02),
		(-2.400335483e-02, -1.289689988e-01, 1.152972341e+00)),
}

S = "resources/scenes/simple/"

# Directly visible emitters only (kd = 0 everywhere), using every input that is
# converted to the working space and the luminance of a colour (a mix amount):
# the image is linear in them, so a render in another space converted back to
# Rec.709 matches the Rec.709 render
EMITTERS_SCENE = """
scene.camera.lookat.orig = 8.34339 -4.09766 5.74567
scene.camera.lookat.target = -2.1735 1.36445 -0.197534
scene.textures.grid.type = imagemap
scene.textures.grid.file = resources/img/grid.png
scene.textures.grid.gamma = 2.2
scene.textures.grid.storage = float
scene.textures.bb.type = blackbody
scene.textures.bb.temperature = 3500
scene.textures.bb.normalize = 1
scene.textures.irr.type = irregulardata
scene.textures.irr.wavelengths = 400 500 600 700
scene.textures.irr.data = 0.2 0.8 0.4 0.1
scene.textures.irr.emission = 0
scene.textures.cgam.type = constfloat3
scene.textures.cgam.value = 0.8 0.3 0.1
scene.textures.cgam.colorspace = luxcore
scene.textures.cgam.colorspace.gamma = 2.2
scene.textures.lum.type = mix
scene.textures.lum.amount = irr
scene.textures.lum.texture1 = 0 0 0
scene.textures.lum.texture2 = 1 1 1
scene.materials.room.type = matte
scene.materials.room.kd = 0 0 0
scene.materials.room.emission = grid
scene.materials.cube1.type = matte
scene.materials.cube1.kd = 0 0 0
scene.materials.cube1.emission = bb
scene.materials.cube1.emission.gain = 3 3 3
scene.materials.sphere.type = matte
scene.materials.sphere.kd = 0 0 0
scene.materials.sphere.emission = cgam
scene.materials.cube2.type = matte
scene.materials.cube2.kd = 0 0 0
scene.materials.cube2.emission = 1 1 1
scene.materials.cube2.emission.temperature = 4000
scene.materials.cube2.emission.temperature.normalize = 1
scene.materials.back2.type = matte
scene.materials.back2.kd = 0 0 0
scene.materials.back2.emission = lum
scene.objects.room.ply = {S}room-nowall.ply
scene.objects.room.material = room
scene.objects.cube1.ply = {S}simple-mat-cube1.ply
scene.objects.cube1.material = cube1
scene.objects.cube2.ply = {S}simple-mat-cube2.ply
scene.objects.cube2.material = cube2
scene.objects.sphere.ply = {S}simple-mat-sphere1.ply
scene.objects.sphere.material = sphere
scene.objects.back2.ply = {S}simple-mat-back2.ply
scene.objects.back2.material = back2
scene.lights.sky.type = sky2
scene.lights.sky.dir = 0.166974 -0.59908 0.783085
scene.lights.sky.turbidity = 2.2
scene.lights.sky.gain = .00002 .00002 .00002
""".replace("{S}", S)

# A Disney sphere with its specular and sheen tints, lit directly by white
# light: linear in the base colour if the tint follows the working space
DISNEY_SCENE = """
scene.camera.lookat.orig = 8.34339 -4.09766 5.74567
scene.camera.lookat.target = -2.1735 1.36445 -0.197534
scene.textures.base.type = constfloat3
scene.textures.base.value = 0.8 0.3 0.1
scene.textures.base.colorspace = luxcore
scene.textures.base.colorspace.gamma = 2.2
scene.materials.grey.type = matte
scene.materials.grey.kd = 0.5 0.5 0.5
scene.materials.disney.type = disney
scene.materials.disney.basecolor = base
scene.materials.disney.roughness = 0.4
scene.materials.disney.specular = 1
scene.materials.disney.speculartint = 1
scene.materials.disney.sheen = 1
scene.materials.disney.sheentint = 1
scene.materials.light.type = matte
scene.materials.light.kd = 0 0 0
scene.materials.light.emission = 20 20 20
scene.objects.room.ply = {S}room-nowall.ply
scene.objects.room.material = grey
scene.objects.lights.ply = {S}simple-lights.ply
scene.objects.lights.material = light
scene.objects.sphere.ply = {S}simple-mat-sphere1.ply
scene.objects.sphere.material = disney
""".replace("{S}", S)

# A 4000 K blackbody emitter on black
BLACKBODY_SCENE = """
scene.camera.lookat.orig = 8.34339 -4.09766 5.74567
scene.camera.lookat.target = -2.1735 1.36445 -0.197534
scene.textures.bb.type = blackbody
scene.textures.bb.temperature = 4000
scene.textures.bb.normalize = 1
scene.materials.bb.type = matte
scene.materials.bb.kd = 0 0 0
scene.materials.bb.emission = bb
scene.objects.sphere.ply = {S}simple-mat-sphere1.ply
scene.objects.sphere.material = bb
""".replace("{S}", S)

def ParseScene(text):
	props = pyluxcore.Properties()
	props.SetFromString(text)
	scene = pyluxcore.Scene()
	scene.Parse(props)

	return scene

def RenderRGB(space, sceneText=EMITTERS_SCENE, directOnly=False, imagePipeline=None):
	scene = ParseScene("scene.colorspace.working = " + space + "\n" + sceneText)

	# TILEPATHCPU with 1 thread is deterministic
	width, height = 64, 48
	cfgProps = pyluxcore.Properties()
	cfgProps.SetFromString("""
		renderengine.type = TILEPATHCPU
		renderengine.seed = 1
		sampler.type = TILEPATHSAMPLER
		native.threads.count = 1
		tile.multipass.enable = 0
		tilepath.sampling.aa.size = 2
		film.width = {}
		film.height = {}
		film.outputs.0.type = RGB
		film.outputs.0.filename = rgb.exr
		""".format(width, height))
	if directOnly:
		cfgProps.Set(pyluxcore.Property("path.pathdepth.total", 1))
		cfgProps.Set(pyluxcore.Property("path.pathdepth.diffuse", 1))
	if imagePipeline:
		cfgProps.SetFromString(imagePipeline)
	config = pyluxcore.RenderConfig(cfgProps, scene)
	session = pyluxcore.RenderSession(config)
	session.Start()
	while not session.HasDone():
		time.sleep(0.05)
		session.UpdateStats()
	session.Stop()

	buf = pyarray.array("f", [0.0] * (width * height * 3))
	session.GetFilm().GetOutputFloat(pyluxcore.FilmOutputType.RGB_IMAGEPIPELINE if imagePipeline else pyluxcore.FilmOutputType.RGB, buf)

	return buf

class TestWorkingColorSpace(unittest.TestCase):
	def test_WorkingColorSpace_Default(self):
		scene = ParseScene(EMITTERS_SCENE)
		self.assertFalse(scene.ToProperties().IsDefined("scene.colorspace.working"))

	def test_WorkingColorSpace_Export(self):
		for space in ("rec2020", "acescg"):
			scene = ParseScene("scene.colorspace.working = " + space + "\n" + EMITTERS_SCENE)
			self.assertEqual(scene.ToProperties().Get("scene.colorspace.working").GetString(), space)

	def test_WorkingColorSpace_Invalid(self):
		with self.assertRaises(RuntimeError):
			ParseScene("scene.colorspace.working = srgb\n")

	def test_WorkingColorSpace_Locked(self):
		scene = ParseScene("scene.colorspace.working = rec2020\n" + EMITTERS_SCENE)

		# Setting the same value again is fine
		props = pyluxcore.Properties()
		props.SetFromString("scene.colorspace.working = rec2020\n")
		scene.Parse(props)

		# Changing it once the scene has content is not
		props.SetFromString("scene.colorspace.working = acescg\n")
		with self.assertRaises(RuntimeError):
			scene.Parse(props)

	# meanError: compare the mean of the pixel errors instead of their maximum
	def CheckLinear(self, sceneText, directOnly, tolerance=1e-4, meanError=False):
		ref = RenderRGB("rec709", sceneText, directOnly)
		scale = max(abs(v) for v in ref)
		self.assertGreater(scale, 0.0)

		for space, m in TO_REC709.items():
			img = RenderRGB(space, sceneText, directOnly)

			maxConverted = 0.0
			sumConverted = 0.0
			maxRaw = 0.0
			for i in range(0, len(ref), 3):
				for c in range(3):
					v = m[c][0] * img[i] + m[c][1] * img[i + 1] + m[c][2] * img[i + 2]
					maxConverted = max(maxConverted, abs(v - ref[i + c]) / scale)
					sumConverted += abs(v - ref[i + c]) / scale
					maxRaw = max(maxRaw, abs(img[i + c] - ref[i + c]) / scale)
			if meanError:
				maxConverted = sumConverted / len(ref)

			# The same image, up to the float rounding of the conversions
			self.assertLess(maxConverted, tolerance, space)
			# ...but not the same pixel values
			self.assertGreater(maxRaw, 1e-2, space)

	def test_WorkingColorSpace_LinearInputs(self):
		self.CheckLinear(EMITTERS_SCENE, False)

	def test_WorkingColorSpace_DisneyTint(self):
		# Disney is not exactly linear in the colour since 2026-10-18: the
		# energy compensation scale 1 + Fms (1 - E) / E depends on the average
		# Fresnel of each channel, and the coloured specular layer scales the
		# diffuse one per channel. On average the image changes by 0.04% with
		# the working space, but single pixels differ by up to 2% (1.4% with 16
		# times the samples, so it is partly noise): the test compares the mean
		# error. The tint, the sheen colour clamp and the lobe probabilities
		# are computed in Rec.709 or by luminance, so they don't depend on the
		# working space.
		self.CheckLinear(DISNEY_SCENE, True, 1e-3, meanError=True)

	def CheckImagePipeline(self, imagePipeline):
		# The image pipeline converts to Rec.709 before the gamma, so its output
		# doesn't depend on the working space when the image is linear in the
		# inputs. (The 4096 entries of the gamma table round the result.)
		ref = RenderRGB("rec709", imagePipeline=imagePipeline)
		self.assertGreater(max(ref), 0.0)

		for space in TO_REC709:
			img = RenderRGB(space, imagePipeline=imagePipeline)
			maxDiff = max(abs(a - b) for a, b in zip(img, ref))
			self.assertLess(maxDiff, 2e-3, space)

	def test_WorkingColorSpace_GammaPipeline(self):
		self.CheckImagePipeline("""
			film.imagepipelines.0.0.type = TONEMAP_LINEAR
			film.imagepipelines.0.0.scale = 1.0
			film.imagepipelines.0.1.type = GAMMA_CORRECTION
			film.imagepipelines.0.1.value = 2.2
			""")

	def test_WorkingColorSpace_AutoLinear(self):
		self.CheckImagePipeline("""
			film.imagepipelines.0.0.type = TONEMAP_AUTOLINEAR
			film.imagepipelines.0.1.type = GAMMA_CORRECTION
			film.imagepipelines.0.1.value = 2.2
			""")

	def test_WorkingColorSpace_WhiteBalance(self):
		# A white balance at a temperature makes a blackbody emitter at that
		# temperature neutral, in every working space (it is a per channel
		# scale, so the other colours depend on the space)
		pipeline = """
			film.imagepipelines.0.0.type = TONEMAP_LINEAR
			film.imagepipelines.0.0.scale = 1.0
			film.imagepipelines.0.1.type = WHITE_BALANCE
			film.imagepipelines.0.1.temperature = 4000
			film.imagepipelines.0.1.normalize = 1
			"""
		for space in ["rec709"] + list(TO_REC709):
			img = RenderRGB(space, BLACKBODY_SCENE, imagePipeline=pipeline)
			maxValue = max(img)
			self.assertGreater(maxValue, 0.0, space)
			maxDiff = max(max(abs(img[i] - img[i + 1]), abs(img[i + 1] - img[i + 2])) for i in range(0, len(img), 3))
			self.assertLess(maxDiff / maxValue, 1e-4, space)

	#---------------------------------------------------------------------------
	# Blackbody temperatures normalized to luminance 1
	#---------------------------------------------------------------------------

	# Blender 5.2.2's Light.temperature_color (Cycles' light temperature and
	# Blackbody node) in each working space
	CYCLES_TEMPERATURE_COLOR = {
		"rec709": {
			2000: (2.520095, 0.646851, 0.020413),
			3000: (1.770841, 0.844285, 0.272223),
			6500: (1.042552, 0.984053, 1.035272),
			10000: (0.871384, 0.994925, 1.430509)},
		"rec2020": {
			2000: (1.794998, 0.769169, 0.116521),
			3000: (1.400831, 0.901807, 0.347136),
			6500: (1.022973, 0.988676, 1.030883),
			10000: (0.936280, 0.991336, 1.383007)},
	}

	# The colour of a constant infinite light with the temperature, seen
	# directly (every pixel has it)
	def TemperatureLightRGB(self, space, temperature, normalizationProps):
		img = RenderRGB(space, """
			scene.camera.lookat.orig = 0 0 0
			scene.camera.lookat.target = 0 1 0
			scene.lights.l.type = constantinfinite
			scene.lights.l.color = 1 1 1
			scene.lights.l.temperature = {}
			""".format(temperature) + normalizationProps.replace("PREFIX", "scene.lights.l.temperature"))
		return tuple(img[0:3])

	def LuminanceWeights(self, space):
		w = (0.212671, 0.715160, 0.072169)
		if space == "rec709":
			return w
		m = TO_REC709[space]
		return tuple(sum(m[y][x] * w[y] for y in range(3)) for x in range(3))

	def test_WorkingColorSpace_TemperatureLuminance(self):
		for space in ("rec709", "rec2020", "acescg"):
			w = self.LuminanceWeights(space)
			for t in (2000, 3000, 6500, 10000):
				c = self.TemperatureLightRGB(space, t, "PREFIX.normalization = luminance\n")
				# Luminance 1 in the working space
				self.assertAlmostEqual(sum(a * b for a, b in zip(w, c)), 1.0, delta=1e-4, msg=(space, t))
				# Cycles' colour (its fit of the blackbody against the
				# spectrum here: within 0.3% from 2000 to 12000 K)
				if space in self.CYCLES_TEMPERATURE_COLOR:
					ref = self.CYCLES_TEMPERATURE_COLOR[space][t]
					for a, b in zip(c, ref):
						self.assertAlmostEqual(a, b, delta=3e-3 * max(b, 0.1), msg=(space, t))

	def test_WorkingColorSpace_TemperatureModes(self):
		w = self.LuminanceWeights("rec709")
		# normalize = 1 is the same as normalization = max, as before: the
		# luminance follows the temperature (0.72 at 6500 K)
		maxRGB = self.TemperatureLightRGB("rec709", 6500, "PREFIX.normalize = 1\n")
		self.assertEqual(maxRGB, self.TemperatureLightRGB("rec709", 6500, "PREFIX.normalization = max\n"))
		self.assertAlmostEqual(sum(a * b for a, b in zip(w, maxRGB)), 0.7244, delta=1e-3)
		# normalization takes precedence over normalize
		self.assertEqual(maxRGB, self.TemperatureLightRGB("rec709", 6500, "PREFIX.normalize = 0\nPREFIX.normalization = max\n"))
		noneRGB = self.TemperatureLightRGB("rec709", 6500, "PREFIX.normalize = 1\nPREFIX.normalization = none\n")
		self.assertGreater(min(noneRGB), 1000.0)
		self.assertEqual(noneRGB, self.TemperatureLightRGB("rec709", 6500, ""))

		with self.assertRaises(RuntimeError):
			ParseScene("""
				scene.lights.l.type = constantinfinite
				scene.lights.l.temperature = 6500
				scene.lights.l.temperature.normalization = lum
				""")

	def test_WorkingColorSpace_TemperatureEmitters(self):
		# The blackbody texture and the emission temperature use the same
		# luminance normalization as the lights: the same image as a constant
		# emission of the light's colour
		for space in ("rec709", "rec2020"):
			c = self.TemperatureLightRGB(space, 3000, "PREFIX.normalization = luminance\n")
			ref = RenderRGB(space, BLACKBODY_SCENE.replace("scene.materials.bb.emission = bb",
					"scene.materials.bb.emission = {} {} {}".format(*c)))
			scale = max(ref)
			self.assertGreater(scale, 0.0)

			tex = RenderRGB(space, BLACKBODY_SCENE.replace("scene.textures.bb.temperature = 4000",
					"scene.textures.bb.temperature = 3000\nscene.textures.bb.normalization = luminance"))
			emission = RenderRGB(space, BLACKBODY_SCENE.replace("scene.materials.bb.emission = bb",
					"scene.materials.bb.emission = 1 1 1\n"
					"scene.materials.bb.emission.temperature = 3000\n"
					"scene.materials.bb.emission.temperature.normalization = luminance"))
			for img in (tex, emission):
				self.assertLess(max(abs(a - b) for a, b in zip(img, ref)) / scale, 1e-5, space)

	def test_WorkingColorSpace_TemperatureExport(self):
		# .normalization is exported only for luminance, so the scenes of the
		# other modes stay readable by the older builds
		text = """
			scene.textures.bb.type = blackbody
			scene.textures.bb.temperature = 3000
			scene.textures.bb.normalization = {0}
			scene.materials.m.type = matte
			scene.materials.m.kd = 0 0 0
			scene.materials.m.emission = bb
			scene.materials.m.emission.temperature = 3000
			scene.materials.m.emission.temperature.normalization = {0}
			scene.objects.sphere.ply = {1}simple-mat-sphere1.ply
			scene.objects.sphere.material = m
			scene.lights.l.type = constantinfinite
			scene.lights.l.temperature = 3000
			scene.lights.l.temperature.normalization = {0}
			"""
		prefixes = ("scene.textures.bb", "scene.materials.m.emission.temperature", "scene.lights.l.temperature")
		for mode, normalize in (("none", False), ("max", True), ("luminance", True)):
			props = ParseScene(text.format(mode, S)).ToProperties()
			for prefix in prefixes:
				self.assertEqual(props.Get(prefix + ".normalize").GetBool(), normalize, (mode, prefix))
				if mode == "luminance":
					self.assertEqual(props.Get(prefix + ".normalization").GetString(), "luminance", prefix)
				else:
					self.assertFalse(props.IsDefined(prefix + ".normalization"), (mode, prefix))
