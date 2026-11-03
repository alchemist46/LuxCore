# -*- coding: utf-8 -*-
################################################################################
# Copyright 1998-2025 by authors (see AUTHORS.txt)
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

# The COLOR_ABERRATION image pipeline plugin (ROADMAP F10). The SPECTRAL mode
# leaves a flat image unchanged, gives the image of the model below (an
# isotropic magnification about the center, measured from the pixel centers)
# and the same image on a hardware device; a saved film keeps it

import os
import tempfile
import unittest
import pyluxcore

from pyluxcoreunittests.tests.utils import *
from pyluxcoreunittests.tests.build import *
from pyluxcoreunittests.tests.basic.testimagepipelinehw import HWDeviceIndex

# After the imports above: utils does "from array import *"
import array as pyarray

# Pipeline 0 is the image in front of the plugin
PIPELINES = """
	film.imagepipelines.0.0.type = NOP
	film.imagepipelines.1.0.type = COLOR_ABERRATION
	film.imagepipelines.1.0.amount = 0.05
	film.imagepipelines.2.0.type = COLOR_ABERRATION
	film.imagepipelines.2.0.mode = LEGACY
	film.imagepipelines.2.0.amount = 0.05
	"""

def Render(width, height, flat, hwEnable=0, hwDevice=0):
	cfgProps = pyluxcore.Properties()
	cfgProps.SetFromString("""
		renderengine.type = TILEPATHCPU
		sampler.type = TILEPATHSAMPLER
		tilepath.sampling.aa.size = 2
		native.threads.count = 1
		renderengine.seed = 7
		batch.haltspp = 4
		film.width = %d
		film.height = %d
		film.opencl.enable = %d
		film.hw.device = %d
		""" % (width, height, hwEnable, hwDevice) + PIPELINES)

	scnProps = pyluxcore.Properties()
	scnProps.SetFromString("""
		scene.camera.lookat.orig = 0.0 5.0 2.0
		scene.camera.lookat.target = 0.0 0.0 0.0
		scene.lights.sky.type = constantinfinite
		scene.lights.sky.color = 0.1 0.2 0.4
		""")
	if not flat:
		# A sharp edge (the horizon of the plane) and a gradient
		scnProps.SetFromString("""
			scene.materials.whitematte.type = matte
			scene.materials.whitematte.kd = 0.75 0.5 0.25
			scene.lights.distl.type = sharpdistant
			scene.lights.distl.direction = 1.0 1.0 -1.0
			""")
		scnProps.Set(BuildPlane("plane1", "whitematte"))
	scene = pyluxcore.Scene()
	scene.Parse(scnProps)

	session = pyluxcore.RenderSession(pyluxcore.RenderConfig(cfgProps, scene))
	session.Start()
	session.WaitForDone()
	session.Stop()

	return session.GetFilm()

def GetImages(film, count=3):
	width, height = film.GetWidth(), film.GetHeight()
	images = []
	for index in range(count):
		buffer = pyarray.array("f", [0.0] * (width * height * 3))
		film.GetOutputFloat(pyluxcore.FilmOutputType.RGB_IMAGEPIPELINE, buffer, index, True)
		images.append(buffer)
	return images

def MaxDiff(a, b):
	return max(abs(x - y) for x, y in zip(a, b))

#-------------------------------------------------------------------------------
# A model of the SPECTRAL mode in the default Rec.709 working space
#-------------------------------------------------------------------------------

# GlassMaterial's WaveLength2RGB() (src/slg/materials/glass.cpp)
def WaveLength2RGB(l):
	if 380 <= l < 440:
		r, g, b = -(l - 440) / (440 - 380), 0, 1
	elif 440 <= l < 490:
		r, g, b = 0, (l - 440) / (490 - 440), 1
	elif 490 <= l < 510:
		r, g, b = 0, 1, -(l - 510) / (510 - 490)
	elif 510 <= l < 580:
		r, g, b = (l - 510) / (580 - 510), 1, 0
	elif 580 <= l < 645:
		r, g, b = 1, -(l - 645) / (645 - 580), 0
	elif 645 <= l < 780:
		r, g, b = 1, 0, 0
	else:
		return (0, 0, 0)

	if 380 <= l < 420:
		f = .3 + .7 * (l - 380) / (420 - 380)
	elif 420 <= l < 700:
		f = 1
	else:
		f = .3 + .7 * (780 - l) / (780 - 700)
	# The normalization of glass.cpp cancels in the weights of a channel
	return (r * f, g * f, b * f)

def SpectralTable(amount, samples):
	d, F, C = 587.56, 486.13, 656.27
	table = []
	for i in range(samples):
		l = 380 + (i + .5) * 400 / samples
		dispersion = (1 / l**2 - 1 / d**2) / (1 / F**2 - 1 / C**2)
		table.append((amount * 2**.5 * dispersion, WaveLength2RGB(l)))
	sums = [sum(w[j] for _, w in table) for j in range(3)]
	return [(s, [w[j] / sums[j] for j in range(3)]) for s, w in table]

def Bilinear(image, width, height, x, y, channel):
	x1 = min(max(int(x // 1), 0), width - 1)
	y1 = min(max(int(y // 1), 0), height - 1)
	x2 = min(x1 + 1, width - 1)
	y2 = min(y1 + 1, height - 1)
	tx = min(max(x - x1, 0.), 1.)
	ty = min(max(y - y1, 0.), 1.)
	def p(px, py):
		return image[(py * width + px) * 3 + channel]
	return ((1 - tx) * (1 - ty) * p(x1, y1) + tx * (1 - ty) * p(x2, y1) +
			(1 - tx) * ty * p(x1, y2) + tx * ty * p(x2, y2))

def SpectralModel(image, width, height, amount, samples=16):
	table = SpectralTable(amount, samples)
	cx, cy = .5 * width, .5 * height
	result = [0.] * (width * height * 3)
	for y in range(height):
		dy = y + .5 - cy
		for x in range(width):
			dx = x + .5 - cx
			for s, w in table:
				for c in range(3):
					result[(y * width + x) * 3 + c] += w[c] * Bilinear(image, width, height,
							x + s * dx, y + s * dy, c)
	return result

class ColorAberration(unittest.TestCase):
	def test_ColorAberration_FlatImage(self):
		# 16:9, and not a multiple of the 8 pixels of the CPU vectors
		images = GetImages(Render(52, 29, True))
		self.assertGreater(min(images[0]), 0.05)
		for index in (1, 2):
			self.assertLess(MaxDiff(images[0], images[index]), 1e-6, "pipeline %d" % index)

	def test_ColorAberration_SpectralModel(self):
		width, height = 64, 36
		images = GetImages(Render(width, height, False))
		model = SpectralModel(images[0], width, height, .05)
		# The edge moves: the plugin changes the image
		self.assertGreater(MaxDiff(images[0], images[1]), .01)
		self.assertLess(MaxDiff(model, images[1]), 1e-5)

	@unittest.skipUnless(HWDeviceIndex() is not None, "no HIP or CUDA device")
	def test_ColorAberration_SameAsCPU(self):
		hwDevice = HWDeviceIndex()
		# 151x101 is not a multiple of the 256 pixels of a work group
		width, height = 151, 101
		cpu = GetImages(Render(width, height, False, 0, hwDevice))
		hw = GetImages(Render(width, height, False, 1, hwDevice))
		for index in (1, 2):
			self.assertLess(MaxDiff(cpu[index], hw[index]), 1e-5, "pipeline %d" % index)

	def test_ColorAberration_SaveLoad(self):
		filmA = Render(64, 36, False)
		with tempfile.TemporaryDirectory() as tmpDir:
			fileName = os.path.join(tmpDir, "coloraberration.flm")
			filmA.SaveFilm(fileName)
			filmB = pyluxcore.Film(fileName)
		imagesA = GetImages(filmA)
		imagesB = GetImages(filmB)
		for index in (1, 2):
			self.assertLess(MaxDiff(imagesA[index], imagesB[index]), 1e-6, "pipeline %d" % index)
