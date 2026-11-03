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

# The GLARE image pipeline plugin (ROADMAP F10): it keeps the energy of a point
# source and a flat image, its diffraction has the Airy core of the f-number
# and 2N spikes for N odd straight blades (N for N even), a hardware device
# gives the same image and a saved film keeps it

import math
import os
import tempfile
import unittest
import pyluxcore

from pyluxcoreunittests.tests.utils import *
from pyluxcoreunittests.tests.basic.testimagepipelinehw import HWDeviceIndex

# After the imports above: utils does "from array import *"
import array as pyarray

SIZE = 128

def Render(pipelines, flat=False, size=SIZE, hwEnable=0, hwDevice=0):
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
		film.filter.type = BOX
		film.filter.width = 0.5
		film.imagepipelines.0.0.type = NOP
		film.opencl.enable = %d
		film.hw.device = %d
		""" % (size, size, hwEnable, hwDevice))
	for i, plugin in enumerate(pipelines):
		cfgProps.SetFromString("film.imagepipelines.%d.0.type = GLARE\n" % (i + 1))
		for k, v in plugin.items():
			cfgProps.SetFromString("film.imagepipelines.%d.0.%s = %s\n" % (i + 1, k, v))

	scnProps = pyluxcore.Properties()
	if flat:
		scnProps.SetFromString("""
			scene.camera.lookat.orig = 0.0 5.0 2.0
			scene.camera.lookat.target = 0.0 0.0 0.0
			scene.lights.sky.type = constantinfinite
			scene.lights.sky.color = 0.1 0.2 0.4
			""")
	else:
		# A bright square of about one pixel at the center of a black image
		half = 0.5 / size
		scnProps.SetFromString("""
			scene.camera.type = orthographic
			scene.camera.lookat.orig = 0.0 0.0 1.0
			scene.camera.lookat.target = 0.0 0.0 0.0
			scene.camera.up = 0.0 1.0 0.0
			scene.camera.screenwindow = -0.5 0.5 -0.5 0.5
			scene.materials.emit.type = matte
			scene.materials.emit.kd = 0.0 0.0 0.0
			scene.materials.emit.emission = 1000.0 1000.0 1000.0
			scene.objects.dot.material = emit
			scene.objects.dot.vertices = -{h} -{h} 0.0  -{h} {h} 0.0  {h} {h} 0.0  {h} -{h} 0.0
			scene.objects.dot.faces = 0 3 2  2 1 0
			""".replace("{h}", repr(half)))
	scene = pyluxcore.Scene()
	scene.Parse(scnProps)

	session = pyluxcore.RenderSession(pyluxcore.RenderConfig(cfgProps, scene))
	session.Start()
	session.WaitForDone()
	session.Stop()

	return session.GetFilm()

def GetImages(film, count):
	width, height = film.GetWidth(), film.GetHeight()
	images = []
	for index in range(count):
		buffer = pyarray.array("f", [0.0] * (width * height * 3))
		film.GetOutputFloat(pyluxcore.FilmOutputType.RGB_IMAGEPIPELINE, buffer, index, True)
		images.append(buffer)
	return images

def Channel(image, c):
	return image[c::3]

# The mean over the angle of the channel c at the distance r from the center
def RadialProfile(image, size, c, radii):
	center = size / 2
	profile = []
	for r in radii:
		total, count = 0., 0
		for x in range(size):
			for y in range(size):
				d = math.hypot(x + .5 - center, y + .5 - center)
				if abs(d - r) < .5:
					total += image[(y * size + x) * 3 + c]
					count += 1
		profile.append(total / count)
	return profile

# The bilinear value of the channel c at the angle a (radians) on the circle of
# radius r about the center
def RingSample(image, size, c, r, a):
	x = size / 2 + r * math.cos(a) - .5
	y = size / 2 + r * math.sin(a) - .5
	x0, y0 = int(math.floor(x)), int(math.floor(y))
	tx, ty = x - x0, y - y0
	def p(px, py):
		return image[(py * size + px) * 3 + c]
	return ((1 - tx) * (1 - ty) * p(x0, y0) + tx * (1 - ty) * p(x0 + 1, y0) +
			(1 - tx) * ty * p(x0, y0 + 1) + tx * ty * p(x0 + 1, y0 + 1))

def CountSpikes(image, size, r):
	n = 720
	ring = [sum(RingSample(image, size, c, r, 2 * math.pi * i / n) for c in range(3)) for i in range(n)]
	# The arcs of the ring above 60% of its maximum
	threshold = 0.6 * max(ring)
	return sum(1 for i in range(n) if ring[i] > threshold and ring[i - 1] <= threshold)

class Glare(unittest.TestCase):
	def test_Glare_Energy(self):
		# The point spread function stays in the image (radius 0.1 * diagonal)
		images = GetImages(Render([{}, {"scale": 20, "scatter": 0.1}]), 3)
		for c in range(3):
			energy = sum(Channel(images[0], c))
			self.assertGreater(energy, 100.)
			for index in (1, 2):
				# The plugin moved the light
				self.assertGreater(max(abs(a - b) for a, b in zip(images[0], images[index])), 1.)
				self.assertAlmostEqual(sum(Channel(images[index], c)) / energy, 1., delta=1e-4)

	def test_Glare_FlatImage(self):
		images = GetImages(Render([{}, {"scale": 20, "scatter": 0.1}], True, 52), 3)
		self.assertGreater(min(images[0]), 0.05)
		for index in (1, 2):
			maxDiff = max(abs(a - b) / b for a, b in zip(images[index], images[0]))
			self.assertLess(maxDiff, 1e-4, "pipeline %d" % index)

	def test_Glare_AiryCore(self):
		# A circular aperture: the first minimum of the radial profile (the
		# first dark ring of the Airy pattern, blurred by the bandwidth of the
		# channel) at 1.22 lambda N / pitch, lambda the mean wavelength of
		# the channel
		# A 3.6 mm sensor: a core of about 5.7 pixels
		fstop, scale, sensorWidth = 8., 30., 3.6
		images = GetImages(Render([{"blades": 0, "scatter": 0, "scale": scale,
				"fstop": fstop, "sensorwidth": sensorWidth}]), 2)
		pitch = sensorWidth / SIZE
		# The green channel of WaveLength2RGB() is centered near 545 nm
		expected = 1.22 * 545e-6 * fstop * scale / pitch
		radii = [0.5 * i for i in range(2, 30)]
		profile = RadialProfile(images[1], SIZE, 1, radii)
		minimum = next(radii[i] for i in range(1, len(radii) - 1)
				if profile[i] < profile[i - 1] and profile[i] <= profile[i + 1])
		self.assertAlmostEqual(minimum / expected, 1., delta=0.15)

	def test_Glare_Spikes(self):
		# Far from the core the straight blades make spikes: 2N for N odd, N
		# for N even
		# (a core of about 1.9 pixels, the kernel 54 pixels wide)
		images = GetImages(Render([{"blades": 5, "scatter": 0, "scale": 10, "sensorwidth": 3.6, "radius": 0.3},
				{"blades": 6, "scatter": 0, "scale": 10, "sensorwidth": 3.6, "radius": 0.3}]), 3)
		self.assertEqual(CountSpikes(images[1], SIZE, 20.), 10)
		self.assertEqual(CountSpikes(images[2], SIZE, 20.), 6)

	@unittest.skipUnless(HWDeviceIndex() is not None, "no HIP or CUDA device")
	def test_Glare_SameAsCPU(self):
		# The GPU FFTs have other padded sizes (the factors 2, 3 and 5): the
		# same image up to the rounding. 151 is padded to 200 on the GPU, the
		# flat 101 to 135.
		hwDevice = HWDeviceIndex()
		pipelines = [{}, {"scale": 20, "scatter": 0.1}, {"blades": 5, "scale": 10}]
		for flat, size in ((False, 151), (True, 101)):
			cpu = GetImages(Render(pipelines, flat, size, 0, hwDevice), 4)
			hw = GetImages(Render(pipelines, flat, size, 1, hwDevice), 4)
			for index in (1, 2, 3):
				peak = max(cpu[index])
				maxDiff = max(abs(a - b) for a, b in zip(cpu[index], hw[index]))
				self.assertLess(maxDiff / peak, 1e-5, "size %d pipeline %d" % (size, index))

	def test_Glare_SaveLoad(self):
		filmA = Render([{"scale": 20}])
		with tempfile.TemporaryDirectory() as tmpDir:
			fileName = os.path.join(tmpDir, "glare.flm")
			filmA.SaveFilm(fileName)
			filmB = pyluxcore.Film(fileName)
		imageA = GetImages(filmA, 2)[1]
		imageB = GetImages(filmB, 2)[1]
		# The loaded film runs its image pipeline on a hardware device if there
		# is one (film.opencl.enable is on by default): the rounding of the GPU
		# FFTs (see test_Glare_SameAsCPU)
		maxDiff = max(abs(a - b) for a, b in zip(imageA, imageB))
		self.assertLess(maxDiff / max(imageA), 1e-5)
