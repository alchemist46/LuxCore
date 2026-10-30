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

# The film image pipeline on a hardware device (film.opencl.enable = 1) gives
# the image of the CPU pipeline: the merged image stays on the device until a
# plugin runs on the CPU (ROADMAP E6 (b)), and a film size that is not a
# multiple of the work group size must not write past the buffers

import unittest
import pyluxcore

from pyluxcoreunittests.tests.utils import *
from pyluxcoreunittests.tests.build import *

# After the imports above: utils does "from array import *"
import array as pyarray

# MIST runs on the CPU only, the other plugins on the device
PIPELINES = """
	film.imagepipelines.0.0.type = TONEMAP_LINEAR
	film.imagepipelines.0.0.scale = 0.8
	film.imagepipelines.0.1.type = GAMMA_CORRECTION
	film.imagepipelines.0.1.value = 2.2
	film.imagepipelines.1.0.type = MIST
	film.imagepipelines.1.0.amount = 0.3
	film.imagepipelines.1.0.enddistance = 8
	film.imagepipelines.1.1.type = TONEMAP_LINEAR
	film.imagepipelines.1.1.scale = 0.8
	film.imagepipelines.2.0.type = TONEMAP_LINEAR
	film.imagepipelines.2.0.scale = 0.8
	film.imagepipelines.2.1.type = MIST
	film.imagepipelines.2.1.amount = 0.3
	film.imagepipelines.2.1.enddistance = 8
	film.imagepipelines.2.2.type = GAMMA_CORRECTION
	film.imagepipelines.2.2.value = 2.2
	film.outputs.0.type = DEPTH
	film.outputs.0.filename = depth.exr
	"""

def HWDeviceIndex():
	# The film picks its device in the list of the hardware devices; the
	# OpenCL image pipeline programs crash some drivers (AMD on Windows), so
	# only a HIP or CUDA device
	if not LuxCoreHasOpenCL():
		return None
	for i, desc in enumerate(pyluxcore.GetOpenCLDeviceList()):
		if desc[1] in ("HIP_GPU", "CUDA_GPU"):
			return i
	return None

def RenderImagePipelines(width, height, hwEnable, hwDevice):
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
		scene.materials.whitematte.type = matte
		scene.materials.whitematte.kd = 0.75 0.5 0.25
		scene.lights.distl.type = sharpdistant
		scene.lights.distl.direction = 1.0 1.0 -1.0
		scene.lights.sky.type = constantinfinite
		scene.lights.sky.color = 0.1 0.2 0.4
		""")
	scnProps.Set(BuildPlane("plane1", "whitematte"))
	scene = pyluxcore.Scene()
	scene.Parse(scnProps)

	session = pyluxcore.RenderSession(pyluxcore.RenderConfig(cfgProps, scene))
	session.Start()
	session.WaitForDone()
	session.Stop()

	film = session.GetFilm()
	images = []
	for index in range(3):
		buffer = pyarray.array("f", [0.0] * (width * height * 3))
		film.GetOutputFloat(pyluxcore.FilmOutputType.RGB_IMAGEPIPELINE, buffer, index, True)
		images.append(buffer)
	return images

@unittest.skipUnless(HWDeviceIndex() is not None, "no HIP or CUDA device")
class ImagePipelineHW(unittest.TestCase):
	def test_ImagePipelineHW_SameAsCPU(self):
		hwDevice = HWDeviceIndex()
		# 151x101 is not a multiple of the 256 pixels of a work group
		for width, height in ((151, 101), (128, 128)):
			cpu = RenderImagePipelines(width, height, 0, hwDevice)
			hw = RenderImagePipelines(width, height, 1, hwDevice)
			for index in range(3):
				self.assertGreater(sum(cpu[index]) / len(cpu[index]), 0.1)
				maxDiff = max(abs(a - b) for a, b in zip(cpu[index], hw[index]))
				self.assertLess(maxDiff, 1e-5, "%dx%d pipeline %d" % (width, height, index))
