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

import os
import shutil
from array import *
import unittest
from PIL import Image, ImageChops, ImageStat

import pyluxcore

import logging
logger = logging.getLogger("pyunittests")

class LuxCoreTest(unittest.TestCase):
	customConfigProps = pyluxcore.Properties()

def LuxCoreHasOpenCL():
	if pyluxcore.GetPlatformDesc().Get("compile.LUXRAYS_DISABLE_OPENCL").GetBool():
		return False

	# A build with OpenCL support can still run on a machine without any
	# GPU (i.e. a CI runner): the *OCL engines can not run there
	return len(pyluxcore.GetOpenCLDeviceList()) > 0

def AddTests(cls, testFunc, opts):
	for input in opts:
		test = lambda self, i=input: testFunc(self, i)
		if isinstance(input, tuple):
			paramName = ""
			for i in [input[0], input[1]]:
				if i:
					paramName += ("" if paramName == "" else "_") + str(i)
		else:
			paramName = str(input)
		test.__name__ = "test_%s_%s" % (cls.__name__, paramName)
		setattr(cls, test.__name__, test)
	return cls

engineProperties = {
	"PATHCPU" : pyluxcore.Properties().SetFromString(
		"""
		film.width = 150
		film.height = 100
		batch.haltspp = 0
		batch.halttime = 0
		batch.haltthreshold = 0.075
		"""),
	"BIDIRCPU" : pyluxcore.Properties().SetFromString(
		"""
		film.width = 150
		film.height = 100
		batch.haltspp = 0
		batch.halttime = 0
		batch.haltthreshold = 0.075
		"""),
	"TILEPATHCPU" : pyluxcore.Properties().SetFromString(
		"""
		film.width = 150
		film.height = 100
		batch.haltspp = 0
		batch.halttime = 0
		batch.haltthreshold = 0.075
		tilepath.sampling.aa.size = 2
		"""),
	"PATHOCL" : pyluxcore.Properties().SetFromString(
		"""
		film.width = 150
		film.height = 100
		batch.haltspp = 0
		batch.halttime = 0
		batch.haltthreshold = 0.075
		"""),
	"TILEPATHOCL" : pyluxcore.Properties().SetFromString(
		"""
		film.width = 150
		film.height = 100
		batch.haltspp = 0
		batch.halttime = 0
		batch.haltthreshold = 0.075
		tilepath.sampling.aa.size = 2
		"""),
}

def GetDefaultEngineProperties(engineType):
	return engineProperties[engineType]

# The tuple is:
#  (<engine name>, <sampler name>, <render config additional properties>)
USE_SUBSET = False
def GetTestCases():
	el = [
		("PATHCPU", "SOBOL", GetDefaultEngineProperties("PATHCPU")),
	]

	if (not USE_SUBSET):
		el += [
			("TILEPATHCPU", "TILEPATHSAMPLER", GetDefaultEngineProperties("TILEPATHCPU")),
			("PATHCPU", "METROPOLIS", GetDefaultEngineProperties("PATHCPU")),
			("BIDIRCPU", "SOBOL", GetDefaultEngineProperties("BIDIRCPU")),
			("BIDIRCPU", "METROPOLIS", GetDefaultEngineProperties("BIDIRCPU")),
		]
	
		if LuxCoreHasOpenCL():
			el += [
			("PATHOCL", "SOBOL", GetDefaultEngineProperties("PATHOCL")),
			("PATHOCL", "METROPOLIS", GetDefaultEngineProperties("PATHOCL")),
			("TILEPATHOCL", "TILEPATHSAMPLER", GetDefaultEngineProperties("TILEPATHOCL"))
			]

	return el

def ConvertToImage(size, imageBufferFloat):
	imageBufferUChar = array('B', [max(0, min(255, int(v * 255.0 + 0.5))) for v in imageBufferFloat])
	return Image.frombuffer("RGB", size, bytes(imageBufferUChar), "raw", "RGB", 0, 1).transpose(Image.FLIP_TOP_BOTTOM)

def CompareImage(a, b):
	diff = ImageChops.difference(a, b)
	diff = diff.convert('L')
	# Map all no black pixels
	pointTable = ([0] + ([255] * 255))
	diff = diff.point(pointTable)
	
	(width, height) = diff.size
	count = 0
	for y in range(height):
		for x in range(width):
			if diff.getpixel((x, y)) != 0:
				count += 1

	if count > 0:
		new = diff.convert('RGB')
		new.paste(b, mask=diff)
		return False, count, new
	else:
		return True, 0, None

# The test renderings are halted at a noise threshold, so two runs never
# produce the same image: the comparison must tolerate the Monte Carlo noise
# but catch the real differences (wrong materials, lights, crashes, etc.).
#
# Two checks are done:
# - local differences: the images are down-sampled (to average the noise
#   out), a pixel differs if any of its channels differs more than
#   PIXEL_TOLERANCE (out of 255) and the test fails if more than
#   MAX_DIFFERENT_PIXELS of the pixels differ;
# - global differences (i.e. a light a bit brighter, an energy loss): the test
#   fails if the mean brightness of the images differs more than
#   MAX_MEAN_DIFFERENCE (relative, the noise averages out on the whole image).
#
# Calibrated on 3 runs of the PATHCPU tests: the worst noise was 1.3% of the
# pixels above the tolerance and 0.4% of mean difference, while a uniform 3%
# brightness change is detected.
DOWNSAMPLE_FACTOR = 2
PIXEL_TOLERANCE = 16
MAX_DIFFERENT_PIXELS = 0.03
MAX_MEAN_DIFFERENCE = 0.015

# Tests with a larger, known, mean brightness noise (image name: maximum)
MAX_MEAN_DIFFERENCE_OVERRIDES = {
}

# Set by the --strict command line option: fail if a reference is missing
# (instead of copying the current result as reference)
FAIL_ON_MISSING_REFERENCE = False
# Set by the --update-references command line option: overwrite the
# references with the current results (after an intended change)
UPDATE_REFERENCES = False

# Returns (same image, fraction of different pixels, relative mean difference)
def CompareImageNoisy(a, b, maxMeanDifference = MAX_MEAN_DIFFERENCE):
	if a.size != b.size:
		return False, 1.0, 1.0

	a = a.convert("RGB").reduce(DOWNSAMPLE_FACTOR)
	b = b.convert("RGB").reduce(DOWNSAMPLE_FACTOR)

	# The maximum channel difference of each pixel
	diff = ImageChops.difference(a, b)
	(r, g, bl) = diff.split()
	diff = ImageChops.lighter(ImageChops.lighter(r, g), bl)

	histogram = diff.histogram()
	diffCount = sum(histogram[PIXEL_TOLERANCE + 1:])
	diffFraction = diffCount / (diff.width * diff.height)

	# The mean brightness difference (with a floor for the very dark images)
	meanA = sum(ImageStat.Stat(a).mean) / 3.0
	meanB = sum(ImageStat.Stat(b).mean) / 3.0
	meanDiff = abs(meanA - meanB) / max(meanB, 8.0)

	sameImage = (diffFraction <= MAX_DIFFERENT_PIXELS) and (meanDiff <= maxMeanDifference)
	return sameImage, diffFraction, meanDiff

def CompareImageFiles(testCase, resultImageName, refImageName):
	if UPDATE_REFERENCES:
		shutil.copyfile(resultImageName, refImageName)
	elif os.path.isfile(refImageName):
		resultImage = Image.open(resultImageName)
		refImage = Image.open(refImageName)

		maxMeanDifference = MAX_MEAN_DIFFERENCE_OVERRIDES.get(os.path.basename(refImageName), MAX_MEAN_DIFFERENCE)
		(sameImage, diffFraction, meanDiff) = CompareImageNoisy(resultImage, refImage, maxMeanDifference)

		if not sameImage:
			# Save the differences
			if resultImage.size == refImage.size:
				(_, _, diffImage) = CompareImage(resultImage.convert("RGB"), refImage.convert("RGB"))
				(head, tail) = os.path.split(resultImageName)
				diffImage.save(head + "/diff-" + tail)

			testCase.fail("%s differs from %s (%.2f%% of the pixels differ, the maximum is %.2f%%; "
					"the mean brightness differs %.2f%%, the maximum is %.2f%%)" % (
					refImageName, resultImageName, 100.0 * diffFraction, 100.0 * MAX_DIFFERENT_PIXELS,
					100.0 * meanDiff, 100.0 * maxMeanDifference))
	elif FAIL_ON_MISSING_REFERENCE:
		testCase.fail("Missing reference image \"" + refImageName + "\"")
	else:
		# Copy the current image as reference
		logger.info("\nWARNING: missing reference image \"" + refImageName + "\". Copying the current result as reference.")
		shutil.copyfile(resultImageName, refImageName)
