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

# The lens models of the cameras: the thin lens controls of the perspective
# camera (ROADMAP F11) and the realistic camera (ROADMAP F12). Small emitting
# dots on a black background, their centroids and spot sizes against the
# geometry of the lens; the light tracing against the path tracing; the CPU
# against the GPU

import math
import os
import tempfile
import time
import unittest
import pyluxcore

from pyluxcoreunittests.tests.utils import *

# After the imports above: utils does "from array import *"
import array as pyarray

SIZE = 96
# One pass of TILEPATHCPU on one thread (16 samples per pixel): the same
# samples in the same order, for the bit identical renders
TILE_ONE_PASS = """
	native.threads.count = 1
	renderengine.type = TILEPATHCPU
	sampler.type = TILEPATHSAMPLER
	tilepath.sampling.aa.size = 4
	batch.haltspp = 1
	"""
# The perspective camera: at the origin, looking at +y, z up
FOV = 45.0
TAN_HALF = math.tan(math.radians(FOV / 2))

def Render(sceneProps, size=SIZE, spp=64, engine="PATHCPU", extraCfg="", seed=7):
	cfgProps = pyluxcore.Properties()
	cfgProps.SetFromString("""
		renderengine.type = %s
		sampler.type = SOBOL
		renderengine.seed = %d
		batch.haltspp = %d
		film.width = %d
		film.height = %d
		film.filter.type = BOX
		film.filter.width = 0.5
		film.imagepipelines.0.0.type = NOP
		""" % (engine, seed, spp, size, size))
	if engine.endswith("OCL"):
		cfgProps.Set(LuxCoreTest.customConfigProps)
		cfgProps.Set(pyluxcore.Property("opencl.native.threads.count", 0))
	cfgProps.SetFromString(extraCfg)

	scene = pyluxcore.Scene()
	scene.Parse(sceneProps)

	session = pyluxcore.RenderSession(pyluxcore.RenderConfig(cfgProps, scene))
	session.Start()
	# The halt conditions of the non-tile engines are checked by UpdateStats()
	while not session.HasDone():
		time.sleep(.01)
		session.UpdateStats()
	session.Stop()

	film = session.GetFilm()
	buffer = pyarray.array("f", [0.0] * (size * size * 3))
	film.GetOutputFloat(pyluxcore.FilmOutputType.RGB_IMAGEPIPELINE, buffer, 0, True)
	return buffer

def Camera(extra="", target="0.0 1.0 0.0"):
	return """
		scene.camera.type = perspective
		scene.camera.lookat.orig = 0.0 0.0 0.0
		scene.camera.lookat.target = %s
		scene.camera.up = 0.0 0.0 1.0
		scene.camera.fieldofview = %f
		%s
		""" % (target, FOV, extra)

# Small emitting squares facing the camera: (x, y, z, half size)
def Dots(dots, emission="1000.0 1000.0 1000.0"):
	s = """
		scene.materials.emit.type = matte
		scene.materials.emit.kd = 0.0 0.0 0.0
		scene.materials.emit.emission = %s
		""" % emission
	for i, (x, y, z, h) in enumerate(dots):
		s += """
			scene.objects.dot%d.material = emit
			scene.objects.dot%d.vertices = %f %f %f  %f %f %f  %f %f %f  %f %f %f
			scene.objects.dot%d.faces = 0 1 2  0 2 3
			""" % (i, i, x - h, y, z - h, x + h, y, z - h, x + h, y, z + h, x - h, y, z + h, i)
	return s

def Props(s):
	props = pyluxcore.Properties()
	props.SetFromString(s)
	return props

# The pixel (x right, y down, centers at .5) of the point (x, y, z) through the
# pinhole of Camera()
def PinholePixel(x, y, z, size=SIZE):
	return (size / 2 * (1 + x / y / TAN_HALF), size / 2 * (1 - z / y / TAN_HALF))

# The centroid and the RMS radius of the channel c (all if None) in the window
# of half size r about (cx, cy)
def Spot(image, cx, cy, r, c=None, size=SIZE):
	total, sx, sy = 0.0, 0.0, 0.0
	values = []
	for y in range(max(0, int(cy - r)), min(size, int(cy + r) + 1)):
		for x in range(max(0, int(cx - r)), min(size, int(cx + r) + 1)):
			# The film rows are bottom up
			i = ((size - 1 - y) * size + x) * 3
			v = sum(image[i:i + 3]) if c is None else image[i + c]
			values.append((x + .5, y + .5, v))
			total += v
			sx += v * (x + .5)
			sy += v * (y + .5)
	mx, my = sx / total, sy / total
	rms = math.sqrt(sum(v * ((x - mx) ** 2 + (y - my) ** 2) for x, y, v in values) / total)
	return mx, my, rms, total

# Lens_NormalizedDispersion() and the channel weights of
# GlassMaterial::WaveLength2RGB()
def NormalizedDispersion(l):
	return (1 / l ** 2 - 1 / 587.56 ** 2) / (1 / 486.13 ** 2 - 1 / 656.27 ** 2)

def WaveLength2RGB(l):
	if l < 440: r, g, b = -(l - 440) / 60, 0, 1
	elif l < 490: r, g, b = 0, (l - 440) / 50, 1
	elif l < 510: r, g, b = 0, 1, -(l - 510) / 20
	elif l < 580: r, g, b = (l - 510) / 70, 1, 0
	elif l < 645: r, g, b = 1, -(l - 645) / 65, 0
	else: r, g, b = 1, 0, 0
	f = .3 + .7 * (l - 380) / 40 if l < 420 else (1 if l < 701 else .3 + .7 * (780 - l) / 80)
	return r * f, g * f, b * f

def ChannelDispersion(c):
	ls = [380 + 400 * (i + .5) / 4000 for i in range(4000)]
	w = [WaveLength2RGB(l)[c] for l in ls]
	return sum(wi * NormalizedDispersion(l) for wi, l in zip(w, ls)) / sum(w)

class ThinLens(unittest.TestCase):
	def test_ThinLens_DefaultsUnchanged(self):
		# The controls at their defaults change nothing, with the polygon
		# aperture and the depth of field
		dofScene = Dots([(0.3, 5.0, 0.2, 0.05)]) + Camera("""
			scene.camera.lensradius = 0.1
			scene.camera.focaldistance = 3.0
			scene.camera.bokeh.blades = 6
			scene.camera.bokeh.distribution.type = UNIFORM
			""")
		oneThread = TILE_ONE_PASS
		a = Render(Props(dofScene), spp=16, extraCfg=oneThread)
		b = Render(Props(dofScene + """
			scene.camera.bokeh.rotation = 0
			scene.camera.bokeh.roundness = 0
			scene.camera.bokeh.obstruction = 0
			scene.camera.opticalvignetting.distance = 0
			scene.camera.distortion.k1 = 0
			scene.camera.aberration.coma = 0
			scene.camera.tilt.x = 0
			scene.camera.dispersion.lateral = 0
			"""), spp=16, extraCfg=oneThread)
		self.assertEqual(max(abs(p - q) for p, q in zip(a, b)), 0.)

	def test_ThinLens_Distortion(self):
		# The image height h of a film point sees the pinhole height
		# h (1 + k1 h^2): the dot is found back by Newton
		x, y, z = 1.2, 5.0, 0.9
		px, py = PinholePixel(x, y, z)
		halfDiagonal = SIZE / 2 * math.sqrt(2)
		for k1 in (0.2, -0.1):
			image = Render(Props(Dots([(x, y, z, 0.02)]) + Camera("scene.camera.distortion.k1 = %f" % k1)), spp=16)
			hu = math.hypot(px - SIZE / 2, py - SIZE / 2) / halfDiagonal
			h = hu
			for i in range(20):
				h -= (h * (1 + k1 * h * h) - hu) / (1 + 3 * k1 * h * h)
			ex = SIZE / 2 + (px - SIZE / 2) * h / hu
			ey = SIZE / 2 + (py - SIZE / 2) * h / hu
			mx, my, rms, total = Spot(image, ex, ey, 4)
			self.assertAlmostEqual(mx, ex, delta=0.3, msg="k1 %f" % k1)
			self.assertAlmostEqual(my, ey, delta=0.3, msg="k1 %f" % k1)

	def test_ThinLens_LightTracing(self):
		# A checkerboard lit from the camera: the light paths (LIGHTCPU) reach
		# the film through the distortion, the tilt and the optical vignetting
		# as the eye paths do (8x8 pixel blocks: the noise of the light
		# tracing, no bias)
		scene = Camera("""
			scene.camera.lensradius = 0.2
			scene.camera.focaldistance = 4.0
			scene.camera.distortion.k1 = 0.15
			scene.camera.tilt.x = 10
			scene.camera.opticalvignetting.distance = 0.5
			""") + """
			scene.lights.l.type = point
			scene.lights.l.position = 0.0 0.0 0.0
			scene.lights.l.color = 1.0 1.0 1.0
			scene.lights.l.gain = 20.0 20.0 20.0
			scene.textures.check.type = checkerboard2d
			scene.textures.check.texture1 = 0.8 0.8 0.8
			scene.textures.check.texture2 = 0.2 0.2 0.2
			scene.textures.check.mapping.uvscale = 8 8
			scene.materials.wall.type = matte
			scene.materials.wall.kd = check
			scene.objects.wall.material = wall
			scene.objects.wall.vertices = -4.0 5.0 -4.0  4.0 5.0 -4.0  4.0 5.0 4.0  -4.0 5.0 4.0
			scene.objects.wall.uvs = 0 0  1 0  1 1  0 1
			scene.objects.wall.faces = 0 1 2  0 2 3
			"""
		size = 64
		a = Render(Props(scene), size=size, spp=256, engine="PATHCPU")
		b = Render(Props(scene), size=size, spp=256, engine="LIGHTCPU")
		def Blocks(img):
			out = []
			for by in range(0, size, 8):
				for bx in range(0, size, 8):
					out.append(sum(img[((by + y) * size + bx + x) * 3 + c] for y in range(8) for x in range(8) for c in range(3)))
			return out
		ba, bb = Blocks(a), Blocks(b)
		mean = sum(ba) / len(ba)
		self.assertGreater(mean, 0.)
		self.assertLess(sum(abs(p - q) for p, q in zip(ba, bb)) / len(ba) / mean, 0.02)
		self.assertLess(abs(sum(bb) - sum(ba)) / sum(ba), 0.005)

	def test_ThinLens_OpticalVignetting(self):
		# A white background through a round aperture: each pixel gets the
		# share of the aperture inside the barrel (two circles: radius 1 and
		# scale, the offset distance tan / lens radius)
		lensRadius, distance, scale = 0.05, 0.08, 1.0
		base = """
			scene.lights.sky.type = constantinfinite
			scene.lights.sky.color = 1.0 1.0 1.0
			""" + Camera("""
			scene.camera.lensradius = %f
			scene.camera.focaldistance = 10.0
			""" % lensRadius)
		size = 32
		a = Render(Props(base), size=size, spp=64)
		b = Render(Props(base + """
			scene.camera.opticalvignetting.distance = %f
			scene.camera.opticalvignetting.scale = %f
			""" % (distance, scale)), size=size, spp=256)
		def Pixel(img, x, y):
			i = ((size - 1 - y) * size + x) * 3
			return sum(img[i:i + 3]) / 3
		def Share(d, r1, r2):
			if d >= r1 + r2: return 0.
			if d <= abs(r1 - r2): return min(r1, r2) ** 2 / r1 ** 2
			a1 = r1 ** 2 * math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
			a2 = r2 ** 2 * math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
			a3 = .5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
			return (a1 + a2 - a3) / (math.pi * r1 ** 2)
		for (x, y) in ((size // 2, size // 2), (0, 0), (size - 1, 0), (4, 20)):
			tanX = ((x + .5) / size * 2 - 1) * TAN_HALF
			tanY = (1 - (y + .5) / size * 2) * TAN_HALF
			share = Share(distance * math.hypot(tanX, tanY) / lensRadius, 1., scale)
			self.assertAlmostEqual(Pixel(a, x, y), 1., delta=0.02)
			self.assertAlmostEqual(Pixel(b, x, y), share, delta=0.03, msg="pixel %d %d" % (x, y))

	def test_ThinLens_Aberrations(self):
		# A dot on the plane of focus: sharp without aberrations, blurred by
		# the spherical one. Off the axis the coma moves the centroid of the
		# spot outward by coma h lens radius (its mean over the aperture)
		lensRadius, focus = 0.5, 5.0
		dof = "scene.camera.lensradius = %f\nscene.camera.focaldistance = %f\n" % (lensRadius, focus)
		image = Render(Props(Dots([(0., focus, 0., 0.01)]) + Camera(dof)), spp=64)
		sharp = Spot(image, SIZE / 2, SIZE / 2, 8)[2]
		image = Render(Props(Dots([(0., focus, 0., 0.01)]) + Camera(dof + "scene.camera.aberration.spherical = 0.3")), spp=64)
		blurred = Spot(image, SIZE / 2, SIZE / 2, 12)[2]
		# (0.71 for a dot between 4 pixels)
		self.assertLess(sharp, 0.75)
		self.assertGreater(blurred, 1.5 * sharp)

		x, z = 1.4, 0.0
		px, py = PinholePixel(x, focus, z)
		# (small: the shift grows with the image height across the comet, a
		# second order effect of 15% at 0.5)
		coma = 0.25
		image = Render(Props(Dots([(x, focus, z, 0.01)]) + Camera(dof + "scene.camera.aberration.coma = %f" % coma)), spp=256)
		mx, my, rms, total = Spot(image, px, py, 10)
		# The shift d at the image height h of the pixels the dot is seen
		# from: d = coma h(d) lens radius / focus, in pixels
		halfDiagonal = SIZE / 2 * math.sqrt(2)
		k = coma * lensRadius / focus / TAN_HALF * SIZE / 2
		expected = k * (px - SIZE / 2) / halfDiagonal / (1 - k / halfDiagonal)
		self.assertAlmostEqual(mx - px, expected, delta=0.15 * expected)
		self.assertAlmostEqual(my, py, delta=0.2)

	def test_ThinLens_Tilt(self):
		# A dot off the plane of focus but on the tilted one: blurred, then
		# sharp. The plane turned by tilt.x is at the depth focus + height
		# tan(tilt): the top farther
		lensRadius, focus, tilt = 1.0, 5.0, 30.0
		height = 1.5
		depth = focus + height * math.tan(math.radians(tilt))
		dot = Dots([(0., depth, height, 0.01)])
		dof = "scene.camera.lensradius = %f\nscene.camera.focaldistance = %f\n" % (lensRadius, focus)
		px, py = PinholePixel(0., depth, height)
		blurred = Spot(Render(Props(dot + Camera(dof)), spp=64), px, py, 12)[2]
		sharp = Spot(Render(Props(dot + Camera(dof + "scene.camera.tilt.x = %f" % tilt)), spp=64), px, py, 12)[2]
		self.assertLess(sharp, 0.75)
		self.assertGreater(blurred, 1.5 * sharp)

	def test_ThinLens_LateralDispersion(self):
		# The red image is larger than the blue one by the dispersion of their
		# channels
		x, y, z = 1.4, 5.0, 0.0
		px, py = PinholePixel(x, y, z)
		lateral = 0.05
		image = Render(Props(Dots([(x, y, z, 0.01)]) + Camera("scene.camera.dispersion.lateral = %f" % lateral)), spp=256)
		mr = Spot(image, px, py, 6, c=0)[0] - SIZE / 2
		mb = Spot(image, px, py, 6, c=2)[0] - SIZE / 2
		er = (px - SIZE / 2) * (1 + lateral * ChannelDispersion(0))
		eb = (px - SIZE / 2) * (1 + lateral * ChannelDispersion(2))
		self.assertAlmostEqual(mr - mb, er - eb, delta=0.15 * abs(er - eb))

	def test_ThinLens_NoLightTracing(self):
		# The aberrations have no inverse for the light paths
		scene = Props(Dots([(0., 5., 0., 0.01)]) + Camera("""
			scene.camera.lensradius = 0.1
			scene.camera.focaldistance = 5.0
			scene.camera.aberration.coma = 0.5
			"""))
		for engine in ("LIGHTCPU", "BIDIRCPU"):
			with self.assertRaises(RuntimeError, msg=engine):
				Render(scene, engine=engine, spp=1)

# The realistic camera (ROADMAP F12): the double-Gauss 50 mm f/2 on a 36 mm
# sensor
def Realistic(extra=""):
	return Camera("""
		scene.camera.type = realistic
		scene.camera.realistic.lens.preset = dgauss
		scene.camera.realistic.focallength = 50.0
		scene.camera.realistic.sensorwidth = 36.0
		""" + extra)

# The pixel of the angle (tan x, tan z) through a lens of focal length f (mm)
# on a 36 mm sensor
def LensPixel(tanX, tanZ, f=50.0, size=SIZE):
	return (size / 2 * (1 + tanX * f / 18.), size / 2 * (1 - tanZ * f / 18.))

# The built-in lens designs: their own focal length (mm), a field angle in the
# design (degrees, the image about 8-14 mm from the center) and its image
# height (mm) of a Python trace of the prescription at the d line, the focus at
# infinity (not in the tree)
PRESETS = {
	"triplet": (50., 15.0, 13.350),
	"tessar": (50., 15.375, 13.699),
	"heliar": (120., 5.0, 10.490),
	"petzval": (85., 5.25, 7.771),
	"telephoto": (200., 2.5, 8.695),
	"retrofocus": (27., 25.0, 11.528),
	"fisheye": (10., 41.0, 6.814),
}

class RealisticLens(unittest.TestCase):
	def test_RealisticLens_Exposure(self):
		# A white background: 1 at the center (the exposure of the thin lens),
		# darker toward the corners (the natural and the optical vignetting)
		scene = """
			scene.lights.sky.type = constantinfinite
			scene.lights.sky.color = 1.0 1.0 1.0
			""" + Realistic("scene.camera.focaldistance = 10.0")
		size = 32
		image = Render(Props(scene), size=size, spp=256)
		def Pixel(x, y):
			i = ((size - 1 - y) * size + x) * 3
			return sum(image[i:i + 3]) / 3
		center = sum(Pixel(size // 2 - i, size // 2 - j) for i in (0, 1) for j in (0, 1)) / 4
		self.assertAlmostEqual(center, 1., delta=0.02)
		corner = Pixel(0, 0)
		self.assertLess(corner, 0.8 * center)
		self.assertGreater(corner, 0.)

	def test_RealisticLens_FocusAndFieldOfView(self):
		# A dot at the focus distance: sharp, where the focal length puts it
		# (the lens distorts it by less than 1.5%); out of focus with the focus
		# at 0.8 m
		focus = 5.0
		tanX, tanZ = 0.15, 0.1
		dot = Dots([(tanX * focus, focus, tanZ * focus, 0.004)])
		image = Render(Props(dot + Realistic("scene.camera.focaldistance = %f\nscene.camera.realistic.fstop = 2.0" % focus)), spp=128)
		ex, ey = LensPixel(tanX, tanZ)
		mx, my, rms, total = Spot(image, ex, ey, 6)
		self.assertLess(rms, 0.8)
		# The focus at 5 m magnifies by about 1%
		self.assertAlmostEqual((mx - SIZE / 2) / (ex - SIZE / 2), 1., delta=0.03)
		self.assertAlmostEqual((my - SIZE / 2) / (ey - SIZE / 2), 1., delta=0.03)

		image = Render(Props(dot + Realistic("scene.camera.focaldistance = 0.8\nscene.camera.realistic.fstop = 2.0")), spp=128)
		self.assertGreater(Spot(image, ex, ey, 10)[2], 2 * rms)

	def test_RealisticLens_FStop(self):
		# The blur of a dot out of focus is about inversely proportional to
		# the f-number (a blur of about 5 pixels at f/4, a dot of 0.4 pixel,
		# the pixel adds its own spread)
		dot = Dots([(0., 0.3, 0., 0.0005)])
		spots = []
		for fstop in (4.0, 8.0):
			image = Render(Props(dot + Realistic("scene.camera.focaldistance = 5.0\nscene.camera.realistic.fstop = %f" % fstop)), spp=256)
			spots.append(Spot(image, SIZE / 2, SIZE / 2, 16)[2])
		self.assertAlmostEqual(spots[0] / spots[1], 1.9, delta=0.3)

	def test_RealisticLens_LensFile(self):
		# The preset written in a lens file (pbrt's 4 columns: the aperture
		# diameter, eta 0 for the stop) renders the same image
		rows = [
			(29.475, 3.76, 1.67, 25.2), (84.83, .12, 1, 25.2), (19.275, 4.025, 1.67, 23),
			(40.77, 3.275, 1.699, 23), (12.75, 5.705, 1, 18), (0, 4.5, 0, 17.1),
			(-14.495, 1.18, 1.603, 17), (40.77, 6.065, 1.658, 20), (-20.385, .19, 1, 20),
			(437.065, 3.22, 1.717, 20), (-39.73, 36, 1, 20)]
		dot = Dots([(0.3, 5.0, 0.2, 0.01)])
		oneThread = TILE_ONE_PASS
		a = Render(Props(dot + Realistic("scene.camera.focaldistance = 4.0")), spp=16, extraCfg=oneThread)
		with tempfile.TemporaryDirectory() as tmpDir:
			fileName = os.path.join(tmpDir, "dgauss.dat")
			with open(fileName, "w") as f:
				f.write("# The double-Gauss\n")
				for r in rows:
					f.write("%s %s %s %s\n" % r)
			b = Render(Props(dot + Realistic("scene.camera.focaldistance = 4.0\nscene.camera.realistic.lens.file = " + fileName.replace("\\", "/"))), spp=16, extraCfg=oneThread)
		self.assertEqual(max(abs(p - q) for p, q in zip(a, b)), 0.)

	def test_RealisticLens_Dispersion(self):
		# The lateral color of the lens: the red and blue images of an off-axis
		# dot (the field tan 0.3, 15 mm from the center of the sensor) separate
		# with the dispersion of the glasses (13 um between the F and C lines
		# for the chief ray), not without. The screen window crops 3.6 mm of
		# the sensor about the dot: 37.5 um pixels
		focus = 5.0
		tanX = 0.3
		dot = Dots([(tanX * focus, focus, 0., 0.001)])
		center = tanX * 50. / 18.
		window = "scene.camera.screenwindow = %f %f -0.1 0.1\n" % (center - 0.1, center + 0.1)
		separations = []
		for dispersion in (0, 1):
			image = Render(Props(dot + Realistic(window + "scene.camera.focaldistance = %f\nscene.camera.realistic.fstop = 4.0\nscene.camera.realistic.dispersion = %d" % (focus, dispersion))), spp=256)
			separations.append(abs(Spot(image, SIZE / 2, SIZE / 2, 12, c=0)[0] - Spot(image, SIZE / 2, SIZE / 2, 12, c=2)[0]))
		self.assertLess(separations[0], 0.05)
		self.assertGreater(separations[1], 0.15)

	def test_RealisticLens_Presets(self):
		# Each built-in lens at its own focal length and widest aperture: the
		# exposure at the center of the film 1, and a dot at 100 m, focused,
		# off the axis, sharp, where a Python trace of the
		# prescription puts its image (the image height in mm on the 36 mm
		# sensor, 0.375 mm a pixel, with the distortion of the lens)
		sky = """
			scene.lights.sky.type = constantinfinite
			scene.lights.sky.color = 1.0 1.0 1.0
			"""
		for name, (focal, field, height) in PRESETS.items():
			with self.subTest(preset=name):
				camera = Camera("""
					scene.camera.type = realistic
					scene.camera.realistic.lens.preset = %s
					scene.camera.realistic.sensorwidth = 36.0
					scene.camera.focaldistance = 100.0
					""" % name)
				size = 32
				image = Render(Props(sky + camera), size=size, spp=128)
				center = sum(image[((size // 2 - j) * size + size // 2 - i) * 3 + c] for i in (0, 1) for j in (0, 1) for c in range(3)) / 12
				self.assertAlmostEqual(center, 1., delta=0.03)

				# A square of 1.5 pixels (0.375 mm on the film): its centroid
				# between the pixels, its RMS radius 0.74 pixel with the box
				# filter of a pixel
				t = math.tan(math.radians(field))
				half = .75 * .375 / focal * 100.
				image = Render(Props(Dots([(t * 100., 100., 0., half)]) + camera), spp=64)
				ex = SIZE / 2 + height / 0.375
				mx, my, rms, total = Spot(image, ex, SIZE / 2, 6)
				self.assertAlmostEqual(mx, ex, delta=0.3)
				self.assertAlmostEqual(my, SIZE / 2, delta=0.3)
				self.assertLess(rms, 0.9)

	def test_RealisticLens_NoLightTracing(self):
		scene = Props(Dots([(0., 5., 0., 0.01)]) + Realistic("scene.camera.focaldistance = 5.0"))
		for engine in ("LIGHTCPU", "BIDIRCPU"):
			with self.assertRaises(RuntimeError, msg=engine):
				Render(scene, engine=engine, spp=1)

# The CPU and the GPU render the lenses the same: the means of two renders
# (two seeds) of each differ by less than 2 sqrt(2) / 2 of the noise of the
# two renders of a device (their 8x8 pixel blocks; at 1024 spp each feature
# alone is 1.1-1.45 times the noise)
GPU_CASES = {
	"thinlens": Camera("""
		scene.camera.lensradius = 0.15
		scene.camera.focaldistance = 4.0
		scene.camera.bokeh.blades = 6
		scene.camera.bokeh.distribution.type = UNIFORM
		scene.camera.bokeh.rotation = 15
		scene.camera.bokeh.roundness = 0.3
		scene.camera.bokeh.obstruction = 0.4
		scene.camera.opticalvignetting.distance = 0.5
		scene.camera.distortion.k1 = -0.1
		scene.camera.aberration.spherical = 0.05
		scene.camera.aberration.coma = 0.3
		scene.camera.aberration.astigmatism = 0.05
		scene.camera.aberration.fieldcurvature = 0.05
		scene.camera.tilt.x = 10
		scene.camera.dispersion.longitudinal = 0.05
		scene.camera.dispersion.lateral = 0.03
		"""),
	"realistic": Realistic("""
		scene.camera.focaldistance = 4.0
		scene.camera.realistic.fstop = 2.8
		scene.camera.bokeh.blades = 5
		scene.camera.realistic.dispersion = 1
		"""),
}

def TestLensGPU(cls, name):
	scene = GPU_CASES[name] + Dots([(x * 0.6, 5.0 + y, z * 0.6, 0.06) for x in (-2, 0, 2) for z in (-1, 1) for y in (-2, 2)])
	size = 64
	def Blocks(img):
		return [sum(img[((by + y) * size + bx + x) * 3 + c] for y in range(8) for x in range(8))
				for by in range(0, size, 8) for bx in range(0, size, 8) for c in range(3)]
	a, a2, b, b2 = [Blocks(Render(Props(scene), size=size, spp=256, engine=engine, seed=seed))
			for engine in ("PATHCPU", "PATHOCL") for seed in (7, 8)]
	cls.assertGreater(sum(a), 0.)
	noise = (sum(abs(p - q) for p, q in zip(a, a2)) + sum(abs(p - q) for p, q in zip(b, b2))) / 2
	difference = sum(abs((p + p2) - (q + q2)) / 2 for p, p2, q, q2 in zip(a, a2, b, b2))
	cls.assertLess(difference, 2 * math.sqrt(2) / 2 * noise)

class LensGPU(unittest.TestCase):
	pass

if LuxCoreHasOpenCL():
	LensGPU = AddTests(LensGPU, TestLensGPU, list(GPU_CASES))
