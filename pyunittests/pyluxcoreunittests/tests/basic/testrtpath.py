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

# RTPATHOCL pause, resume and camera edits (ROADMAP E6 (a)); the film
# generation and WaitNewFrame(generation, timeoutMs) (E6 (c))

import statistics
import threading
import time
import unittest
import pyluxcore

from pyluxcoreunittests.tests.utils import *
from pyluxcoreunittests.tests.build import *

# After the imports above: utils does "from array import *"
import array as pyarray

FILM_SIZE = 128

def CreateSession(engine="RTPATHOCL"):
	cfgProps = pyluxcore.Properties()
	sampler = "TILEPATHSAMPLER" if engine == "RTPATHOCL" else "RTPATHCPUSAMPLER"
	cfgProps.SetFromString("""
		renderengine.type = %s
		sampler.type = %s
		film.width = %d
		film.height = %d
		film.opencl.enable = 0
		opencl.native.threads.count = 0
		film.imagepipelines.0.0.type = TONEMAP_LINEAR
		film.imagepipelines.0.0.scale = 1.0
		""" % (engine, sampler, FILM_SIZE, FILM_SIZE))
	cfgProps.Set(LuxCoreTest.customConfigProps)

	scnProps = pyluxcore.Properties()
	scnProps.SetFromString("""
		scene.camera.lookat.orig = 0.0 5.0 2.0
		scene.camera.lookat.target = 0.0 0.0 0.0
		scene.materials.whitematte.type = matte
		scene.materials.whitematte.kd = 0.75 0.75 0.75
		scene.lights.distl.type = sharpdistant
		scene.lights.distl.color = 1.0 1.0 1.0
		scene.lights.distl.gain = 10.0 10.0 10.0
		scene.lights.distl.direction = 1.0 1.0 -1.0
		""")
	scnProps.Set(BuildPlane("plane1", "whitematte"))

	scene = pyluxcore.Scene()
	scene.Parse(scnProps)

	return pyluxcore.RenderSession(pyluxcore.RenderConfig(cfgProps, scene))

def SampleCount(session):
	session.UpdateStats()
	return session.GetStats().Get("stats.renderengine.total.samplecount").GetFloat()

def WaitForSamples(session, condition, timeout=10.0):
	end = time.time() + timeout
	while time.time() < end:
		count = SampleCount(session)
		if condition(count):
			return count
		time.sleep(0.001)
	raise AssertionError("No new samples in %g s" % timeout)

def FilmGeneration(session):
	session.UpdateStats()
	return session.GetStats().Get("stats.renderengine.film.generation").GetUnsignedLongLong()

def ImageMean(session):
	film = session.GetFilm()
	buffer = pyarray.array("f", [0.0] * (film.GetWidth() * film.GetHeight() * 3))
	film.GetOutputFloat(pyluxcore.FilmOutputType.RGB_IMAGEPIPELINE, buffer)
	return sum(buffer) / len(buffer)

def MoveCamera(session, lookUp):
	scene = session.GetRenderConfig().GetScene()
	session.BeginSceneEdit()
	if lookUp:
		# Nothing but the black sky above the plane
		scene.Parse(pyluxcore.Properties().SetFromString("""
			scene.camera.lookat.orig = 0.0 5.0 2.0
			scene.camera.lookat.target = 0.0 5.0 10.0
			scene.camera.up = 0.0 1.0 0.0
			"""))
	else:
		scene.Parse(pyluxcore.Properties().SetFromString("""
			scene.camera.lookat.orig = 0.0 5.0 2.0
			scene.camera.lookat.target = 0.0 0.0 0.0
			scene.camera.up = 0.0 0.0 1.0
			"""))
	session.EndSceneEdit()

@unittest.skipUnless(LuxCoreHasOpenCL(), "no OpenCL/HIP/CUDA device")
class RTPathOCL(unittest.TestCase):
	def test_RTPathOCL_CameraEditWhilePaused(self):
		# A camera edit while paused: nothing is rendered until Resume(), then
		# the film has the new camera only. (Before, the first frame after
		# Resume() still had the old camera, but the film was reset right
		# after it: what showed was one frame more of latency.)
		session = CreateSession()
		session.Start()
		try:
			WaitForSamples(session, lambda c: c > 0)
			time.sleep(0.2)
			self.assertGreater(ImageMean(session), 0.05)

			session.Pause()
			# Let the frame in flight end
			time.sleep(0.1)
			pausedCount = SampleCount(session)
			MoveCamera(session, True)
			# Nothing is rendered while paused
			time.sleep(0.1)
			self.assertEqual(SampleCount(session), pausedCount)

			session.Resume()
			WaitForSamples(session, lambda c: (c > 0) and (c != pausedCount))
			self.assertLess(ImageMean(session), 1e-3)
		finally:
			session.Stop()

	def test_RTPathOCL_ResumeLatency(self):
		# Render thread 0 waits for Resume() on a condition variable, not in
		# 100 ms sleeps: the next frame starts right away
		session = CreateSession()
		session.Start()
		try:
			WaitForSamples(session, lambda c: c > 0)

			latencies = []
			for i in range(10):
				session.Pause()
				time.sleep(0.05)
				pausedCount = SampleCount(session)
				MoveCamera(session, (i % 2) == 0)

				start = time.time()
				session.Resume()
				WaitForSamples(session, lambda c: (c > 0) and (c != pausedCount))
				latencies.append(time.time() - start)

			self.assertLess(statistics.median(latencies), 0.03)
		finally:
			session.Stop()

	def test_RTPathOCL_SceneEditWhilePaused(self):
		# A scene edit while paused syncs with render thread 0 right away: it
		# takes as long as the same edit while rendering (it waited for the
		# end of a 100 ms sleep)
		session = CreateSession()
		session.Start()
		try:
			WaitForSamples(session, lambda c: c > 0)
			scene = session.GetRenderConfig().GetScene()

			def Edit(i):
				start = time.time()
				session.BeginSceneEdit()
				scene.Parse(pyluxcore.Properties().SetFromString(
					"scene.lights.distl.color = %f 1.0 1.0" % (0.5 + 0.05 * i)))
				session.EndSceneEdit()
				return time.time() - start

			# The first edit can compile the kernels again
			Edit(0)

			# The edits while rendering first: after a Resume(), render thread 0
			# could still be in its sleep
			running = []
			for i in range(5):
				time.sleep(0.05)
				running.append(Edit(i + 1))

			paused = []
			for i in range(5):
				session.Pause()
				time.sleep(0.05)
				paused.append(Edit(i + 6))
				session.Resume()

			self.assertLess(statistics.median(paused), statistics.median(running) + 0.02)
		finally:
			session.Stop()

def CheckFilmGeneration(test, session):
	# The generation moves as the frames arrive, WaitNewFrame() returns as
	# soon as it moved, and it stays still while paused
	generation = session.WaitNewFrame(0, 10000)
	test.assertGreater(generation, 0)
	for i in range(5):
		newGeneration = session.WaitNewFrame(generation, 5000)
		test.assertGreater(newGeneration, generation)
		generation = newGeneration
	test.assertGreaterEqual(FilmGeneration(session), generation)

	session.Pause()
	# Let the frame in flight end
	time.sleep(0.1)
	generation = FilmGeneration(session)
	start = time.time()
	test.assertEqual(session.WaitNewFrame(generation, 100), generation)
	test.assertGreaterEqual(time.time() - start, 0.09)
	test.assertEqual(FilmGeneration(session), generation)

	session.Resume()
	start = time.time()
	test.assertGreater(session.WaitNewFrame(generation, 5000), generation)
	test.assertLess(time.time() - start, 1.0)

	# Stop() ends a wait in another thread
	session.Pause()
	time.sleep(0.1)
	generation = FilmGeneration(session)
	waitTime = []
	def Wait():
		start = time.time()
		session.WaitNewFrame(generation, 10000)
		waitTime.append(time.time() - start)
	waiter = threading.Thread(target=Wait)
	waiter.start()
	time.sleep(0.1)
	session.Stop()
	waiter.join(5.0)
	test.assertFalse(waiter.is_alive())
	test.assertLess(waitTime[0], 2.0)

@unittest.skipUnless(LuxCoreHasOpenCL(), "no OpenCL/HIP/CUDA device")
class RTPathOCLFilmGeneration(unittest.TestCase):
	def test_RTPathOCL_FilmGeneration(self):
		session = CreateSession()
		session.Start()
		try:
			CheckFilmGeneration(self, session)
		finally:
			if session.IsStarted():
				session.Stop()

	def test_RTPathOCL_FilmGenerationPerFrame(self):
		# One generation per frame merged in the film: a new generation
		# has new samples
		session = CreateSession()
		session.Start()
		try:
			generation = session.WaitNewFrame(0, 10000)
			count = SampleCount(session)
			for i in range(5):
				generation = session.WaitNewFrame(generation, 5000)
				newCount = SampleCount(session)
				self.assertNotEqual(newCount, count)
				count = newCount
		finally:
			session.Stop()

class RTPathCPUFilmGeneration(unittest.TestCase):
	def test_RTPathCPU_FilmGeneration(self):
		session = CreateSession("RTPATHCPU")
		session.Start()
		try:
			CheckFilmGeneration(self, session)
		finally:
			if session.IsStarted():
				session.Stop()

	def test_RTPathCPU_StopEndsFirstFrameWait(self):
		# WaitNewFrame() waits for every render thread's first frame: Stop()
		# from another thread ends the wait (it waited forever before when
		# the threads stopped before they all signalled)
		cfgProps = pyluxcore.Properties()
		cfgProps.SetFromString("""
			renderengine.type = RTPATHCPU
			sampler.type = RTPATHCPUSAMPLER
			film.width = 4000
			film.height = 4000
			rtpathcpu.zoomphase.size = 1
			""")
		scnProps = pyluxcore.Properties()
		scnProps.SetFromString("""
			scene.camera.lookat.orig = 0.0 5.0 2.0
			scene.camera.lookat.target = 0.0 0.0 0.0
			scene.materials.whitematte.type = matte
			scene.materials.whitematte.kd = 0.75 0.75 0.75
			scene.lights.distl.type = sharpdistant
			scene.lights.distl.direction = 1.0 1.0 -1.0
			""")
		scnProps.Set(BuildPlane("plane1", "whitematte"))
		scene = pyluxcore.Scene()
		scene.Parse(scnProps)

		for i in range(3):
			session = pyluxcore.RenderSession(pyluxcore.RenderConfig(cfgProps, scene))
			session.Start()
			waiter = threading.Thread(target=session.WaitNewFrame, daemon=True)
			waiter.start()
			time.sleep(0.05)
			session.Stop()
			waiter.join(5.0)
			self.assertFalse(waiter.is_alive())
