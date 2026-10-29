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

# pyluxcore releases the GIL in its long calls, and LuxCore's log lines reach
# the Python log handler from any thread (ROADMAP E6 (d))

import threading
import time
import unittest
import pyluxcore

from pyluxcoreunittests.tests.utils import *
from pyluxcoreunittests.tests.build import *

# After the imports above: utils does "from array import *"
import array as pyarray

def CreateSession(haltTime, engineType="PATHCPU", extraProps=""):
	cfgProps = pyluxcore.Properties()
	cfgProps.SetFromString("""
		renderengine.type = %s
		sampler.type = SOBOL
		film.width = 64
		film.height = 64
		film.opencl.enable = 0
		native.threads.count = 2
		batch.halttime = %d
		""" % (engineType, haltTime))
	cfgProps.SetFromString(extraProps)

	scnProps = pyluxcore.Properties()
	scnProps.SetFromString("""
		scene.camera.lookat.orig = 0.0 5.0 2.0
		scene.camera.lookat.target = 0.0 0.0 0.0
		scene.materials.whitematte.type = matte
		scene.materials.whitematte.kd = 0.75 0.75 0.75
		scene.lights.distl.type = sharpdistant
		scene.lights.distl.color = 1.0 1.0 1.0
		scene.lights.distl.direction = 1.0 1.0 -1.0
		""")
	scnProps.Set(BuildPlane("plane1", "whitematte"))

	scene = pyluxcore.Scene()
	scene.Parse(scnProps)

	return pyluxcore.RenderSession(pyluxcore.RenderConfig(cfgProps, scene))

class GIL(unittest.TestCase):
	def tearDown(self):
		# Back to the handler of the test runner
		from pyluxcoreunittests.main import LuxCoreLogHandler
		pyluxcore.SetLogHandler(LuxCoreLogHandler)

	def test_GIL_WaitForDoneInThread(self):
		# A thread waits in WaitForDone() while this one polls UpdateStats(),
		# which runs the halt test: with the GIL held by the waiting thread,
		# this one could not run at all
		session = CreateSession(1)
		session.Start()

		worker = threading.Thread(target=session.WaitForDone)
		start = time.time()
		worker.start()

		# The time of a short sleep: the waiting thread holding the GIL
		# would show here (not the time of this thread's own calls)
		maxGap = 0.0
		while worker.is_alive() and time.time() - start < 20.0:
			session.UpdateStats()
			before = time.time()
			time.sleep(0.01)
			maxGap = max(maxGap, time.time() - before)

		worker.join(1.0)
		self.assertFalse(worker.is_alive())
		session.Stop()

		self.assertGreater(time.time() - start, 0.9)
		self.assertLess(maxGap, 0.25)

	def test_GIL_RenderThreadLog(self):
		# The render threads never hold the GIL: their lines are queued and
		# delivered when a pyluxcore call takes it back. BiDir's AOV warm up
		# logs from render thread 0 in a release build too (with an ALBEDO channel).
		lines = []
		pyluxcore.SetLogHandler(lines.append)

		session = CreateSession(0, "BIDIRCPU", """
			path.aovs.warmup.spp = 1
			film.outputs.0.type = ALBEDO
			film.outputs.0.filename = albedo.exr
			""")
		session.Start()
		time.sleep(0.5)
		session.Stop()

		self.assertTrue(any("AOV warmup started" in l for l in lines))

	def test_GIL_LogFromStartInThread(self):
		# Start() in a thread: its own lines reach the handler too
		lines = []
		lineThreads = set()
		def Handler(msg):
			lines.append(msg)
			lineThreads.add(threading.current_thread().name)
		pyluxcore.SetLogHandler(Handler)

		session = CreateSession(0)
		worker = threading.Thread(target=session.Start, name="start")
		worker.start()
		worker.join(20.0)
		self.assertFalse(worker.is_alive())
		session.Stop()

		self.assertTrue(any("CPU render threads" in l for l in lines))
		self.assertIn("start", lineThreads)

	def test_GIL_GetOutputFloatInThread(self):
		# The film read in a thread while this one calls UpdateStats()
		session = CreateSession(0)
		session.Start()
		film = session.GetFilm()
		buffer = pyarray.array("f", [0.0] * (film.GetWidth() * film.GetHeight() * 3))

		errors = []
		reads = [0]
		stop = threading.Event()
		def Read():
			try:
				while not stop.is_set():
					film.GetOutputFloat(pyluxcore.FilmOutputType.RGB_IMAGEPIPELINE, buffer)
					reads[0] += 1
			except Exception as e:
				errors.append(e)

		worker = threading.Thread(target=Read)
		worker.start()
		start = time.time()
		while time.time() - start < 1.0:
			session.UpdateStats()
			time.sleep(0.005)
		stop.set()
		worker.join(20.0)
		self.assertFalse(worker.is_alive())
		session.Stop()

		self.assertEqual(errors, [])
		self.assertGreater(reads[0], 0)
		self.assertGreater(max(buffer), 0.0)
