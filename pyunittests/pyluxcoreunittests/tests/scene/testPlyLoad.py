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

import math
import os
import random
import struct
import tempfile
import unittest
import pyluxcore

# The binary PLY meshes are read by a fast path (ReadPlyBinary() in
# exttrianglemeshfile.cpp), the others by rply. Each mesh here is written in
# both formats: the binary file goes through the fast path (or falls back to
# rply for what the fast path does not read) and the ASCII one through rply,
# and the two loaded meshes must be the same, value for value.

STRUCT_TYPES = {"char": "b", "uchar": "B", "uint8": "B", "short": "h", "ushort": "H",
		"int": "i", "uint": "I", "float": "f", "float32": "f", "double": "d"}

def MakeVertices(props, count, rnd):
	vertices = []
	for i in range(count):
		v = []
		for name, t in props:
			c = STRUCT_TYPES[t]
			if c in "fd":
				# Values exact in a float32, so the ASCII file holds them exactly
				v.append(struct.unpack("f", struct.pack("f", rnd.uniform(-2.0, 2.0)))[0])
			else:
				v.append(rnd.randrange(0, 120))
		vertices.append(v)
	return vertices

def WritePly(fileName, props, vertices, faces, binary, faceAOVs = False):
	header = ["ply", "format %s 1.0" % ("binary_little_endian" if binary else "ascii"),
			"element vertex %d" % len(vertices)]
	header += ["property %s %s" % (t, n) for n, t in props]
	header += ["element face %d" % len(faces), "property list uchar uint vertex_indices"]
	if faceAOVs:
		header += ["element faceaov %d" % len(faces), "property float triaov"]
	header += ["end_header"]

	with open(fileName, "wb") as f:
		f.write(("\n".join(header) + "\n").encode())
		for v in vertices:
			if binary:
				for (n, t), x in zip(props, v):
					f.write(struct.pack("<" + STRUCT_TYPES[t], x))
			else:
				f.write((" ".join(repr(float(x)) if STRUCT_TYPES[t] in "fd" else str(x)
						for (n, t), x in zip(props, v)) + "\n").encode())
		for face in faces:
			if binary:
				f.write(struct.pack("<B", len(face)) + b"".join(struct.pack("<I", i) for i in face))
			else:
				f.write((" ".join(str(i) for i in [len(face)] + list(face)) + "\n").encode())
		if faceAOVs:
			for i in range(len(faces)):
				if binary:
					f.write(struct.pack("<f", i * 0.5))
				else:
					f.write(("%r\n" % (i * 0.5)).encode())

def LoadAndSave(directory, plyName):
	# Load the file as the mesh of an object, then save the mesh: SavePly()
	# writes every layer the mesh has
	scene = pyluxcore.Scene()
	props = pyluxcore.Properties()
	props.SetFromString("""
		scene.materials.mat.type = matte
		scene.objects.obj.material = mat
		""")
	props.Set(pyluxcore.Property("scene.objects.obj.ply", os.path.join(directory, plyName)))
	scene.Parse(props)

	outName = os.path.join(directory, "saved-" + plyName)
	scene.SaveMesh(os.path.join(directory, plyName), outName)
	with open(outName, "rb") as f:
		return f.read()

def ReadSavedVertices(data):
	# The vertex records of a mesh saved by SavePly(): all floats
	end = data.index(b"end_header\n") + len(b"end_header\n")
	header = data[:end].decode().splitlines()
	count = int([l for l in header if l.startswith("element vertex")][0].split()[2])
	names = []
	element = None
	for l in header:
		w = l.split()
		if w[0] == "element":
			element = w[1]
		elif (w[0] == "property") and (element == "vertex"):
			names.append(w[2])
	values = struct.unpack_from("<%df" % (count * len(names)), data, end)
	return [dict(zip(names, values[i * len(names):(i + 1) * len(names)])) for i in range(count)]

P = [("x", "float"), ("y", "float"), ("z", "float")]
N = [("nx", "float"), ("ny", "float"), ("nz", "float")]
UV = [("s", "float"), ("t", "float")]
RGB = [("red", "float"), ("green", "float"), ("blue", "float")]

CASES = {
	"positions": (P, False, False),
	"normals-uvs": (P + N + UV, False, False),
	"normals-uvs-colors": (P + N + UV + RGB, False, False),
	"uchar-colors-alpha": (P + [("red", "uchar"), ("green", "uchar"), ("blue", "uchar"), ("alpha", "uchar")], False, False),
	"uint8-colors": (P + [("red", "uint8"), ("green", "uint8"), ("blue", "uint8")], False, False),
	"two-layers": (P + N + UV + [("s1", "float"), ("t1", "float"), ("red2", "float"), ("green2", "float"),
			("blue2", "float"), ("alpha3", "float")], False, False),
	"skipped-properties": ([("confidence", "float")] + P + [("intensity", "double"), ("s0", "float"),
			("vertaov", "float")] + N + [("flags", "ushort")], False, False),
	"double-positions-int-uvs": ([("x", "double"), ("y", "double"), ("z", "double"), ("s", "short"), ("t", "int")], False, False),
	"other-order": (P + [("nz", "float"), ("nx", "float"), ("ny", "float"), ("t", "float"), ("s", "float")], False, False),
	# Read by rply in both formats
	"partial-uv": (P + [("s", "float")], False, False),
	"quads": (P + N + UV, True, False),
	"triangle-aovs": (P + N, False, True),
}

class PlyLoad(unittest.TestCase):
	def setUp(self):
		self.tmpDir = tempfile.TemporaryDirectory()
		self.directory = self.tmpDir.name

	def tearDown(self):
		self.tmpDir.cleanup()

	def test_PlyLoad_BinaryEqualsAscii(self):
		rnd = random.Random(7)
		for name, (props, quads, faceAOVs) in CASES.items():
			with self.subTest(case = name):
				vertices = MakeVertices(props, 40, rnd)
				faces = [[0, i + 1, i + 2] + ([i + 3] if (quads and (i % 2) and (i + 3 < 40)) else [])
						for i in range(38)]
				WritePly(os.path.join(self.directory, name + "-bin.ply"), props, vertices, faces, True, faceAOVs)
				WritePly(os.path.join(self.directory, name + "-ascii.ply"), props, vertices, faces, False, faceAOVs)

				binaryMesh = LoadAndSave(self.directory, name + "-bin.ply")
				asciiMesh = LoadAndSave(self.directory, name + "-ascii.ply")
				self.assertEqual(binaryMesh, asciiMesh)

	def test_PlyLoad_Values(self):
		# The rules of the rply path: the binary float NaN and infinite values
		# read as 0, the colors and alphas of type "uchar" (not "uint8") scaled
		# to [0, 1]
		nan, inf = float("nan"), float("inf")
		props = P + N + UV + [("red", "uchar"), ("green", "uchar"), ("blue", "uchar"), ("alpha", "uchar"),
				("red1", "uint8"), ("green1", "uint8"), ("blue1", "uint8")]
		vertices = [
			[1.0, 2.0, 3.0, nan, nan, nan, inf, -inf, 255, 0, 51, 102, 255, 0, 51],
			[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.25, 0.5, 0, 0, 0, 0, 1, 2, 3],
			[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.75, 1.0, 0, 0, 0, 0, 1, 2, 3]]
		WritePly(os.path.join(self.directory, "values.ply"), props, vertices, [[0, 1, 2]], True)

		v = ReadSavedVertices(LoadAndSave(self.directory, "values.ply"))
		self.assertEqual([v[0][k] for k in ("x", "y", "z")], [1.0, 2.0, 3.0])
		self.assertEqual([v[0][k] for k in ("nx", "ny", "nz", "s", "t")], [0.0] * 5)
		self.assertEqual([v[1][k] for k in ("s", "t")], [0.25, 0.5])
		self.assertEqual([v[0][k] for k in ("red", "green", "blue", "alpha")],
				[1.0, 0.0, struct.unpack("f", struct.pack("f", 51 / 255.0))[0],
				struct.unpack("f", struct.pack("f", 102 / 255.0))[0]])
		self.assertEqual([v[0][k] for k in ("red1", "green1", "blue1")], [255.0, 0.0, 51.0])
