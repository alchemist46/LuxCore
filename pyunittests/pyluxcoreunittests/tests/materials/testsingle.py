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
from pyluxcoreunittests.tests.imagetest import *

# Note: keep in alphabetical order

################################################################################
# ArchGlass material test
################################################################################

def TestArchGlassMaterial(cls, params):
	StandardSceneTest(cls, params, "simple/mat-archglass-area.cfg", "ArchGlassMaterial")

class ArchGlassMaterial(ImageTest):
    pass

ArchGlassMaterial = AddTests(ArchGlassMaterial, TestArchGlassMaterial, GetTestCases())

################################################################################
# CarPaint material test
################################################################################

def TestCarPaintMaterial(cls, params):
	StandardSceneTest(cls, params, "simple/mat-carpaint-area.cfg", "CarPaintMaterial")

class CarPaintMaterial(ImageTest):
    pass

CarPaintMaterial = AddTests(CarPaintMaterial, TestCarPaintMaterial, GetTestCases())

################################################################################
# CarPaintGGX material test
################################################################################

def TestCarPaintGGXMaterial(cls, params):
	StandardSceneTest(cls, params, "simple/mat-carpaint-ggx-area.cfg", "CarPaintGGXMaterial")

class CarPaintGGXMaterial(ImageTest):
    pass

CarPaintGGXMaterial = AddTests(CarPaintGGXMaterial, TestCarPaintGGXMaterial, GetTestCases())

################################################################################
# Cloth material test
################################################################################

def TestClothMaterial(cls, params):
	StandardSceneTest(cls, params, "simple/mat-cloth-area.cfg", "ClothMaterial")

class ClothMaterial(ImageTest):
    pass

ClothMaterial = AddTests(ClothMaterial, TestClothMaterial, GetTestCases())

################################################################################
# Disney material test
################################################################################

def TestDisneyMaterial(cls, params):
	StandardSceneTest(cls, params, "simple/mat-disney-area.cfg", "DisneyMaterial")

class DisneyMaterial(ImageTest):
    pass

DisneyMaterial = AddTests(DisneyMaterial, TestDisneyMaterial, GetTestCases())

################################################################################
# Glass material test
################################################################################

def TestGlassMaterial(cls, params):
	StandardSceneTest(cls, params, "simple/mat-glass-area.cfg", "GlassMaterial")

class GlassMaterial(ImageTest):
    pass

GlassMaterial = AddTests(GlassMaterial, TestGlassMaterial, GetTestCases())

################################################################################
# Glass+Dispersion material test
################################################################################

def TestGlassDispMaterial(cls, params):
	StandardSceneTest(cls, params, "simple/mat-glass-disp-area.cfg", "GlassDispMaterial")

class GlassDispMaterial(ImageTest):
    pass

GlassDispMaterial = AddTests(GlassDispMaterial, TestGlassDispMaterial, GetTestCases())

################################################################################
# Glossy2 material test
################################################################################

def TestGlossy2Material(cls, params):
	StandardSceneTest(cls, params, "simple/mat-glossy2-area.cfg", "Glossy2Material")

class Glossy2Material(ImageTest):
    pass

Glossy2Material = AddTests(Glossy2Material, TestGlossy2Material, GetTestCases())

################################################################################
# Glossy2GGX material test
################################################################################

def TestGlossy2GGXMaterial(cls, params):
	StandardSceneTest(cls, params, "simple/mat-glossy2-ggx-area.cfg", "Glossy2GGXMaterial")

class Glossy2GGXMaterial(ImageTest):
    pass

Glossy2GGXMaterial = AddTests(Glossy2GGXMaterial, TestGlossy2GGXMaterial, GetTestCases())

################################################################################
# GlossyCoating material test
################################################################################

def TestGlossyCoatingMaterial(cls, params):
	StandardSceneTest(cls, params, "simple/mat-glossycoating-area.cfg", "GlossyCoatingMaterial")

class GlossyCoatingMaterial(ImageTest):
    pass

GlossyCoatingMaterial = AddTests(GlossyCoatingMaterial, TestGlossyCoatingMaterial, GetTestCases())

################################################################################
# GlossyCoatingGGX material test
################################################################################

def TestGlossyCoatingGGXMaterial(cls, params):
	StandardSceneTest(cls, params, "simple/mat-glossycoating-ggx-area.cfg", "GlossyCoatingGGXMaterial")

class GlossyCoatingGGXMaterial(ImageTest):
    pass

GlossyCoatingGGXMaterial = AddTests(GlossyCoatingGGXMaterial, TestGlossyCoatingGGXMaterial, GetTestCases())

################################################################################
# GlossyTranslucent material test
################################################################################

def TestGlossyTranslucentMaterial(cls, params):
	StandardSceneTest(cls, params, "simple/mat-glossytranslucent-area.cfg", "GlossyTranslucentMaterial")

class GlossyTranslucentMaterial(ImageTest):
    pass

GlossyTranslucentMaterial = AddTests(GlossyTranslucentMaterial, TestGlossyTranslucentMaterial, GetTestCases())

################################################################################
# GlossyTranslucentGGX material test
################################################################################

def TestGlossyTranslucentGGXMaterial(cls, params):
	StandardSceneTest(cls, params, "simple/mat-glossytranslucent-ggx-area.cfg", "GlossyTranslucentGGXMaterial")

class GlossyTranslucentGGXMaterial(ImageTest):
    pass

GlossyTranslucentGGXMaterial = AddTests(GlossyTranslucentGGXMaterial, TestGlossyTranslucentGGXMaterial, GetTestCases())

################################################################################
# Matte material test
################################################################################

def TestMatteMaterial(cls, params):
	StandardSceneTest(cls, params, "simple/mat-matte-area.cfg", "MatteMaterial")

class MatteMaterial(ImageTest):
    pass

MatteMaterial = AddTests(MatteMaterial, TestMatteMaterial, GetTestCases())

################################################################################
# Metal2 material test
################################################################################

def TestMetal2Material(cls, params):
	StandardSceneTest(cls, params, "simple/mat-metal2-area.cfg", "Metal2Material")

class Metal2Material(ImageTest):
    pass

Metal2Material = AddTests(Metal2Material, TestMetal2Material, GetTestCases())

################################################################################
# Metal2GGX material test
################################################################################

def TestMetal2GGXMaterial(cls, params):
	StandardSceneTest(cls, params, "simple/mat-metal2-ggx-area.cfg", "Metal2GGXMaterial")

class Metal2GGXMaterial(ImageTest):
    pass

Metal2GGXMaterial = AddTests(Metal2GGXMaterial, TestMetal2GGXMaterial, GetTestCases())

################################################################################
# RoughGlass material test
################################################################################

def TestRoughGlassMaterial(cls, params):
	StandardSceneTest(cls, params, "simple/mat-roughglass-area.cfg", "RoughGlassMaterial")

class RoughGlassMaterial(ImageTest):
    pass

RoughGlassMaterial = AddTests(RoughGlassMaterial, TestRoughGlassMaterial, GetTestCases())

################################################################################
# RoughGlassGGX material test
################################################################################

def TestRoughGlassGGXMaterial(cls, params):
	StandardSceneTest(cls, params, "simple/mat-roughglass-ggx-area.cfg", "RoughGlassGGXMaterial")

class RoughGlassGGXMaterial(ImageTest):
    pass

RoughGlassGGXMaterial = AddTests(RoughGlassGGXMaterial, TestRoughGlassGGXMaterial, GetTestCases())
