# Credits to Blender Cycles

# Environment variables that control the build:
#   LUX_SOURCE_DIR - The path to the top level of the source tree (default: .)
#   LUX_BINARY_DIR - The path to the top level of the build tree (default: ./out)
#   LUX_BUILD_TYPE - Specifies the build type (default: Release)


ifndef PYTHON
	PYTHON:=python3
endif

LUX-CMAKE = $(PYTHON) -m build-system.luxmake

build-targets = pyluxcore luxcoreui luxcoreconsole luxcore doc

.PHONY: deps list-presets config spirv hiprt luxcore pyluxcore luxcoreui luxcoreconsole install clean clear doc wheel repair-wheel

all: luxcore pyluxcore luxcoreui luxcoreconsole

deps:
	$(LUX-CMAKE) deps

list-presets:
	$(LUX-CMAKE) list-presets

config:
	$(LUX-CMAKE) config

# Bake the OpenCL kernels in SPIR-V format (dependency-free: no 'make deps'
# needed). The baked modules are automatically used by the next 'make config'
# (i.e. by all the build targets).
spirv:
	$(LUX-CMAKE) spirv

# Download the pinned HIPRT build (see build-system/luxmake/hiprt.py). It is
# automatically shipped by the next 'make config'.
hiprt:
	$(LUX-CMAKE) hiprt

$(build-targets): %: config
	$(LUX-CMAKE) build-and-install $*

install:
	$(LUX-CMAKE) install all

package:
	$(LUX-CMAKE) build-and-install package

# Build the pyluxcore wheel (into out/install/<build type>/wheel)
wheel:
	$(LUX-CMAKE) wheel

# Repair a raw wheel: make repair-wheel WHEEL=<wheel> DEST=<dir>
repair-wheel:
	$(LUX-CMAKE) repair-wheel $(WHEEL) $(DEST)

clean:
	$(LUX-CMAKE) clean

clear:
	$(LUX-CMAKE) clear
