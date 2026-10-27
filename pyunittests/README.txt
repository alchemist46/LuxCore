Dependencies
============

The tests run against the pyluxcore module installed in the Python
environment. To test a local build:

make wheel   (LUX_BUILD_TYPE=Debug for a Debug build)
pip install out/install/<build type>/wheel/*.whl

List of python packages required to run the tests and demos:

- Pillow (to install "pip install pillow")

Unit tests
==========

To run the tests:

python3 unittests.py

To run only a set of tests:

python3 unittests.py --filter PATHCPU

To run only the PATHCPU tests (the ones run on every CI build):

python3 unittests.py --subset

To have the list of options:

python3 unittests.py --help

To run the test with a custom configuration:

python3 unittests.py --config testconfigs/dade.cfg

Add the following properties to --config file to enable
https://github.com/LuxCoreRender/LuxCoreTestScenes test scenes:

luxcoretestscenes.directory = /home/david/projects/luxcorerender/LuxCoreTestScenes

The *OCL engines tests are run only if a GPU device is available.

Reference images
================

The renderings are compared with the reference images in referenceimages
directory. The comparison tolerates the Monte Carlo noise (see
CompareImageNoisy() in pyluxcoreunittests/tests/utils.py); the result and a
"diff-" image showing the different pixels are saved in images directory.

A missing reference image is created from the current result (with --strict,
used by the CI, it is an error instead).

After an intended change of the renderings (i.e. a new sampler), update the
reference images with:

python3 unittests.py --update-references

and commit them with the change.
