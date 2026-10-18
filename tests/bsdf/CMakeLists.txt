################################################################################
# Copyright 1998-2026 by Authors (see AUTHORS.txt)
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

#############################################################################
#
# bsdftest: consistency checks of the CPU material BSDFs (no rendering)
#
#############################################################################

add_executable(bsdftest bsdftest.cpp)
# The slg headers pull in the same third-party headers as luxcore_static
TARGET_LINK_LIBRARIES(bsdftest PRIVATE
	luxrays
	luxcore_static
	boost::boost
	robin_hood::robin_hood
	spdlog::spdlog_header_only
	fmt::fmt
	openimageio::openimageio
	embree::embree
	TBB::tbbmalloc_proxy)

# "make bsdftest_run" (or the build tool's target) runs every case
add_custom_target(bsdftest_run
	COMMAND bsdftest
	DEPENDS bsdftest
	COMMENT "Running the BSDF consistency test"
	VERBATIM)
