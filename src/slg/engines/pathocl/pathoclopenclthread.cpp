/***************************************************************************
 * Copyright 1998-2020 by authors (see AUTHORS.txt)                        *
 *                                                                         *
 *   This file is part of LuxCoreRender.                                   *
 *                                                                         *
 * Licensed under the Apache License, Version 2.0 (the "License");         *
 * you may not use this file except in compliance with the License.        *
 * You may obtain a copy of the License at                                 *
 *                                                                         *
 *     http://www.apache.org/licenses/LICENSE-2.0                          *
 *                                                                         *
 * Unless required by applicable law or agreed to in writing, software     *
 * distributed under the License is distributed on an "AS IS" BASIS,       *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.*
 * See the License for the specific language governing permissions and     *
 * limitations under the License.                                          *
 ***************************************************************************/

#if !defined(LUXRAYS_DISABLE_OPENCL)

#include <mutex>
#include <boost/lexical_cast.hpp>

#include "luxrays/core/geometry/transform.h"
#include "luxrays/core/randomgen.h"
#include "luxrays/utils/ocl.h"
#include "luxrays/devices/ocldevice.h"
#include "luxrays/kernels/kernels.h"

#include "slg/slg.h"
#include "slg/kernels/kernels.h"
#include "slg/renderconfig.h"
#include "slg/engines/pathocl/pathocl.h"

using namespace std;
using namespace luxrays;
using namespace slg;
using namespace std::literals::chrono_literals;

//------------------------------------------------------------------------------
// PathOCLRenderThread
//------------------------------------------------------------------------------

PathOCLOpenCLRenderThread::PathOCLOpenCLRenderThread(const u_int index,
		HardwareIntersectionDeviceRef device, PathOCLRenderEngine *re) :
		PathOCLBaseOCLRenderThread(index, device, re) {
}

PathOCLOpenCLRenderThread::~PathOCLOpenCLRenderThread() {
}

void PathOCLOpenCLRenderThread::GetThreadFilmSize(u_int *filmWidth, u_int *filmHeight,
		u_int *filmSubRegion) {
	PathOCLRenderEngine *engine = (PathOCLRenderEngine *)renderEngine;
	FilmConstRef engineFilm = engine->GetFilm();

	*filmWidth = engineFilm.GetWidth();
	*filmHeight = engineFilm.GetHeight();

	const u_int *subRegion = engineFilm.GetSubRegion();
	filmSubRegion[0] = subRegion[0];
	filmSubRegion[1] = subRegion[1];
	filmSubRegion[2] = subRegion[2];
	filmSubRegion[3] = subRegion[3];
}

void PathOCLOpenCLRenderThread::StartRenderThread() {
	PathOCLRenderEngine *engine = (PathOCLRenderEngine *)renderEngine;

	// I have to load the start film otherwise it is overwritten at the first
	// merge of all thread films
	if (engine->hasStartFilm && (threadIndex == 0))
		threadFilms[0]->GetFilm().AddFilm(engine->GetFilm());

	PathOCLBaseOCLRenderThread::StartRenderThread();
}

static void PGICUpdateCallBack(CompiledScene *compiledScene) {
	compiledScene->RecompilePhotonGI();
}

void PathOCLOpenCLRenderThread::RenderThreadImpl(std::stop_token stop_token) {
	//SLG_LOG("[PathOCLRenderThread::" << threadIndex << "] Rendering thread started");

	PathOCLRenderEngine *engine = (PathOCLRenderEngine *)renderEngine;
	const u_int taskCount = engine->taskCount;

	intersectionDevice.PushThreadCurrentDevice();

	//----------------------------------------------------------------------
	// Execute initialization kernels
	//----------------------------------------------------------------------

	// Clear the frame buffer
	const u_int filmPixelCount = threadFilms[0]->GetFilm().GetWidth() * threadFilms[0]->GetFilm().GetHeight();
	intersectionDevice.EnqueueKernel(filmClearKernel,
		HardwareDeviceRange(RoundUp<u_int>(filmPixelCount, filmClearWorkGroupSize)),
		HardwareDeviceRange(filmClearWorkGroupSize));

	// Initialize random number generator seeds
	intersectionDevice.EnqueueKernel(initSeedKernel,
			HardwareDeviceRange(engine->taskCount), HardwareDeviceRange(initWorkGroupSize));

	// Initialize the tasks buffer
	intersectionDevice.EnqueueKernel(initKernel,
			HardwareDeviceRange(engine->taskCount), HardwareDeviceRange(initWorkGroupSize));

	// Check if I have to load the start film
	if (engine->hasStartFilm && (threadIndex == 0))
		threadFilms[0]->SendFilm(intersectionDevice);

	//----------------------------------------------------------------------
	// Rendering loop
	//----------------------------------------------------------------------

	// The film refresh time target
	const double targetTime = 0.2; // 200ms

	u_int iterations = 4;
	u_int totalIterations = 0;

	// The halt spp: the transfers predict when the next batch of kernels
	// reaches it, from the spp done per kernel iteration (see below)
	const u_int haltSPP = engine->GetEyeHaltSPP();
	double haltCheckSPP = 0.0, sppPerIteration = 0.0;
	u_int iterationsSinceHaltCheck = 0;
	// The first transfer with samples: the spp per iteration is averaged from
	// there (before, the first paths are not done yet)
	double rateStartSPP = -1.0;
	u_int rateStartIterations = 0;

	double totalTransferTime = 0.0;
	double totalKernelTime = 0.0;
	u_int transferCount = 0;

	const std::function<void()> pgicUpdateCallBack = std::bind(PGICUpdateCallBack, engine->compiledScene);

	while (!stop_token.stop_requested()) {
		//if (threadIndex == 0)
		//	SLG_LOG("[DEBUG] =================================");

		// Check if we are in pause mode
		if (engine->pauseMode) {
			// Check every 100ms if I have to continue the rendering
			while (!stop_token.stop_requested() && engine->pauseMode)
				std::this_thread::sleep_for(100ms);

			if (stop_token.stop_requested())
				break;
		}

		//------------------------------------------------------------------

		//------------------------------------------------------------------

		// Transfer the film only if I have already spent enough time running
		// rendering kernels. This is very important when rendering very high
		// resolution images (for instance at 4961x3508). The host buffers are
		// pinned on CUDA and HIP, so the copies are real DMA transfers.
		bool transfer = (totalTransferTime < totalKernelTime * (1.0 / 100.0));

		// With a halt spp, the film is also transferred (and the halt
		// checked) before a batch that would reach it. With the time rule
		// alone, a small film ran many batches between two transfers and
		// went far past the halt (96x96 at batch.haltspp = 64: 1844 spp).
		// The film is transferred at every batch until the spp per iteration
		// is known, then when the batches since the last transfer would have
		// done half of the samples still missing: each check halves the
		// distance to the halt, so an estimate up to 2x too low (it is low
		// while the first paths finish) doesn't overshoot it.
		if (haltSPP > 0) {
			if (sppPerIteration > 0.0) {
				const double halfIterationsLeft = .5 * (haltSPP - haltCheckSPP) / sppPerIteration;
				transfer = transfer || (iterationsSinceHaltCheck + iterations >= halfIterationsLeft);
			} else
				transfer = true;
		}

		if (transfer) {
			const double timeTransferStart = WallClockTime();

			// Async. transfer of the Film buffers
			threadFilms[0]->RecvFilm(intersectionDevice);
			// The pixels to converge of the per-pixel noise threshold halt
			// test (uploaded with the NOISE channel, by RecvFilm())
			UpdateSamplerTodoPixels();

			// Async. transfer of GPU task statistics
			intersectionDevice.EnqueueReadBuffer(
				taskStatsBuff,
				CL_FALSE,
				sizeof(slg::ocl::pathoclbase::GPUTaskStats) * taskCount,
				gpuTaskStats.get());

			intersectionDevice.FinishQueue();

			// I need to update the film samples count

			double totalCount = 0.0;
			for (size_t i = 0; i < taskCount; ++i)
				totalCount += gpuTaskStats[i].sampleCount;
			threadFilms[0]->GetFilm().SetSampleCount(totalCount, totalCount, 0.0);

			totalTransferTime += WallClockTime() - timeTransferStart;
			++transferCount;

			//SLG_LOG("[DEBUG] film transferred");

			if (haltSPP > 0) {
				// The halt spp is checked here on the thread films: the host
				// may poll the statistics (and run the film tests) rarely,
				// and a GPU renders well past the halt point in the meantime
				const double spp = engine->GetThreadFilmsEyeSPP();
				if (rateStartSPP < 0.0) {
					if (spp > 0.0) {
						rateStartSPP = spp;
						rateStartIterations = totalIterations;
					}
				} else if (totalIterations > rateStartIterations)
					sppPerIteration = Max(0.0, (spp - rateStartSPP) / (totalIterations - rateStartIterations));
				haltCheckSPP = spp;
				iterationsSinceHaltCheck = 0;

				if (spp > haltSPP)
					break;
			}
		}

		// Near the halt spp, the batch is cut to the next check (half of the
		// samples still missing)
		u_int batchIterations = iterations;
		if ((haltSPP > 0) && (sppPerIteration > 0.0)) {
			const double halfIterationsLeft = .5 * (haltSPP - haltCheckSPP) / sppPerIteration;
			batchIterations = (u_int)Clamp(ceil(halfIterationsLeft - iterationsSinceHaltCheck), 1.0, (double)iterations);
		}

		//------------------------------------------------------------------

		const double timeKernelStart = WallClockTime();

		// This is required for updating film denoiser parameter
		if (threadFilms[0]->GetFilm().GetDenoiser().IsEnabled()) {
			std::unique_lock<std::mutex> lock(engine->setKernelArgsMutex);
			SetAllAdvancePathsKernelArgs(0);
		}

		for (u_int i = 0; i < batchIterations; ++i) {
			// Trace rays
			intersectionDevice.EnqueueTraceRayBuffer(raysBuff, hitsBuff, taskCount);

			// Advance to next path state
			EnqueueAdvancePathsKernel();
		}
		totalIterations += batchIterations;
		iterationsSinceHaltCheck += batchIterations;

		intersectionDevice.FinishQueue();
		const double timeKernelEnd = WallClockTime();
		const double kernelTime = timeKernelEnd - timeKernelStart;
		totalKernelTime += kernelTime;

		/*if (threadIndex == 0)
			SLG_LOG("[DEBUG] kernel time: " << kernelTime * 1000.0 << "ms "
					"iterations: " << iterations << " #"<< taskCount << ")");*/

		// Check if I have to adjust the number of kernel enqueued
		if (kernelTime > targetTime)
			iterations = Max<u_int>(iterations - 1, 1);
		else
			iterations = Min<u_int>(iterations + 1, 128);

		// Check halt conditions (the halt spp is checked after the transfers)
		if (engine->GetFilm().GetConvergence() == 1.f)
			break;

		if (engine->photonGICache) {
                    if (engine->photonGICache->Update(threadIndex, engine->GetTotalEyeSPP(), pgicUpdateCallBack)) {
                            InitPhotonGI();
                            SetKernelArgs();
                    }
		}
	} // ~while

		//SLG_LOG("[PathOCLRenderThread::" << threadIndex << "] Rendering thread halted");
        if (stop_token.stop_requested())
		SLG_LOG("[PathOCLRenderThread::" << threadIndex << "] Rendering thread halted");

	const double timeFinalTransferStart = WallClockTime();
	threadFilms[0]->RecvFilm(intersectionDevice);
	intersectionDevice.FinishQueue();
	SLG_LOG("[PathOCLRenderThread::" << threadIndex << "] Film transfers: " << transferCount <<
			" in " << int(totalTransferTime * 1000.0) << "ms, the last one in " <<
			int((WallClockTime() - timeFinalTransferStart) * 1000.0) << "ms, kernels " <<
			int(totalKernelTime * 1000.0) << "ms");
	
	threadDone = true;

	// This is done to stop threads pending on barrier wait
	// inside engine->photonGICache->Update(). This can happen when an
	// halt condition is satisfied.
	if (engine->photonGICache)
		engine->photonGICache->FinishUpdate(threadIndex);
	
	intersectionDevice.PopThreadCurrentDevice();
}

#endif
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
