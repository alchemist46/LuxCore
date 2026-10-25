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

// At the halt, the paths in flight finish instead of being dropped: the ones
// still running are the long paths (deep bounces, subsurface walks), so the
// finished samples alone would be biased. The tasks start no new sample
// (SobolSamplerSharedData::drain) and the kernels run until all are done.
void PathOCLOpenCLRenderThread::DrainPaths() {
	PathOCLRenderEngine *engine = (PathOCLRenderEngine *)renderEngine;
	const u_int taskCount = engine->taskCount;

	const u_int drain = 1;
	intersectionDevice.EnqueueWriteBufferRange(samplerSharedDataBuff, CL_TRUE,
			offsetof(slg::ocl::SobolSamplerSharedData, drain), sizeof(u_int), &drain);

	// The longest path: 2 iterations per vertex, plus the steps of its
	// subsurface walks, up to SSS_MAX_BOUNCES (65536, subsurface_funcs.cl)
	// each. The loop ends as soon as every task is done.
	const u_int maxPathDepth = engine->pathTracer.maxPathDepth.depth;
	const u_int maxIterations = 2 * maxPathDepth + 2 +
			(engine->compiledScene->hasSubsurface ? maxPathDepth * 65536 : 0);
	const u_int checkIterations = 8;
	vector<slg::ocl::pathoclbase::PathState> pathStates(taskCount);
	for (u_int iterations = 0; iterations < maxIterations; iterations += checkIterations) {
		for (u_int i = 0; i < checkIterations; ++i) {
			EnqueueTraceRays();
			EnqueueAdvancePathsKernel();
		}

		intersectionDevice.EnqueueReadBuffer(pathStatesBuff, CL_TRUE,
				sizeof(slg::ocl::pathoclbase::PathState) * taskCount, &pathStates[0]);

		bool done = true;
		for (const auto state : pathStates) {
			if (state != slg::ocl::pathoclbase::MK_DONE) {
				done = false;
				break;
			}
		}
		if (done)
			break;
	}

	// The sample count of the finished paths
	intersectionDevice.EnqueueReadBuffer(taskStatsBuff, CL_TRUE,
			sizeof(slg::ocl::pathoclbase::GPUTaskStats) * taskCount,
			gpuTaskStats.get());
	double totalCount = 0.0;
	for (size_t i = 0; i < taskCount; ++i)
		totalCount += gpuTaskStats[i].sampleCount;
	threadFilms[0]->GetFilm().SetSampleCount(totalCount, totalCount, 0.0);
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

	// The tasks start new samples again: a restart after a scene edit follows
	// a halt whose paths were drained (DrainPaths()), and the sampler shared
	// data is initialized only once, by InitRender(). Left set, each task
	// finished the sample it had and the render stalled at taskCount samples
	// (20 spp on a 160x160 film).
	if (engine->oclSampler->type == slg::ocl::SOBOL) {
		const u_int noDrain = 0;
		intersectionDevice.EnqueueWriteBufferRange(samplerSharedDataBuff, CL_TRUE,
				offsetof(slg::ocl::SobolSamplerSharedData, drain), sizeof(u_int), &noDrain);
	}

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
	// With the Sobol sampler, the paths in flight finish at the halt
	// (DrainPaths()): one sample per task of every device is still to come,
	// so the halt is reached that many samples earlier
	const bool drainAtHalt = (engine->oclSampler->type == slg::ocl::SOBOL);
	bool drain = false;
	const u_int *subRegion = engine->GetFilm().GetSubRegion();
	const double regionPixelCount = (subRegion[1] - subRegion[0] + 1) * (subRegion[3] - subRegion[2] + 1);
	const double haltTargetSPP = drainAtHalt ?
		(haltSPP - double(taskCount) * engine->renderOCLThreads.size() / regionPixelCount) : haltSPP;
	double haltCheckSPP = 0.0, sppPerIteration = 0.0;
	u_int iterationsSinceHaltCheck = 0;
	// The first transfer with samples: the spp per iteration is averaged from
	// there (before, the first paths are not done yet)
	double rateStartSPP = -1.0;
	u_int rateStartIterations = 0;

	double totalTransferTime = 0.0;
	double totalKernelTime = 0.0;
	u_int transferCount = 0;

	// Run by the first thread after a caustic cache update, while the others wait
	const std::function<void()> pgicUpdateCallBack = [engine]() {
		engine->compiledScene->RecompilePhotonGI();
		// The new radius and photon count
		engine->taskConfig.pathTracer.pgic = engine->compiledScene->compiledPathTracer.pgic;
	};
	// The caustic map this device has (uploaded at the start)
	u_int pgicUpdateGeneration = engine->photonGICache ? engine->photonGICache->GetUpdateGeneration() : 0;

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
				const double halfIterationsLeft = .5 * (haltTargetSPP - haltCheckSPP) / sppPerIteration;
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

			UpdateLightTreeLearning(totalCount);

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

				if (drainAtHalt ? (spp >= haltTargetSPP) : (spp > haltSPP)) {
					drain = drainAtHalt;
					break;
				}
			}
		}

		// Near the halt spp, the batch is cut to the next check (half of the
		// samples still missing). Until the spp per iteration is known, the
		// batches are of one iteration: on a small film the first batches
		// alone went 16 spp past a halt of 64.
		u_int batchIterations = iterations;
		if (haltSPP > 0) {
			if (sppPerIteration > 0.0) {
				const double halfIterationsLeft = .5 * (haltTargetSPP - haltCheckSPP) / sppPerIteration;
				batchIterations = (u_int)Clamp(ceil(halfIterationsLeft - iterationsSinceHaltCheck), 1.0, (double)iterations);
			} else
				batchIterations = 1;
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
			EnqueueTraceRays();

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
		if (engine->GetFilm().GetConvergence() == 1.f) {
			drain = drainAtHalt;
			break;
		}

		if (engine->photonGICache) {
			engine->photonGICache->Update(threadIndex, engine->GetTotalEyeSPP(), pgicUpdateCallBack);

			// Every device uploads the new caustic map, not only the one of
			// the thread that updated it
			if (engine->photonGICache->GetUpdateGeneration() != pgicUpdateGeneration) {
				pgicUpdateGeneration = engine->photonGICache->GetUpdateGeneration();

				InitPhotonGI();
				UpdateGPUTaskConfigurationBuffer();
				SetKernelArgs();
			}
		}
	} // ~while

	if (drain)
		DrainPaths();

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
	const string kernelTimingReport = intersectionDevice.GetKernelTimingReport();
	if (kernelTimingReport.length() > 0)
		SLG_LOG("[PathOCLRenderThread::" << threadIndex << "] " << kernelTimingReport);

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
