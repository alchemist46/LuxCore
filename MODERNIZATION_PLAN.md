# LuxCore modernization roadmap

## Context
LuxCore has a solid core: a wavefront GPU path tracer, BiDir/VCM on the CPU, PhotonGI and DLS caches, noise-aware adaptive sampling, OIDN/OptiX/BCD denoisers, OCIO, and Embree 4, OptiX and (on this branch) HIPRT. It has fallen behind Cycles, Arnold, RenderMan XPU, Karma and V-Ray in three areas: **architecture** (every feature is written twice), **modern sampling and light transport research** (2017 onwards), and **pipeline interop** (USD, MaterialX, OpenPBR).

The survey found these specific gaps:
- **GPU is a separate port.** The CPU `PathTracer` (`src/slg/engines/pathtracer.cpp`) is reimplemented in about 165 OpenCL C files (`include/slg/**/**.cl`). These are run on CUDA/HIP through an emulation prelude (`cudadevice_oclemul_*.cl`). Every material, light and texture costs two implementations, which is the main reason feature velocity stalled.
- **Hardware RT is uneven.** OptiX uses the 7.1 headers (`deps/optix-7.1.0`). OpenCL has no hardware RT. There is no Apple or Intel GPU path. Ray tracing runs as a separate kernel over a ray buffer, so there is no shader-execution reordering or material sorting.
- **Sampling** is Sobol with only Cranley-Patterson rotation (`src/slg/samplers/sobolsequence.cpp:60`). There is no Owen scrambling and no blue noise. *(Owen scrambling done, see Phase 1.)*
- **Light transport** has no light tree, path guiding, ReSTIR or MIS compensation.
- **Volumes** use jittered ray marching (`src/slg/volumes/heterogenous.cpp:68`). There is no delta or null-scattering tracking and no NanoVDB.
- **Materials:**
  - The only uber shader is Disney 2012.
  - Microfacet lobes have no multi-scatter energy compensation.
  - There is no random-walk SSS, no hair BSDF and no native curves (`strands.cpp` tessellates to ribbons), and no OSL.
- **Other features missing:** UDIM and `.tx` mip streaming, light linking, deformation motion blur, point clouds, USD/Hydra and Alembic, realistic lenses.
- **Infrastructure:**
  - CI only builds wheels and runs no tests. *(Fixed, see Phase 0.1.)*
  - C++ tests are off by default. *(Fixed, see Phase 0.1.)*
  - `boost::serialization` is used in 88 files.
  - The C++ standard disagrees between `CMakeLists.txt` (23) and `build-settings.json` (20). *(Both now set in `build-settings.json`, see Phase 0.3.)*
  - OpenCL still uses the 1.2-era queue API.

## Recommended roadmap (ordered by leverage)

**Status (2026-10-13, branch `hip-backend`):** items are tagged **[DONE]**, **[PARTIAL]** or **[BLOCKED]**; untagged items are not started. See also "Done along the way" and "Known open issues" below.

### Phase 0: Foundations (enables everything else)
1. **[DONE] CI with tests.** Turn on `LUXCORE_TESTS` and run `pyunittests/` on every PR. Add a fixed **perf benchmark scene set** with timed CPU/CUDA/HIP runs, building on the BENCH timing already used on this branch, so speed work can be measured.
   - *Done in `13c2e0bcd`.* The reference images did **not** exist (only a placeholder): 532 references were created for all the CPU and GPU engine/sampler combinations (8MB).
   - The image comparison no longer needs perceptualdiff/FreeImage: a Pillow comparison tolerant to the Monte Carlo noise (local and mean brightness checks, calibrated on repeated renderings). `--strict` fails on a missing reference, `--update-references` rewrites them.
   - CI: the Linux wheel is tested on every push and PR (PATHCPU tests; all the CPU tests on manual runs; the runners have no GPU) and a benchmark is recorded in the job summary. `LUXCORE_TESTS` is ON by default and the CMake test targets work again.
   - Benchmark: `tests/benchmark/renderbench.py` (6 scenes, samples/sec, setup time, time to a noise threshold, per engine and device, JSON output, `--baseline` comparison).
2. **[PARTIAL] Single-source kernels.** This is the biggest strategic move.
   - Convert the `.cl` code into a restricted C++ dialect: `#include`-able headers with `DEVICE` macros, compiled natively for CPU, by NVRTC for CUDA and by hiprtc for HIP. This is Cycles' model since 3.0.
   - The existing emulation layer already proves the code compiles as CUDA/HIP C++. The work is mostly making the CPU build consume the same files and retiring the duplicate C++ in `src/slg/materials/*.cpp` etc.
   - Migrate incrementally, one subsystem at a time: textures, then materials, then lights, then the integrator. Use the pyunittests render diffs as the safety net.
   - Decide OpenCL's fate. Keep it as a compatibility tier, or drop it once CUDA, HIP and (later) SYCL or Metal cover the vendors.
   - *How the CPU compiles the kernel sources (done):* the `.cl` files are `#include`d in C++ inside `namespace slg { namespace ocl { namespace {`, after `luxrays/devices/cpudevice_oclemul.h`. That header is the CPU counterpart of the CUDA/HIP prelude: the OpenCL qualifiers and vector types (`cpudevice_oclemul_types.h`), then the same `cudadevice_math.cl` and `cudadevice_oclemul_funcs.cl` used by CUDA and HIP, with `LUXRAYS_CPU_DEVICE` guards around the CUDA-only parts. The kernel data types are the `slg::ocl` structs the host already uses to compile the scene, and the unnamed namespace gives the kernel functions and `__constant` tables internal linkage, so any number of `.cpp` files can include the same kernel sources. The `.cl` files stay plain OpenCL C for the GPU compilers (no `#include`).
   - *Migration, in two steps per subsystem:*
     1. Split each GPU function into the math on plain values, shared by the CPU and the GPU, and the GPU glue that reads the compiled scene (`HitPoint`, `TextureMapping3D`, the texture eval-op stack), guarded by `#if !defined(LUXRAYS_CPU_DEVICE)`. The CPU classes keep their own data and call the shared function, e.g. `BlenderCloudsTexture_Evaluate(P, ...)` with `P` mapped by the C++ mapping. The guards mark exactly what is not single-source yet.
     2. The CPU evaluates the compiled scene data itself (`slg::ocl::HitPoint`, `Texture`, `TextureEvalOp`), so the mapping code, the eval-op glue and the host-to-kernel conversions in `compile*.cpp` stop being duplicated. This is the step XPU needs (Phase 4.1), and the one that removes the conversion bugs, like the Minkowski one below. It needs the CPU `HitPoint` in the kernel layout (mesh and volume pointers replaced by indices).
   - *Done with step 1:* the 10 Blender textures and the Blender noise library. The CPU noise library (`blender_noiselib.cpp`, 1857 lines) is deleted, and `blender_texture.cpp` lost ~340 lines of evaluation code. Verified with a deterministic render of 203 texture configurations (every texture type with every noise basis and variant, TILEPATHCPU, ALBEDO AOV): the CPU images match the old CPU code within 2.4e-7 (1-2 float ULPs), most of them bit for bit. New `BlenderTextures` unit test (there was no procedural texture test).
   - *Found by the CPU/GPU comparison:* the GPU compiled the Voronoi "minkowski" distance metric as Manhattan (`compiletextures.cpp`), so blender_voronoi with that metric always rendered differently on the GPU.
   - *Next:* the Perlin noise textures (`texture_noise_funcs.cl` against `texture.cpp`: fbm, marble, wrinkled, windy, dots...) and the other simple textures of `texture_others_funcs.cl`, then the texture mappings, then step 2 for the textures before starting the materials.
3. **[PARTIAL]** Clean up the build: pick one C++ standard, replace `boost::serialization` (e.g. cereal or a custom binary format), and update the OpenCL API usage.
   - *C++ standard done in `7032edd3d`:* one standard isn't possible yet. The dependencies are built with C++20 while LuxCore code uses C++23 (`std::views::join_with`), so `build-settings.json` now has both (`cxx` and `cxx-luxcore`) and CMake reads them. A single C++23 needs LuxCoreDeps rebuilt with `cppstd=23`.
   - `boost::serialization` is not replaced, but the serialization regressions of the smart pointer conversion were fixed (`45ba1ad07`, see below).
   - OpenCL API: unchanged. There is a single `clCreateCommandQueue()` call, still supported by OpenCL 3.0, and clew has no OpenCL 2.0 entry points.

### Phase 1: Quick speed and quality wins (each is small and independent)
- **[PARTIAL] Owen-scrambled Sobol / ZSampler plus blue-noise pixel decorrelation.** This gives visibly lower noise at equal spp and sits in `src/slg/samplers/` plus `sampler_*_funcs.cl`.
  - *Owen scrambling done in `dfa3d6a18`* (Burley 2020 hash-based nested uniform scrambling, one seed per pixel, the same code on CPU and GPU, SOBOL and TILEPATHSAMPLER samplers). RMSE at equal spp is 9-21% lower on cornell and luxball-hdr (16 to 1024 spp), with samples/sec within -2%/+2%.
  - It also fixed multi-GPU rendering: all the PATHOCL devices rendered **the same samples** (same seed, own pass counters), so a second GPU added no information. Each device now has its own seed.
  - Blue-noise pixel decorrelation: not started.
- **[BLOCKED] GPU OIDN 2.x.** Run the denoiser on CUDA/HIP/SYCL devices instead of CPU (`intel_oidn.cpp:169` creates a CPU device). Bump the OptiX SDK to 8/9 for both acceleration and the denoiser.
  - GPU OIDN: LuxCoreDeps ships only `libLuxOpenImageDenoise_device_cpu`; the GPU device modules have to be added to the dependencies first. The code also passes host pointers to `newBuffer()`, which GPU devices don't accept.
  - OptiX 8/9: no NVIDIA GPU on the development machine to test it.
- **Kulla-Conty energy compensation** for metal2/glossy2/roughglass/disney. This uses precomputed albedo LUTs.
- **Material/shader sorting in the wavefront.** Sort or compact paths by material before the shading `MK_` kernels in `pathoclbase_kernels_micro.cl`. This is a large win on divergent GPU scenes. *Re-prioritized: it is the last rung of the ladder in Phase 1b (G6); the compact state array (G2) and the shadow any-hit (G1) come first.*
- **Fused trace on hardware-RT backends.** Call OptiX/HIPRT trace inline from the state kernels to avoid the ray-buffer round trip. This is partly enabled by the HIPRT work on this branch. *Deferred: the benchmark shows rays/sec flat across scenes, so the trace launch is not what limits throughput today (see Phase 1b).*

### Phase 1b: Measured speed wins (analysis of 2026-10-13)

**Why this section exists.** The Phase 1 speed items are big-ticket (sorting, fused trace, GPU OIDN). A code review of the hot paths, guided by the `renderbench.py` numbers of this branch, found a list of fixed per-sample overheads that are cheaper to remove and are either bit-identical or statistically identical in output. They are ordered by gain per unit of risk.

**Evidence** (`tests/benchmark/renderbench.py`, 640x480, Ryzen 5 3600 / RX 6950 XT, HIP without HIPRT loaded):

| Run | samples/s | rays/s |
|---|---:|---:|
| cornell PATHCPU | 2.25M | 10.5M |
| luxball-hdr PATHCPU | 3.53M | 7.8M |
| classroom PATHCPU | 0.94M | 10.2M |
| cornell PATHOCL HIP | 11.1M | 54M |
| luxball-hdr PATHOCL HIP | 21.4M | 48M |
| classroom PATHOCL HIP | 4.4M | 48M |

1. **Rays/sec is flat across scenes on both backends.** Classroom has far more geometry than cornell and the same rays/sec: BVH traversal is not the bottleneck, a fixed per-ray and per-sample cost is (shading, BSDF construction, film writes, kernel launches).
2. **The GPU numbers were taken without HIPRT** (`compile.LUXRAYS_ENABLE_HIPRT` is a runtime check of the library). Hardware RT has never been benchmarked: Step 0 below.

**Verified findings** (checked in the source):
- *Both backends:* shadow rays are closest-hit rays with a full BSDF build on every blocker. CPU: there is no `rtcOccluded*` call; `Scene::Intersect` (`src/slg/scene/scene.cpp`) runs `BSDF::Init` (normals, UVs, differentials, bump, volume, light lookup) on every shadow-ray hit just to read the pass-through transparency. GPU: the trace kernel cannot tell shadow rays from path rays, so HIPRT, OptiX and BVH all do closest-hit, and `MK_RT_DL` builds `tmpBsdf` for every shadow hit. Sobol recomputes the per-sample pass scramble for each of the ~68 dimensions (`sobolsequence.cpp:72`, `sampler_sobol_funcs.cl:25`).
- *CPU:* the Embree scene is built with `RTC_SCENE_FLAG_DYNAMIC` for every render (`embreeaccel.cpp:156`) and the instance sub-scenes get no flags. The film atomic add executes `pause` before the first CAS attempt (`atomic.h:41`), per float per sample, and the weight add next to it is a non-atomic `+=` (`framebuffer.h:189`), a data race. The per-thread sample counters are false-shared (`filmsamplescounts.cpp:78`). No `-march` and no LTO on Linux/macOS (only MSVC gets `/arch:AVX`). Materials re-read all their textures in `Evaluate`, `Sample` and `Pdf` (Disney: 11-14 reads each).
- *GPU:* 11 full-width launches per iteration with no compaction; each `MK_*` kernel loads `tasksState[gid].state` from a 392-byte AoS struct and returns if it is not its state (~64 MB of traffic per launch with 512K tasks). The per-task state is ~1.8 KB in global memory. `__noinline__` is applied to ~160 kernel functions on HIP/CUDA (`cudadevice_oclemul_types.cl:28`) as a compile-time workaround. No per-scene kernel specialization (`kernelsources.cpp:222`): sampler type, film channels, volumes and pass-through are runtime branches. The film readback is a blocking `FinishQueue` into pageable memory.
- *Film (all engines):* `Film::RunTests` (`film.cpp:1408`) runs the image pipeline up to three times per stats poll; with OIDN in pipeline 0 that is 2-3 denoises per poll, which starves the render threads. OIDN creates a new device and commits it on every `FilterImage` call (`intel_oidn.cpp:169`), three calls per `Apply`. The noise estimation and the convergence test are single-threaded full-film loops with an 8x8 box filter per pixel.

**Step 0: real baselines.** Build the Release wheel with `LUXRAYS_HIPRT_LIBRARY` set, confirm `[HiprtAccel] Compilation completed` in the log, record `base-cpu.json` and `base-hiprt.json` for the 6 scenes with `--repeat 3`. Optionally `rocprof --stats` on a short PATHOCL render for the time share of the trace versus each `AdvancePaths_MK_*` kernel (it decides whether G6 is worth doing).

**CPU track**
- **C1. Occlusion-only shadow rays** (largest CPU win, bit-identical). `Accelerator::Occluded` with an `rtcOccluded1` override in `embreeaccel.cpp` and `IntersectionDevice::TraceOccluded` (keeps rays/sec comparable). `Material::HasPassThroughTransparency` (transparency textures; always true for null and archglass; union of the children for mix, twosided, glossycoating). A cached `shadowOpaque` flag per `SceneObject` (no bevel, no pass-through, black shadow transparency) recomputed in `Scene::Preprocess` before `lightDefs.Preprocess` (DLSC and the env visibility cache trace shadow rays there). In `Scene::Intersect`: (a) all objects shadow-opaque, no default world volume and no current volume → return `TraceOccluded`; (b) otherwise, after the trace and before `BSDF::Init`, return true if the hit object is a shadow blocker. Exact because `PathVolumeInfo::ContinueToTrace` cannot fire without a current volume (`CompareVolumePriorities` returns false for a null first argument) and the callers only read `connectionThroughput` on a miss. Rejected: a cut-down `BSDF::InitForShadow` (archglass needs the full frame) and Embree filter callbacks (not front-to-back).
- **C2. Film write path** (bit-identical, fixes a race). `AtomicAdd(float*)` with `std::atomic_ref` and `pause` only on retry; atomic weight add; one `alignas(64)` struct of counters per thread (keep the serialized format; fix the `[0]`/`[i]` bug at `filmsamplescounts.cpp:62`). No per-thread films for PATHCPU: they would delay the adaptive-sampling feedback for little gain.
- **C3. Sobol** (bit-identical). Cache `NestedUniformScramble(pass, seed)` once per sample in `SobolSampler::InitNewSample`; a `Sampler::GetSamples(first, count, out)` batch call used per bounce by the `PathTracer`.
- **C4. Embree flags** (statistically identical). `RTC_SCENE_FLAG_NONE` unless `accelerator.embree.dynamic` (default true only for the `RTPATH*` engines); HIGH quality and NONE on the instance sub-scenes.
- **C5. Compiler flags** (statistically identical; last, so it does not confound the benchmarks of C1-C4). `LUXCORE_X86_ARCH` default `x86-64-v3` on x86_64 Linux/macOS for Release, `-fno-math-errno`, never `-ffast-math` (the film relies on NaN checks), never `native` by default (the wheels are distributed); an AVX2 guard in the pyluxcore module init compiled for the baseline arch; `LUXCORE_ENABLE_LTO` via `CheckIPOSupported`, kept only if the benchmark improves.
- **C6. Later.** A per-BSDF material parameter cache shared by `Evaluate` and `Sample`; `Schlick_Weight` as multiplications; one modulo and the wrap switch hoisted out of the bilinear taps; a pointer instead of a copied `Transform` in `BSDF::Init` for static meshes and a cached motion-blur inverse.

**GPU track**
- **G1. Shadow-ray any-hit** (largest GPU win, identical visibility). `RAY_FLAGS_SHADOW` set in `MK_DL_SAMPLE_BSDF` after `Light_Illuminate` initialised the ray; a compiled `shadowRaysNeedBSDF` scene flag (null, archglass, transparency textures, shadow transparency, interior/exterior/default volumes) → `taskConfig->scene.shadowRayAnyHit`; `MK_RT_DL` skips `Scene_Intersect` entirely when the flag is set. HIPRT: `hiprtGeomTraversalAnyHitCustomStack` / `hiprtSceneTraversalAnyHitCustomStack` with `hiprtTraversalHintShadowRays` for flagged rays in the same kernel. OptiX: `OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT`. BVH/MBVH: break on the first hit. Scenes with transparency or volumes take the current path.
- **G2. Compact path-state array and trace skip** (bit-identical, 1-2 days). A `uint pathStates[taskCount]` buffer through a `PathState_Set` helper, `state` removed from `GPUTaskState` so any missed site fails to compile; a `uchar rayActive[]` mask passed to `EnqueueTraceRayBuffer` so the trace kernels skip loading inactive rays; `rayCount` increment moved after the state check.
- **G3. Inlining on HIP** (last-ulp differences only). Experiment: `OPENCL_FORCE_NOT_INLINE` → `inline` on HIP only, measure cold compile time and scratch (`HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`); if it regresses, an `OPENCL_HOT_FUNCTION` whitelist for the one- or two-caller functions (`Scene_Intersect`, `BSDF_Init`, `Material_Evaluate/Sample`, `Light_Illuminate`, `Sampler_NextSample`), keeping the big `EvalOp` switches non-inlined.
- **G4. Per-scene kernel specialization** (bit-identical). A coarse macro set in `GetKernelParamters` (`PARAM_SAMPLER_TYPE`, `PARAM_FILM_USE_PIXEL_ATOMICS`, `PARAM_FILM_HAS_AOVS`, `PARAM_HAS_VOLUMES`, `PARAM_HAS_PASSTHROUGH`, `PARAM_PGIC_ENABLED`), tens of combinations; HIP/CUDA only (the embedded OpenCL SPIR-V would always miss); the cache key already hashes the options.
- **G5. Pinned host memory, one sync per batch, GPU-side halt.** `hipHostRegister` / `cuMemHostRegister` RAII-owned by the `ThreadFilm`; the readback enqueued before the single `FinishQueue`; leave the loop on `haltSPP` from the task stats. Small at 640x480, real at 4K.
- **G6. Full wavefront compaction** (statistically identical, 2-4 weeks; only after rocprof on the post-G2 build shows the shading kernels dominating). Per-state index queues with wave-aggregated atomics, kernels launched over `taskCount` with a device-side bound check, `gid = stateQueue[state][get_global_id(0)]`; ~40 helper functions derive the task index from `get_global_id(0)` and the sampler state must take a `taskIndex`; index-buffer trace variants for the four accelerators.

**Film track** (all engines; matters most in Blender and luxcoreui sessions with a denoiser)
- **F1. `Film::RunTests`:** execute each distinct pipeline index the due tests read once, never unconditionally. Unobservable: `GetChannel(IMAGEPIPELINE)` executes the pipeline on demand and the test timing is sample-count based.
- **F2. OIDN:** a transient `DeviceRef` created on first use, cached `FilterRef`s keyed by filter, size and aux presence.
- **F3. Noise estimation and convergence test:** TBB-parallel; the box filter as two separable prefix-sum passes with exactly the current bounds and divisor (the NOISE channel drives the adaptive sampling, so the numbers must match).

**Execution order and status.** Step 0 → F1 → C1 → C2 → C3 → C4 → G1 → G2 → F2 → F3 first, each as a tested and committed phase; C5, C6, G3, G4, G5 and G6 later. All not started.

**Verification.** Per item: the pyunittests of the touched engines and features, the full CPU suite plus all PATHOCL/TILEPATHOCL cases before each commit, and for the bit-identical items a 1-thread fixed-seed checksum before and after. Speed: `renderbench.py --baseline` after each item with `--repeat 3` and `--max-regression 3`, on an idle machine. Quality for the statistically-identical items: equal-spp RMSE against a high-spp reference, as done for the Owen scrambling.

### Phase 2: Modern light transport
- **Light tree / many-light BVH** as a new light strategy in `src/slg/lights/strategies/`. Use it alongside or instead of the DLS cache, and it works on GPU too.
- **Path guiding with Intel OpenPGL** on the CPU engines first, as Cycles did. It complements PhotonGI for caustics and indirect light.
- **Volumes:** null-scattering / delta / ratio tracking with majorant grids, **NanoVDB** on GPU, emissive volumes and random-walk SSS. Random-walk SSS shares the same machinery.
- **ReSTIR DI** in RTPATH engines for interactive viewport quality.
- Optional and long-term: **hero-wavelength spectral rendering**. The `SWCSpectrum` code already exists for thin film.

### Phase 3: Features and interop (adoption)
- **OpenPBR surface** as the new uber material, which supersedes Disney. Add **MaterialX** import.
- **Hair:** native curve primitives (Embree/OptiX/HIPRT curves) and a Chiang/d'Eon hair BSDF.
- **Textures:** UDIM and OIIO TextureSystem with `.tx` mip streaming and ray differentials. This cuts memory for large scenes.
- **Lights:** light linking and shadow linking, plus the Prague/Nishita sky.
- **Geometry:** deformation (vertex) motion blur, point clouds.
- **USD Hydra render delegate.** This is the single biggest adoption lever: Houdini, Maya, Blender and Omniverse would all get LuxCore without bespoke plugins.
- **Backends:** add Metal (Apple has no OpenCL path) and SYCL/Level Zero with Embree GPU for Intel.

### Phase 4: XPU (CPU + GPU rendering one image, with identical results)
**What exists today.** PATHOCL is already hybrid: it starts CPU render threads (`opencl.native.threads.count`, which defaults to all hardware threads, `src/slg/engines/oclrenderengine.cpp:249`) alongside the GPU devices.
- The scheduling model is already the right one. Every device renders the *full frame* into its own film, with no per-iteration barrier between devices.
- Films are merged lazily (`PathOCLRenderEngine::MergeThreadFilms`, `src/slg/engines/pathocl/pathocl.cpp:176`).
- GPU-to-host film transfers are budgeted at 1% of kernel time (`pathoclopenclthread.cpp:149`).
- So a slow device never blocks a fast one, which is the property that lets Karma XPU and RenderMan XPU gain speed where Cycles and Redshift hybrid modes lose it.

**Why hybrid usually loses (lessons to avoid):**
- **Synchronization barriers.** Each sample pass waits for every device (Cycles-style slices plus a sync per iteration, needed for adaptive sampling and display), so the fast GPU idles waiting for the CPU.
- **Bucket tail.** In a bucket renderer (Redshift-style), the last buckets handed to the slow CPU hold up the end of the frame.
- **Host-thread starvation.** CPU render threads compete with the threads that feed the GPU and with the driver, which drops GPU occupancy.
- **Small ceiling.** The gain is at most CPU throughput divided by GPU throughput, often only 5–15%, so any overhead (duplicate BVH builds, memory, power or thermal limits) wipes it out.

**Work items:**
1. **Single-source kernels (Phase 0.2) are the prerequisite.** CPU and GPU must run the same integrator and shading code, so that averaging their samples cannot introduce bias.
2. **Core reservation.** By default, leave about 1 core per GPU for its feeder thread and the driver, rather than all hardware threads, and pin CPU render threads away from them.
3. **Automatic enable/disable.** Benchmark samples/sec per device during the first seconds of a render and drop the CPU when its share is below the cost of its overhead (e.g. under 3%).
4. **Merge strategy for adaptive sampling.** Keep the noise/convergence estimation on the merged film. Make sure per-device sample counts and the adaptive importance map are shared without adding a barrier.
5. **Avoid tile-based hybrid.** Hybrid TILEPATHOCL has the bucket-tail problem. Use work stealing with small tiles there, or recommend PATHOCL for hybrid.
6. **Interactive XPU.** Allow native threads in RTPATHOCL (currently rejected at `src/slg/engines/rtpathocl/rtpathocl.cpp:38`) with asynchronous merging per frame.
7. **Parity tests.** Render each test scene CPU-only, GPU-only and hybrid, and fail on statistically significant differences.
8. **Feature gating.** Scenes using CPU-only features (BiDir, path guiding) either fall back to CPU-only or warn.

## Done along the way
Bugs found by the new tests and fixed:
- `95f8e5c7c` Subdiv: heap overflow of the UVs and colors (crashed every subdiv, displacement and simplify shape).
- `3f3f6a6d9` ImageMap: the "white" wrap returned red on the CPU (only the first channel of the pixel was set).
- `87656a4ec` pyluxcore: `RenderConfig.GetScene()` always failed, and the Scene methods failed on a returned scene (scene editing from Python and pyluxcoretools).
- `45ba1ad07` Serialization: loading any `.flm` film failed, render state load and resume from a state + film crashed, `RenderConfig` save dereferenced a null pointer.
- `c5dc429b2` DataSet: data race of the accelerator lookup (unlocked `std::unordered_map` read while another thread inserted).
- `deps/hipew/src/hiprtew.cpp`: no `RTLD_DEEPBIND` in sanitized builds, so ASan/UBSan builds (`LUX_SANITIZE`) can run (part of `13c2e0bcd`).

## Known open issues
- Intermittent render-thread segfaults when the whole test suite runs in a single process (roughly once every 20-160 tests); with each test in its own process, all 509 tests passed without a crash. Not found yet. The development machine also showed nondeterministic GCC crashes under load, so part of it may be hardware.
- `.bcf` and `.rsm` serialization of scenes with instances crashes (`observer_ptr<TriangleMesh>` of `InstanceTriangleMesh`), and `RenderConfig::serialize()` is empty (`.rsm` files never contained the configuration and the scene).
- The CannelleEtFromage scene renders much darker with PATHCPU than with PATHOCL (in the old build too): a CPU/GPU parity problem.
- `LaserLight_PATHOCL_METROPOLIS` has a larger mean brightness tolerance (Metropolis normalization noise on the GPU).
- PATHOCL with the RANDOM and SOBOL samplers spreads the first samples very unevenly when the task count is large compared with the film: all the tasks that share a pixel bucket walk its 16 pixels in lockstep, so a 96x96 film halted at `batch.haltspp = 16` (512K tasks, ~145 spp in total) has half of its pixels without any sample. Small and low spp renders (material previews, thumbnails) are affected. It needs a per-task offset in the bucket, or a task count capped to the film size.
- pyluxcore crashes at exit when a Python log handler is set (`pyluxcore.Init(lambda s: None)`) and not reset with `pyluxcore.SetLogHandler(None)` before the interpreter shuts down.

## Language: stay in C++
A Rust rewrite was considered and rejected:
- Performance would be about the same.
- The GPU kernel side has no viable Rust path to OptiX/HIPRT hardware RT.
- Every major dependency (Embree, OIIO, OCIO, OpenVDB, OpenSubdiv, OIDN, USD/Hydra, MaterialX, OpenPGL) is C++.

Get the safety benefits in C++ instead: run ASan/UBSan/TSan in CI, use clang-tidy, and fuzz the scene parsers.
- **[PARTIAL]** ASan/UBSan: local sanitized builds work (`LUX_SANITIZE`, HIPRT loading fixed for them); not in CI yet. TSan, clang-tidy and fuzzing: not started.

## Suggested first milestone
Phase 0.1 (CI, tests and a benchmark harness), then the Phase 1 quick wins: Owen/blue-noise sampling, GPU OIDN, and Kulla-Conty. These deliver visible results in weeks, while the single-source kernel migration (0.2) runs as the long-term track that unblocks Phases 2 and 3.

**Milestone status:** Phase 0.1 done, Owen sampling done (blue noise not yet), GPU OIDN blocked on the dependencies, Kulla-Conty not started. Next: the Phase 1b speed items (Step 0, F1, C1, C2, C3, C4, G1, G2, F2, F3).

## Verification approach (applies to every item)
- **Correctness:** `pyunittests/unittests.py` render diffs against reference images (a noise tolerant Pillow comparison, see Phase 0.1), on CPU and on every GPU backend.
- **Quality:** equal-time and equal-spp RMSE against a high-spp reference on the benchmark scene set, especially for sampler, light tree, guiding and volume changes.
- **Speed:** samples/sec and time-to-threshold (the convergence test in `include/slg/film/convtest/filmconvtest.h`) per backend, tracked in CI.
