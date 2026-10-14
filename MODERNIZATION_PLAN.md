# LuxCore modernization roadmap

## Context
LuxCore has a solid core: a wavefront GPU path tracer, BiDir/VCM on the CPU, PhotonGI and DLS caches, noise-aware adaptive sampling, OIDN/OptiX/BCD denoisers, OCIO, and Embree 4, OptiX and (on this branch) HIPRT. It has fallen behind Cycles, Arnold, RenderMan XPU, Karma and V-Ray in three areas: **architecture** (every feature is written twice), **modern sampling and light transport research** (2017 onwards), and **pipeline interop** (USD, MaterialX, OpenPBR).

The survey found these specific gaps:
- **GPU is a separate port.** The CPU `PathTracer` (`src/slg/engines/pathtracer.cpp`) is reimplemented in about 165 OpenCL C files (`include/slg/**/**.cl`). These are run on CUDA/HIP through an emulation prelude (`cudadevice_oclemul_*.cl`). Every material, light and texture costs two implementations, which is the main reason feature velocity stalled.
- **Hardware RT is uneven.** OptiX uses the 7.1 headers (`deps/optix-7.1.0`). OpenCL has no hardware RT. There is no Apple or Intel GPU path. Ray tracing runs as a separate kernel over a ray buffer, so there is no shader-execution reordering or material sorting.
- **Sampling** is Sobol with only Cranley-Patterson rotation (`src/slg/samplers/sobolsequence.cpp:60`). There is no Owen scrambling and no blue noise. *(Owen scrambling done, see Phase 1.)*
- **Adaptive sampling and the noise halt** use two unrelated metrics, and the halt waits for every single pixel (see Phase 1). Cycles' per-pixel stopping is more predictable.
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
   - *Done with step 1 (`db05c8c15`):* the 10 Blender textures and the Blender noise library. The CPU noise library (`blender_noiselib.cpp`, 1857 lines) is deleted, and `blender_texture.cpp` lost ~340 lines of evaluation code. Verified with a deterministic render of 203 texture configurations (every texture type with every noise basis and variant, TILEPATHCPU, ALBEDO AOV): the CPU images match the old CPU code within 2.4e-7 (1-2 float ULPs), most of them bit for bit. New `BlenderTextures` unit test (there was no procedural texture test).
   - *Found by the CPU/GPU comparison (`a84d54d73`):* the GPU compiled the Voronoi "minkowski" distance metric as Manhattan (`compiletextures.cpp`), so blender_voronoi with that metric always rendered differently on the GPU.
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
    - *Why:* each pixel's Sobol sequence gets an independent seed (`SobolSequence::Hash(seedBase, pixelIndex)` in `sobol.cpp`), so the error is white noise from pixel to pixel. Correlating the sequences of neighbouring pixels moves the error to high frequencies. RMSE stays the same, but the noise is less visible at low spp and OIDN gets a better input.
    - *Candidate method:* the hierarchical pixel ordering of Ahmed & Wonka 2020 ("Screen-space blue-noise diffusion of Monte Carlo sampling error"), the approach Cycles 4.2 adopted. Pixels along a Morton- or Hilbert-style order take consecutive sample indices of one Owen-scrambled sequence, so it builds on the existing Owen code. It is a sampler-only change (`sobol.cpp`, `sampler_sobol_funcs.cl`, TILEPATHSAMPLER) with no light transport changes.
    - *Caveats:* the effect fades at high spp and when adaptive sampling gives pixels different sample counts. The PATHOCL bucket lockstep (see "Known open issues") decides which pixels get their first samples, so check the two together.
    - *Measure:* equal-spp RMSE must not regress; plus the error power spectrum, or the RMSE after OIDN, at 1-16 spp.
- **Cycles-style adaptive sampling and noise halt.** The noise threshold halt (`batch.haltthreshold`, used by PATHCPU and PATHOCL) is unpredictable in practice: it often stops much later than expected, and the quality it stops at varies by scene. The code (`filmconvtest.cpp`, `filmnoiseestimation.cpp`, `sobol.cpp:210`) shows four causes:
  - **Every pixel must pass.** The test compares two snapshots of the image pipeline output taken `step` spp apart and halts only when *no* pixel changed by more than the threshold. One firefly anywhere during a step blocks the halt. The count is recomputed from scratch at each test, so a pixel can go from done back to not done.
  - **The snapshot difference is not the error.** Its expected size is about `error × sqrt(step / N)`, so the test is strict early and looser later. The quality at the halt depends on the step size and on how fast the scene renders.
  - **The sampler targets a different metric.** The adaptive sampler reads the NOISE channel: a relative error (divided by the square root of brightness), rescaled to 0-1 across the film, so it only ranks pixels against each other. Bright pixels, which hold up the absolute halt test, get fewer samples. Pixels that already pass keep at least `1 - adaptive.strength` (5%) of the samples.
  - **Both tests read the tonemapped output.** The tonemapper, gamma, exposure and any denoiser in that pipeline change the result, and fireflies count at full strength because the output isn't clamped.

  TILEPATH already does it the Cycles way, per tile (`Tile::CheckConvergence`: even passes against all passes). Cycles keeps a buffer of every other sample, computes `|full - half| / sqrt(luminance)` per pixel on linear radiance, stops sampling converged pixels (dilated to their neighbours), derives a minimum spp from the threshold and caps at max samples.

  *Plan, in two steps:*
  1. **Quick fix, no new channel.** A pixel that passes stays converged until the film is reset. The halt tolerates a small fraction of unconverged pixels (a new property; its default to be chosen from the measurements). The sampler and the halt share one metric (relative error), so samples go where the halt is waiting.
  2. **Cycles model.** Add a film channel that accumulates every other sample on the CPU and the GPU (`filmaddsample.cpp`, the PATHOCL film merge). Compute the per-pixel error on linear radiance. SOBOL and RANDOM skip converged pixels instead of keeping a 5% floor. Use `batch.haltspp` as the max samples. METROPOLIS can't skip pixels and keeps the film-wide test.

  The threshold value changes meaning, so the BlendLuxCore UI and the docs need updating with step 2.

  *Measure:* time to halt and RMSE at the halt (`renderbench.py` already measures time to a noise threshold), on the benchmark scenes plus one scene with caustics or fireflies, against the current test.
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
- *CPU:* the Embree scene is built with `RTC_SCENE_FLAG_DYNAMIC` for every render (`embreeaccel.cpp:156`) and the instance sub-scenes get no flags. The film atomic add executes `pause` before the first CAS attempt (`atomic.h:41`), per float per sample, and the weight add next to it is a non-atomic `+=` (`framebuffer.h:189`), a data race. The per-thread sample counters are false-shared (`filmsamplescounts.cpp:78`). No LTO (`-march=x86-64-v3` comes from the Conan profiles, see C5). Materials re-read all their textures in `Evaluate`, `Sample` and `Pdf` (Disney: 11-14 reads each).
- *GPU:* 11 full-width launches per iteration with no compaction; each `MK_*` kernel loads `tasksState[gid].state` from a 392-byte AoS struct and returns if it is not its state (~64 MB of traffic per launch with 512K tasks). The per-task state is ~1.8 KB in global memory. `__noinline__` is applied to ~160 kernel functions on HIP/CUDA (`cudadevice_oclemul_types.cl:28`) as a compile-time workaround (see G3: it also cost registers). No per-scene kernel specialization (`kernelsources.cpp:222`): sampler type, film channels, volumes and pass-through are runtime branches. The film readback is a blocking `FinishQueue` into pageable memory.
- *Film (all engines):* `Film::RunTests` (`film.cpp:1408`) runs the image pipeline up to three times per stats poll; with OIDN in pipeline 0 that is 2-3 denoises per poll, which starves the render threads. OIDN creates a new device and commits it on every `FilterImage` call (`intel_oidn.cpp:169`), three calls per `Apply`. The noise estimation and the convergence test are single-threaded full-film loops with an 8x8 box filter per pixel.

**[DONE] Step 0: real baselines.** Build the Release wheel with `LUXRAYS_HIPRT_LIBRARY` set, confirm `[HiprtAccel] Compilation completed` in the log, record `base-cpu.json` and `base-hiprt.json` for the 6 scenes with `--repeat 3`. Optionally `rocprof --stats` on a short PATHOCL render for the time share of the trace versus each `AdvancePaths_MK_*` kernel (it decides whether G6 is worth doing).
- *Done:* `out/bench/base-cpu.json` and `out/bench/base-hiprt.json` (build `db05c8c15`, `--repeat 3`, HIPRT loaded and used). `rocprof` not run yet.

  | Scene | PATHCPU samples/s | rays/s | PATHOCL HIP+HIPRT samples/s | rays/s |
  |---|---:|---:|---:|---:|
  | cornell | 2.213M | 10.55M | 11.054M | 53.42M |
  | luxball-hdr | 3.449M | 7.61M | 22.821M | 50.62M |
  | classroom | 0.943M | 10.15M | 4.690M | 50.64M |
  | kitchen | 1.310M | 9.81M | 4.855M | 36.60M |
  | manylights | 5.858M | 7.70M | 33.390M | 44.03M |
  | luxball-hdr-vol | 1.875M | 9.77M | 9.104M | 47.83M |

  HIPRT gives the same rays/s as the BVH numbers above: it confirms that the trace is not the GPU bottleneck.
- *Measurement noise:* two builds of the same code differ by up to ~1.5% in samples/s (code layout); a single build is stable within ~1%. Changes below that are reported as "no change".

**CPU track**
- **[DONE] C1. Occlusion-only shadow rays** (bit-identical). *`43a3aaead`: +2.2% cornell, +2.8% luxball-hdr, +6.7% classroom, +3.0% kitchen, +1.3% manylights, 0% luxball-hdr-vol (volumes), less than expected because the shortcut only saves work on blocked shadow rays. The flags are in `Scene` (`shadowBlockers`), not in `SceneObject` (serialized).* `Accelerator::Occluded` with an `rtcOccluded1` override in `embreeaccel.cpp` and `IntersectionDevice::TraceOccluded` (keeps rays/sec comparable). `Material::HasPassThroughTransparency` (transparency textures; always true for null and archglass; union of the children for mix, twosided, glossycoating). A cached `shadowOpaque` flag per `SceneObject` (no bevel, no pass-through, black shadow transparency) recomputed in `Scene::Preprocess` before `lightDefs.Preprocess` (DLSC and the env visibility cache trace shadow rays there). In `Scene::Intersect`: (a) all objects shadow-opaque, no default world volume and no current volume → return `TraceOccluded`; (b) otherwise, after the trace and before `BSDF::Init`, return true if the hit object is a shadow blocker. Exact because `PathVolumeInfo::ContinueToTrace` cannot fire without a current volume (`CompareVolumePriorities` returns false for a null first argument) and the callers only read `connectionThroughput` on a miss. Rejected: a cut-down `BSDF::InitForShadow` (archglass needs the full frame) and Embree filter callbacks (not front-to-back).
- **[DONE] C2. Film write path** (bit-identical, fixes a race). *`7840b1a1f`: -0.5% to +4% (+4.0% manylights, +3.1% luxball-hdr-vol, the others within the noise).* `AtomicAdd(float*)` with `std::atomic_ref` and `pause` only on retry; atomic weight add; one `alignas(64)` struct of counters per thread (keep the serialized format; fix the `[0]`/`[i]` bug at `filmsamplescounts.cpp:62`). No per-thread films for PATHCPU: they would delay the adaptive-sampling feedback for little gain.
- **[DONE] C3. Sobol** (bit-identical). *`85954bbc6`: the scramble of the pass is cached in `SobolSequence` (bit-identical checksums). Benchmarked together with C4 against C2: within the noise (-2% to +0.2%, two builds of the same code differ by up to 1.5%). All the 348 CPU unit tests pass. The `GetSamples()` batch call is dropped: the ~68 virtual calls per sample are far below 1% of a sample's cost.* Cache `NestedUniformScramble(pass, seed)` once per sample in `SobolSampler::InitNewSample`; a `Sampler::GetSamples(first, count, out)` batch call used per bounce by the `PathTracer`.
- **[DONE] C4. Embree flags** (statistically identical). *`d721623d5`: no measurable change in samples/sec (see C3), which is consistent with the traversal not being the bottleneck. All the CPU unit tests pass.* `RTC_SCENE_FLAG_NONE` unless `accelerator.embree.dynamic` (default true only for the `RTPATH*` engines); HIGH quality and NONE on the instance sub-scenes.
- **[DONE] C5. Compiler flags** (last-ulp identical). *The "no `-march`" finding above was wrong: the Conan profiles (`build-system/conan`, `tools.build:cxxflags`) already compile LuxCore and the dependencies with `-march=x86-64-v3` on x86_64 Linux and macOS (`/arch:AVX` on Windows), so all the baselines had it. What was missing: (1) `-fno-math-errno` (GCC/Clang, `sqrt` and co. as plain instructions): +0.4% to +2.7% on five scenes, -1.6% on luxball-hdr-vol (within the 1.5% noise band), against a fresh same-day baseline `head-cpu.json`; four scenes bit-identical, classroom differs by one float ULP (max 6e-8), manylights is not deterministic. (2) A CPU check in `python/pyluxcore/__init__.py` before the native module is loaded (x86-64-v3 features from `/proc/cpuinfo`, `sysctl` or `IsProcessorFeaturePresent`), so an old CPU gets an `ImportError` instead of an illegal instruction; `LUXCORE_SKIP_CPU_CHECK=1` disables it. A Python check is safer than a guard in the module init: no native code runs before it. (3) `LUXCORE_X86_ARCH` (CMake cache) appends a `-march` for LuxCore code only, e.g. `native` for a local build; empty by default. (4) `LUXCORE_ENABLE_LTO` via `CheckIPOSupported`, measured and **left off**: with GCC 16, cornell +3.6% and luxball-hdr-vol +2.1% but luxball-hdr -4.2%, classroom -2.9%, manylights -3.3% (also last-ulp identical); the wheel is 2% smaller and the build time is the same (~280 s). `GetPlatformDesc()` reports `compile.LUXCORE_ENABLE_LTO`.* `LUXCORE_X86_ARCH` default `x86-64-v3` on x86_64 Linux/macOS for Release, `-fno-math-errno`, never `-ffast-math` (the film relies on NaN checks), never `native` by default (the wheels are distributed); an AVX2 guard in the pyluxcore module init compiled for the baseline arch; `LUXCORE_ENABLE_LTO` via `CheckIPOSupported`, kept only if the benchmark improves.
- **C6. Later.** A per-BSDF material parameter cache shared by `Evaluate` and `Sample`; `Schlick_Weight` as multiplications; one modulo and the wrap switch hoisted out of the bilinear taps; a pointer instead of a copied `Transform` in `BSDF::Init` for static meshes and a cached motion-blur inverse.

**GPU track**
- **[DONE] G1. Shadow-ray any-hit** (identical visibility). *`8b482993e`, RX 6950 XT with HIPRT: cornell +2.7%, luxball-hdr +5.1%, classroom +12.5%, kitchen +15.0%, manylights +6.3%, luxball-hdr-vol +1.0% (volumes, so the old path). The `GPUTaskConfiguration` is now also rebuilt and re-uploaded after a camera or material edit (the default volume index was never refreshed before). The PATHOCL and TILEPATHOCL unit tests pass on the OpenCL and on the HIP device.* `RAY_FLAGS_SHADOW` set in `MK_DL_SAMPLE_BSDF` after `Light_Illuminate` initialised the ray; a compiled `shadowRaysNeedBSDF` scene flag (null, archglass, transparency textures, shadow transparency, interior/exterior/default volumes) → `taskConfig->scene.shadowRayAnyHit`; `MK_RT_DL` skips `Scene_Intersect` entirely when the flag is set. HIPRT: `hiprtGeomTraversalAnyHitCustomStack` / `hiprtSceneTraversalAnyHitCustomStack` with `hiprtTraversalHintShadowRays` for flagged rays in the same kernel. OptiX: `OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT`. BVH/MBVH: break on the first hit. Scenes with transparency or volumes take the current path.
- **[DONE] G2. Compact path-state array** (bit-identical). *`a655364d4`, RX 6950 XT with HIPRT, on top of G1: cornell +13.9%, luxball-hdr +14.6%, classroom +11.6%, kitchen +8.2%, manylights +16.0%, luxball-hdr-vol +8.6% (with G1, 17-26% over the morning baseline on the non-volume scenes). The trace skip was not needed: in PATHOCL every task ends an iteration in a trace state, and the tile engines already mask their done rays. The ray counter now counts the traced rays only. Unit tests pass on both devices.* A `uint pathStates[taskCount]` buffer through a `PathState_Set` helper, `state` removed from `GPUTaskState` so any missed site fails to compile; a `uchar rayActive[]` mask passed to `EnqueueTraceRayBuffer` so the trace kernels skip loading inactive rays; `rayCount` increment moved after the state check.
- **[DONE] G3. Inlining on HIP** (last-ulp differences only). *RX 6950 XT with HIPRT, against the C5 build, `g3-hiprt-v2.json` vs `c5-hiprt-v2.json`: cornell +4.3%, luxball-hdr +5.3%, classroom +6.2%, kitchen +17.2%, manylights +9.9%, luxball-hdr-vol +18.2%. With adaptive sampling on, the time to the noise threshold 0.05 goes from 8.0-9.0 s to 4.5-5.0 s on manylights and from 31.7-35.0 s to 27.0-29.5 s on kitchen. Equal-spp RMSE against a 240 s reference is the same (manylights 0.00252 vs 0.00248-0.00256), kitchen is bit-identical at equal spp, the CPU renders are bit-identical (the CPU prelude inlines everything as before). The 174 OCL tests pass on the HIP device.*
  - *Full inlining first:* removing `__noinline__` on HIP gives kitchen +25% and luxball-hdr-vol +8% but the cold compile goes from 28 s to 110 s and the big kernels sit at the 256-VGPR cap with 864 bytes of scratch.
  - *The whitelist:* `OPENCL_HOT_FUNCTION` marks the 18 one- or two-caller functions (`Scene_Intersect`, `Material_Evaluate/Sample/Albedo/GetPassThroughTransparency/GetEmittedRadiance/GetInterior/ExteriorVolume`, `Sampler_NextSample`, `Camera_GenerateRay`, `Volume_Scatter`, `EnvLight_GetRadiance`, the texture slow paths, `Texture_Bump`, the three `TextureMapping*_Map`); it expands to nothing on HIP (the compiler decides) and to `__noinline__` on CUDA, `noinline` on OpenCL and `inline` on the CPU, i.e. no change elsewhere. The `EvalOp` switches, the image map and the noise functions stay non-inlined. Cold compile 57 s (28 s before; 17 ms cached), and the scratch of the `MK_*` kernels drops from 688-1232 to 528 bytes with 248 VGPRs (the forced calls were costing registers, not saving them). `BSDF_Init` and `Light_Illuminate` were never marked non-inlined.
  - *Benchmark pitfall found on the way:* `renderbench.py` left adaptive sampling on (`film.noiseestimation.warmup = 8`), and the Sobol sampler then skips converged pixels at a rate that depends on the timing of the noise estimation; samples/sec measured the skip rate as much as the renderer and showed manylights at -14% for a build that is +10% at equal spp. The script now sets `sampler.*.adaptive.strength = 0` unless `--threshold` is given; the older `*-hiprt.json` files are not comparable on manylights. A long cold compile also perturbs the first numbers after it: benchmark with a warm kernel cache.
  - *Tooling added:* each kernel's VGPRs and scratch bytes are logged when it is loaded (HIP via a new `hipFuncGetAttribute` entry in hipew, CUDA via `cuFuncGetAttribute`), and `LUXRAYS_KERNEL_DUMP_DIR=<dir>` writes the HIP program sources with their options, so they can be compiled offline (e.g. with the ROCm clang and `-Wuninitialized`, which found nothing).
- **G4. Per-scene kernel specialization** (bit-identical). A coarse macro set in `GetKernelParamters` (`PARAM_SAMPLER_TYPE`, `PARAM_FILM_USE_PIXEL_ATOMICS`, `PARAM_FILM_HAS_AOVS`, `PARAM_HAS_VOLUMES`, `PARAM_HAS_PASSTHROUGH`, `PARAM_PGIC_ENABLED`), tens of combinations; HIP/CUDA only (the embedded OpenCL SPIR-V would always miss); the cache key already hashes the options.
- **G5. Pinned host memory, one sync per batch, GPU-side halt.** `hipHostRegister` / `cuMemHostRegister` RAII-owned by the `ThreadFilm`; the readback enqueued before the single `FinishQueue`; leave the loop on `haltSPP` from the task stats. Small at 640x480, real at 4K.
- **G6. Full wavefront compaction** (statistically identical, 2-4 weeks; only after rocprof on the post-G2 build shows the shading kernels dominating). Per-state index queues with wave-aggregated atomics, kernels launched over `taskCount` with a device-side bound check, `gid = stateQueue[state][get_global_id(0)]`; ~40 helper functions derive the task index from `get_global_id(0)` and the sampler state must take a `taskIndex`; index-buffer trace variants for the four accelerators.

**Film track** (all engines; matters most in Blender and luxcoreui sessions with a denoiser)
- **[DONE] F1. `Film::RunTests`:** *`2cae0e821`: PATHCPU with OIDN in the pipeline and `UpdateStats()` every 0.1 s (a Blender-like session): 1.58M → 1.89M samples/s (+19%); no change without a denoiser. It also fixed the convergence test, which executed the pipeline 0 while reading the pipeline of `batch.haltnoisethreshold.index`.* execute each distinct pipeline index the due tests read once, never unconditionally. Unobservable: `GetChannel(IMAGEPIPELINE)` executes the pipeline on demand and the test timing is sample-count based.
- **[DONE, as a quality option] F2. OIDN:** *measured first: creating the device and committing the filter is 14 ms of a 1.9 s pass at 1280x720, so the planned caching is pointless; the cost is the three network executions. `426021c50` adds `film.imagepipelines.N.M.quality` (default, high, balanced, fast; the default is unchanged): balanced denoises the image in 0.4 s instead of 0.9 s, the prefilter passes stay at 0.7 s each.* The original idea: a transient `DeviceRef` created on first use, cached `FilterRef`s keyed by filter, size and aux presence.
- **[DONE] F3. Noise estimation and convergence test:** *`426021c50`: TBB parallel, separable box, double accumulators, 0 instead of NaN with an unchanged image. The NOISE channel differs from the old build as much as two old runs differ from each other (mean abs diff 0.031-0.038 vs 0.033, correlation 0.964 vs 0.962); the slowest stats poll of a 4K PATHCPU render goes from 0.60 s to 0.44 s. All the CPU tests and the OpenCL-device tests pass.* TBB-parallel; the box filter as two separable prefix-sum passes with exactly the current bounds and divisor (the NOISE channel drives the adaptive sampling, so the numbers must match).

**Execution order and status.** Step 0 → F1 → C1 → C2 → C3 → C4 → G1 → G2 → F2 → F3 first, each as a tested and committed phase; C5, C6, G3, G4, G5 and G6 later. *Status:* the first round (Step 0, F1, C1, C2, C3, C4, G1, G2, F2, F3), C5 and G3 are done and tested. Next: G4, G5, then G6 after a rocprof profile (no ROCm profiler is installed on the development machine yet). Also worth doing now: a GPU-side halt (part of G5), since the GPU renders well past the halt point before the host notices (see the Metropolis note in the open issues).

**Verification.** Bit-identical check: PATHCPU is not deterministic even with 1 thread and a fixed seed (the halt overshoots), and TILEPATHCPU with multipass tiles races with the halt test; a deterministic render is TILEPATHCPU, `native.threads.count = 1`, `tile.multipass.enable = 0`, a fixed seed and `tilepath.sampling.aa.size`. manylights is not deterministic even then (cause not found yet). Per item: the pyunittests of the touched engines and features, the full CPU suite plus all PATHOCL/TILEPATHOCL cases before each commit, and for the bit-identical items a 1-thread fixed-seed checksum before and after. Speed: `renderbench.py --baseline` after each item with `--repeat 3` and `--max-regression 3`, on an idle machine. Quality for the statistically-identical items: equal-spp RMSE against a high-spp reference, as done for the Owen scrambling.

### Phase 2: Modern light transport
- **Light tree / many-light BVH** as a new light strategy in `src/slg/lights/strategies/`. Use it alongside or instead of the DLS cache, and it works on GPU too.
  - *Why it matters here:* every emissive triangle is its own `TriangleLight` (`sceneobjectdefs.cpp:68`), so one emissive mesh means thousands of lights. The current strategies either pick from one global distribution that ignores where the shading point is (uniform, power, log-power), or use the DLS cache. The cache needs a pre-pass and memory. It also drops BSDF hits on a light it gave a zero pick probability (`pathtracer.cpp:304`, `pathoclbase_funcs.cl:251`), which darkens slightly wherever the cache is wrong.
  - *Design:* a BVH over the lights with power, bounds and orientation cones (Conty & Kulla 2018, the basis of Cycles' light tree since 3.5). `SampleLights` and `SampleLightPdf` traverse it from the shading point and normal. The pdf for a BSDF hit must come from the same traversal so the MIS weights stay correct. Environment lights stay outside the tree with their own pick probability. One option keeps the visibility knowledge, which Cycles' tree lacks: build the DLS cache entries from tree importance instead of from scratch.
  - *GPU cost:* after G2 the kernels already use 256 VGPRs and spill to scratch, and the traversal adds code to the direct-light state. Compare the cached HIP kernels' VGPR and scratch use before and after.
  - *Measure first:* the `manylights` benchmark scene plus an interior with a few hundred emissive triangles, at equal time, RMSE against log-power and the DLS cache.
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
- Intermittent render-thread segfaults when the whole test suite runs in a single process (roughly once every 20-160 tests); with each test in its own process, all 509 tests passed without a crash. Not found yet. The development machine also showed nondeterministic GCC crashes under load, so part of it may be hardware. *2026-10-13: the `--filter OCL` subset crashes every 20-90 tests on this and on the previous builds (SIGSEGV, and SIGBUS in the glass material tests on the HIP device), always between tests; `tests/benchmark`-style resuming from the crashed test (writing `totaltestsdone.txt` and `--resume`) completes the run, and the crashed tests pass when re-run.*
- `.bcf` and `.rsm` serialization of scenes with instances crashes (`observer_ptr<TriangleMesh>` of `InstanceTriangleMesh`), and `RenderConfig::serialize()` is empty (`.rsm` files never contained the configuration and the scene).
- The CannelleEtFromage scene renders much darker with PATHCPU than with PATHOCL (in the old build too): a CPU/GPU parity problem.
- `LaserLight_PATHOCL_METROPOLIS` has a larger mean brightness tolerance (Metropolis normalization noise on the GPU). `SceneEditRendering_PATHOCL_METROPOLIS` can also exceed the 1.5% mean brightness tolerance (1.68% once, then 3 out of 3 reruns passed). *2026-10-13, measured: on this scene the PATHOCL METROPOLIS brightness depends on the sample count, identically on the old and the new builds (image mean 14 at ~100 passes, 18 at ~300-800, 21 at ~1200, 28 at ~4500, against 24.7 for PATHOCL SOBOL and 24.8 for the CPU engines; the reference is 25.7). The test outcome depends on how many passes the GPU does before the host notices the halt, so the faster builds (and the HIP device) fail it more often. It is a GPU Metropolis normalization problem, not a parity one, and not a regression of the Phase 1b changes.*
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

**Milestone status:** Phase 0.1 done, Owen sampling done (blue noise not yet), GPU OIDN blocked on the dependencies, Kulla-Conty not started. Phase 1b: the first round (Step 0, F1, C1-C4, G1, G2, F2, F3), C5 and G3 are done; the GPU is 22-45% faster than in the morning (G1+G2+G3: cornell +22%, luxball-hdr +27%, classroom +33%, kitchen +46%, manylights +35%, luxball-hdr-vol +30%). Next: G4, G5, then G6 after a rocprof profile.

**Quality items, in order (added 2026-10-14):** the Cycles-style adaptive sampling and noise halt (Phase 1, step 1 then step 2), blue-noise pixel decorrelation (Phase 1), then the light tree (Phase 2) once the many-light measurement shows the gain.

## Verification approach (applies to every item)
- **Correctness:** `pyunittests/unittests.py` render diffs against reference images (a noise tolerant Pillow comparison, see Phase 0.1), on CPU and on every GPU backend.
- **Quality:** equal-time and equal-spp RMSE against a high-spp reference on the benchmark scene set, especially for sampler, light tree, guiding and volume changes.
- **Speed:** samples/sec and time-to-threshold (the convergence test in `include/slg/film/convtest/filmconvtest.h`) per backend, tracked in CI.
