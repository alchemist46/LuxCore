# Known bugs

Bugs and issues found during a task that were not fixed right away. See "Keep the bug list" in [CLAUDE.md](CLAUDE.md) for the rules. Remove an entry when it is fixed.

The first entries were collected on 2026-10-20 from "Known open issues" in [MODERNIZATION_PLAN.md](docs/history/MODERNIZATION_PLAN.md) (the "Found" date is the date given there) and from [SPEEDUP_CHANGES.md](docs/history/SPEEDUP_CHANGES.md); the plan has the full measurements. Entries the plan already marks as fixed are not listed. The open work items are in [ROADMAP.md](ROADMAP.md). Bugs in the BlendLuxCore addon are in the addon's own `BUGS.md`.

## Entry format

```
### <short title>
- Found: YYYY-MM-DD
- Area: engine | material | texture | light | film | build | tests | ...
- Problem: what happens, and what was expected
- Reproduce: scene, engine, command, spp
- Notes: cause, suspects, what was ruled out
```

## Open

### Render-thread segfaults when many tests run in one process
- Found: 2026-10-13
- Area: engine, tests
- Problem: SIGSEGV or SIGBUS in a CPU render thread in the middle of a render, roughly once every 20-160 tests (every 20-90 with `--filter OCL`). Each crashed test passes when re-run alone.
- Reproduce: `pyunittests/unittests.py --filter OCL` on the HIP device; `--filter HaltEdit` (PATHCPU, 12 threads, RANDOM sampler, `simple` scene) crashed in 2 of 3 runs. Single `*_PATHOCL_METROPOLIS` tests on the HIP device (`GlassMaterial`, `ImageMapTexture`) crash on their own.
- Notes: the frames look overwritten (float-like values in saved pointers). Not a stop-time use-after-free. With `opencl.native.threads.count = 0` 0 of 10 runs crashed, so the suspect is the native CPU threads that PATHOCL runs beside the GPU (Metropolis sampler); the GPU kernels are not involved. A `-fstack-protector-all` build passed 8 runs. `HaltEdit_Threshold` re-creates the `FilmConvTest` in `Film::Parse()` while the threads render: no race with the render threads found by reading (2026-10-22). Next step: a LuxCore-only ASan build (MSVC has `/fsanitize=address` on Windows) (needs `libasan`/`libubsan` on the development machine).

### PATHOCL halt bias with the subsurface material
- Found: 2026-10-18
- Area: engine (PATHOCL)
- Problem: a subsurface walk takes one iteration per step, so the long walks (the darker ones) are the paths in flight at the halt, and a subsurface sphere reads up to 1.8% too bright at 64 spp (0.2% at 512).
- Reproduce: subsurface sphere, PATHOCL, `batch.haltspp = 64`.
- Notes: the general halt bias (paths in flight) was fixed on 2026-10-18; this is the opposite effect of the subsurface walks.

### Dense scattering media behind a delta glass read wrong in the furnace
- Found: 2026-10-18 (the sigma_s 1 loss, fixed 2026-10-21); this part 2026-10-21
- Area: material, engine (volumes)
- Problem: white furnace, a unit sphere with a white homogeneous interior (`multiscattering = 1`) behind a delta `glass` (kr = kt = 1, ior 1.5): at sigma_s 4 PATHCPU reads 1.11-1.16 and PATHOCL 0.91 (1024 spp, relative std 1.6-2.9 per pixel: fireflies). At sigma_s 1 both read 0.996-0.999 and the `null` boundary is lossless at both densities. The rough OpenPBR glass read 0.894 at sigma_s 4 (0.910 with roughness 0) before the fix: not re-measured. BIDIRCPU reads 1.059 at sigma_s 1 and 1.145 at sigma_s 4 (2026-10-22, 256 spp).
- Reproduce: `tests/furnace/furnace.py PATHCPU glass-vol-s4 spp=1024` (and PATHOCL).
- Notes: the sigma_s 1 loss (0.630) was the correlation of the volume's scatter decision with the glass lobe choice (both read the same pass-through event), fixed by remapping the event for the surface. The CPU and GPU disagree at sigma_s 4 in opposite directions, so it is not a shared estimator bias; the fireflies come from long internal paths with the lobe probability clamped to [0.25, 0.75].

### BiDir reads 2-4% low on rough glass
- Found: 2026-10-18
- Area: engine (BIDIRCPU), material
- Problem: BiDir is 2-4% low on rough glass and on the OpenPBR glass alike (OpenPBR 0.990/0.973/0.966 at alpha 0.2/0.5/1, half glass over a white base 0.955). The path engines read 0.997/0.992 at alpha 0.5/1.
- Reproduce: white furnace with a rough glass sphere, BIDIRCPU against PATHCPU.
- Notes: not the interface's compensation. The `fromLight` branches of `Sample()` have not been derived against the adjoint BSDF. Leaving the Russian roulette out of the MIS weights (2026-10-22) did not change it: 0.969/0.974/0.961 for rough glass, OpenPBR glass and half glass at alpha 0.5 (256 spp).

### Caustics seen through a specular surface are missing with light tracing
- Found: 2026-10-16
- Area: engine (light tracing)
- Problem: a floor caustic reflected in a mirror or seen through a window is produced by neither side: the eye path drops it as caustic and the light tracer connects only directly to the lens. Missing energy with light tracing on, unless PhotonGI's caustic cache is on too.
- Reproduce: caustic on a floor, seen in a mirror, with `path.hybridbackforward.enable`.
- Notes: MNEE covers part of it; the rest is a bidirectional connection, out of scope for the hybrid. On PATHCPU the hybrid on and off agree on a floor caustic seen in a mirror and through a window (2026-10-22, 512 spp), so it is not reproduced there; PATHOCL not checked yet (the GPU was down).

### Direct light sampling is off for emissive meshes above 256 triangles
- Found: 2026-10-16
- Area: light
- Problem: `emission.directlightsampling.type = AUTO` (the default) disables direct light sampling for emissive meshes with more than 256 triangles; they are found by BSDF rays only (noisy). The classroom's lamp meshes are such lights.
- Reproduce: classroom scene.
- Notes: the light tree keeps the limit (measured 2026-10-16): lifting it lets decorative lamp meshes take the picks of the real lights next to them.

### GPU Metropolis sampler bias with short chains
- Found: 2026-10-17
- Area: engine (PATHOCL)
- Problem: each task estimates the mean image luminance (the Kelemen normalization) from its own large steps, and each chain starts at a uniform sample, not at one drawn proportionally to the luminance. Both bias short chains in opposite directions (simple scene, 150x100, one task per pixel, at the test halt: +0.9%; with a fixed exact luminance and 512K tasks the start-up alone is -20% at 267 spp).
- Reproduce: `*_PATHOCL_METROPOLIS` tests; small films or few spp.
- Notes: a global luminance estimate (a reduction over the tasks) and a resampled start (PSSMLT bootstrap) would remove both. Only matters when the chains are short: a chain gets about spp x pixels / tasks samples, about 1000 for a 1080p render at 256 spp with the default 512K tasks (a small bias), 8 for the 150x100 test film. Decided 2026-10-22 to leave it as a known limitation (the GPU renders use Sobol by default, and the hybrid light tracing, PhotonGI's caustic cache and the light tree cover most of Metropolis' scenes). Workarounds for small films or previews: fewer tasks (`opencl.task.count`) for longer chains, or the CPU engines, whose few chains are long.

### BiDir darkens the silhouette of the furnace sphere
- Found: 2026-10-22
- Area: engine (BIDIRCPU)
- Problem: in the white furnace the pixels on the outline of the sphere read 0.6-0.7 with BIDIRCPU, whatever the material (matte too), and the image mean is 0.973 against 0.9999 with PATHCPU; the inner disk is right (0.999). The "mu < .2" band of `tests/furnace/furnace.py` shows it (0.40 for matte).
- Reproduce: a white matte sphere under a constant white environment, orthographic camera, 48x48, no pixel filter, BIDIRCPU 512 spp: the first and the last pixel on the sphere of the centre row and column read 0.6.
- Notes: in the build before the camera row fix (b22905df5) the dark ring was one row off (row 0 on top, the third last row at the bottom); now it is symmetric. Suspects: the light-traced connections to the camera at grazing angles. Not investigated.

### TILEPATHOCL is slow with dense, nearly white subsurface media
- Found: 2026-10-22
- Area: engine (TILEPATHOCL)
- Problem: a random walk goes on past 256 bounces with a Russian roulette, up to 65536 (`subsurface_funcs.cl`), and TILEPATHOCL runs the trace and `MK_RT` kernels of a tile pass until its last task is done, one walk step per iteration (about 0.37 ms each with 4096 tasks, launch-bound): the longest walk of the pass sets its time. Measured 2026-10-22 (furnace, 96x96, 256 spp, TILEPATHOCL against the old hard limit of 256 bounces): a white sphere of mean free path 0.01 of its radius takes 24 s instead of 1 s (it read 0.864 with the limit, 0.997 now), mean free path 1.0 6 s instead of 1 s, albedo 0.95 10 s instead of 1 s (0.922 with the limit, 0.962 now). Albedo 0.8 and below, and the coloured and coated cases, take the same time. PATHOCL is barely affected (4 s for the densest), since its tasks start new samples independently.
- Reproduce: `tests/furnace/furnace.py TILEPATHOCL subsurface-rad0.01 spp=256` against `PATHOCL`.
- Notes: the energy the limit lost is real (4% at albedo 0.95), so the fix is to take several walk steps per kernel launch rather than to shorten the walks: that needs the ray traversal inside `MK_RT` (HIPRT can traverse from any kernel) instead of the separate trace kernel. A TILEPATHOCL-only steeper roulette (survival (b / (b + 1))^2, reaching n bounces with (256 / n)^2) would shorten the tail at the cost of noise in the densest media; not tried.
### One pixel of luxball-hdr-vol differs between TILEPATHOCL runs at a fixed seed
- Found: 2026-10-22
- Area: engine
- Problem: two runs of the same build should differ by at most 1e-6 per value (the atomic adds of the film), but pixel (x 143, y 101) of luxball-hdr-vol comes out with one of two values that differ by 1.5e-2 (relative); the rest of the image is identical.
- Reproduce: `tests/benchmark/parity.py render <dir> luxball-hdr-vol` (TILEPATHOCL, seed 1, aa 8, 320x240, HIP device) six times with the same build, then `compare`: 2 of 6 runs at HEAD `aa2fa98c1` had the other value.
- Notes: always the same pixel and the same two values, so one sample takes one of two discrete outcomes depending on timing, not accumulation order. Suspects: something read across tasks or threads in the volume path (the null-shell boundary init of V4, the pass-through of V1, the ray-hit buffers of the volume continuation). It can make a bit-identical check of `parity.py` fail on this scene: re-run it before looking for a change.

### PhotonGI: the GPU drops light hits after the caustic cache fired
- Found: 2026-10-22
- Area: engine (PATHOCL), caches
- Problem: `pgic_funcs.cl:50` tests `!isNearlyCaustic` where the CPU tests `!IsCausticPath()` (`isNearlyCaustic && depth > 1`, `photongicache.cpp:101`, `pathinfo.h:102`), so once `photonGICausticCacheUsed` is set at the first vertex the GPU adds no emission at a light hit after one diffuse bounce (the BSDF half of the direct light); the CPU does.
- Reproduce: cornell-glass with `path.photongi.caustic.enabled = 1`, PATHCPU against PATHOCL means, 256 spp.
- Notes: found by reading (ROADMAP CA1), magnitude not measured. Fix: `EyePathInfo_IsCausticPath()` (`pathinfo_funcs.cl:143-145`).

### PhotonGI is not rebuilt on scene edits and its caustic update stops after one
- Found: 2026-10-22
- Area: engine, caches
- Problem: the cache is built in `StartLockLess` only (`pathcpu.cpp:158-163`, `pathoclbase.cpp:522-535`), so it goes stale on geometry, light and material edits; `FinishUpdate()` sets `finishUpdateFlag` (`pgicupdate.cpp:91-100`) and nothing resets it, so after any edit the periodic caustic update never runs again (and `filmSPP - lastUpdateSpp` underflows after the film reset).
- Reproduce: a PATHCPU session with the caustic cache and `updatespp = 8`; edit a light; render on.
- Notes: ROADMAP CA1.

### PhotonGI multi-GPU keeps the old caustic map
- Found: 2026-10-22
- Area: engine (PATHOCL), caches
- Problem: `Update()` returns true to thread 0 only, so only the first device re-runs `InitPhotonGI()` and `SetKernelArgs()` (`pathoclopenclthread.cpp:341-346`); the other devices keep the previous caustic map and radius.
- Reproduce: two OpenCL/HIP devices, the caustic cache with the periodic update.
- Notes: ROADMAP CA1; not reproducible on this machine (one GPU).

### PhotonGI persistent file and resume
- Found: 2026-10-22
- Area: engine, serialization
- Problem: a `.pgi` file is loaded with no scene or parameter check and its stored params override the config's (`pgicserialize.cpp:31-86`); `samplerType`, `photon.time.*` and `finishUpdateFlag` are not serialized (uninitialized after a load, `photongicache.h:225-260`); the `.rsm` resume re-creates the update barrier with the saved thread count (`pgicserialize.cpp:99-119`), so a resume with another thread count desynchronizes the update.
- Reproduce: render with `path.photongi.persistent.file`, change the lights, render again: the old cache is used.
- Notes: ROADMAP CA1; see also K5.

### A null PhotonGI indirect lookup ends the path
- Found: 2026-10-22
- Area: engine, caches
- Problem: when the cache is enabled at a vertex but no radiance photon is within the radius, `GetIndirectRadiance()` returns null and the path still breaks (`pathtracer.cpp:714-718`, `pathoclbase_kernels_micro.cl:699-705`): lost energy where the visibility pass placed no entry, and wherever `path.regularization.enable` makes a mirror or a near-specular glossy surface eligible (the gate reads the regularized glossiness, `photongicache.cpp:70`; the caches are built without regularization).
- Reproduce: a scene with the indirect cache and a mirror, regularization on; compare with the cache off.
- Notes: ROADMAP CA1.

### DLSC: BVH leak, stale radius and a pick floor on DLS-disabled lights
- Found: 2026-10-22
- Area: light
- Problem: on an edit `Build()` allocates a new `DLSCBvh` without freeing the old (`dlscacheimpl.cpp:544`), the radius is not re-estimated (no longer 0 after the first build), an edit that removes every light keeps the stale cache (`:486-487`), and CAMERA_EDIT does not rebuild it although it depends on the camera (`scenepreprocess.cpp:95-103`); the 2.5% floor goes to every light, DLS-disabled and always-shadowed ones included (`:265-282, :361-362`), so DLS-disabled lights can be picked; one radical-inverse sequence serves every entry (`:246-326`), with a pass-0 division by zero (`:296`) and the luminance average divided by `pass` after `pass+1` samples (`:321`); the GPU shadow catchers that only see infinite lights take the DLSC branch (`pathoclbase_funcs.cl:309-318`) where the CPU uses the infinite-only strategy.
- Reproduce: manylights with `lightstrategy.type = DLS_CACHE`, a light edit in a session.
- Notes: ROADMAP CA2 retires the cache. The manylights non-determinism of the deterministic TILEPATHCPU setup (MODERNIZATION_PLAN "Verification") is probably this build.

### ELVC: rebuilt on every preprocess, one per scene on the GPU, tile distributions lost from the file, sky2 transform, pdf 0 with no entry
- Found: 2026-10-22
- Area: light
- Problem: `UpdateVisibilityMaps()` runs on every `Scene::Preprocess` whatever the edit flags (`scenepreprocess.cpp:106`), camera-only edits included; `CompileELVC` keeps one cache per scene and drops the others with a warning (`compilelights.cpp:417-425`); a persistent file skips `BuildTileDistributions()` (`envlightvisibilitycache.cpp:686-696`), so it samples uniformly within a tile; sky2 with a `.transformation` is binned in light-local lat-long (`:346`) while it samples in world lat-long (`sky2light.cpp:345, :430`), an efficiency loss; `Sample()` returns pdf 0 where no entry is near, so the env light gets no light sampling there (the comment promises a fallback to the map that does not exist); the DLSC build traces its env shadow rays before the ELVC is rebuilt.
- Reproduce: a sky2 light with `visibilitymapcache.enable = 1` and a transform; an RTPATHCPU session with a camera edit (the rebuild time).
- Notes: ROADMAP CA3.

### The caches' eye-path tracer differs from the render's
- Found: 2026-10-22
- Area: engine, caches
- Problem: `SceneVisibility<T>` (`scenevisibility.cpp:92-131`) has its own Russian roulette (depth 3, cap 0.5), 4 sampler dimensions per vertex and its own depth rules, uses `GetHardwareThreadCount()` threads whatever the engine's count, and camera times 0..1 for DLSC and ELVC (the shutter ignored), so the visibility particles of the three caches do not follow the render's depths, roulette, regularization or subsurface rules.
- Reproduce: by reading; the particle placement differs from where the render's paths land.
- Notes: deferred with K1 (ROADMAP CA, deferred).

### PATHOCL renders the sun 0.1% brighter than PATHCPU
- Found: 2026-10-22
- Area: light (sun), engine (PATHOCL)
- Problem: a grey matte sphere and floor lit only by a sun read 1.0011 of PATHCPU in every channel with PATHOCL (2 seeds each, 256 spp, 96x96; the seed noise is under 1e-4). The other lights of the same check (sky2, a point light and an emissive quad with a temperature, a constant environment) agree within 0.02%.
- Reproduce: `tests/benchmark/cpugpu.py <outdir> cases=sun` (a `sun` light, `dir = 0.3 -0.4 1`, gain 0.01, path depth 8).
- Notes: not investigated. Suspects: the sun disk sampling or its pdf (cone angle from the relative size), CPU against GPU.

### The bump of a TwoSided sub-material differs between the CPU and the GPU
- Found: 2026-10-22
- Area: material (twosided, bump)
- Problem: the CPU bumps a TwoSided with its front or back material's `Bump()`, so with that material's `bumpsamplingdistance` and, for a nested TwoSided, recursively; the GPU (`Material_GetBumpTexIndex()`) takes only the bump texture of the front or back material and uses the TwoSided's own `bumpsamplingdistance`. The images differ only when the sampling distances differ or TwoSideds are nested.
- Reproduce: a TwoSided whose front material has a bump texture and `bumpsamplingdistance = 0.01` (the TwoSided left at the default 0.001); PATHCPU against PATHOCL.
- Notes: found reading the code for A6, not rendered. The GPU rule is also the slot rule of the bump cache (`CompiledScene::MaterialHasBump()`), so a fix on the GPU side changes both.
