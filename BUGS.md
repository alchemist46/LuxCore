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
- Reproduce: on the Linux development machine (GCC build): `pyunittests/unittests.py --filter OCL` on the HIP device; `--filter HaltEdit` (PATHCPU, 12 threads, RANDOM sampler, `simple` scene) crashed in 2 of 3 runs. Single `*_PATHOCL_METROPOLIS` tests on the HIP device (`GlassMaterial`, `ImageMapTexture`) crash on their own.
- Notes: the frames look overwritten (float-like values in saved pointers). Not a stop-time use-after-free. With `opencl.native.threads.count = 0` 0 of 10 runs crashed, so the suspect is the native CPU threads that PATHOCL runs beside the GPU (Metropolis sampler); the GPU kernels are not involved. A `-fstack-protector-all` build passed 8 runs. `HaltEdit_Threshold` re-creates the `FilmConvTest` in `Film::Parse()` while the threads render: no race with the render threads found by reading (2026-10-22). Not reproduced on the Windows machine (MSVC build, RX 6950 XT), 2026-10-23: the normal build passed `--filter PATHOCL_METROPOLIS` (56 tests, native threads on) 3 times, and the HIP `--filter OCL` runs of the previous items passed (168 tests); an ASan build (`LUX_SANITIZE=1`, MSVC `/fsanitize=address`, see ROADMAP K4) passed `--filter OCL` on the HIP device (168), every CPU test (`--subset` 113 and the other 224) and `--filter HaltEdit` without a report. ASan reports a heap overflow on a worker thread of a DLL loaded in the uninstrumented Python (checked with a probe), but it does not see writes from uninstrumented code (Embree, TBB, the other deps) into LuxCore's stack frames, which the overwritten frames with Embree-like float runs could be. So it is either specific to the Linux build (GCC, glibc, the Embree/TBB builds of the Linux LuxCoreDeps) or the development machine's hardware (it also showed GCC crashes under load). Next step: the same ASan+UBSan build on the Linux machine (`LUX_SANITIZE=1`; the Python needs `LD_PRELOAD` of `libasan` there), with the reproducers above; if it is silent, a memory test of that machine.

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

### The GPU Mix texture bump uses the second texture's bump for the amount's
- Found: 2026-10-23
- Area: texture (mix, bump)
- Problem: `MixTexture_Bump()` (`texture_bump_funcs.cl`) computes the amount's slope terms `dua`/`dva` from `bumbNTex2`, the second texture's bump, where the CPU (`MixTexture::Bump()`) uses the amount texture's bump; `bumbNAmount` is unused, so the GPU evaluates the amount's bump ops and drops them. The bumps differ wherever the amount texture varies and the two textures differ.
- Reproduce: a `mix` texture with a varying amount (i.e. an image map) and different `texture1`/`texture2` as a material's `bumptex`; PATHCPU against PATHOCL.
- Notes: found reading the code for A5, not rendered. A fix changes the GPU images of every scene with a Mix of bumps (the G0 classroom has 46 Mix textures), so it is not bit-identical; the A5 sharing does not depend on it.

### PATHOCL renders far more than the halt spp on a small film
- Found: 2026-10-23
- Area: engine (PATHOCL halt)
- Problem: on a 64x64 film, `batch.haltspp = 64` gives 141-147 samples per pixel and `256` gives 416-431 (the SAMPLECOUNT channel; every pixel within 4 of the others, so the image is fine, only slower). The halt counts the samples in flight, one per task, and the GPU runs 524288 tasks (the SampleResult buffer of the log), i.e. 128 samples per pixel of this film, whatever the halt.
- Reproduce: `lc-work/d1/toy/toy.py PATHOCL aquad 64 64 256 --set opencl.devices.select=01 --set film.opencl.enable=0` (a floor lit by one quad light, depth 1), HIP.
- Notes: the task count does not follow the film size (the Sobol bucket size does since 2026-10-18). At 540x540 it is about two samples per pixel (524288 tasks for 291600 pixels). Found during D1; not investigated further.

### PhotonGI drops the light hits after a diffuse then a non-diffuse bounce
- Found: 2026-10-23
- Area: engine, caches
- Problem: with either PhotonGI cache on, `IsDirectLightHitVisible()` (`photongicache.cpp:101`, `pgic_funcs.cl:46`) drops every light hit whose last bounce is not diffuse once the path had a diffuse bounce (`!(lastBSDFEvent & DIFFUSE) && diffuseDepth > 0`), a firefly cut. Nothing adds these paths back: the caustic cache stores only the nearly specular chains, and the indirect cache only where it ends a path. Diffuse -> glossy -> light and the light leaving a scattering medium through glass are lost.
- Reproduce: `tests/furnace/furnace.py PATHCPU glass-vol-s1 path.photongi.indirect.enabled=1` (or `caustic.enabled=1`): disk 0.209 instead of 1.000 (1.000 with the caches off), same before and after CA1.
- Notes: found during CA1 (2026-10-23). The fix (drop only the nearly caustic paths the caustic cache covers, the third test of the function) changes the look of every PhotonGI render with glossy surfaces (brighter, more fireflies): a decision for the user, see ROADMAP CA1.

### PhotonGI caustic lookups on the GPU vary up to 15x in time between runs
- Found: 2026-10-23
- Area: engine (PATHOCL), caches
- Problem: the same render takes 24 s or 341 s of kernel time depending on the run: each run traces its own caustic photons (Metropolis) and the GPU lookup visits every photon within the radius (`PGICPhotonBvh_ConnectAllNearEntries()`), so a dense caustic spot costs a lot.
- Reproduce: cornell-glass with `path.photongi.caustic.enabled = 1`, `caustic.lookup.radius = 0.1`, `photon.maxcount = 4000000`, `caustic.updatespp = 0`, hybrid off, PATHOCL 200x200 at 64 spp: kernels 23.6 s in one run, 341 s in another (same build, `lc-work/ca1/timing.py`).
- Notes: not investigated: not a regression of CA1 (the base build shows both). A cap on the photons per lookup (the old `caustic.lookup.maxcount`, commented out in `scenes/causticcube/render.cfg`) or a k-nearest lookup would bound it.

### PATHOCL stops at about 20 spp after a scene edit with a halt spp
- Found: 2026-10-23
- Area: engine (PATHOCL halt, scene edits)
- Problem: a session with `batch.haltspp = 64` renders its 64 spp, then after `BeginSceneEdit()`/`EndSceneEdit()` (a camera move) the film stops at 20 spp: `HasDone()` stays false and the pass count does not move for minutes. With `batch.haltspp = 32` the session reports done at 20 spp. Same with every cache off and on the build before CA1 (244d25614), so not PhotonGI.
- Reproduce: `EDIT="scene.camera.lookat.orig = -2.5 -8. 2.9" SPP=64 out/venvtest/Scripts/python -u lc-work/ca1/edit_halt_repro.py PATHOCL "path.photongi.indirect.enabled = 0;path.photongi.caustic.enabled = 0"` (cornell-glass 160x160, HIP).
- Notes: suspect the per-thread halt check (`PathOCLOpenCLRenderThread::RenderThreadImpl()`: `haltSPP`, `GetThreadFilmsEyeSPP()`, the spp-per-iteration estimate) counting the samples of before the edit. Found during CA1; the PhotonGI edit test there halts by hand instead.
