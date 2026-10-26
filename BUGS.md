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

### The subsurface walk loses up to 10% of a white medium under a strong boundary
- Found: 2026-10-25
- Area: material (subsurface walk)
- Problem: a white walk (color 1) should return all the light that enters, but the walks longer than `SSS_MAX_BOUNCES` (65536, `subsurface_funcs.cl`) are dropped. Under the internal reflection of the boundary a walk reaches the surface many times before it leaves (about 1 / 0.08 times at eta 3), so the long walks are many more: a white slab 2000 mean free paths deep returns 0.991 of the light that enters at eta 1, 0.976 at eta 1.5 and 0.898 at eta 3 (PATHCPU, 256 spp; a sphere of 100 mean free paths loses less, 0.75% at eta 3).
- Reproduce: `tests/furnace/subsurface_boundary.py PATHCPU colors=1` (the `d1` of the slab: `(d(1) - d(0)) / (1 - d(0))`), or the furnace `subsurface-rad0.01` case with `ior = 3`.
- Notes: found while fixing the color of the walk under the boundary (SSS_BoundaryColor()). Raising the limit costs time on the GPU (see the TILEPATHOCL entry); a cheaper fix would be to finish a walk that is deep in a nearly white medium analytically (its escape probability from diffusion theory) instead of dropping it.

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
- Notes: always the same pixel and the same two values, so one sample takes one of two discrete outcomes depending on timing, not accumulation order. Suspects: something read across tasks or threads in the volume path (the null-shell boundary init of V4, the pass-through of V1, the ray-hit buffers of the volume continuation, the homogeneous volume's scattering, the film write of a sample from two tasks). It can make a bit-identical check of `parity.py` fail on this scene: re-run it before looking for a change. Seen again on 2026-10-24 with the build of the GGX minimum roughness fix (1 of 3 runs, the red channel of the same pixel; the scene has only matte and null materials, so that change cannot reach it).

### ELVC: one cache per scene on the GPU
- Found: 2026-10-22
- Area: light
- Problem: `CompileELVC` keeps one visibility map cache per scene and drops the others with a warning (`compilelights.cpp:417-425`): with two environment lights that have `visibilitymapcache.enable = 1`, the second one samples its own distribution on the GPU (unbiased since the cache falls back to it where it has no map, an efficiency loss only).
- Reproduce: an infinite light and a sky2 light, both with the cache on, PATHOCL: the log prints "OpenCL rendering supports only one EnvLightVisibilityCache".
- Notes: ROADMAP CA3. The GPU would need the cache data of each light at its own offsets (entries, BVH nodes, tile distributions, radius, normal angle) in the light record. The other parts of the old entry were fixed on 2026-10-25 (rebuilt only on visibility edits, tile distributions rebuilt after loading a file, binning in each light's own uv mapping, the light's own distribution where the cache has no map).

### The caches' eye-path tracer differs from the render's
- Found: 2026-10-22
- Area: engine, caches
- Problem: `SceneVisibility<T>` (`scenevisibility.cpp:92-131`) has its own Russian roulette (depth 3, cap 0.5), 4 sampler dimensions per vertex and its own depth rules, uses `GetHardwareThreadCount()` threads whatever the engine's count, and camera times 0..1 for ELVC (the shutter ignored), so the visibility particles of the two caches do not follow the render's depths, roulette, regularization or subsurface rules.
- Reproduce: by reading; the particle placement differs from where the render's paths land.
- Notes: deferred with K1 (ROADMAP CA, deferred).

### PhotonGI caustic lookups on the GPU vary up to 15x in time between runs
- Found: 2026-10-23
- Area: engine (PATHOCL), caches
- Problem: the same render takes 24 s or 341 s of kernel time depending on the run: each run traces its own caustic photons (Metropolis) and the GPU lookup visits every photon within the radius (`PGICPhotonBvh_ConnectAllNearEntries()`), so a dense caustic spot costs a lot.
- Reproduce: cornell-glass with `path.photongi.caustic.enabled = 1`, `caustic.lookup.radius = 0.1`, `photon.maxcount = 4000000`, `caustic.updatespp = 0`, hybrid off, PATHOCL 200x200 at 64 spp: kernels 23.6 s in one run, 341 s in another (same build, `lc-work/ca1/timing.py`).
- Notes: not investigated: not a regression of CA1 (the base build shows both). A cap on the photons per lookup (the old `caustic.lookup.maxcount`, commented out in `scenes/causticcube/render.cfg`) or a k-nearest lookup would bound it.

### The light tree's learning biases the direct light
- Found: 2026-10-23
- Area: light (light tree learning), engine (PATHCPU)
- Problem: with the learning on (the default) PATHCPU reads the direct light of a light of a few triangles too bright: +0.07% at depth 1 on a box lit by a tilted 0.6x0.4 rectangle exported as two triangle lights (0.070276-0.070284 against 0.070224-0.070227 with `lightstrategy.lighttree.learning.enable = 0`, with UNIFORM and with the rectangle as one quad light, ±0.000003 per seed pair at 1024 spp), the same on one thread; +0.01% at depth 3 on the box with a horizontal quad as two triangles. PATHOCL reads +0.006% on the tilted case (0.070231 against 0.070227, ±0.000001).
- Reproduce: `lc-work/d3/qtest.py PATHCPU 64 1024 1 --light tilted --set lightstrategy.trianglelight.quad=0` against the same with `--set lightstrategy.lighttree.learning.enable=0` (64x64 SDL Cornell box, HIP not needed).
- Notes: not a thread race (one thread the same). A light that is one leaf of the tree (a Blender area light since D3 samples its quad as one light) has nothing to learn and is exact. Found during D3.
- Cause (2026-10-25): the correlation of the learned probabilities with the Sobol sample of the pick. Measured on the tilted case at depth 1 (light sampling only, no MIS): the learned weight 0 is exact (0.070226), 0.3 reads +0.055%, the default +0.07%, cells frozen after a few samples (`learning.samples = 4`) +0.13%, smaller cells (`cellsize = 0.011`) +0.10%; both triangles (each its own light group) read +0.10-0.16%. The pick itself is right: its frequency per slot matches its pdf within 1e-4. The probabilities are learned from the pixel's earlier samples and the pick reads the pixel's next Owen-scrambled Sobol sample, which is correlated with them, so E[f/P] is not I. (`sampler.type = RANDOM` is SOBOL since 979b505b4, so an "i.i.d." run is not one.) Tried: (1) the pick number from the render thread's RNG when the strategy learns: unbiased (0.070241 +- 9, 0.070215 +- 10, 0.070227 +- 8 against 0.070227), but 8x the median variance of the learning-off render on this scene (the pick loses its stratification); (2) the Sobol number shifted by a hash of the rebuild number, new at each rebuild: biased the other way (-0.016%, -0.08% with the frozen cells), since the shift is shared by every sample of an epoch. Not tried: learning the cells of a pixel only from the other pixels (two sets of cells by pixel parity), which keeps the stratification but needs the pixel parity at every pick, pdf and learning call on both devices. Learning also raised the variance on this two-triangle scene (0.0042 against 0.0027 var*spp with the Sobol pick).
- The GPU too (2026-10-25, merged from "PATHOCL renders a matte plane under a point light 0.2% brighter than PATHCPU", found the same day): a 4x4 matte plane (kd 0.7) under a point light (`position = 1.5 -1 1.2`, gain 6) and a constant environment of 0.2, 160x120, 256 spp, 3 seeds: GPU/CPU 1.00066 +- 0.00001 with the learning, 0.99999 without; the GPU reads +0.074% with the learning against without, the CPU +0.006%; each light alone is exact on both (1.00000). Script: `C:\Users\Ricardo\lc-work\fix\pointplane.py`.
- Also found: the learning sums `contribution / pickPdf` (`AddSample()`, `LightTree_LearnAddContribution()`), but the contribution is already divided by the pick pdf (`factor` in `DirectLightSampling()`, `pathoclbase_funcs.cl`), so a cell learns I_j / P_j and settles on P proportional to sqrt(I_j). The fixed target (`contribution` alone) measured the same on the tilted case (0.0076 against 0.0072 var*spp, with the independent pick); not measured on the native scenes, so not changed.

### PhotonGI caustic cache reads darker than the light it replaces
- Found: 2026-10-23
- Area: caches (PhotonGI)
- Problem: with the caustic cache a render reads darker than with every cache off, beyond the noise: cornell-glass with the right sphere's shadow transparency off, a large ceiling light (`Grey.ply`, emission 60) and the middle sphere a mirror: -0.39% (+-0.06%); the middle sphere a 0.04-rough silver metal2: -0.60% (+-0.09%); the small light and glass only: -0.04% (+-0.28%) (PATHCPU 200x200, 128 spp, 4 seeds, a new cache each run; photon.maxcount 4M, caustic.maxsize 1M, radius 0.1, updatespp 0, hybrid off). With one persistent cache file and the lookup at the first vertex only, the cache adds about 0.0014-0.0019 of the image mean where the path-traced light it replaces is about 0.003-0.004 (noisy).
- Reproduce: `lc-work/pgicut/pt.py render <out> cmm-caustic --engine PATHCPU --spp 128 --seed N` against `cmm-none` (cases `cmb-*` for the metal).
- Notes: found while removing the firefly cut (CA1). Not a double count or a missing path class as far as checked: the paths the render drops follow the photon definition (the same glossiness threshold, the same surfaces, 1 to photon.maxdepth - 1 bounces). Ruled out: the light sample skip (the mirror has none), the photon sampler (RANDOM the same), the radius (0.03 the same), D3's quad pairing (`lightstrategy.trianglelight.quad = 0`: -0.26%, 2 seeds). Suspects: photons are stored only near a visibility particle (`tracephotonsthread.cpp`, `allNearEntryIndices`), the lookup's normal tests, the edge loss of the disk. The old firefly cut hid it (it dropped far more: -8.6% on the metal case). In a volume too (2026-10-25, after the volume fix): the furnace `glass-vol-s1` reads 0.973 with the caustic cache against 1.004 without (PATHCPU 256 spp, 4M photons), 0.193 against 0.200 with absorption 1; part of it is the blur of the lookup sphere at the medium's boundary.

### PhotonGI indirect cache in a volume uses the surface estimate (suspected)
- Found: 2026-10-25
- Area: caches (PhotonGI), volumes
- Problem: `PGICVisibilityParticle::ComputeRadiance()` (`photongicache.h`) turns the photons of a volume visibility particle into radiance with the surface formula: the flux over an area (pi r^2, or the one estimated from the hit distances) times `EvaluateTotal() / pi`. A volume needs the flux over a volume (4/3 pi r^3), the phase function instead of 1/pi, and the photon flux without the sigma_t of the volume's scatter weight (as the caustic cache does since 2026-10-25, `TracePhotonsThread::TracePhotonPath()`).
- Reproduce: not reproduced yet. `tests/furnace/furnace.py PATHCPU glass-vol-s1 path.photongi.indirect.enabled=1 scene.volumes.v.photongi.enable=1` (and `null-vol-s1`) render bit-identical images with and without the cache (2026-10-25: the cache is built, 4 s -> 15 s, but no volume vertex reads it: the lookup needs the previous vertex cache-enabled and a ray longer than the usage threshold, `pathtracer.cpp:706`). A larger medium (a room-sized fog) is needed to reach the formula.
- Notes: found while fixing the caustic cache in volumes (10x too bright); by reading only. Volumes have `photongi.enable` false by default.

### CannelleEtFromage: PATHOCL reads 3% brighter than PATHCPU without clamping
- Found: 2026-10-25
- Area: engine (PATHOCL against PATHCPU), material (glass caustics?)
- Problem: with the variance clamping off, quarter film, 256 spp, the green mean is 0.2210/0.2213 (seeds 1/2) on PATHCPU with the scene's hybrid (its partition of 0 replaced by 0.8), 0.2202 on PATHCPU with the hybrid off (seed 1), and 0.2272/0.2270 on PATHOCL (the eye paths render the caustics, no native threads). The seed noise is about 0.1%. The build before the hybrid fix, whose GPU eye paths dropped the caustics, read 0.2226 on PATHOCL (seed 1): the GPU without its caustics is already 0.7% above the CPU with them, and the GPU's eye-path caustics add 2% where the CPU's eye paths (hybrid off) read 0.4% below its light-traced caustics.
- Reproduce: `tests/benchmark/native_render.py --scene CannelleEtFromage --scale 0.25 --spp 256 --repeats 1 --set path.clamping.variance.maxvalue=0`, with and without `--engine PATHCPU --device cpu` (add `--set path.hybridbackforward.enable=0` for the CPU without the hybrid).
- Notes: found while fixing the hybrid with a partition of 0 (2026-10-25); before that fix PATHCPU rendered only the light-traced caustics (mean 0.006) and the GPU dropped the caustics, which hid it. MODERNIZATION_PLAN "Verification" (2026-10-17) had the four engines within the noise on the CPU export of the scene with the clamping off: so it may be newer than that, or specific to this export. Not split by material or path class yet.

### Glossy-heavy native scenes render 1.5x slower than LuxCore 2.11 on the CPU
- Found: 2026-10-24
- Area: material (glossy2, metal2, glossycoating; CPU)
- Problem: `scenes/LuxCore2.1Benchmark` (13 glossy2, 3 metal2, 2 glossycoating) with the PhotonGI cache off, PATHCPU, quarter film, 128 spp: 8.1-8.6 s in upstream pyluxcore 2.11.2, af4f833ea and 16e85154b, 12.3 s at HEAD; DanishMood is 1.45x slower the same way. The GPU is not affected: with the cache off (af4f833ea's was broken), half film, 256 spp, HEAD traces the same 1.15-1.28x longer paths but 1.5-2.2x more rays/s than af4f833ea, so it is faster per sample on every native scene (LuxCore2.1Benchmark 22.5 against 28.9 ms per pass, DanishMood 33 against 43, LynxSpider 98 against 149, CannelleEtFromage 61 against 84, OrangeJuice 21 against 35) and 2-10x less noisy per second; the per-channel coating lookups of 351123475 bring nothing there (the compiler merges them already: 11.50 against 11.51 s, the same image). The PhotonGI part of the first measurement was the KdTree bug fixed in 700c091d6.
- Reproduce: `tests/benchmark/native_render.py --scene LuxCore2.1Benchmark --engine PATHCPU --device cpu --scale 0.25 --spp 128 --set path.photongi.indirect.enabled=0` with `out/venvup2112` and HEAD.
- Notes: one build per commit (`C:/Users/Ricardo/lc-work/bis`, venvs `out/vb-<commit>`), two alternating rounds on an idle machine, with the rays per sample (the path length) and the rays/s (the code speed) from the engine statistics. Of the 1.45x, 1.27x is longer paths and 1.17x slower code. Longer paths, the price of the physics fixes: 48c670678 "GGX is the only microfacet distribution" (5.22 -> 6.18 rays/sample: the Schlick lobe lost energy, the image is 8% brighter; its code is faster, 4.03 -> 4.55 Mrays/s) and e137b7fcf "Path depth as in Arnold and Cycles" (6.22 -> 6.61 rays/sample, 5% brighter). Slower code: d6e008a0c "Layered glossy family: reciprocal albedo-scaled coats" (4.40 -> 3.75 Mrays/s, -15%, the image 5% darker), 1de435c9d "Energy compensation always on" (-3%), the commits between d6e008a0c and 4443f1574 (-4%) and between e137b7fcf and HEAD (-4%). c56bba156 costs nothing (the first single-run bisection had blamed it and 48c670678; a build ran in parallel). After 351123475 (the coating albedo lookups shared between the channels, +12%) HEAD runs at 3.92 Mrays/s against 4.10 for 2.11.2: about 4% slower code, the rest of the time is the longer paths. Shorter paths through the Russian roulette (2026-10-24, PATHOCL, scenes as authored, 3 seed pairs at 256 spp, variance x time against the default BOUNCE from depth 3 with cap 0.5): PATH from depth 2 with cap 0.25 is 2-13% better on the five native scenes but 4-12% worse on cornell, kitchen, cornell-glass and luxball-glossycoat; PATH or PATH_SQRT from depth 1 with a cap of 0.05-0.1 is 25-33% better on CannelleEtFromage, OrangeJuice and LynxSpider and up to 8% worse on LuxCore2.1Benchmark, and with the scenes' clamping it darkens OrangeJuice by 6-9%. No setting wins everywhere, so the default stays; the rule is unbiased (the means agree within 0.01% with the clamping off, except LynxSpider, see the transparent emitter entry), so `path.russianroulette.type/depth/cap` can be tuned per scene.

### LynxSpider: the Russian roulette depth changes the image by 1%
- Found: 2026-10-24 (in "Direct light sampling ignores the transparency of an emissive material", fixed 2026-10-25); split from it 2026-10-25
- Area: engine (Russian roulette, MIS?)
- Problem: `scenes/LynxSpider` with the clamping off, PATHOCL, half film, 256 spp: green mean 0.1088 with `path.russianroulette.depth = 100` and 0.1098 with `= 1` (+0.9%, 4 seeds each, seed spread 0.03%). The roulette is unbiased, so the image must not depend on its depth; every other native scene agrees within 0.01%.
- Reproduce: `tests/benchmark/native_render.py --scene LynxSpider --spp 256 --repeats 4 --set path.clamping.variance.maxvalue=0 --set path.russianroulette.depth=100` against `=1`.
- Notes: not the transparent emitters: the build before the opacity fix of the light sampling gives the same +1.0% (0.1087 against 0.1097), and the fix leaves the image unchanged within the noise (0.1088 against 0.1087 at depth 100). Ruled out before: the roulette's MIS factor (the same on the light and the BSDF sides), the sampler (an i.i.d. one gives the same; but `sampler.type = RANDOM` has been SOBOL since 979b505b4, 2026-10-16, so if that run used it, the sampler is not ruled out), the volumes. Suspects: the area light with `transparency.shadow = 1` (its direct hits through shadow-transparent surfaces are left to the light sampling), the spider's `mattetranslucent` and `mix` materials with transparency textures under a glossy coating, the Background card (opaque from the front, transparent from the back).
### HIPRT pinned archive: the old trace-kernel path misses hits (pre-G9 builds only)
- Found: 2026-10-24
- Area: build (HIPRT)
- Problem: the pinned HIPRT archive (PRECOMPILE + BAKE_COMPILED_KERNEL, `C:/Users/Ricardo/lc-hiprt`) with a build from before G9 step 4 (3f79c20ba, which builds its trace kernel through `hiprtBuildTraceKernels`) misses a regular pattern of hits on gfx1030 / Windows: the 320x240 cornell has 37% non-black pixels, mean 0.156 instead of 0.385, deterministic. The BAKE_KERNEL build of the same HIPRT commit (`out/hiprt-dist-backup/Release`) is fine, and HEAD (its own trace kernel) is fine with the archive.
- Reproduce: `out/venvb0` (af4f833ea) with `B0_HIPRT_DIR=C:\Users\Ricardo\lc-hiprt\dist\bin\Release`, `opencl.native.threads.count=0`, `C:\Users\Ricardo\lc-work\b0\render_b0.py 16 320 240`.
- Notes: found while building the pre-optimization wheel; HEAD is not affected. It matters only for old builds or a return to `hiprtBuildTraceKernels`. Ruled out: the work group size, the DataSet race fixed in c5dc429b2, stale caches.

### PhotonGI indirect cache: walls and ceiling 2-5% darker than path tracing
- Found: 2026-10-24
- Area: caches (PhotonGI)
- Problem: LuxCore2.1Benchmark (the scene's own PhotonGI settings, PATHOCL, half size, 1024 spp, clamping off) with the indirect cache against without: whole image -2.5%, walls -2.7%, ceiling -5.4%, floor -1%, a little warmer (R/B +0.3%). With the scene's clamping: -1.6%, -1.7%, -4.5%.
- Reproduce: `tests/benchmark/native_render.py --scene LuxCore2.1Benchmark --spp 1024 --set path.clamping.variance.maxvalue=0` against the same with `--set path.photongi.indirect.enabled=0`.
- Notes: The ceiling is lit only indirectly, so the cache's share is largest there. Suspects: the area estimate of a visibility particle (`PGICVisibilityParticle::ComputeRadiance()`, from the mean distance of its hits), the 10 degree normal angle and the radius blur at the panel mouldings, the stop at 15M photons with the error still at 11%. Cannelle with the cache: mean within 0.05%.
- Notes (2026-10-25, Cornell Box Luxcore and the SDL `cornell`, PATHCPU, deep paths): two estimator errors of opposite sign, measured with test switches in a worktree build (`LUX_PGI_AREA=disc`, `LUX_PGI_ALBEDO=albedo` in `C:\Users\Ricardo\lc-pgiarea`, not committed).
  1. The hit-based area `pi (2 * mean hit distance)^2` is not the area the photons are gathered over: each photon goes to every particle within `visibility.lookup.radius` r, so a flat surface needs `pi r^2`. It reads too small and depends on the viewing angle: with Lambertian walls and the emitter left out of the cache, the cache's own radiance (`debug.type = showindirect`) is +15-20% on the walls, floor and ceiling. With `pi r^2` they match path tracing (0.95-1.00), but the disc is cut at edges and corners: the SDL `cornell` goes from +5.5% to -1.2%, ceiling -5% (like LuxCore2.1Benchmark above). A fix needs the clipped disc area, not the plain one.
  2. `Material::EvaluateTotal()` (generic, 64 points, the fixed and sampled directions both from the same `u1, u2`) reads the EON `roughmatte` albedo 16% low at sigma 0.5 and 25% low at sigma 1 (0.574 against 0.680 for kd 0.70; `matte` has its exact fast path). That makes the cache darker and hides part of error 1. `Albedo()` (kd) is within 3% of the EON albedo.
  With both corrected (and the emitter photons fixed, see the commit that removed "PhotonGI indirect cache too bright around an emitter in a ceiling hole"), Cornell Box Luxcore with Lambertian walls reads 0.996-0.998 of path tracing. With the scene's own sigma 0.5 walls and neither corrected, the two cancel there: 1.002.

### The CPU light tree costs 11-16% more per sample than the DLS cache did
- Found: 2026-10-25
- Area: light (CPU engines)
- Problem: in the CA2 gate (PATHCPU, 256 spp, quarter size, two seeds), the light tree's time per sample was above the DLS cache's on the scenes with many lights: manylights (direct light only, depth 1) 17.64 against 15.27 ms/spp (+15.5%; 16.97 with the learning off, so the walk, not the learning, is the cost), the SDL Cornell box with a point, spot, sphere, sun and constant infinite light added to its mesh light 33.83 against 29.09 (+16%). Its noise per sample was equal or lower (manylights 0.632 against 0.636 var*spp, Cornell 0.132 against 0.149), so variance x time is 1.15x on manylights (setup excluded: the cache took 11.5 s to build, the tree 0.2 s) and 1.03x on the Cornell box. On the GPU the tree is faster per sample on every scene of the gate (1.8 against 1.9 ms/spp on manylights).
- Reproduce: `tests/benchmark/native_render.py --scene scenes/manylights/render.cfg --engine PATHCPU --device cpu --scale 0.25 --spp 256 --set lightstrategy.type=LIGHT_TREE`; the cache's figures need a build before its removal (`73337a80d`).
- Notes: not profiled. Suspects: the CPU tree walk of `LightTree_SampleLights()` (the `ocl::` functions compiled for the CPU, with the per-node importance of every child), the pdf walk of `SampleLightPdf()` at each emitter hit, and the shared learning sums written from every thread.
### Light tree with learning but no learned cells: a 7e6 firefly on manylights
- Found: 2026-10-25
- Area: light (light tree, GPU)
- Problem: RTPATHOCL before CA2 (`73337a80d`: the learning on by default, its sums accumulated, the cells never rebuilt, so every cell has no learned weight) on manylights, 30 s, seed 2: one pixel, (116, 305) of 480x480, reads 618 against ~0.2 around it (one sample of about 7e6), mean +1.3%, var*spp 245,000 against 4.5 with the learning off. The same pixel and value in every run (deterministic); seed 1 has no outlier, nor any run of 10 s (about 4000 spp).
- Reproduce: the CA2 scratch script `rt_learn.py` (RTPATHOCL, TILEPATHSAMPLER, the resolution reduction off, half size, `lightstrategy.lighttree.learning.enable = 1`, 30 s, seeds 1 and 2) with a build before CA2. TILEPATHOCL is in the same state today (it does not call `UpdateLightTreeLearning()` either).
- Notes: with the learning on, `LightTree_SampleLights()` picks a slot first from `LightTree_SlotBoundsProbabilities()` (the bounds with the `minImportance` mix), then walks inside the slot; with the learning off it walks the whole tree. So the two give different pick pdfs even with empty cells, and the slot step probably gives some light a tiny pick pdf at that point. After CA2, RTPATHOCL rebuilds the cells and three 30 s runs showed no outlier, but the cells only blend with the bounds probabilities, so the cause may still be there. PATHOCL is in the empty-cell state until its first epoch (a quarter sample per pixel). Not investigated further.
### Loading a scene with 18M triangle lights takes 8.5 s, starting it 5.4 s, deleting it 4 s
- Found: 2026-10-25
- Area: scene (light definition)
- Problem: `scenes/lightinstances` (4,501 instances of a 3,968-triangle mesh with an emissive material) takes 8.5 s in `Scene(file)`, 5.4 s in the PATHOCL `Start()` (3.5 s of it `LightSourceDefinitions::Preprocess()`, 0.7 s the light compile) and 4.2 s to delete (`del scene`), nearly all of it the triangle lights: one `TriangleLight` object per triangle of each instance, 17.9M of them, each a heap object with its own name (`TL<hash of the object name>_<triangle index>`, about 20 characters, more than the 15 of the MSVC small-string buffer) and an entry in the light definitions keyed by that name. Before the E1 work of 2026-10-25 it was 14.3 s, 16.8 s and 20.6 s.
- Reproduce: `pyluxcore.Scene("scenes/lightinstances/scene.scn")`, a PATHOCL session on it, then delete it.
- Notes: measured on 2026-10-25 with timers around the two halves of `DefineIntersectableLights()`: building the lights 6.3 s, inserting them 8.0 s, deleting the scene 21.8 s. Done since: the map's key is a `string_view` of the light's own name (one allocation, copy and free less per light; hashed as a std::string, so the same light order), the lights of an object are built in parallel (3.4 s; the heap allocations contend) and freed in parallel when there are more than 65,536 of them. Then the key keeps the hash of its name (the table hashed every key again at each growth, reading every name again; standalone, 17.9M names: 5.1 s of inserts, 2.8 s with the hash kept, the same table) and the names are hashed in parallel (`DefineLightSources()`); and in the engine start every pass over the lights runs in parallel, in the same order (the `Preprocess()` loop 5.5 s, the quad pairing 1.4 s, the three strategies 4.7 s, the zero fill of the 6 GB of `slg::ocl::LightSource` 0.7 s and their translation 3.3 s). Left: the inserts (a cache miss each in a table of 17.9M entries: reserving the final size halves them but moves 17,961 of the light indices, i.e. the iteration order, `Preprocess()`, and so the noise of every render with triangle lights), the 3.5 s left in `Preprocess()` (not split yet), the 6 GB of GPU light records (each `slg::ocl::LightSource` is ~348 bytes, the size of the union of every light type), the delete. The real fix is structural: the triangle lights of an object in one array of their own, without names (the names are only used by `DeleteLightSourceStartWith()`, `Scene::UpdateObjectTransformation()` and the light order), or instanced emitters sharing one light per mesh triangle with the instance transform. With direct light sampling on (the material's AUTO turns it off above 256 triangles) the light tree's own build is serial (`BuildNode()`), not measured. Not a native benchmark scene.
