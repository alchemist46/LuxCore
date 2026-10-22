# Known bugs

Bugs and issues found during a task that were not fixed right away. See "Keep the bug list" in [CLAUDE.md](CLAUDE.md) for the rules. Remove an entry when it is fixed.

The first entries were collected on 2026-10-20 from "Known open issues" in [MODERNIZATION_PLAN.md](docs/history/MODERNIZATION_PLAN.md) (the "Found" date is the date given there) and from [SPEEDUP_CHANGES.md](docs/history/SPEEDUP_CHANGES.md); the plan has the full measurements. Entries the plan already marks as fixed are not listed. The open work items are in [ROADMAP.md](ROADMAP.md).

## Entry format

```
### <short title>
- Found: YYYY-MM-DD
- Area: engine | material | texture | light | film | addon | build | tests | ...
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

### CPU/GPU gap of 2% on the working colour space test scene
- Found: 2026-10-15
- Area: engine, material
- Problem: PATHOCL renders 2.2% darker in red and 1.7% in green than PATHCPU.
- Reproduce: `wcs.py --scene full` (gold fresnel-preset metal2, Cauchy-fresnel metal2, dispersive glass with a thin film, image-map matte, sun, sky2, a point light with a temperature, emission with a temperature), 2 seeds each, 256 spp.
- Notes: the same in `rec709` and in the build before. A scene with only the glass, rough glass, Disney and sky2 agrees to 0.3%, so the gap is in one of the other elements. Not bisected yet. The Disney 0.1-0.2% gap was not re-measured after the Disney rebuild (the furnace agrees within 0.03%).

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
- Notes: a global luminance estimate (a reduction over the tasks) and a resampled start (PSSMLT bootstrap) would remove both. Only matters when the chains are short.

### GPU AOV problems: SHADING_NORMAL not deterministic, NaN in DIRECT_DIFFUSE_REFLECT
- Found: 2026-10-15 (G7 step 4 AOV check; the plan gives no exact date)
- Area: film (AOV), engine (OpenCL/HIP)
- Problem: on the GPU the SHADING_NORMAL output is not deterministic: the sample that wins the depth test overwrites the pixel, and with depth ties (pass-through surfaces, volumes) the winner depends on the timing (2 to 10 pixels of transparencyball at 160x120 differ between two runs of the same build). The same check shows a NaN in DIRECT_DIFFUSE_REFLECT on the Cornell box (TILEPATHOCL, both builds), not looked into.
- Reproduce: transparencyball at 160x120, SHADING_NORMAL AOV, two runs; Cornell box with TILEPATHOCL and DIRECT_DIFFUSE_REFLECT.
- Notes: the NaN has not been investigated.

### BiDir darkens the silhouette of the furnace sphere
- Found: 2026-10-22
- Area: engine (BIDIRCPU)
- Problem: in the white furnace the pixels on the outline of the sphere read 0.6-0.7 with BIDIRCPU, whatever the material (matte too), and the image mean is 0.973 against 0.9999 with PATHCPU; the inner disk is right (0.999). The "mu < .2" band of `tests/furnace/furnace.py` shows it (0.40 for matte).
- Reproduce: a white matte sphere under a constant white environment, orthographic camera, 48x48, no pixel filter, BIDIRCPU 512 spp: the first and the last pixel on the sphere of the centre row and column read 0.6.
- Notes: in the build before the camera row fix (b22905df5) the dark ring was one row off (row 0 on top, the third last row at the bottom); now it is symmetric. Suspects: the light-traced connections to the camera at grazing angles. Not investigated.

### TILEPATHOCL may wait long for the subsurface walks
- Found: 2026-10-22
- Area: engine (TILEPATHOCL)
- Problem (suspected, not measured): a random walk now goes on past 256 bounces with a Russian roulette, up to 65536 (`subsurface_funcs.cl`), so in a dense, nearly white medium the longest walk of a tile pass can take thousands of iterations, and TILEPATHOCL runs the kernels until every task of the pass is done. The PATHOCL subsurface halt bias (above) may grow as well.
- Reproduce: a white subsurface sphere with a radius of 0.01 of its size, TILEPATHOCL: the time per pass against the build before.
- Notes: a walk reaches n bounces with the probability 256 / n past 256. The hard limit of 256 lost 13% of a white sphere of radius 100 mean free paths, 4096 lost 2%. Not measured on the GPU (it was down).

### One pixel of luxball-hdr-vol differs between TILEPATHOCL runs at a fixed seed
- Found: 2026-10-22
- Area: engine
- Problem: two runs of the same build should differ by at most 1e-6 per value (the atomic adds of the film), but pixel (x 143, y 101) of luxball-hdr-vol comes out with one of two values that differ by 1.5e-2 (relative); the rest of the image is identical.
- Reproduce: `tests/benchmark/parity.py render <dir> luxball-hdr-vol` (TILEPATHOCL, seed 1, aa 8, 320x240, HIP device) six times with the same build, then `compare`: 2 of 6 runs at HEAD `aa2fa98c1` had the other value.
- Notes: always the same pixel and the same two values, so one sample takes one of two discrete outcomes depending on timing, not accumulation order. Suspects: something read across tasks or threads in the volume path (the null-shell boundary init of V4, the pass-through of V1, the ray-hit buffers of the volume continuation). It can make a bit-identical check of `parity.py` fail on this scene: re-run it before looking for a change.
