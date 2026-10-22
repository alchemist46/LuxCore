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
- Notes: the frames look overwritten (float-like values in saved pointers). Not a stop-time use-after-free. With `opencl.native.threads.count = 0` 0 of 10 runs crashed, so the suspect is the native CPU threads that PATHOCL runs beside the GPU (Metropolis sampler); the GPU kernels are not involved. A `-fstack-protector-all` build passed 8 runs. Next step: a LuxCore-only ASan build (needs `libasan`/`libubsan` on the development machine).

### CPU/GPU gap of 2% on the working colour space test scene
- Found: 2026-10-15
- Area: engine, material
- Problem: PATHOCL renders 2.2% darker in red and 1.7% in green than PATHCPU.
- Reproduce: `wcs.py --scene full` (gold fresnel-preset metal2, Cauchy-fresnel metal2, dispersive glass with a thin film, image-map matte, sun, sky2, a point light with a temperature, emission with a temperature), 2 seeds each, 256 spp.
- Notes: the same in `rec709` and in the build before. A scene with only the glass, rough glass, Disney and sky2 agrees to 0.3%, so the gap is in one of the other elements. Not bisected yet. The Disney 0.1-0.2% gap was not re-measured after the Disney rebuild (the furnace agrees within 0.03%).

### Dark values of byte image maps are quantized after decoding
- Found: 2026-10-17
- Area: texture
- Problem: byte image maps decoded by a gamma or an OCIO transform are stored back in bytes, already linear, so the dark values are quantized (sRGB 30/255 becomes 3/255 = 0.0118 instead of 0.013). Banding in dark gradients, up to 10% off Cycles on small channels.
- Reproduce: render a dark sRGB byte texture and compare with Cycles.
- Notes: the gamma path always did it. Storing decoded byte images as half floats would fix it at twice the memory.

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

### Dispersion draws one wavelength per vertex
- Found: 2026-10-19
- Area: material
- Problem: a photon keeps its wavelength along the path; here every vertex draws its own, so a vertex is the average of its wavelengths. Exact for one interface, not along a chain of interfaces. Smooth dispersive glass passes the furnace (0.997); OpenPBR rough glass at roughness 0.3 reads 1.5-2% low near the total internal reflection. The glass material does the same (its -1.3%).
- Reproduce: furnace with dispersive rough glass at roughness 0.3.
- Notes: tinting the reflection too multiplies independent colours (relative std 0.07 -> 0.50). The fix is a wavelength per path: a sampler dimension kept in the path state of every engine and passed to the materials through the hit point, applied once at the first dispersive vertex.

### OpenPBR subsurface is up to 10% brighter than Cycles
- Found: 2026-10-19
- Area: material
- Problem: skin-like color (0.8, 0.5, 0.4), radius (1, 0.2, 0.1) x 0.2 reads 1.02/1.10/1.10 of Cycles' Principled BSDF. Without the specular layer 1.02-1.05; a white medium 1.003.
- Reproduce: `tests/test_openpbr_principled.py`-style comparison against Cycles.
- Notes: Cycles' walk exits through the rough dielectric (Fresnel, some light goes back into the medium), OpenPBR's exits diffusely.

### OpenPBR thin film differs from Cycles
- Found: 2026-10-19
- Area: material
- Problem: the thin film is `CalcFilmColor()`, LuxCore's old model, not Cycles' (Belcour & Barla 2017, with the conductor phase shift). 400 nm on a dark dielectric reads 1.22/0.87/0.89 of Cycles; on a metal the interference color is another one (magenta against yellow-green).
- Reproduce: OpenPBR thin film at 400 nm, compare with Cycles.
- Notes: implementing the Belcour & Barla model would fix it.

### Caustics seen through a specular surface are missing with light tracing
- Found: 2026-10-16
- Area: engine (light tracing)
- Problem: a floor caustic reflected in a mirror or seen through a window is produced by neither side: the eye path drops it as caustic and the light tracer connects only directly to the lens. Missing energy with light tracing on, unless PhotonGI's caustic cache is on too.
- Reproduce: caustic on a floor, seen in a mirror, with `path.hybridbackforward.enable`.
- Notes: MNEE covers part of it; the rest is a bidirectional connection, out of scope for the hybrid.

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

### Cloth (Irawan) gains energy at grazing angles
- Found: 2026-10-21
- Area: material
- Problem: the specular part of the cloth material is not energy conserving: with the diffuse colours at 0 and the specular colours at 1, the white furnace gives 0.73 at normal incidence and 1.23-1.32 at grazing view angles (mu < 0.4): it reflects more light than it receives there. The default denim preset: 0.87 at normal incidence, 1.12-1.15 at grazing.
- Reproduce: `tests/furnace/furnace.py PATHCPU cloth-denim "scene.materials.m.weft_kd=0 0 0" "scene.materials.m.warp_kd=0 0 0" "scene.materials.m.weft_ks=1 1 1" "scene.materials.m.warp_ks=1 1 1"` (PATHOCL the same: disk 0.865 against 0.862).
- Notes: the Irawan-Marschner weave model is a fit, not a conserving BSDF; one scale cannot fix it (too dark head-on, too bright at grazing). A per-angle albedo normalization (a table of the directional albedo per preset, as the energy compensation of the microfacet materials) would. The diffuse part alone passes (cloth-diffuse 0.9999).

### BiDir darkens the silhouette of the furnace sphere
- Found: 2026-10-22
- Area: engine (BIDIRCPU)
- Problem: in the white furnace the pixels on the outline of the sphere read 0.6-0.7 with BIDIRCPU, whatever the material (matte too), and the image mean is 0.973 against 0.9999 with PATHCPU; the inner disk is right (0.999). The "mu < .2" band of `tests/furnace/furnace.py` shows it (0.40 for matte).
- Reproduce: a white matte sphere under a constant white environment, orthographic camera, 48x48, no pixel filter, BIDIRCPU 512 spp: the first and the last pixel on the sphere of the centre row and column read 0.6.
- Notes: in the build before the camera row fix (b22905df5) the dark ring was one row off (row 0 on top, the third last row at the bottom); now it is symmetric. Suspects: the light-traced connections to the camera at grazing angles. Not investigated.
