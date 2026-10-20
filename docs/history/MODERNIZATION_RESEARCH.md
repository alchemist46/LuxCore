# LuxCore modernization: plan review and research survey

> Companion of the plans now superseded by [ROADMAP.md](../../ROADMAP.md) (2026-10-21); kept for the papers and engine comparisons.

Date: 2026-10-14. Companion to `MODERNIZATION_PLAN.md`. Sources were checked against the engines' own docs, source trees and the papers; items that could only be found second-hand are marked *(unverified)*. Code facts below were checked in this tree (branch `hip-backend`).

## 1. Review of the plan

The plan's diagnosis matches the code, the Phase 1b measurements are sound, and the order of leverage (single-source kernels, then sampling and light transport, then interop) is the same one Cycles followed from 2.8 to 4.x. The points below are corrections, refinements and gaps found while checking it against the current state of the art.

### 1.1 Corrections to statements in the plan

- **The microfacet lobes are not GGX.** `metal2`, `roughglass`, `carpaint` and the glossy coating helpers (`SchlickBSDF_*` used by `glossy2`, `glossycoating`, `glossytranslucent`) use the Schlick 1994 distribution (`SchlickDistribution_SchlickZ` times `SchlickDistribution_SchlickA`, `src/slg/materials/material.cpp:430`) and sample the half vector from `D(h)`, not from the visible normals. Only Disney uses GGX. This matters for the Kulla-Conty item: the published albedo tables and every engine's energy compensation assume GGX plus Smith. *(Checked while implementing it: the isotropic Schlick Z term is in fact the GGX NDF with alpha^2 = u * v, and its sampling is the GGX NDF sampling; only the anisotropy term, the masking term and the reported pdf differ. See the Phase 1 GGX item of the plan for the pdf bug this uncovered.)* The practical route is a GGX switch for those materials (with a compatibility property to keep the old look) together with visible-normal sampling, then the compensation. See 2.4.
- **GPU OIDN is a packaging problem, not a library gap.** LuxCoreDeps 2.2.4 pins OIDN 2.5.0 (CUDA, HIP, SYCL and Metal device modules exist since 2.0/2.1; 2.5 adds external semaphore import on CUDA and HIP), Embree 4.4.1, OpenVDB 12.1.1 (NanoVDB included), OpenImageIO 3.1.14 (TextureSystem, `.tx`, UDIM), OpenSubdiv 3.7.0 and OpenColorIO 2.5.2. So NanoVDB, the texture cache and GPU OIDN need no new dependency; the deps build only has to ship `libOpenImageDenoise_device_{cuda,hip}` and `intel_oidn.cpp:171` has to create a device matching the render device and pass device buffers.
- **Textures already read `.tx` mip levels, but pick one level at load.** `imagemap.cpp:918` lists the sub-images and keeps a single level chosen from a size hint; there is no per-lookup level selection or filtering. So "mip streaming" is two separate items: per-lookup LOD (cheap, see 2.7) and out-of-core streaming (expensive, CPU-only via OIIO).
- **"Fused trace" and "material sorting" are smaller wins than the plan's wording suggests.** Cycles' shader sorting is worth 2-3% in their tracker and is turned off above ~300 shaders; compaction is "not clearly a win" when only one or two kernels run per bounce; a 2026 Vulkan study found wavefront ~16% faster than a megakernel and both latency-bound. The plan's guard (rocprof before G6) is the right one. Shader Execution Reordering is NVIDIA Ada+ only, reachable only from OptiX programs, and its gain shrinks with local memory use, so it is a late option for the OptiX path only.
- **ReSTIR has no offline adopter.** No evidence that Cycles, Karma, Arnold, V-Ray or Vantage use it; it lives in Falcor/RTXDI and Omniverse's real-time mode. The only offline-oriented variant (conditional ReSTIR, Kettunen 2023) is research. Recommend removing it from Phase 2 and keeping it as a note for the RTPATH viewport only.
- **OpenPGL risk changed.** Intel discontinued it in August 2025; the ASWF took it over in April 2026 (v0.7.1, July 2025, is the last release). It has no GPU device (`Device` means SIMD width). Still the right choice for the CPU engines, as in Cycles, V-Ray 6.1 and Karma, but plan for maintaining a fork and for an in-house GPU guiding structure.
- **OSL is a dead end for the HIP backend.** OSL 1.15/1.16 has OptiX/CUDA only; no HIP, Metal or Vulkan backend exists or is announced. Drop "no OSL" from the gap list; the realistic route is a fixed OpenPBR node fed from MaterialX (see 2.8).

### 1.2 Refinements to planned items

- **Light tree (Phase 2).** Cycles' implementation (`intern/cycles/kernel/light/tree.h`) differs from Conty-Kulla 2018 in ways that matter for the GPU: no splitting (one light per shading point); the selection probability is the average of the probabilities computed with the node's maximum and minimum possible contribution, which keeps the pdf away from zero; the pdf for a BSDF hit is recomputed by re-walking root to leaf with a per-emitter bit trail (plus a top-level tree for instanced emissive meshes); distant lights are a separate leaf group; the environment light is a tree node with its own importance. pbrt-v4's `BVHLightSampler` is the same design and runs in its wavefront GPU path. Port this design rather than the paper's. The DLSC can later become a per-cell cut of the tree. Tokuyoshi et al. 2024 (AMD) is a drop-in upgrade of the node importance that includes the BRDF.
- **Adaptive sampling (Phase 1).** The plan's step 2 is Cycles' scheme; three details to copy: the odd/even split uses `popcount(sample & 0xaaaaaaaa) & 1` (class A/B from Christensen 2018 §10.2.1, which the Owen-scrambled Sobol already supports); the error is `|full - half| / max(sqrt(I), I)` on linear radiance; converged pixels get their accumulated value rescaled as if fully sampled so the denoiser needs no spp map. For a halt criterion on the *denoised* image, Firmino et al. 2024 is the model (per-pixel error estimate of the denoised result); Firmino 2023 needs a derivative through the denoiser that OIDN does not expose.
- **Blue noise (Phase 1).** Cycles 4.2's implementation is ~50 lines on top of Burley scrambling: one global sequence, each pixel takes the chunk at index `nested_uniform_scramble_base4(morton2d(x, y), seed)`. Its `BLUE_NOISE_FIRST` mode (sample 0 from a separate 1-spp blue-noise sequence, then chunks) is the answer to the plan's adaptive-sampling caveat and also improves the 1-spp viewport. The Chizhov 2022 caveat applies: kernel-predicting denoisers are trained on white noise; measure RMSE after OIDN as the plan says.
- **GPU track.** Cycles' state is strictly SoA with one buffer per member and a "compact integrator state" rule. After G2, the next VGPR/scratch lever is shrinking the 1.8 KB per-task state and splitting the big state kernels, not hardware features (RDNA4 dynamic VGPRs do not help RDNA2). HIPRT 3.0.3 re-enabled hardware intersection on RDNA2 (RTIP 1.1 fix), so check the bundled `libhiprt0300164.so` is at least 3.0.3. HIPRT has no curve primitive (triangles and AABB lists only), so hair on HIP stays tessellated or custom-intersected.
- **Volumes (Phase 2).** Blender 5.0 made null-scattering the default and removed the step-size settings; the biased ray marcher is kept as an option. Mitsuba 3 `volpath` (delta tracking with a global majorant, ratio tracking for transmittance) is the compact reference. LuxCore's interior volumes are already a random walk, so random-walk SSS is a material wrapper (dielectric entry, albedo-to-coefficients inversion, Dwivedi guiding) rather than new transport.
- **XPU (Phase 4).** RenderMan XPU is now described in a paper (Christensen, Fong et al., HPG 2025): most code shared, materials and transport compiled from one C++ source for CPU and CUDA, low-level tracing and texture caching per platform, OSL through LLVM on both. Karma XPU and V-Ray hybrid both state device-independent results but not bit-exactness; plan to tolerate float differences (FMA, transcendentals) rather than require them identical. V-Ray reports 13-25% from adding the CPU.
- **OpenPBR (Phase 3).** v1.1.1 (April 2026) is the current spec; Blender's dedicated OpenPBR node targets Blender 5.4 and has a cross-renderer test suite that found coat, thin film and fuzz inconsistent across engines. Implementing OpenPBR as a new material (slabs with albedo scaling, EON diffuse, F82 metal, the existing thin film) with Adobe's `openpbr-bsdf` as the correctness reference, and timing it with the Blender node, is the single best adoption lever for BlendLuxCore.
- **Hydra (Phase 3).** Hydra 2.0 (scene index) is the default since OpenUSD 25.11 and scene delegates are deprecated at runtime from 26.03; write the delegate against the scene-index API. hdCycles (`intern/cycles/hydra`, 46 files) is the closest template since it wraps a standalone session API like LuxCore's.

### 1.3 Gaps: items missing from the plan

Ordered by gain per unit of work. The first four are each a few days and fit Phase 1.

1. **MIS compensation on the environment and sky importance maps** (Karlík 2019). A second CDF with the cosine-weighted average subtracted, used only by the light-sampling technique; pbrt-v4 and Mitsuba 3 ship it. Free noise reduction on every HDRI-lit scene.
2. **GGX with visible-normal sampling for the Schlick-distribution materials** (Heitz 2018; Dupuy and Benyoub 2023 spherical caps; Eto and Tokuyoshi 2023 bounded VNDF for reflection). Fewer wasted samples, lower variance on rough metals and glass, and the prerequisite for item 3.
3. **Energy compensation** is in the plan; add the **EON rough diffuse** (Portsmouth 2025, used by OpenPBR and Cycles 4.2) so `roughmatte` also passes the furnace test.
4. **Ray-cone texture LOD with per-lookup mip selection** (Akenine-Möller et al. 2021) plus **stochastic trilinear filtering** (Pharr et al. 2024). One float per path, identical CPU and GPU code, one texel fetch per lookup, fixes the load-time single-level choice and cuts GPU texture bandwidth. UDIM is a loader-side change on top of it.
5. **Manifold next-event estimation for shadow caustics** (Hanika 2015; Cycles 3.1 "shadow caustics"). Needs no cache, works in the wavefront kernels, fixes the common caustic-under-glass case without PhotonGI.
6. **Dwivedi-guided random walk** for SSS (Meng 2016, Cycles 2.93) when random-walk SSS is added.
7. **A denoised-image halt** (Firmino 2024) once the Cycles-style adaptive sampling is in.
8. **Hair BSDF choice:** Chiang 2016 (pbrt-v4 reference, MaterialX `chiang_hair_bsdf`) rather than Huang 2022, which Cycles ships but with known energy loss at high roughness.

### 1.4 Suggested re-ordering

Phase 1 quality items, in order: adaptive sampling step 1 and 2, blue-noise pixel ordering, MIS compensation, GGX/VNDF switch, Kulla-Conty plus EON, ray-cone LOD. Phase 2: light tree, null-scattering volumes with NanoVDB, OpenPGL on CPU, MNEE, random-walk SSS. Phase 3 unchanged except: OpenPBR first (aligned with Blender 5.4), OSL removed, Hydra 2.0 explicitly. Phase 4 unchanged, with "device-independent within tolerance" instead of "identical".

## 2. Research survey by topic

Each entry: idea in one line, who ships it, cost for LuxCore.

### 2.1 Sampling

| Paper | Idea | Shipped in |
|---|---|---|
| Burley, *Practical Hash-based Owen Scrambling*, JCGT 2020. https://jcgt.org/published/0009/04/01/ | Hash-based nested uniform scrambling of Sobol. | Cycles, pbrt-v4, LuxCore (done). |
| Ahmed, Wonka, *Screen-Space Blue-Noise Diffusion of Monte Carlo Sampling Error via Hierarchical Ordering of Pixels*, TOG 39(6) 2020. https://doi.org/10.1145/3414685.3417881 | Pixels along a hierarchically scrambled Z-order take consecutive chunks of one sequence; error becomes blue noise. | pbrt-v4 `ZSobolSampler`, Cycles 4.2 (PR #118479). |
| Christensen, Kensler, Kilpatrick, *Progressive Multi-Jittered Sample Sequences*, EGSR 2018. https://graphics.pixar.com/library/ProgressiveMultiJitteredSampling | pmj02/pmj02bn; the class A/B sample split used for half buffers. | RenderMan tables, Cycles up to 3.4, pbrt-v4. |
| Helmer, Christensen, Kensler, *Stochastic Generation of (t,s) Sample Sequences*, EGSR 2021. https://diglib.eg.org/items/cd326c49-1f97-437b-8280-f989181e52e4/full | Owen-scrambled Sobol generated as stratified samples in <30 lines; pmj02 and Owen-Sobol are one family. | Reference. |
| Heitz, Belcour et al., *A Low-Discrepancy Sampler that Distributes Monte Carlo Errors as a Blue Noise in Screen Space*, SIGGRAPH Talks 2019; Belcour, Heitz, *Lessons Learned...*, 2021. https://arxiv.org/abs/2105.12620 | Optimized per-pixel scramble tables. A LuxCore forum test in 2019 saw little effect. | Unity. Superseded by Ahmed-Wonka for path tracers. |
| Chizhov et al., *Perceptual Error Optimization for Monte Carlo Rendering*, TOG 2022. https://arxiv.org/abs/2012.02344 | Caveat: denoisers trained on white noise may over-smooth blue-noise error. | Caveat only. |

Cycles 4.x default is "tabulated Sobol" (pretabulated 4D Owen-scrambled Sobol, padded in 4D blocks) with the blue-noise ordering on top. Mitsuba 3 and Falcor have no screen-space blue noise. Arnold's own work is Georgiev and Fajardo, *Blue-noise Dithered Sampling*, SIGGRAPH Talks 2016 *(whether it ships is unverified)*.

### 2.2 Adaptive sampling and stopping

| Paper | Idea | Shipped in |
|---|---|---|
| Dammertz, Hanika, Keller, Lensch, *A Hierarchical Automatic Stopping Condition for Monte Carlo Global Illumination*, WSCG 2010. | Two independent sample sets per block; split and stop hierarchically. | Cycles, MoonRay (Jeffery, SIGGRAPH Talks 2019). |
| Cycles `kernel/film/adaptive_sampling.h` (design notes in commit 29afadcb). | Odd/even half buffer, `error = |I-A| / max(sqrt I, I)`, converged flag, neighbour dilation, rescale after convergence. | Cycles. |
| Rousselle, Knaus, Zwicker, *Adaptive Sampling and Reconstruction using Greedy Error Minimization*, SIGGRAPH Asia 2011. | Error from two half images drives both filter choice and sample allocation. | Reference. |
| Firmino, Frisvad, Jensen, *Denoising-Aware Adaptive Sampling for Monte Carlo Ray Tracing*, SIGGRAPH 2023. https://doi.org/10.1145/3588432.3591537 | Variance of the denoiser output via a JVP through the network, demonstrated with OIDN. Needs the network's derivative. | Research. |
| Firmino, Ramamoorthi, Frisvad, Jensen, *Practical Error Estimation for Denoised Monte Carlo Image Synthesis*, SIGGRAPH 2024. | Per-pixel squared-error estimate of the denoised image as a stopping criterion. | Research; the right model for a denoised halt. |
| Denisova, Bocchi, *Converging Algorithm-Agnostic Denoising for Monte Carlo Rendering*, HPG 2024. https://doi.org/10.1145/3675384 | Black-box denoiser, blend weights learned during progressive rendering, adaptive sampling from the denoised variance. | Research; lower-cost alternative. |

Arnold stops at an estimated-noise threshold per pixel between min and max AA samples; RenderMan 25/26 uses a "variance" metric on gamma-weighted differences with a minimum sample count; Karma's "pixel oracle" compares each pixel with its neighbours (XPU since 20.5).

### 2.3 Many lights

| Paper | Idea | Shipped in |
|---|---|---|
| Conty Estevez, Kulla, *Importance Sampling of Many Lights with Adaptive Tree Splitting*, HPG 2018. https://fpsunflower.github.io/ckulla/data/many-lights-hpg2018.pdf | Light BVH with energy, orientation bounds and distance; stochastic descent; splitting. | Imageworks Arnold; Cycles 3.5 and pbrt-v4 without splitting. |
| Cycles `intern/cycles/kernel/light/tree.h`; pbrt-v4 ch. 12.6. https://pbr-book.org/4ed/Light_Sources/Light_Sampling.html | Min/max-averaged selection pdf, bit-trail pdf re-walk, distant group, environment as a node, instance tree. | Cycles, pbrt-v4. |
| Yuksel, *Stochastic Lightcuts*, HPG 2019; Lin, Yuksel, *Real-Time Stochastic Lightcuts*, I3D 2020. | One light per cut node chosen stochastically; GPU build. | Research. |
| Moreau, Pharr, Clarberg, *Dynamic Many-Light Sampling for Real-Time Ray Tracing*, HPG 2019. | Two-level light BVH refit per frame. | Falcor. |
| Tokuyoshi, Ikeda, Kulkarni, Harada, *Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting*, SIGGRAPH Asia 2024. https://gpuopen.com/download/publications/Hierarchical_Light_Sampling_with_Accurate_Spherical_Gaussian_Lighting.pdf | Node importance with a spherical-Gaussian BRDF term; better on glossy highlights. | Research (AMD); contained upgrade. |
| Figueiredo, He, Bako, Kalantari, *Neural Importance Sampling of Many Lights*, SIGGRAPH 2025. https://arxiv.org/abs/2505.11729 | Small network predicts a visibility-aware light distribution per point. | Research. |
| Wang et al., *Learning to Cluster for Rendering with Many Lights*, SIGGRAPH Asia 2021. | Online adaptation of clustering and per-cluster distribution. | Research. |

Karma uses a light tree with dome, distant, point and IES lights excluded (sampled uniformly). Autodesk Arnold's and RenderMan's schemes are not documented publicly.

### 2.4 Materials

| Paper / spec | Idea | Shipped in |
|---|---|---|
| OpenPBR Surface v1.1.1, 2026. https://academysoftwarefoundation.github.io/OpenPBR/ ; Portsmouth, Kutz, Hill, *OpenPBR: Novel Features and Implementation Details*, 2025. https://arxiv.org/abs/2512.23696 | Slab layering with albedo scaling, EON diffuse, F82 metal, coat, fuzz, thin film. Reference: MaterialX graph in the spec repo; Adobe `openpbr-bsdf` C++ library. | Arnold, Karma 21, V-Ray 7, Omniverse, Redshift; Blender 5.4 node in testing. |
| Kulla, Conty, *Revisiting Physically Based Shading at Imageworks*, SIGGRAPH 2017 course. https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_slides_v2.pdf | Tabulated albedo E(mu, alpha), compensating lobe (1-E)/(1-Eavg). | Cycles 4.0 (replaced stochastic MultiGGX, PR #107958), OpenPBR recommendation. |
| Turquin, *Practical Multiple Scattering Compensation for Microfacet Models*, 2019. https://blog.selfshadow.com/publications/turquin/ms_comp_final.pdf | Scale the lobe by 1/E(mu) with a Fresnel tint. Simplest form. | ILM and many production shaders. |
| Sforza, Pellacini, *Enforcing Energy Preservation in Microfacet Models*, STAG 2022. | Analytic fits of E, no tables; GPU-friendly. | Research. |
| Portsmouth, Kutz, Hill, *EON: A Practical Energy-Preserving Rough Diffuse BRDF*, JCGT 2025. https://arxiv.org/abs/2410.18026 | Energy-preserving Oren-Nayar. | OpenPBR, Cycles 4.2 diffuse roughness. |
| Dupuy, Benyoub, *Sampling Visible GGX Normals with Spherical Caps*, HPG 2023. https://arxiv.org/abs/2306.05044 ; Eto, Tokuyoshi, *Bounded VNDF Sampling for the Smith-GGX BRDF*, SIGGRAPH Asia 2023. https://gpuopen.com/download/publications/Bounded_VNDF_Sampling_for_the_Smith-GGX_BRDF.pdf | Cheaper VNDF sampling; bounded variant rejects fewer samples for reflection. | Cycles PR #109757 still open. |
| Zeltner, Burley, Chiang, *Practical Multiple-Scattering Sheen Using Linearly Transformed Cosines*, SIGGRAPH Talks 2022. | Sheen as an LTC fit. | Cycles 4.0 "microfiber" sheen, OpenPBR fuzz. |
| Guo, Hasan, Zhao, *Position-Free Monte Carlo Simulation for Arbitrary Layered BSDFs*, TOG 2018; Belcour, *Efficient Rendering of Layered Materials...*, SIGGRAPH 2018. | Stochastic and analytic layering. | pbrt-v4 `LayeredBxDF`. Not worth the GPU register cost here. |
| Burley, *Extending the Disney BRDF to a BSDF with Integrated Subsurface Scattering*, SIGGRAPH 2015 course. | Disney 2015. | RenderMan PxrDisneyBsdf; superseded by OpenPBR as an interchange target. |

### 2.5 Volumes and subsurface

| Paper | Idea | Shipped in |
|---|---|---|
| Novák, Georgiev, Hanika, Jarosz, *Monte Carlo Methods for Volumetric Light Transport Simulation*, EG STAR 2018. | The reference survey: delta, ratio, residual tracking, majorant grids. | Reference. |
| Kutz, Habel, Li, Novák, *Spectral and Decomposition Tracking for Rendering Heterogeneous Volumes*, SIGGRAPH 2017. https://www.jannovak.info/publications/SDTracking/ | Spectral null-collision tracking; decomposition into homogeneous plus residual. | Hyperion. |
| Miller, Georgiev, Jarosz, *A Null-Scattering Path Integral Formulation of Light Transport*, SIGGRAPH 2019. | Null-collision tracking as path sampling with pdfs, so MIS works in volumes. | Mitsuba 3, Cycles 5.0. |
| Kettunen, d'Eon, Pantaleoni, Novák, *An Unbiased Ray-Marching Transmittance Estimator*, SIGGRAPH 2021. https://research.nvidia.com/publication/2021-06_An-Unbiased-Ray-Marching | Unbiased power-series ray marching, zero variance for constant extinction. | Research. |
| Misso et al., *Progressive Null-Tracking for Volumetric Rendering*, SIGGRAPH 2023. | Progressively tightens bad majorants. | Research. |
| Herholz et al., *Volume Path Guiding Based on Zero-Variance Random Walk Theory*, TOG 2019; Xu et al., *Volume Scattering Probability Guiding*, SIGGRAPH Asia 2024. | Guided scattering in volumes; the latter is in OpenPGL. | OpenPGL. |
| Christensen, Burley, *Approximate Reflectance Profiles for Efficient Subsurface Scattering*, 2015. https://graphics.pixar.com/library/ApproxBSSRDF | Two-exponential profile; also the albedo inversion used by random walks. | RenderMan, Cycles SSS node. |
| Wrenninge, Villemin, Hery, *Path Traced Subsurface Scattering using Anisotropic Phase Functions and Non-Exponential Free Flights*, 2017. https://graphics.pixar.com/library/PathTracedSubsurface/paper.pdf | Random-walk SSS with non-exponential flights. | RenderMan. |
| Křivánek, d'Eon 2014; Meng, Hanika, Dachsbacher, *Improving the Dwivedi Sampling Scheme*, EGSR 2016. https://cg.ivd.kit.edu/1951.php | Zero-variance-based direction guiding for dense media, with MIS. | Cycles 2.93 random walk. |

Blender 5.0: null-scattering default, NanoVDB on CUDA, OptiX, HIP and Metal, stochastic grid interpolation, multi-bounce random walk. Arnold is still a ray marcher with `step_size` (7.5.1). OpenVDB 12.1 adds a NanoVDB streaming block manager; 13.0 adds GPU topology ops and half grids.

### 2.6 Path guiding, MIS, caustics

| Paper | Idea | Shipped in |
|---|---|---|
| Müller, Gross, Novák, *Practical Path Guiding for Efficient Light-Transport Simulation*, EGSR 2017; SIGGRAPH 2019 course *Path Guiding in Production*. https://cgg.mff.cuni.cz/~jaroslav/papers/2019-path-guiding-course/ | Spatial tree of directional quadtrees learned online. The quadtree is the kernel-friendly structure for a GPU port. | Research; the course covers Weta and Disney practice. |
| Ruppert, Herholz, Lensch, *Robust Fitting of Parallax-Aware Mixtures for Path Guiding*, SIGGRAPH 2020. | VMM mixtures with parallax; OpenPGL's core. | OpenPGL: Cycles 3.4 (CPU), V-Ray 6.1 (CPU), Karma, Hyperion. |
| Huang et al., *Online Neural Path Guiding with Normalized Anisotropic Spherical Gaussians*, TOG 2024; Figueiredo, He, Kalantari, *Neural Path Guiding with Distribution Factorization*, EGSR 2025. | Small online MLPs predicting mixtures or factorized pdfs. | Research. |
| Yalçıner, Akyüz, *Path Guiding for Wavefront Path Tracing*, Computers & Graphics 2024. https://arxiv.org/abs/2405.06997 | Guiding structure sized for wavefront memory limits. | Research; relevant to PATHOCL. |
| Karlík, Šik, Vévoda, Skřivan, Křivánek, *MIS Compensation: Optimizing Sampling Techniques in Multiple Importance Sampling*, SIGGRAPH Asia 2019. https://cgg.mff.cuni.cz/~jaroslav/papers/2019-mis-compensation/ | Subtract the BSDF-covered part from the env-map pdf. Unbiased, no runtime cost. | pbrt-v4 `ImageInfiniteLight`, Mitsuba 3 `envmap`. |
| Kondapaneni et al., *Optimal Multiple Importance Sampling*, SIGGRAPH 2019. | Variance-optimal MIS weights via a linear system. | No engine; skip. |
| Hanika, Droske, Fascione, *Manifold Next Event Estimation*, EGSR 2015. | Newton-walk a shadow connection through refractive interfaces. | Cycles 3.1 shadow caustics (up to 4 bounces, opt-in per light, caster and receiver). |
| Zeltner et al., *Specular Manifold Sampling*, SIGGRAPH 2020; Fan et al., *Specular Polynomials*, SIGGRAPH 2024. | Unbiased manifold sampling; polynomial root finding for one-bounce chains. | Research. |

ReSTIR (Bitterli 2020; Ouyang 2021; Lin 2022 GRIS; Kettunen 2023 conditional; Lin, Kettunen, Wyman, *ReSTIR PT Enhanced*, 2026) is a correlated low-spp real-time technique with no offline adopter; the SIGGRAPH 2023 course (Wyman et al.) is the entry point if the RTPATH viewport ever wants it.

### 2.7 Textures

| Paper | Idea | Shipped in |
|---|---|---|
| Akenine-Möller et al., *Texture Level of Detail Strategies for Real-Time Ray Tracing*, Ray Tracing Gems 2019; *Improved Shader and Texture Level of Detail Using Ray Cones*, JCGT 2021. https://jcgt.org/published/0010/01/01/paper.pdf | One cone width per ray instead of full ray differentials; single trilinear lookup. | Falcor and real-time path tracers. |
| Pharr, Wronski, Salvi, Fajardo, *Filtering After Shading With Stochastic Texture Filtering*, I3D 2024. https://research.nvidia.com/labs/rtr/publication/pharr2024stochtex/stochtex.pdf | One texel fetch per lookup; the pixel filter does the rest. | Research; natural fit for a path tracer. |
| Vaidyanathan et al., *Random-Access Neural Compression of Material Textures*, SIGGRAPH 2023; RTXNTC SDK 0.10 beta. | Neural texture compression, cooperative vectors. | NVIDIA only; premature. |

OIIO 3.x TextureSystem is CPU-only; Cycles flattens textures for the GPU and its texture-cache branch never merged; Arnold and Redshift page textures out of core with a 2-3x slowdown *(secondary sources)*. Karma XPU uses tiled on-demand textures with GPU eviction.

### 2.8 Hair and interop

- Chiang, Bitterli, Tappan, Burley, *A Practical and Controllable Hair and Fur Model for Production Path Tracing*, EGSR 2016 (pbrt-v4 `HairBxDF`, Cycles "Chiang", MaterialX `chiang_hair_bsdf`). Huang, Hullin, Hanika, *A Microfacet-based Hair Scattering Model*, EGSR 2022 (Cycles 4.0 "Huang", energy loss at high roughness). Curves: Embree 4.4 flat, oriented and round curves in four bases; OptiX curves since 7.1, hardware linear curves on Blackwell in 8.1/9.0; HIPRT 3.1 has none (AABB list plus custom intersector, which is what Cycles does on HIP RT).
- Christensen, Fong et al., *RenderMan XPU: A Hybrid CPU+GPU Renderer for Interactive and Final-frame Rendering*, HPG 2025, CGF 44(8). https://graphics.pixar.com/library/RenderManXPU . The one paper on a shipping XPU design.
- OpenUSD 26.08, Hydra 2.0 scene index default since 25.11; MaterialX 1.39.5 (OpenPBR, Chiang hair, Disney graphs; OSL, GLSL, MSL, WGSL and Slang generators); OSL 1.15.7 has OptiX only. Slang targets have no HIP backend and OptiX and Metal support is unfinished, so it is not a kernel language for LuxCore today.

### 2.9 GPU architecture and denoising

- Laine, Karras, Aila, *Megakernels Considered Harmful: Wavefront Path Tracing on GPUs*, HPG 2013; Cycles "Kernel Scheduling" and "Kernel Language" developer docs; pbrt-v4 ch. 15. These define the design LuxCore is converging on (indexed launches over gathered queues, SoA state, shadow paths split off, shader sorting by partitioned bucket sort).
- Padilla, Webster, Kim, *Megakernel vs Wavefront GPU Path Tracing*, 2026. https://arxiv.org/abs/2605.27323 (wavefront ~16% faster, both latency-bound). Meister et al., *On Ray Reordering Techniques for Faster GPU Ray Tracing*, 2020/2025. https://arxiv.org/abs/2506.11273 (ray sorting does not pay against hardware trace).
- NVIDIA SER whitepaper 2022 and OptiX 8 `optixReorder`; Karma XPU uses it with driver 555+. OptiX 9.0 clusters (Mega Geometry) and cooperative vectors; 9.1 improved denoiser. AMD DGF and H-PLOC (HPG 2024) arrive through HIPRT updates; relevant only for animation or huge meshes.
- OIDN 2.3 quality modes, 2.4 and 2.5 performance and external semaphores; no temporal or variance inputs. OptiX denoiser has AOV, temporal and 2x upscale modes. Research: Balint et al., *Neural Partitioning Pyramids*, SIGGRAPH 2023; Zhang et al., *Neural Denoising for Deep-Z Monte Carlo Renderings*, EG 2024; Yan et al., survey in ACM Computing Surveys 2025. None changes the recommendation: GPU OIDN in `high` mode with Cycles-style prefilter options, then retire BCD.
- Dr.Jit / Mitsuba 3 (Jakob et al., SIGGRAPH 2022) is the JIT alternative to a macro C++ subset; cite only as precedent for per-scene specialization (the plan's G4).
