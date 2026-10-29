# Material test kit

Five checks, used for every material change (see the "Material standards pass"
entry of `docs/history/MODERNIZATION_PLAN.md`):

| Check | Tool | Renders? | Target |
|---|---|---|---|
| Sample/Evaluate/Pdf consistency, chi-square, directional albedo | `bsdftest` (`tests/bsdf/`, built with the project) | no | every case passes |
| White furnace | `furnace.py <ENGINE>` | yes, small | 1.000 ± 0.005 (opaque), ± 0.01 (transmissive) |
| Estimator consistency | `furnace.py <ENGINE> env=box` against `env=boxnodls` | yes, small | equal within the noise |
| CPU/GPU parity | `parity.py` twice, then `paritycmp.py` | yes, small | mean and median block within 1% |
| Subsurface albedo under a boundary | `subsurface_boundary.py <ENGINE>` | yes, small | the slab albedo at each IOR within 3% of the index-matched one |

The lossless parameter sets are the same in `bsdftest.cpp` and `furnace.py`.
Keep the two lists in sync when a material is added.

## Pitfalls (found in earlier runs)

- Set all four `path.pathdepth.*` values. The glossy depth of 4 capped the old
  rough glass furnace. `furnace.py` sets 128 for each: the white medium
  behind the glass (`glass-vol-s4`) reads 0.957 at 48, 0.9995 at 128.
- Glass materials need `interiorior` set explicitly. Without it the IOR is 1
  and the furnace passes trivially.
- Cloth reads the UVs; `furnace.py`'s sphere has them.
- The disk mean (r < 0.9) never sees view cosines below 0.44, so check the
  grazing band (`mu<.2`) too.
- PATHOCL on small films with many tasks stops with paths in flight and is
  biased dark. Both scripts use `opencl.task.count = 4096`.
- Run the scripts from the repository root with the venv of the build under
  test. Do not use `python -I`.

## Examples

```
python3 tests/furnace/furnace.py PATHCPU glossy2 out=/tmp/furnace.tsv
python3 tests/furnace/furnace.py BIDIRCPU roughglass
python3 tests/furnace/furnace.py PATHCPU roughglass env=boxnodls
python3 tests/furnace/parity.py PATHCPU cpu.npz && python3 tests/furnace/parity.py PATHOCL gpu.npz
python3 tests/furnace/paritycmp.py cpu.npz gpu.npz
python3 tests/furnace/furnace_ref.py
python3 tests/furnace/subsurface_boundary.py PATHCPU
out/build/.../bsdftest --filter glossycoating
```
