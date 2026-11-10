# Paper experiment entry points

Run from this directory's parent (`PD-Adapt/new`) after restoring the external
source-grasp, object, robot-mesh, checkpoint and teacher data. These commands
write new results under `new/results/`; the repository does not ship results.

## Table I: original and adapted grasps

`DRO/generate_64.py` and `GEN/generate_64.py` merge 20 existing and 44 newly
generated source grasps for each hand/object/generator. Their source manifests
are created during `merge`; they are not bundled without the input `.pt` files.

```bash
python DRO/run_raw_native_six.py all --line all

# Only use these existing Inspire entries on the two Inspire lines:
python DRO/run_cem288_six.py all --line inspire-dro
python DRO/run_cem288_six.py all --line inspire-gen
python DRO/run_sample288_six.py search --line inspire-dro
python DRO/run_sample288_six.py recheck --line inspire-dro
python DRO/run_sample288_six.py summarize --line inspire-dro
python DRO/run_pd288_five.py prepare --line inspire-dro
python DRO/run_pd288_five.py search --line inspire-dro
python DRO/run_pd288_five.py recheck --line inspire-dro
python DRO/run_pd288_five.py summarize --line inspire-dro
# Repeat Sample and PD with --line inspire-gen.

# Ability/SVH Table I requires the consistent LSQ teacher + retraining flow:
python results/supplement_lsq_ability_svh_v1/run.py --help
```

Why the hand split matters: old `DRO/run_{cem288_six,sample288_six,pd288_five}.py`
also supports Ability and SVH, but those historical search pools used the
`none` controller target projection. Table I uses the independently rebuilt
`lsq` teacher, student, search and verifier for those hands. Do not merge old
and new protocols into one percentage column.

All three hands' **direct** Raw/naive-Φ columns come from
`DRO/run_raw_native_six.py`, with original active commands versus an LSQ fit
to all original finger joints. Both reconstruct dependent targets through
the coupling function and validate the pre-open/close controller targets with
LSQ under the standard six-force battery. Each direct source has one physical
trial. Raw is not the historical full-joint `projection_mode=none` baseline.

## Table II: Inspire ablations

```bash
# First generate the main Inspire teacher + train/val/test model and its
# complete 10×64 PD-9 baseline, as described in the root README.

python supplement_inspire_dro_1/ablate_no_set.py train

python DRO/run_inspire_pd288_ablations.py all --variant no-wrist --line inspire-dro
python DRO/run_inspire_pd288_ablations.py all --variant no-wrist --line inspire-gen
python DRO/run_inspire_pd288_ablations.py all --variant no-set --line inspire-dro
python DRO/run_inspire_pd288_ablations.py all --variant no-set --line inspire-gen

python DRO/run_inspire_k1_equal9.py train --line all
python DRO/run_inspire_k1_equal9.py all --line inspire-dro
python DRO/run_inspire_k1_equal9.py all --line inspire-gen

python DRO/run_inspire_random9_gaussian.py prepare --line all
python DRO/run_inspire_random9_gaussian.py all --line inspire-dro --all-objects
python DRO/run_inspire_random9_gaussian.py all --line inspire-gen --all-objects
```

The random-start script uses the shared search/recheck helper
`DRO/run_inspire_random9_spatial.py` but replaces that helper's shell sampler
with eight validation-calibrated fixed-scale Gaussian starts and local P32
search. The K=1 experiment repeats one trained head's
prediction with eight independent random search streams so that its default
plus eight slots still spends 288 GPU evaluations per source. Each method
uses CPU-state batch32 tail-padded, once-only rechecks for its 10×64 sources.

`../tools/reproduce_tables.py` builds measured Table I and Table II views
only from completed, source-matched outputs. It never supplies missing values
from older `none`-projection summaries or hardcodes paper percentages.
