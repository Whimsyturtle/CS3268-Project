# Shared cluster runtime

All new CS3268 project construction, tuning, and evaluation runs on the school
cluster use this shared runtime: CAA, ICV, SAE, ReFT-r1, and the unsteered baseline.
Follow the [cluster guide](SoC_Cluster_Guide.md) for access and connection steps;
its separate guide-repository Conda example is not the project environment.

| Component | Required source or value |
| --- | --- |
| Python | CPython version in [`.python-version`](../.python-version) (3.13.18) |
| Libraries | Exact platform-specific versions and hashes in [`uv.lock`](../uv.lock); install with `uv sync --locked` |
| PyTorch CUDA build | CUDA 12.6 (`cu126`), selected in [`pyproject.toml`](../pyproject.toml) |
| Slurm resource | `h100-47`, one device |
| Device visible to PyTorch | `NVIDIA H100 NVL MIG 3g.47gb`, 60 [SMs](../GLOSSARY.md#sm) |

The Python and GPU choice follows the [verified MIG rerun](../steering/sae/verification.md#eight-condition-mig-tuning-rerun).
The [SoC GPU resource table](https://www.comp.nus.edu.sg/~cs3210/student-guide/soc-gpus/)
documents `h100-47`. Its name alone is insufficient: the
[allocation investigation](../steering/sae/verification.md#baseline-reproduction-check)
observed full-device access on `xgpi15` and `xgpi17`. Exclude those observed
mismatches and check the actual device on every allocation.

From `xlogin`, outside any existing allocation:

```bash
cd ~/CS3268-Project
srun --gpus=h100-47:1 --exclude=xgpi15,xgpi17 --mem=32G --time=02:00:00 --propagate=NONE --pty bash
```

Inside the allocated compute shell, at the repository root:

```bash
uv sync --locked
uv run --locked python -m cluster_runtime
```

Stop if either command fails. Release a mismatched allocation with `exit` and
request another; do not continue with a full H100, another MIG profile, or CPU
fallback. Installation also belongs on a compute node. If `uv` cannot find the
pinned interpreter, update `uv` on a compute node and retry; do not choose another
Python version. [`uv` uses `.python-version`](https://docs.astral.sh/uv/concepts/python-versions/#python-version-files)
as the default interpreter request.

After the check passes, use the same shell for the selected command:

```bash
# Replace placeholders; each command needs its own fresh run ID.
uv run --locked python tuning.py --config configs/tuning.json --method <method> --run-id <run-id>
uv run --locked python -m evaluation.runner --config configs/main.json --method <method-or-baseline> --benchmark <benchmark> --run-id <run-id>
```

Method implementations, interventions, candidate strengths, and final settings
must still be ready as required by the [tuning](../steering/README.md#strength-hyperparameter-tuning)
and [evaluation](../evaluation/README.md) workflows. Use the same allocation
request for batch jobs. Request longer wall time if needed, retaining the GPU
resource and runtime check. This setup does not launch or authorize a new
experiment or change the [proposal](../CS3268_Project_Proposal_Group_6.md).

[`cluster_runtime.py`](../cluster_runtime.py) checks the exact interpreter,
uses [`uv sync --locked --check`](https://docs.astral.sh/uv/reference/cli/#uv-sync)
to verify the running virtual environment against the complete lock without
installing anything, then checks CUDA and the visible device. The check runs
automatically before model downloads through the shared model loader on Slurm
jobs and known SoC login/GPU hosts. New construction/training entrypoints must
use that loader and save its runtime evidence as existing ICV and SAE builders
do. Local development remains available outside the cluster; CPU-only judging
or summarizing saved generations does not require a GPU allocation.

New cluster tuning, coherence generation, benchmark runs, and construction
artifacts retain `cluster_runtime.json`: interpreter, installed package versions,
lockfile hash, CUDA build, GPU profile, memory and SM count, plus host, job and
driver evidence for each generation allocation. Keep it with the raw artifacts.
Partial generation resumes must match the recorded runtime; a partial run
without that evidence needs a fresh run ID. Completed generation and historical
judging can still use their recorded workflow without regenerating responses.

The system NVIDIA driver and system CUDA toolkit are **not pinned** by the
Python lockfile. `nvidia-smi`'s CUDA label describes driver support, not the CUDA
build used by PyTorch. Record the driver; do not attempt to install a cluster
driver. The existing lock includes a newer shared dependency fix than the
historical SAE environment, so that old package list is not a replacement lock.

Matching this runtime reduces a demonstrated source of variation; it does not
guarantee identical text. Keep the existing model/data revisions, prompt order,
batch size, seed and decoding settings fixed as specified in
[the configurations](../configs/README.md). Do not change numerical backend flags
or experiment settings as part of setup. NVIDIA's
[cuBLAS reproducibility conditions](https://docs.nvidia.com/cuda/archive/12.6.0/cublas/index.html#results-reproducibility)
include the GPU architecture and SM count. Earlier results retain their original
provenance and must be checked for compatibility before a combined comparison.
