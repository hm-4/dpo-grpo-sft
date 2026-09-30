# Local environment for HM.ipynb

The project uses a Python 3.12 virtual environment in `.venv`. All notebook
dependencies are isolated from the other Python installations on this computer.

From PowerShell:

```powershell
cd C:\Users\Admin\Desktop\local_mahamathi2\HM
.\.venv\Scripts\python.exe start_notebook.py
```

The launcher opens `HM.ipynb` in JupyterLab with `HM` as its working directory.
It keeps Hugging Face and Jupyter caches inside this folder. Select the
**HM (.venv)** kernel.

In VS Code, use **Select Kernel > Select Another Kernel... > Python Environments**
and choose **.venv (3.12.7)** whose interpreter is `HM\.venv\Scripts\python.exe`.
When HM is the open folder, the path may display as `.venv\Scripts\python.exe`.
The Python environment picker uses the folder name `.venv`; the registered
Jupyter kernelspec has the separate display name **HM (.venv)** under
**Jupyter Kernels**. The first menu can show only recently used kernels. If the new
entry has not appeared yet, save your work, run **Developer: Reload Window**
from the Command Palette (`Ctrl+Shift+P`), and open the kernel picker again.
The kernel is registered in your Windows user profile and points to
`HM\.venv\Scripts\python.exe`; the environment and its packages remain in `HM`.
See [VS Code's kernel picker documentation](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management).

Activation is optional when using the explicit Python path. To activate it in
an existing PowerShell session:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Recreating the environment

With Python 3.12 available:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-torch.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name hm --display-name "HM (.venv)"
```

For VS Code discovery, also register the same environment for your Windows
user. Run this from the `HM` folder:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name hm --display-name "HM (.venv)" --env HF_HOME "$PWD\.cache\huggingface"
```

PyTorch 2.7.1 with CUDA 11.8 is selected for this computer's Quadro P2000.
The wheel is provided by the [official PyTorch index](https://pytorch.org/get-started/previous-versions/#v271).
Transformers is kept on version 4 for compatibility with the existing notebook.
The requirements files contain the installation dependencies; the environment's
fully resolved versions are recorded in `requirements-lock.txt` after setup.
For an exact reinstall, install `requirements-torch.txt` first and then
`requirements-lock.txt` instead of the version ranges in `requirements.txt`.

## Data and execution

All six JSONL files referenced by the notebook are downloaded into `data/`.
See [DATASET.md](DATASET.md) for their schemas, row counts, shared questions,
and source links. The notebook skips downloading files that already exist.

The notebook uses `deadMarkov/distilgpt2-math` as its initial model. The first
execution of `BaseModel()` downloads its pretrained weights from Hugging Face;
the launcher stores them in `.cache/huggingface/`. Its GPU selection is
automatic when CUDA is available. With 4 GB of GPU memory, full
SFT/DPO training may require smaller batches than their notebook defaults.
Installing the environment does not run training epochs.

The optional GRPO stage is complete. Run SFT and DPO first to create the local
`dpo_model` checkpoint, or set `GRPO_INIT_MODEL` to an existing compatible local
checkpoint. GRPO uses a batch of one prompt, two sampled completions, and gradient
checkpointing to reduce memory use. It saves its trained model to `grpo_model`.
Increase group size only when GPU memory allows. Full training memory use still
depends on prompt and completion lengths.

The environment, downloaded data, caches, and generated model folders are
excluded from Git by `.gitignore`.

## Verified on this computer

Setup was verified on 2026-09-29 with Python 3.12.7, PyTorch 2.7.1+cu118,
Transformers 4.57.6, and JupyterLab 4.6.4. `pip check` found no broken
requirements. CUDA matrix multiplication and a tiny randomly initialized
Transformer's forward/backward pass succeeded on the Quadro P2000.
The `hm` Jupyter kernel started, selected this virtual environment, and detected
CUDA. The notebook launcher also passed its JupyterLab version check.

No full notebook execution or training on the math dataset was run during setup.
The machine-readable local setup report is `.cache/environment-check.json`.

GRPO correctness checks were added on 2026-09-30. Run them offline with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

These checks use tiny random GPT-2 models to verify sampling, reward calculation,
loss and gradients, one optimizer update, a frozen reference, evaluation, and
checkpoint save/reload. They do not measure math-task performance.
