"""Open HM.ipynb using this folder's virtual environment and local caches."""

import os
from pathlib import Path
import sys


project = Path(__file__).resolve().parent
expected_python = project / ".venv" / "Scripts" / "python.exe"
if Path(sys.executable).resolve() != expected_python.resolve():
    raise SystemExit(f'Run with: "{expected_python}" "{__file__}"')

os.chdir(project)
os.environ.setdefault("HF_HOME", str(project / ".cache" / "huggingface"))
os.environ.setdefault("JUPYTER_CONFIG_DIR", str(project / ".jupyter" / "config"))
os.environ.setdefault("JUPYTER_DATA_DIR", str(project / ".jupyter" / "data"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(project / ".jupyter" / "runtime"))

from jupyterlab.labapp import main

sys.argv = ["jupyter-lab", "HM.ipynb", *sys.argv[1:]]
main()
