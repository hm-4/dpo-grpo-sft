# DPO, GRPO, and SFT: RL Methods for NLP

[Open the notebook](HM.ipynb) | [Open in Google Colab](https://colab.research.google.com/github/hm-4/dpo-grpo-sft/blob/main/HM.ipynb)

DS-207: Introduction to NLP, Assignment 3, converted from `HM.py` into a
Jupyter notebook. It covers supervised fine-tuning (SFT), Direct Preference
Optimization (DPO), and Group Relative Policy Optimization (GRPO) for math word
problems. Assignment attribution and evaluation markers are retained.

The optional GRPO stage is implemented: online group sampling, answer rewards,
group-normalized advantages, token-level clipped policy loss, frozen-reference
KL regularization, evaluation, and checkpoint saving. It also handles left-padded
prompts, completion-only scoring, and shared EOS/padding tokens.

## Run the notebook

Use Python 3 in Colab or Jupyter and run cells in order. The notebook downloads
its datasets and initial model. Finish SFT and DPO before running GRPO, which
starts from the local `dpo_model` checkpoint and saves to `grpo_model`.

GRPO defaults to one prompt and two completions per batch, with gradient
checkpointing to reduce memory use. Training performs one optimizer update per
fresh rollout. Its sampling distribution uses temperature scaling without
top-k/top-p truncation or repetition penalties so rollout and training
probabilities agree. Pass@1/Pass@k retain their separate evaluation settings.

See [LOCAL_SETUP.md](LOCAL_SETUP.md) for the virtual environment, dependencies,
and kernel selection. [DATASET.md](DATASET.md) documents the six data files and
their verified GSM8K provenance. The assignment uses a custom split that includes
some official GSM8K test questions in training; its scores are not standard
GSM8K benchmark scores.

## Verify the implementation

From the project folder, with dependencies installed:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests load the GRPO definitions directly from the notebook and use tiny random
GPT-2 models offline. They check loss/gradient calculations, sampling, masks,
policy updates, reference freezing, evaluation, and checkpoint save/reload.
No dataset or pretrained checkpoint download is needed for these tests.

The notebook schema and code syntax have also been validated. Full SFT/DPO/GRPO
training on the math dataset has not been run as part of this implementation;
no benchmark improvement is claimed. Environment files, downloaded data,
caches, and trained weights are excluded from Git.
