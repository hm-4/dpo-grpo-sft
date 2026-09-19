# DPO, GRPO, and SFT — RL Methods for NLP

[Open the notebook](HM.ipynb) · [Open in Google Colab](https://colab.research.google.com/github/hm-4/dpo-grpo-sft/blob/main/HM.ipynb)

Jupyter notebook converted from `HM.py`, covering DS-207: Introduction to NLP, Assignment 3 — RL Methods for NLP. It includes supervised fine-tuning (SFT), Direct Preference Optimization (DPO), and optional Group Relative Policy Optimization (GRPO) for mathematical word problems.

The original explanatory text, attribution, assignment instructions, Python code, and evaluation markers are preserved. The Colab export's text blocks are restored as Markdown cells.

To run it, open the notebook in Colab or Jupyter with Python 3. Its imports require `torch`, `transformers`, `datasets`, `gdown`, `numpy`, and `tqdm`. Run the cells in order; a GPU is recommended for training. The notebook downloads its datasets and base model when executed. Run the optional GRPO section and its model archive cell only if you want to train GRPO.

Notebook structure and Python cell syntax were validated during conversion. Training and evaluation were not executed, so the notebook contains no saved outputs. Dataset files and model weights are not included in this repository.
