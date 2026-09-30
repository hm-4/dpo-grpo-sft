r"""Offline correctness checks for the GRPO implementation in HM.ipynb.

Run from HM with: .\.venv\Scripts\python.exe -m unittest discover -s tests -v
Only the GRPO definitions are loaded through AST; notebook setup, downloads,
training cells, datasets, and saved project checkpoints are never executed.
"""

import ast
import copy
import json
import math
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel


def load_grpo_definitions():
    notebook_path = Path(__file__).resolve().parents[1] / "HM.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    names = {"GRPODataset", "grpo_collate_fn", "GRPOTrainer"}
    definitions = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell["source"])
        if not any(f"class {name}" in source or f"def {name}(" in source for name in names):
            continue
        definitions.extend(
            node for node in ast.parse(source).body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
        )
    if {node.name for node in definitions} != names:
        raise RuntimeError("The notebook is missing required GRPO definitions.")
    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "F": F,
        "np": np,
        "re": re,
        "math": math,
        "Dataset": Dataset,
        "tqdm": lambda iterable, **kwargs: iterable,
    }
    module = ast.Module(body=definitions, type_ignores=[])
    exec(compile(module, str(notebook_path), "exec"), namespace)
    return namespace["GRPOTrainer"], namespace["grpo_collate_fn"]


GRPOTrainer, grpo_collate_fn = load_grpo_definitions()


class TinyTokenizer:
    """A deliberately small decoder; no tokenizer files or downloads required."""

    eos_token_id = 0
    pad_token_id = 0
    padding_side = "left"

    vocabulary = {
        2: "prompt",
        3: "42",
        5: "Answer:",
        6: "42",
        7: "41",
        8: "uncertain",
        9: "another-question",
    }

    def batch_decode(self, sequences, skip_special_tokens=True):
        decoded = []
        for sequence in sequences:
            words = [
                self.vocabulary.get(int(token), f"word{int(token)}")
                for token in sequence
                if not (skip_special_tokens and int(token) == self.eos_token_id)
            ]
            decoded.append(" ".join(words))
        return decoded

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "test_tokenizer.json").write_text(
            json.dumps({"eos_token_id": self.eos_token_id}), encoding="utf-8"
        )


class TinyBaseModel:
    def __init__(self, model):
        self.model = model
        self.tokenizer = TinyTokenizer()
        self.device = "cpu"

    def save(self, directory):
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)


class GRPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(17)

    def make_trainer(self, group_size=2, **overrides):
        config = GPT2Config(
            vocab_size=32, n_positions=64, n_ctx=64, n_embd=16,
            n_layer=1, n_head=2, bos_token_id=1, eos_token_id=0,
            pad_token_id=0, resid_pdrop=0.5, embd_pdrop=0.5,
            attn_pdrop=0.5,
        )
        policy = TinyBaseModel(GPT2LMHeadModel(config))
        reference = TinyBaseModel(copy.deepcopy(policy.model))
        optimizer = torch.optim.AdamW(policy.model.parameters(), lr=0.01, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        options = dict(
            base_model=policy, ref_model=reference, train_loader=[], eval_loader=[],
            optimizer=optimizer, scheduler=scheduler, device="cpu", num_epochs=1,
            group_size=group_size, max_new_tokens=8, temperature=0.7,
            clip_eps=0.2, kl_beta=0.02,
        )
        options.update(overrides)
        return GRPOTrainer(**options)

    def make_batch(self):
        return grpo_collate_fn(
            [
                {"input_ids": torch.tensor([2, 3]),
                 "attention_mask": torch.tensor([1, 1]), "answer": "42"},
                {"input_ids": torch.tensor([9]),
                 "attention_mask": torch.tensor([1]), "answer": "41"},
            ],
            pad_token_id=0,
        )

    def install_deterministic_rollouts(self, trainer):
        def generate(input_ids, attention_mask):
            completions = torch.tensor(
                [[5, 6, 0, 0], [8, 0, 0, 0], [5, 7, 0, 0], [5, 6, 0, 0]],
                device=input_ids.device,
            )
            return torch.cat((input_ids.repeat_interleave(2, dim=0), completions), dim=1)
        trainer.sample_responses = generate

    def test_collator_left_pads_unequal_prompts(self):
        batch = self.make_batch()
        self.assertEqual(batch["input_ids"].tolist(), [[2, 3], [0, 9]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1], [0, 1]])
        self.assertEqual(batch["answers"], ["42", "41"])

    def test_log_probs_use_causal_shift_and_only_unmasked_targets(self):
        trainer = self.make_trainer()
        logits = torch.randn(2, 5, 7, requires_grad=True)
        labels = torch.tensor([[-100, -100, 3, 4, -100], [-100] * 5])
        result = trainer.get_log_probs(logits, labels)
        log_probabilities = logits.log_softmax(dim=-1)
        expected = torch.stack(((log_probabilities[0, 1, 3] + log_probabilities[0, 2, 4]) / 2,
                                torch.tensor(0.0)))
        torch.testing.assert_close(result, expected)
        result.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[1].abs().sum().item(), 0.0)
        self.assertEqual(logits.grad[0, [0, 3, 4]].abs().sum().item(), 0.0)
        self.assertGreater(logits.grad[0, [1, 2]].abs().sum().item(), 0.0)

    def test_rewards_exact_match_unparseable_and_mismatched_lengths(self):
        trainer = self.make_trainer()
        rewards = trainer.compute_rewards(
            ["The answer is 42", "#### 41", "uncertain", "Answer: -2", "Answer: 1.0", "uncertain"],
            ["42", "42", "5", "-2", "1", "None"],
        )
        self.assertEqual(rewards.tolist(), [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(rewards.dtype, torch.float32)
        self.assertEqual(rewards.device.type, "cpu")
        with self.assertRaises(ValueError):
            trainer.compute_rewards(["Answer: 42"], [])

    def test_rewards_preserve_negative_sign_and_normalize_thousands_separators(self):
        trainer = self.make_trainer()
        cases = [
            ("#### 1,000", "1000", 1.),
            ("#### 1000", "1,000", 1.),
            ("Final Answer -5", "5", 0.),
            ("Final Answer -5", "-5", 1.),
            ("#### -1,200", "-1200", 1.),
            ("#### 42\nThis solution used 3 steps.", "42", 1.),
        ]
        for response, answer, expected in cases:
            with self.subTest(response=response, answer=answer):
                reward = trainer.compute_rewards([response], [answer])
                self.assertEqual(reward.item(), expected)

    def test_advantages_stay_within_groups_and_handle_equal_rewards(self):
        trainer = self.make_trainer(group_size=4)
        rewards = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.])
        advantages = trainer.compute_advantages(rewards)
        expected = torch.tensor([math.sqrt(3), *([-1 / math.sqrt(3)] * 3), *([0.] * 8)])
        torch.testing.assert_close(advantages, expected, atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.isfinite(advantages).all())
        with self.assertRaises(ValueError):
            trainer.compute_advantages(torch.tensor([1., 0., 1.]))

    def test_completion_mask_keeps_first_eos_when_eos_is_padding(self):
        trainer = self.make_trainer()
        ids = torch.tensor([[6, 0, 0, 0], [0, 0, 0, 0], [5, 6, 7, 8]])
        mask = trainer._completion_mask(ids)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.tolist(), [[True, True, False, False],
                                         [True, False, False, False],
                                         [True, True, True, True]])

    def test_clipping_uses_old_policy_and_handles_both_advantage_signs(self):
        trainer = self.make_trainer()
        old_logps = torch.full((2, 2), -2.0, requires_grad=True)
        policy_logps = (-2 + torch.log(torch.tensor([[1.5, 0.5], [1.5, 0.5]]))).requires_grad_()
        ref_logps = policy_logps.detach().clone().requires_grad_()
        advantages = torch.tensor([1., -1.])
        loss = trainer._grpo_objective(policy_logps, old_logps, ref_logps, advantages,
                                       torch.ones((2, 2), dtype=torch.bool))
        # Positive group: -(1.2 + 0.5)/2. Negative group: (1.5 + 0.8)/2.
        self.assertAlmostEqual(loss.item(), 0.15, places=6)
        loss.backward()
        torch.testing.assert_close(policy_logps.grad, torch.tensor([[0., -0.125], [0.375, 0.]]),
                                   atol=1e-6, rtol=1e-6)
        self.assertIsNone(old_logps.grad)
        self.assertIsNone(ref_logps.grad)

    def test_kl_estimator_and_token_mask(self):
        trainer = self.make_trainer()
        policy_logps = torch.tensor([[-0.5, -1.0, -4.0]], requires_grad=True)
        ref_logps = torch.tensor([[-1.5, -0.2, -0.1]], requires_grad=True)
        mask = torch.tensor([[True, True, False]])
        loss = trainer._grpo_objective(policy_logps, policy_logps.detach(), ref_logps,
                                       torch.zeros(1), mask)
        delta = torch.tensor([-1.0, 0.8])
        expected = trainer.kl_beta * (delta.exp() - delta - 1).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(ref_logps.grad)
        self.assertEqual(policy_logps.grad[0, 2].item(), 0.)
        self.assertGreater(policy_logps.grad[0, :2].abs().sum().item(), 0.)

    def test_compute_loss_scores_only_completions_and_preserves_eos_attention(self):
        trainer = self.make_trainer()
        self.install_deterministic_rollouts(trainer)
        with patch.object(trainer, "compute_rewards", wraps=trainer.compute_rewards) as rewards, \
             patch.object(trainer.model, "forward", wraps=trainer.model.forward) as forward, \
             patch.object(trainer, "_token_log_probs", wraps=trainer._token_log_probs) as token_logps:
            loss = trainer.compute_loss(self.make_batch())
        self.assertTrue(torch.isfinite(loss).item())
        responses, answers = rewards.call_args.args
        self.assertEqual(responses, ["Answer: 42", "uncertain", "Answer: 41", "Answer: 42"])
        self.assertEqual(answers, ["42", "42", "41", "41"])
        # In particular, the second response must not inherit "42" from its prompt.
        self.assertEqual(trainer.compute_rewards(responses, answers).tolist(), [1., 0., 1., 0.])
        attention = forward.call_args.kwargs["attention_mask"]
        self.assertEqual(attention.tolist(), [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0],
                                             [0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 0]])
        for call in token_logps.call_args_list:
            labels = call.args[1]
            self.assertEqual(labels.tolist(), [[-100, -100, 5, 6, 0, -100],
                                               [-100, -100, 8, 0, -100, -100],
                                               [-100, -100, 5, 7, 0, -100],
                                               [-100, -100, 5, 6, 0, -100]])

    def test_training_has_finite_gradients_updates_policy_and_freezes_reference(self):
        trainer = self.make_trainer()
        self.install_deterministic_rollouts(trainer)
        trainer.train_loader = [self.make_batch()]
        policy_before = {name: value.detach().clone() for name, value in trainer.model.named_parameters()}
        ref_before = {name: value.detach().clone() for name, value in trainer.ref_model.named_parameters()}
        scheduler_step = trainer.scheduler.last_epoch
        seen_dropout_states = []
        handle = trainer.model.register_forward_pre_hook(
            lambda module, args: seen_dropout_states.append(
                [layer.training and layer.p > 0 for layer in module.modules()
                 if isinstance(layer, torch.nn.Dropout)]
            )
        )
        try:
            loss = trainer.train_epoch()
        finally:
            handle.remove()
        self.assertTrue(math.isfinite(loss))
        self.assertTrue(seen_dropout_states)
        self.assertFalse(any(any(state) for state in seen_dropout_states),
                         "Dropout must remain disabled during rollout and scoring.")
        self.assertTrue(any(not torch.equal(policy_before[name], value)
                            for name, value in trainer.model.named_parameters()))
        gradients = [p.grad for p in trainer.model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0.)
        self.assertEqual(trainer.scheduler.last_epoch, scheduler_step + 1)
        for name, value in trainer.ref_model.named_parameters():
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad)
            torch.testing.assert_close(value, ref_before[name], atol=0, rtol=0)

    def test_actual_generation_retains_contiguous_prompt_groups(self):
        trainer = self.make_trainer()
        batch = self.make_batch()
        with patch.object(trainer.model, "generate", wraps=trainer.model.generate) as generate:
            generated = trainer.sample_responses(batch["input_ids"], batch["attention_mask"])
        self.assertEqual(generated.shape[0], 4)
        self.assertGreater(generated.shape[1], batch["input_ids"].shape[1])
        self.assertLessEqual(generated.shape[1], batch["input_ids"].shape[1] + 8)
        torch.testing.assert_close(generated[:, :2], batch["input_ids"].repeat_interleave(2, dim=0))
        kwargs = generate.call_args.kwargs
        self.assertTrue(kwargs["do_sample"])
        self.assertEqual(kwargs["temperature"], trainer.temperature)
        self.assertEqual(kwargs["top_k"], 0)
        self.assertEqual(kwargs["top_p"], 1.0)
        self.assertEqual(kwargs["repetition_penalty"], 1.0)

    def test_actual_rollout_and_backward_with_gradient_checkpointing(self):
        trainer = self.make_trainer()
        trainer.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        trainer.model.config.use_cache = False
        trainer.model.train()
        self.assertTrue(trainer.model.is_gradient_checkpointing)
        loss = trainer.compute_loss(self.make_batch())
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        gradients = [parameter.grad for parameter in trainer.model.parameters()
                     if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(all(parameter.grad is None for parameter in trainer.ref_model.parameters()))

    def test_train_evaluate_save_and_reload_without_downloads(self):
        trainer = self.make_trainer()
        self.install_deterministic_rollouts(trainer)
        trainer.train_loader = [self.make_batch()]
        trainer.eval_loader = [self.make_batch()]
        with tempfile.TemporaryDirectory(prefix="hm-grpo-test-") as directory:
            trainer.save_path = str(Path(directory) / "checkpoint")
            trainer.train()
            self.assertTrue((Path(trainer.save_path) / "config.json").is_file())
            self.assertTrue((Path(trainer.save_path) / "test_tokenizer.json").is_file())
            reloaded = GPT2LMHeadModel.from_pretrained(trainer.save_path, local_files_only=True)
            for name, value in trainer.model.state_dict().items():
                torch.testing.assert_close(value, reloaded.state_dict()[name], atol=0, rtol=0)
        trainer.optimizer.zero_grad(set_to_none=True)
        policy_before = {name: value.detach().clone() for name, value in trainer.model.named_parameters()}
        self.assertTrue(math.isfinite(trainer.eval_epoch()))
        for name, value in trainer.model.named_parameters():
            self.assertIsNone(value.grad)
            torch.testing.assert_close(value, policy_before[name], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
