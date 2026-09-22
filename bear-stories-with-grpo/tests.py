"""Unit tests: python tests.py."""
import copy
from types import SimpleNamespace
import unittest

import torch

from grpo import sequence_log_probs, group_advantages, grpo_loss, kl_penalty
from eval import completion_metrics
from train import TrainConfig, Rollout, _action_mask, clone_reference, grpo_step


class RewardTests(unittest.TestCase):
    def test_sparse_reward_uses_whole_words(self):
        from rewards import SparseBearReward

        scores = SparseBearReward()(["a bear", "two BEARS", "bearing", "a rabbit"])
        torch.testing.assert_close(scores, torch.tensor([1.0, 1.0, 0.0, 0.0]))

    def test_dense_reward_normalizes_embeddings_and_handles_empty_text(self):
        from rewards import DenseStoryReward

        class FakeDenseReward(DenseStoryReward):
            def _encode(self, texts):
                vectors = {
                    "bear": [3.0, 0.0],
                    "same direction": [2.0, 0.0],
                    "orthogonal": [0.0, 4.0],
                }
                return torch.tensor([vectors[text] for text in texts])

        scores = FakeDenseReward()(["same direction", "orthogonal", ""])
        torch.testing.assert_close(scores, torch.tensor([1.0, 0.0, 0.0]))


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.4]))

    def forward(self, input_ids, attention_mask, use_cache, position_ids=None):
        return SimpleNamespace(logits=self.logits.expand(*input_ids.shape, 3))


class GRPOTests(unittest.TestCase):
    def test_advantages(self):
        result = group_advantages(torch.tensor([1., 3., 5., 5.]), torch.tensor([0, 0, 1, 1]))
        torch.testing.assert_close(result, torch.tensor([-1 / 1.0001, 1 / 1.0001, 0., 0.]))

    def test_eos_and_log_probs(self):
        ids = torch.tensor([[1, 2, 2], [0, 1, 2]])
        mask = _action_mask(ids, 2, 2)
        self.assertEqual(mask.tolist(), [[True, True, False], [True, True, True]])
        policy = TinyPolicy()
        lp = sequence_log_probs(policy, torch.tensor([[2, 0], [0, 1]]), torch.tensor([[0, 1], [1, 1]]), ids, mask)
        expected = policy.logits.log_softmax(0)[ids].masked_fill(~mask, 0)
        torch.testing.assert_close(lp, expected)
        lp.sum().backward()
        self.assertIsNotNone(policy.logits.grad)

    def test_log_probs_use_preceding_position(self):
        class PositionPolicy(TinyPolicy):
            def forward(self, input_ids, attention_mask, use_cache, position_ids=None):
                # Each position predicts a different distribution.
                offsets = torch.arange(input_ids.shape[1], dtype=torch.float32)
                logits = self.logits + offsets[:, None] * torch.tensor([0.7, -0.3, 0.2])
                return SimpleNamespace(logits=logits.unsqueeze(0))

        policy = PositionPolicy()
        result = sequence_log_probs(
            policy, torch.tensor([[0, 1]]), torch.ones(1, 2, dtype=torch.long),
            torch.tensor([[2, 0]]), torch.ones(1, 2, dtype=torch.bool),
        )
        # Positions 1 and 2 predict the two completion tokens, not 2 and 3.
        first = (policy.logits + torch.tensor([0.7, -0.3, 0.2])).log_softmax(0)[2]
        second = (policy.logits + torch.tensor([1.4, -0.6, 0.4])).log_softmax(0)[0]
        torch.testing.assert_close(result, torch.stack([first, second]).unsqueeze(0))

    def test_kl_changes_update_without_reward_signal(self):
        initial = TinyPolicy()
        reference = clone_reference(initial)
        with torch.no_grad():
            reference.logits[0] += 1.0
        reference_before = reference.logits.clone()
        prompt = torch.zeros(1, 1, dtype=torch.long)
        completion = torch.tensor([[0]])
        mask = torch.ones_like(completion, dtype=torch.bool)
        old = sequence_log_probs(initial, prompt, mask, completion, mask).detach()
        rollout = Rollout(['p'], ['bear'], prompt, mask, completion, mask, old, torch.tensor([0]))
        policies = [copy.deepcopy(initial), copy.deepcopy(initial)]
        for policy, beta in zip(policies, [0.0, 1.0]):
            grpo_step(policy, reference, torch.optim.SGD(policy.parameters(), lr=0.1),
                      rollout, torch.zeros(1), TrainConfig(inner_epochs=1, kl_beta=beta))
        torch.testing.assert_close(policies[0].logits, initial.logits)
        self.assertFalse(torch.allclose(policies[1].logits, initial.logits))
        updated_probability = policies[1].logits.softmax(0)[0].detach()
        initial_probability = initial.logits.softmax(0)[0].detach()
        self.assertGreater(float(updated_probability), float(initial_probability))
        torch.testing.assert_close(reference.logits, reference_before)
        self.assertIsNone(reference.logits.grad)

    def test_clipping_and_detach(self):
        current = torch.tensor([[1.5], [0.5]]).log().requires_grad_()
        old = torch.zeros_like(current, requires_grad=True)
        mask = torch.ones_like(current, dtype=torch.bool)
        adv = torch.tensor([1., -1.])
        loss, fraction = grpo_loss(current, old, mask, adv, clip_epsilon=0.2)
        torch.testing.assert_close(loss, torch.tensor(-0.2))
        self.assertEqual(float(fraction), 1.)
        loss.backward()
        self.assertIsNone(old.grad)
        raw, _ = grpo_loss(current, old, mask, adv, clip_epsilon=None)
        torch.testing.assert_close(raw, torch.tensor(-0.5))

    def test_kl(self):
        current = torch.tensor([[0.2, 999.]], requires_grad=True)
        ref = torch.tensor([[0., -999.]], requires_grad=True)
        mask = torch.tensor([[True, False]])
        loss = kl_penalty(current, ref, mask)
        self.assertGreater(float(loss.detach()), 0)
        loss.backward()
        self.assertIsNone(ref.grad)
        self.assertEqual(float(current.grad[0, 1]), 0.)
        self.assertEqual(float(kl_penalty(ref, ref, mask).detach()), 0.)

    def test_partial_microbatch_matches_full_batch(self):
        first = TinyPolicy(); second = copy.deepcopy(first)
        reference = clone_reference(first)
        prompt = torch.zeros(3, 1, dtype=torch.long)
        completion = torch.tensor([[0, 2], [1, 2], [0, 2]])
        mask = torch.ones_like(completion, dtype=torch.bool)
        old = sequence_log_probs(first, prompt, torch.ones_like(prompt), completion, mask).detach()
        rollout = Rollout(['p']*3, ['a']*3, prompt, torch.ones_like(prompt), completion, mask, old, torch.zeros(3, dtype=torch.long))
        for policy, size in ((first, 2), (second, 3)):
            grpo_step(policy, reference, torch.optim.SGD(policy.parameters(), lr=0.1), rollout,
                      torch.tensor([1., -1., 0.5]), TrainConfig(micro_batch_size=size, inner_epochs=2, kl_beta=0.1))
        torch.testing.assert_close(first.logits, second.logits)
        self.assertIsNone(reference.logits.grad)

    def test_perplexity_is_token_weighted(self):
        metrics = completion_metrics([
            dict(bear_mentions=1, reference_nll=2., completion_token_count=1),
            dict(bear_mentions=3, reference_nll=2., completion_token_count=3),
        ])
        self.assertAlmostEqual(metrics['reference_perplexity'], 2.718281828)
        self.assertEqual(metrics['bear_count_mean'], 2.)

class TinyTokenizer:
    padding_side = 'left'
    eos_token_id = 2
    pad_token_id = 2

    def __call__(self, texts, return_tensors, padding=False):
        n = 1 if isinstance(texts, str) else len(texts)
        return dict(input_ids=torch.zeros(n, 1, dtype=torch.long), attention_mask=torch.ones(n, 1, dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        return 'bear' if 0 in ids else 'rabbit'

    def batch_decode(self, rows, skip_special_tokens=True):
        return [self.decode(row) for row in rows]

    def save_pretrained(self, path):
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / 'tokenizer.json').write_text('{}')


class GeneratingPolicy(TinyPolicy):
    def generate(self, input_ids, **kwargs):
        n = input_ids.shape[0]
        completion = torch.stack([torch.arange(n) % 2, torch.full((n,), 2)], dim=1)
        return torch.cat([input_ids, completion], dim=1)

    def save_pretrained(self, path, **kwargs):
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / 'weights.pt')


class IntegrationTests(unittest.TestCase):
    def test_2x2_experiment_config(self):
        from utils import load_grid_config

        grid = load_grid_config()
        self.assertEqual(grid["shared"]["updates"], 20)
        self.assertEqual(grid["shared"]["inner_epochs"], 4)
        self.assertEqual(set(grid["arms"]), {
            "sparse_clip", "sparse_noclip", "dense_clip", "dense_noclip"
        })
        self.assertTrue(all(arm["group_size"] == 16 for arm in grid["arms"].values()))
        self.assertEqual(grid["arms"]["sparse_clip"]["clip_epsilon"], 0.2)
        self.assertIsNone(grid["arms"]["sparse_noclip"]["clip_epsilon"])

    def test_demo_grid_uses_small_overrides(self):
        from pathlib import Path
        from unittest.mock import patch
        from utils import run_grid

        with patch("utils.run_arm", return_value=Path("demo")) as run_arm:
            result = run_grid(demo=True)
        self.assertEqual(result, {"sparse_clip": Path("demo")})
        kwargs = run_arm.call_args.kwargs
        self.assertEqual(kwargs["updates"], 2)
        self.assertEqual(kwargs["eval_count"], 2)
        self.assertEqual(kwargs["max_new_tokens"], 32)
        self.assertEqual(kwargs["group_size"], 4)
        self.assertEqual(kwargs["inner_epochs"], 1)

    def test_intro_sampling_without_evaluation_solution(self):
        from unittest.mock import patch
        from eval import evaluate
        with patch('eval.completion_metrics', side_effect=NotImplementedError):
            policy = GeneratingPolicy()
            samples = evaluate(policy, TinyTokenizer(), ['prompt'], count=2, compute_metrics=False)
        self.assertEqual([s['completion'] for s in samples['samples']], ['bear', 'rabbit'])
        self.assertEqual(samples['aggregate'], {})
        self.assertTrue(policy.training)

    def test_training_saves_evaluation_and_samples(self):
        import json
        import tempfile
        from train import train_grpo
        from rewards import SparseBearReward
        from utils import summarize_runs, summarize_seed_grid
        with tempfile.TemporaryDirectory() as output:
            path = train_grpo(
                SparseBearReward(), TrainConfig(output_dir=output, updates=2, group_size=3,
                    micro_batch_size=2, sample_every=1, kl_beta=0.2),
                policy=GeneratingPolicy(), tokenizer=TinyTokenizer(), eval_count=2,
                semantic_reward_fn=lambda texts: torch.full((len(texts),), 0.25),
            )
            metrics = [json.loads(line) for line in (path / 'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(len(metrics), 2)
            self.assertIn('kl_mean', metrics[0])
            self.assertTrue((path / 'samples_step_0002.jsonl').exists())
            self.assertTrue((path / 'checkpoint' / 'weights.pt').exists())
            rows = summarize_runs({'test': path})
            self.assertEqual(len(rows), 1)
            self.assertGreater(rows[0]['post_reference_perplexity'], 1)
            self.assertEqual(rows[0]['post_semantic_similarity_mean'], 0.25)
            per_seed, aggregate = summarize_seed_grid({17: {'test': path}})
            self.assertEqual(per_seed[0]['seed'], 17)
            self.assertEqual(aggregate[0]['updates'], 2)


if __name__ == '__main__':
    unittest.main()
