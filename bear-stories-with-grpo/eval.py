"""Held-out evaluation: bear story rate + good/bad sample buckets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import math

import torch

from rewards import RewardFn, bear_mentions, extract_words


def classify_story(completion: str) -> str:
    """Heuristic aboutness label (not the training reward)."""
    count = bear_mentions(completion)
    if count <= 0:
        return "no_bear"
    words = extract_words(completion)
    if not words:
        return "no_bear"
    density = count / len(words)
    if density >= 0.08:
        return "keyword_heavy"
    # early mention → more likely main/supporting character
    first_idx = None
    for i, w in enumerate(words):
        if w in {"bear", "bears"}:
            first_idx = i
            break
    if first_idx is not None and first_idx / max(1, len(words)) <= 0.25:
        return "main_or_supporting"
    return "mentioned"


def _truncate(tokens: list[int], eos: int, pad: int) -> list[int]:
    """Drop the first EOS or padding token and everything after it."""
    cut = []
    for token in tokens:
        if token == eos or (pad != eos and token == pad):
            break
        cut.append(token)
    return cut


def completion_metrics(samples):
    """Completion-only bear count, optional semantic score and reference PPL.

    PPL is exp(total reference NLL / total completion tokens), including EOS.
    It is a reference-likelihood proxy, not a direct judgment of story quality.
    """
    metrics = {
        "bear_count_mean": sum(s["bear_mentions"] for s in samples)
        / max(1, len(samples))
    }
    if samples and "semantic_similarity" in samples[0]:
        metrics["semantic_similarity_mean"] = (
            sum(s["semantic_similarity"] for s in samples) / len(samples)
        )
    if samples and "reference_nll" in samples[0]:
        tokens = sum(s["completion_token_count"] for s in samples)
        metrics["reference_perplexity"] = (
            math.exp(sum(s["reference_nll"] for s in samples) / tokens)
            if tokens
            else None
        )
    return metrics


def evaluate(
    policy,
    tokenizer,
    prompts: Sequence[str],
    *,
    count: int = 8,
    max_new_tokens: int = 256,
    seed: int | None = None,
    reward_fn: RewardFn | None = None,
    temperature: float = 1.0,
    compute_metrics: bool = True,
    reference_policy=None,
    semantic_reward_fn: RewardFn | None = None,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Generate stories in batches; ``compute_metrics=False`` skips the aggregate metrics."""
    from grpo import sequence_log_probs
    from train import _action_mask

    tokenizer.padding_side = "left"
    device = next(policy.parameters()).device
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    samples: list[dict[str, Any]] = []
    was = policy.training
    policy.eval()
    reference_was_training = (
        reference_policy.training if reference_policy is not None else False
    )
    if reference_policy is not None:
        reference_policy.eval()

    # A flat job list allows one generation batch to span several openers.
    jobs = [
        (prompt_index, member)
        for prompt_index in range(len(prompts))
        for member in range(count)
    ]

    cuda_devs = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=cuda_devs), torch.no_grad():
        if seed is not None:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
        size = max(1, batch_size)
        for start in range(0, len(jobs), size):
            chunk = jobs[start : start + size]
            texts = [prompts[prompt_index] for prompt_index, _ in chunk]
            enc = tokenizer(texts, return_tensors="pt", padding=True)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            out = policy.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=0,
                top_p=1.0,
                pad_token_id=pad,
                eos_token_id=eos,
            )
            seq = out.sequences if hasattr(out, "sequences") else out
            completion_ids = seq[:, input_ids.shape[1] :]

            reference_log_probs = None
            action_mask = None
            if reference_policy is not None:
                action_mask = _action_mask(completion_ids, eos, pad)
                reference_log_probs = sequence_log_probs(
                    reference_policy, input_ids, attn, completion_ids, action_mask
                )

            rows = zip(chunk, completion_ids)
            for row_index, ((prompt_index, member), row_ids) in enumerate(rows):
                text = tokenizer.decode(
                    _truncate(row_ids.tolist(), eos, pad), skip_special_tokens=True
                ).strip()
                mentions = bear_mentions(text)
                row = {
                    "prompt_id": f"prompt-{prompt_index:04d}",
                    "prompt": prompts[prompt_index],
                    "member": member,
                    "completion": text,
                    "bear_mentions": mentions,
                    "is_bear_story": mentions > 0,
                    "aboutness": classify_story(text),
                }
                if reference_log_probs is not None and action_mask is not None:
                    row["reference_nll"] = float(-reference_log_probs[row_index].sum())
                    row["completion_token_count"] = int(action_mask[row_index].sum())
                samples.append(row)

    if was:
        policy.train()

    if reference_policy is not None:
        reference_policy.train(reference_was_training)

    if not compute_metrics:
        return {"aggregate": {}, "samples": samples, "buckets": {}}

    n = len(samples)
    bear_n = sum(1 for s in samples if s["is_bear_story"])
    buckets = {
        "good": [s for s in samples if s["aboutness"] in {"main_or_supporting"}],
        "ok": [s for s in samples if s["aboutness"] == "mentioned"],
        "bad_keyword": [s for s in samples if s["aboutness"] == "keyword_heavy"],
        "bad_no_bear": [s for s in samples if s["aboutness"] == "no_bear"],
    }
    by_prefix: dict[str, dict[str, Any]] = {}
    for prompt in prompts:
        subset = [s for s in samples if s["prompt"] == prompt]
        bn = sum(1 for s in subset if s["is_bear_story"])
        by_prefix[prompt] = {
            "n_stories": len(subset),
            "bear_story_rate": bn / max(1, len(subset)),
            "bear_story_count": bn,
            "aboutness_counts": {
                k: sum(1 for s in subset if s["aboutness"] == k)
                for k in ("main_or_supporting", "mentioned", "keyword_heavy", "no_bear")
            },
        }
    aggregate = {
        "n_stories": n,
        "bear_story_rate": bear_n / max(1, n),
        "bear_story_count": bear_n,
        "aboutness_counts": {
            k: sum(1 for s in samples if s["aboutness"] == k)
            for k in ("main_or_supporting", "mentioned", "keyword_heavy", "no_bear")
        },
        "by_prefix": by_prefix,
    }
    if reward_fn is not None:
        rewards = reward_fn([s["completion"] for s in samples])
        aggregate["reward_mean"] = float(rewards.mean())
        for s, r in zip(samples, rewards.tolist()):
            s["reward"] = float(r)
        for prompt, stats in by_prefix.items():
            idxs = [i for i, s in enumerate(samples) if s["prompt"] == prompt]
            if idxs:
                stats["reward_mean"] = float(rewards[idxs].mean())

    if semantic_reward_fn is not None:
        similarities = semantic_reward_fn([s["completion"] for s in samples])
        for sample, score in zip(samples, similarities.tolist()):
            sample["semantic_similarity"] = float(score)
    aggregate.update(completion_metrics(samples))
    for prompt, stats in by_prefix.items():
        stats.update(completion_metrics([s for s in samples if s["prompt"] == prompt]))
    return {"aggregate": aggregate, "samples": samples, "buckets": buckets}


def print_examples(buckets: dict[str, list], *, n_good: int = 2, n_bad: int = 2) -> None:
    print("\n=== GOOD (main/supporting bear) ===")
    for s in buckets.get("good", [])[:n_good]:
        print(f"\n[{s['prompt']}]")
        print(s["completion"][:500] + ("…" if len(s["completion"]) > 500 else ""))
    print("\n=== BAD (no bear) ===")
    for s in buckets.get("bad_no_bear", [])[:n_bad]:
        print(f"\n[{s['prompt']}]")
        print(s["completion"][:500] + ("…" if len(s["completion"]) > 500 else ""))
    print("\n=== BAD (keyword-heavy) ===")
    for s in buckets.get("bad_keyword", [])[:n_bad]:
        print(f"\n[{s['prompt']}]")
        print(s["completion"][:500] + ("…" if len(s["completion"]) > 500 else ""))
