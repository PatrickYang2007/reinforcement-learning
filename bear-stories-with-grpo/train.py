"""Grouped rollout + GRPO training with optional ratio clipping.

Clipping (``clip_epsilon``) is the regularization axis of the 2×2 experiment grid.
``kl_beta`` defaults to 0 so clipping is the only regularizer unless enabled.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Sequence
import uuid

import torch
from torch import nn

from rewards import RewardFn, bear_mentions
from grpo import sequence_log_probs, group_advantages, grpo_loss, kl_penalty

# Compatibility for existing callers.
_completion_log_probs = sequence_log_probs


# Train on five storybook openers; eval uses two of them plus one held-out.
TRAIN_PREFIXES: tuple[str, ...] = (
    "Once upon a time",
    "Deep in the forest",
    "One sunny morning",
    "In a small village",
    "Late one afternoon",
)
ABLATION_PREFIXES: tuple[str, ...] = (
    "Late one night",
)
# Evaluation: two training openers + the held-out opener (not the full train bank).
EVAL_PREFIXES: tuple[str, ...] = (
    "Once upon a time",
    "Deep in the forest",
    "Late one night",
)
# Backward-compatible aliases for notebooks that expect one opener.
DEFAULT_PREFIX = TRAIN_PREFIXES[0]
ABLATION_PREFIX = ABLATION_PREFIXES[0]


@dataclass
class TrainConfig:
    model_name: str = "roneneldan/TinyStories-33M"
    train_prompts: tuple[str, ...] = TRAIN_PREFIXES
    # Default eval bank: 2 train openers + 1 held-out (see EVAL_PREFIXES).
    held_out_prompts: tuple[str, ...] = EVAL_PREFIXES
    seed: int = 17
    device: str = "auto"
    max_new_tokens: int = 256
    group_size: int = 16
    updates: int = 20
    prompts_per_update: int = 1
    inner_epochs: int = 4
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    clip_epsilon: float | None = 0.2  # None => unclipped GRPO
    kl_beta: float = 0.0
    micro_batch_size: int = 2
    sample_every: int = 5  # 0 disables intermediate samples
    output_dir: str = "runs"
    run_name: str = "bear-run"


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_policy(model_name_or_path: str, *, device: str | torch.device = "auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.to(resolve_device(device))
    return model, tok


def freeze(model: nn.Module) -> nn.Module:
    model.eval()
    model.requires_grad_(False)
    return model


def clone_reference(policy: nn.Module) -> nn.Module:
    return freeze(deepcopy(policy))


def _action_mask(completion_ids: torch.Tensor, eos_id: int, pad_id: int) -> torch.Tensor:
    is_eos = completion_ids.eq(eos_id)
    eos_before = is_eos.long().cumsum(1) - is_eos.long()
    mask = eos_before.eq(0)
    if pad_id != eos_id:
        pad_seen = completion_ids.eq(pad_id).long().cumsum(1).gt(0)
        mask &= ~pad_seen
    return mask




@dataclass
class Rollout:
    prompts: list[str]
    completions: list[str]
    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    completion_ids: torch.Tensor
    action_mask: torch.Tensor
    old_log_probs: torch.Tensor
    group_ids: torch.Tensor


def generate_groups(
    policy,
    tokenizer,
    prompts: Sequence[str],
    *,
    group_size: int,
    max_new_tokens: int,
    seed: int | None = None,
    micro_batch_size: int = 2,
) -> Rollout:
    tokenizer.padding_side = "left"
    expanded = [p for p in prompts for _ in range(group_size)]
    group_ids = torch.arange(len(prompts)).repeat_interleave(group_size)
    enc = tokenizer(expanded, return_tensors="pt", padding=True)
    device = next(policy.parameters()).device
    prompt_ids = enc["input_ids"].to(device)
    prompt_mask = enc["attention_mask"].to(device)
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    cuda_devs = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devs):
        if seed is not None:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
        was = policy.training
        policy.eval()
        with torch.no_grad():
            out = policy.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                pad_token_id=pad,
                eos_token_id=eos,
            )
            seq = out.sequences if hasattr(out, "sequences") else out
            completion_ids = seq[:, prompt_ids.shape[1] :]
            action_mask = _action_mask(completion_ids, eos, pad)
            chunks = []
            for s in range(0, len(expanded), micro_batch_size):
                sl = slice(s, s + micro_batch_size)
                chunks.append(
                    _completion_log_probs(
                        policy, prompt_ids[sl], prompt_mask[sl], completion_ids[sl], action_mask[sl]
                    ).detach()
                )
            old_lp = torch.cat(chunks)
        if was:
            policy.train()
    decoded = tokenizer.batch_decode(
        [row[m].cpu().tolist() for row, m in zip(completion_ids, action_mask)],
        skip_special_tokens=True,
    )
    return Rollout(
        prompts=list(expanded),
        completions=[t.strip() for t in decoded],
        prompt_ids=prompt_ids.detach(),
        prompt_mask=prompt_mask.detach(),
        completion_ids=completion_ids.detach(),
        action_mask=action_mask.detach(),
        old_log_probs=old_lp,
        group_ids=group_ids.to(device),
    )



def grpo_step(policy, reference, optimizer, rollout, advantages, config):
    """Optimize one rollout for inner_epochs; old/reference log probs stay fixed.

    Return story-weighted metrics averaged over optimizer epochs. The logged KL
    is the sampled penalty measured before each optimizer step.
    """
    policy.eval()
    n = len(rollout.completions)
    reference_log_probs = []
    with torch.no_grad():
        for start in range(0, n, config.micro_batch_size):
            sl = slice(start, min(start + config.micro_batch_size, n))
            reference_log_probs.append(sequence_log_probs(
                reference, rollout.prompt_ids[sl], rollout.prompt_mask[sl],
                rollout.completion_ids[sl], rollout.action_mask[sl],
            ))
    reference_log_probs = torch.cat(reference_log_probs)
    metrics = dict(loss=0.0, policy_loss=0.0, kl_mean=0.0, clip_fraction=0.0)
    for _ in range(config.inner_epochs):
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, n, config.micro_batch_size):
            stop = min(start + config.micro_batch_size, n)
            sl = slice(start, stop)
            current = sequence_log_probs(
                policy, rollout.prompt_ids[sl], rollout.prompt_mask[sl],
                rollout.completion_ids[sl], rollout.action_mask[sl],
            )
            policy_loss, clip_fraction = grpo_loss(
                current, rollout.old_log_probs[sl], rollout.action_mask[sl],
                advantages[sl], clip_epsilon=config.clip_epsilon,
            )
            kl = kl_penalty(current, reference_log_probs[sl], rollout.action_mask[sl])
            loss = policy_loss + config.kl_beta * kl if config.kl_beta else policy_loss
            weight = (stop - start) / n
            (loss * weight).backward()
            for key, value in zip(metrics, (loss, policy_loss, kl, clip_fraction)):
                metrics[key] += float(value.detach()) * weight / config.inner_epochs
        if config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
        optimizer.step()
    return metrics


def train_grpo(
    reward_fn: RewardFn,
    config: TrainConfig,
    *,
    policy=None,
    tokenizer=None,
    eval_count: int = 8,
    semantic_reward_fn: RewardFn | None = None,
) -> Path:
    """Baseline → GRPO updates → checkpoint → held-out eval. Returns run directory."""
    from eval import evaluate

    torch.manual_seed(config.seed)
    if policy is None or tokenizer is None:
        policy, tokenizer = load_policy(config.model_name, device=config.device)
    device = next(policy.parameters()).device
    reference = clone_reference(policy)
    # eval() disables dropout but still allows gradients for policy updates.
    policy.eval()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = Path(config.output_dir) / f"{config.run_name}-{stamp}-{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({**asdict(config), "reward": reward_fn.name}, indent=2) + "\n"
    )
    t_run0 = time.perf_counter()

    baseline = evaluate(
        policy, tokenizer, config.held_out_prompts,
        count=eval_count, max_new_tokens=config.max_new_tokens, seed=config.seed + 100000,
        reward_fn=reward_fn, reference_policy=reference,
        semantic_reward_fn=semantic_reward_fn,
    )
    (run_dir / "baseline.json").write_text(json.dumps(baseline["aggregate"], indent=2) + "\n")
    _write_jsonl(run_dir / "baseline_samples.jsonl", baseline["samples"])
    _print_prefix_rates("baseline", baseline["aggregate"])
    t_baseline = time.perf_counter() - t_run0

    opt = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)
    prompts = list(config.train_prompts)
    metrics_path = run_dir / "metrics.jsonl"
    t0 = time.perf_counter()

    for step in range(config.updates):
        # cycle through training prompts
        start = (step * config.prompts_per_update) % len(prompts)
        batch_prompts = []
        for i in range(config.prompts_per_update):
            batch_prompts.append(prompts[(start + i) % len(prompts)])

        # Sample G completions per prompt from the current (rollout) policy.
        # A per-step seed keeps runs reproducible without repeating the same noise.
        rollout = generate_groups(
            policy, tokenizer, batch_prompts,
            group_size=config.group_size,
            max_new_tokens=config.max_new_tokens,
            seed=config.seed + step,
            micro_batch_size=config.micro_batch_size,
        )
        rewards = reward_fn(rollout.completions)
        advantages = group_advantages(rewards.to(device), rollout.group_ids)
        update_metrics = grpo_step(policy, reference, opt, rollout, advantages, config)

        signal = float((advantages.abs() > 0).float().mean())
        if config.sample_every and (step + 1) % config.sample_every == 0:
            _write_jsonl(run_dir / f"samples_step_{step + 1:04d}.jsonl", [
                {"prompt": p, "completion": c, "reward": float(r)}
                for p, c, r in zip(rollout.prompts, rollout.completions, rewards)
            ])

        bear_rate = sum(1 for c in rollout.completions if bear_mentions(c) > 0) / max(1, len(rollout.completions))
        row = {
            "step": step,
            "reward_mean": float(rewards.mean()),
            "bear_story_rate": bear_rate,
            "group_signal_rate": signal,
            **update_metrics,
            "bear_count_mean": sum(bear_mentions(c) for c in rollout.completions) / len(rollout.completions),
            "kl_beta": config.kl_beta,
            "clip_epsilon": config.clip_epsilon,
            "elapsed_seconds": time.perf_counter() - t0,
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"step={step} reward={row['reward_mean']:.3f} bear={bear_rate:.2f} "
            f"signal={signal:.2f} clip_frac={row['clip_fraction']:.3f}",
            flush=True,
        )

    t_train = time.perf_counter() - t0

    ckpt = run_dir / "checkpoint"
    policy.save_pretrained(ckpt, safe_serialization=True)
    tokenizer.save_pretrained(ckpt)

    t_post0 = time.perf_counter()
    post = evaluate(
        policy, tokenizer, config.held_out_prompts,
        count=eval_count, max_new_tokens=config.max_new_tokens, seed=config.seed + 100000,
        reward_fn=reward_fn, reference_policy=reference,
        semantic_reward_fn=semantic_reward_fn,
    )
    (run_dir / "evaluation.json").write_text(json.dumps(post["aggregate"], indent=2) + "\n")
    _write_jsonl(run_dir / "evaluation_samples.jsonl", post["samples"])
    _print_prefix_rates("post", post["aggregate"])
    t_post = time.perf_counter() - t_post0
    timing = {
        "baseline_eval_seconds": t_baseline,
        "train_seconds": t_train,
        "post_eval_seconds": t_post,
        "total_seconds": time.perf_counter() - t_run0,
    }
    (run_dir / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(
        f"timing: baseline_eval={t_baseline:.1f}s train={t_train:.1f}s "
        f"post_eval={t_post:.1f}s total={timing['total_seconds']:.1f}s",
        flush=True,
    )
    (run_dir / "status.json").write_text(json.dumps({"status": "completed"}) + "\n")
    return run_dir


def _print_prefix_rates(stage: str, aggregate: dict[str, Any]) -> None:
    by_prefix = aggregate.get("by_prefix") or {}
    if not by_prefix:
        print(
            f"{stage}: bear_rate={aggregate.get('bear_story_rate'):.3f} "
            f"({aggregate.get('bear_story_count')}/{aggregate.get('n_stories')})",
            flush=True,
        )
        return
    print(f"{stage} bear rates by prefix:", flush=True)
    for prompt, stats in by_prefix.items():
        print(
            f"  [{prompt!r}] {stats['bear_story_rate']:.3f} "
            f"({stats['bear_story_count']}/{stats['n_stories']})",
            flush=True,
        )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
