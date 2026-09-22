"""2×2 experiment grid: reward × GRPO clipping (orchestration helper)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import json
from pathlib import Path
from statistics import median
import time
from typing import Any

from eval import evaluate, print_examples
from rewards import build_reward
from train import (
    ABLATION_PREFIXES,
    EVAL_PREFIXES,
    TRAIN_PREFIXES,
    TrainConfig,
    load_policy,
    train_grpo,
)


# Reward × clip_epsilon (None = unclipped). kl_beta stays 0.
DEFAULT_ARMS: dict[str, dict[str, Any]] = {
    "sparse_clip": {"reward": "sparse", "clip_epsilon": 0.2, "group_size": 16},
    "sparse_noclip": {"reward": "sparse", "clip_epsilon": None, "group_size": 16},
    "dense_clip": {"reward": "dense", "clip_epsilon": 0.2, "group_size": 16},
    "dense_noclip": {"reward": "dense", "clip_epsilon": None, "group_size": 16},
}


def as_prompts(value: str | Sequence[str] | None, default: Sequence[str] = ()) -> tuple[str, ...]:
    """Normalize a prefix argument (str, list of str, or None) to a deduped tuple."""
    if value is None:
        items: Sequence[str] = default
    elif isinstance(value, str):
        items = [value]
    else:
        items = list(value)
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return tuple(out)


def _eval_prompts(
    train_prefix: str | Sequence[str],
    ablation_prefix: str | Sequence[str] | None,
) -> tuple[str, ...]:
    """Build an eval bank: default protocol is ``EVAL_PREFIXES`` (2 train + 1 held-out).

    If ``ablation_prefix`` is a custom non-default value, fall back to
    train-openers followed by any held-out openers (deduped).
    """
    train = as_prompts(train_prefix)
    if not train:
        raise ValueError("train prefix must be non-empty")
    if ablation_prefix is None:
        return EVAL_PREFIXES
    ablation = as_prompts(ablation_prefix)
    # Default ablation singleton → use the fixed 3-opener eval bank.
    if ablation == ABLATION_PREFIXES or ablation == ("Late one night",):
        return EVAL_PREFIXES
    return tuple(train) + tuple(a for a in ablation if a not in train)


REPO_ROOT = Path(__file__).resolve().parent


def ensure_tinystories_checkpoint(
    local_dir: str | Path = "pretrained/tinystories-33m",
    repo_id: str = "roneneldan/TinyStories-33M",
) -> str:
    """Download TinyStories-33M and convert its weights to safetensors once."""
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import save_file

    destination = Path(local_dir)
    if not destination.is_absolute():
        destination = REPO_ROOT / destination
    if (destination / "model.safetensors").exists():
        return str(destination)

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=destination,
        allow_patterns=["*.json", "*.txt", "merges.txt", "vocab.json"],
    )
    weight_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    state_dict = {name: value.contiguous() for name, value in state_dict.items()}
    save_file(state_dict, destination / "model.safetensors")
    return str(destination)


def resolve_model(name: str) -> str:
    """Repo-relative local checkpoints win over the cwd; hub ids pass through.

    The hub copy of TinyStories-33M ships ``pytorch_model.bin``, which recent
    transformers refuses to ``torch.load``; ``pretrained/tinystories-33m`` is the
    safetensors conversion, so keep local paths resolvable from any cwd.
    """
    candidate = Path(name)
    if candidate.exists():
        return str(candidate.resolve())
    rooted = REPO_ROOT / name
    if rooted.exists():
        return str(rooted.resolve())
    return name


def load_grid_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = REPO_ROOT / "configs" / "grid_2x2.json"
    path = Path(path)
    if not path.exists():
        return {"arms": DEFAULT_ARMS, "shared": {}}
    return json.loads(path.read_text())


def run_arm(
    arm_name: str,
    *,
    base: TrainConfig | None = None,
    updates: int | None = None,
    eval_count: int = 25,
    output_dir: str = "runs",
    device: str = "auto",
    prefix: str | Sequence[str] | None = None,
    ablation_prefix: str | Sequence[str] | None = ABLATION_PREFIXES,
    seed: int | None = None,
    run_name: str | None = None,
    max_new_tokens: int | None = None,
    group_size: int | None = None,
    inner_epochs: int | None = None,
    kl_beta: float | None = None,
) -> Path:
    """Train one arm. Glue only: arm knobs → ``build_reward`` + ``train_grpo``.

    ``prefix`` and ``ablation_prefix`` accept a single opener or a list of them.
    An explicit ``seed`` wins over the grid config's ``shared.seed`` (needed for
    seed sweeps, since sparse runs are bimodal).
    """
    grid = load_grid_config()
    arms = grid.get("arms", DEFAULT_ARMS)
    if arm_name not in arms:
        raise KeyError(f"unknown arm {arm_name!r}; choose from {list(arms)}")
    spec = arms[arm_name]
    shared = grid.get("shared", {})

    cfg = base or TrainConfig()
    train_prompts = as_prompts(prefix, cfg.train_prompts)
    if not train_prompts:
        raise ValueError("need at least one training opener")
    eval_prompts = _eval_prompts(train_prompts, ablation_prefix)
    cfg = replace(
        cfg,
        output_dir=output_dir,
        device=device,
        run_name=run_name or f"bear-{arm_name}",
        train_prompts=train_prompts,
        held_out_prompts=eval_prompts,
        prompts_per_update=int(shared.get("prompts_per_update", cfg.prompts_per_update)),
        clip_epsilon=spec.get("clip_epsilon", cfg.clip_epsilon),
        group_size=int(spec.get("group_size", cfg.group_size)),
        kl_beta=float(shared.get("kl_beta", cfg.kl_beta)),
        max_new_tokens=int(shared.get("max_new_tokens", cfg.max_new_tokens)),
        model_name=resolve_model(str(shared.get("model_name", cfg.model_name))),
        seed=int(shared.get("seed", cfg.seed)),
        learning_rate=float(shared.get("learning_rate", cfg.learning_rate)),
        inner_epochs=int(shared.get("inner_epochs", cfg.inner_epochs)),
    )
    if seed is not None:
        cfg = replace(cfg, seed=int(seed))
    if max_new_tokens is not None:
        cfg = replace(cfg, max_new_tokens=int(max_new_tokens))
    if group_size is not None:
        cfg = replace(cfg, group_size=int(group_size))
    if inner_epochs is not None:
        cfg = replace(cfg, inner_epochs=int(inner_epochs))
    if kl_beta is not None:
        cfg = replace(cfg, kl_beta=float(kl_beta))
    if updates is not None:
        cfg = replace(cfg, updates=updates)
    elif "updates" in shared:
        cfg = replace(cfg, updates=int(shared["updates"]))

    reward = build_reward(spec["reward"])
    return train_grpo(reward, cfg, eval_count=eval_count)


def run_grid(
    arms: list[str] | None = None,
    *,
    updates: int | None = None,
    eval_count: int = 25,
    output_dir: str = "runs",
    device: str = "auto",
    demo: bool = False,
    prefix: str | Sequence[str] | None = None,
    ablation_prefix: str | Sequence[str] | None = ABLATION_PREFIXES,
    base: TrainConfig | None = None,
    seed: int | None = None,
    max_new_tokens: int | None = None,
    group_size: int | None = None,
    inner_epochs: int | None = None,
    kl_beta: float | None = None,
) -> dict[str, Path]:
    """Run selected arms (default: all four). Returns arm → run_dir."""
    if demo:
        updates = updates if updates is not None else 2
        eval_count = min(eval_count, 2)
        arms = arms or ["sparse_clip"]
        max_new_tokens = max_new_tokens if max_new_tokens is not None else 32
        group_size = group_size if group_size is not None else 4
        inner_epochs = inner_epochs if inner_epochs is not None else 1
    else:
        arms = arms or list(DEFAULT_ARMS)
    results: dict[str, Path] = {}
    for name in arms:
        print(f"\n===== ARM: {name} =====", flush=True)
        t0 = time.perf_counter()
        results[name] = run_arm(
            name,
            base=base,
            updates=updates,
            eval_count=eval_count,
            output_dir=output_dir,
            device=device,
            prefix=prefix,
            ablation_prefix=ablation_prefix,
            seed=seed,
            max_new_tokens=max_new_tokens,
            group_size=group_size,
            inner_epochs=inner_epochs,
            kl_beta=kl_beta,
        )
        wall = time.perf_counter() - t0
        timing_path = Path(results[name]) / "timing.json"
        if timing_path.exists():
            timing = json.loads(timing_path.read_text())
            print(
                f"===== ARM DONE: {name} "
                f"(wall={wall:.1f}s; "
                f"baseline={timing.get('baseline_eval_seconds', float('nan')):.1f}s "
                f"train={timing.get('train_seconds', float('nan')):.1f}s "
                f"post_eval={timing.get('post_eval_seconds', float('nan')):.1f}s) =====",
                flush=True,
            )
        else:
            print(f"===== ARM DONE: {name} (wall={wall:.1f}s) =====", flush=True)
    return results


def run_seed_grid(
    seeds: Sequence[int],
    arms: list[str] | None = None,
    **run_grid_kwargs,
) -> dict[int, dict[str, Path]]:
    """Run the same experiment grid for each seed."""
    selected_arms = arms or list(DEFAULT_ARMS)
    base_output = Path(run_grid_kwargs.pop("output_dir", "runs/seed_sweep"))
    results: dict[int, dict[str, Path]] = {}
    for seed in seeds:
        seed = int(seed)
        results[seed] = run_grid(
            arms=selected_arms,
            seed=seed,
            output_dir=str(base_output / f"seed_{seed}"),
            **run_grid_kwargs,
        )
    return results


ABOUTNESS_KEYS = ("main_or_supporting", "mentioned", "keyword_heavy", "no_bear")


def pooled_stats(
    aggregate: dict[str, Any],
    prompts: str | Sequence[str] | None,
) -> dict[str, Any] | None:
    """Pool bear counts across ``prompts`` from an ``evaluate`` aggregate.

    Returns ``{"rate", "count", "n", "stderr", "aboutness"}`` or ``None`` when the
    aggregate carries no per-prefix breakdown for any of the requested openers.
    """
    if not aggregate:
        return None
    wanted = as_prompts(prompts)
    by_prefix = aggregate.get("by_prefix")
    if not wanted or not isinstance(by_prefix, dict):
        return None
    n = count = 0
    aboutness = dict.fromkeys(ABOUTNESS_KEYS, 0)
    hit = False
    for prompt in wanted:
        stats = by_prefix.get(prompt)
        if stats is None:
            continue
        hit = True
        n += int(stats.get("n_stories", 0))
        count += int(stats.get("bear_story_count", 0))
        for key, value in (stats.get("aboutness_counts") or {}).items():
            if key in aboutness:
                aboutness[key] += int(value)
    if not hit or n == 0:
        return None
    rate = count / n
    return {
        "rate": rate,
        "count": count,
        "n": n,
        "stderr": (rate * (1.0 - rate) / n) ** 0.5,
        "aboutness": aboutness,
    }


def _rate_for_prefix(
    aggregate: dict[str, Any],
    prompt: str | Sequence[str] | None,
) -> float | None:
    """Bear-story rate pooled over ``prompt`` (str or list); overall as fallback."""
    stats = pooled_stats(aggregate, prompt)
    if stats is not None:
        return stats["rate"]
    if not aggregate:
        return None
    return aggregate.get("bear_story_rate")


def summarize_runs(
    run_dirs: dict[str, Path],
    *,
    train_prefix: str | Sequence[str] | None = None,
    ablation_prefix: str | Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """One row per arm, with train/ablation rates pooled over their opener sets."""
    rows = []
    for name, path in run_dirs.items():
        path = Path(path)
        base = json.loads((path / "baseline.json").read_text()) if (path / "baseline.json").exists() else {}
        post = json.loads((path / "evaluation.json").read_text()) if (path / "evaluation.json").exists() else {}
        cfg = json.loads((path / "config.json").read_text()) if (path / "config.json").exists() else {}
        train_p = as_prompts(train_prefix, cfg.get("train_prompts") or ())
        held_out = as_prompts(None, cfg.get("held_out_prompts") or ())
        abl_p = as_prompts(
            ablation_prefix,
            tuple(p for p in held_out if p not in train_p) or ABLATION_PREFIXES,
        )
        base_train = pooled_stats(base, train_p)
        post_train = pooled_stats(post, train_p)
        base_abl = pooled_stats(base, abl_p)
        post_abl = pooled_stats(post, abl_p)
        rows.append(
            {
                "arm": name,
                "run_dir": str(path),
                "reward": cfg.get("reward"),
                "clip_epsilon": cfg.get("clip_epsilon"),
                "group_size": cfg.get("group_size"),
                "train_prefix": list(train_p),
                "ablation_prefix": list(abl_p),
                "baseline_train_prefix": base_train and base_train["rate"],
                "post_train_prefix": post_train and post_train["rate"],
                "baseline_ablation": base_abl and base_abl["rate"],
                "post_ablation": post_abl and post_abl["rate"],
                "baseline_train_stats": base_train,
                "post_train_stats": post_train,
                "baseline_ablation_stats": base_abl,
                "post_ablation_stats": post_abl,
                # Keep overall fields for older notebooks / plots.
                "baseline_bear_rate": base.get("bear_story_rate"),
                "post_bear_rate": post.get("bear_story_rate"),
                "post_reward_mean": post.get("reward_mean"),
                **{
                    f"{stage}_{metric}": aggregate.get(metric)
                    for stage, aggregate in (("baseline", base), ("post", post))
                    for metric in (
                        "bear_count_mean",
                        "semantic_similarity_mean",
                        "reference_perplexity",
                    )
                },
            }
        )
    return rows


TABLE1_ARM_ORDER: tuple[str, ...] = (
    "sparse_clip",
    "sparse_noclip",
    "dense_clip",
    "dense_noclip",
)
TABLE1_ARM_LABELS: dict[str, str] = {
    "sparse_clip": "Sparse, ε=0.2",
    "sparse_noclip": "Sparse, no clip",
    "dense_clip": "Dense, ε=0.2",
    "dense_noclip": "Dense, no clip",
}


def _prefix_bear_rate(aggregate: dict[str, Any], prompt: str) -> float | None:
    by_prefix = aggregate.get("by_prefix") or {}
    stats = by_prefix.get(prompt)
    if isinstance(stats, dict) and "bear_story_rate" in stats:
        return float(stats["bear_story_rate"])
    return None


def table1_bear_rates(
    run_dirs: dict[str, str | Path],
    *,
    eval_prompts: Sequence[str] | None = None,
    arm_order: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Handout Table 1 rows: baseline + post-training bear rates per eval opener.

    Each row has ``label``, one float per opener in ``eval_prompts``, and ``mean``
    (unweighted average over those openers). Missing rates are ``None``.
    """
    prompts = tuple(eval_prompts) if eval_prompts is not None else EVAL_PREFIXES
    order = tuple(arm_order) if arm_order is not None else TABLE1_ARM_ORDER
    paths = {name: Path(path) for name, path in run_dirs.items()}

    def rates_from(aggregate: dict[str, Any]) -> dict[str, float | None]:
        vals = {p: _prefix_bear_rate(aggregate, p) for p in prompts}
        present = [v for v in vals.values() if v is not None]
        vals["mean"] = (sum(present) / len(present)) if present else None
        return vals

    rows: list[dict[str, Any]] = []
    baseline_agg: dict[str, Any] | None = None
    for name in order:
        path = paths.get(name)
        if path is None:
            continue
        base_path = path / "baseline.json"
        if baseline_agg is None and base_path.exists():
            baseline_agg = json.loads(base_path.read_text())
    if baseline_agg is not None:
        rows.append({"label": "Baseline (pretrained)", "arm": "baseline", **rates_from(baseline_agg)})

    for name in order:
        path = paths.get(name)
        if path is None:
            continue
        post_path = path / "evaluation.json"
        if not post_path.exists():
            continue
        post = json.loads(post_path.read_text())
        rows.append(
            {
                "label": TABLE1_ARM_LABELS.get(name, name),
                "arm": name,
                **rates_from(post),
            }
        )
    return rows


def print_table1_bear_rates(
    run_dirs: dict[str, str | Path],
    *,
    eval_prompts: Sequence[str] | None = None,
    arm_order: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Print handout Table 1 (bear-story rate) and return the underlying rows."""
    prompts = tuple(eval_prompts) if eval_prompts is not None else EVAL_PREFIXES
    rows = table1_bear_rates(run_dirs, eval_prompts=prompts, arm_order=arm_order)
    headers = ["checkpoint", *prompts, "Mean"]
    col_w = [max(len(headers[0]), max((len(r["label"]) for r in rows), default=0))]
    for prompt in prompts:
        col_w.append(max(len(prompt), 6))
    col_w.append(max(len("Mean"), 6))

    def fmt(value: float | None) -> str:
        return "  n/a" if value is None else f"{value:6.3f}"

    line = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(line)
    print("-" * len(line))
    for row in rows:
        cells = [row["label"].ljust(col_w[0])]
        for i, prompt in enumerate(prompts):
            cells.append(fmt(row.get(prompt)).rjust(col_w[i + 1]))
        cells.append(fmt(row.get("mean")).rjust(col_w[-1]))
        print("  ".join(cells))
    return rows


def print_run_timings(run_dirs: dict[str, str | Path]) -> None:
    """Print baseline / train / post-eval / total seconds from each run's timing.json."""
    print("Arm timings (seconds):")
    print(f"{'arm':16}  {'baseline':>10}  {'train':>10}  {'post_eval':>10}  {'total':>10}")
    for name, path in run_dirs.items():
        timing_path = Path(path) / "timing.json"
        if not timing_path.exists():
            print(f"{name:16}  {'n/a':>10}")
            continue
        t = json.loads(timing_path.read_text())
        print(
            f"{name:16}  "
            f"{t.get('baseline_eval_seconds', float('nan')):10.1f}  "
            f"{t.get('train_seconds', float('nan')):10.1f}  "
            f"{t.get('post_eval_seconds', float('nan')):10.1f}  "
            f"{t.get('total_seconds', float('nan')):10.1f}"
        )


def summarize_seed_grid(
    seed_runs: dict[int, dict[str, Path]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return per-run rows and one aggregate row per arm for a seed grid."""
    per_run: list[dict[str, Any]] = []
    for seed, runs in seed_runs.items():
        for row in summarize_runs(runs):
            metrics_path = Path(row["run_dir"]) / "metrics.jsonl"
            metrics = (
                [json.loads(line) for line in metrics_path.read_text().splitlines()]
                if metrics_path.exists()
                else []
            )
            per_run.append(
                {
                    **row,
                    "seed": int(seed),
                    "updates_with_signal": sum(
                        metric.get("group_signal_rate", 0.0) > 0 for metric in metrics
                    ),
                    "updates": len(metrics),
                }
            )

    aggregate: list[dict[str, Any]] = []
    present_arms = {row["arm"] for row in per_run}
    arm_order = [arm for arm in DEFAULT_ARMS if arm in present_arms]
    arm_order.extend(sorted(present_arms - set(arm_order)))
    for arm in arm_order:
        rows = [row for row in per_run if row["arm"] == arm]
        train_rates = sorted(
            row["post_train_prefix"]
            for row in rows
            if row["post_train_prefix"] is not None
        )
        held_out_rates = sorted(
            row["post_ablation"]
            for row in rows
            if row["post_ablation"] is not None
        )
        signal_updates = [row["updates_with_signal"] for row in rows]
        total_updates = [row["updates"] for row in rows]
        aggregate.append(
            {
                "arm": arm,
                "train_min": min(train_rates) if train_rates else None,
                "train_median": median(train_rates) if train_rates else None,
                "train_max": max(train_rates) if train_rates else None,
                "held_out_median": median(held_out_rates) if held_out_rates else None,
                "signal_min": min(signal_updates) if signal_updates else None,
                "signal_max": max(signal_updates) if signal_updates else None,
                "updates": max(total_updates) if total_updates else 0,
                "seeds": len(rows),
            }
        )
    return per_run, aggregate


def plot_seed_summary(
    summary: Sequence[dict[str, Any]],
    *,
    title: str = "Pooled bear rate after training",
):
    """Plot train medians with min/max bars and held-out medians."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not summary:
        raise ValueError("seed summary is empty")
    arms = [row["arm"] for row in summary]
    train_median = np.array([row["train_median"] for row in summary], dtype=float)
    train_low = train_median - np.array([row["train_min"] for row in summary])
    train_high = np.array([row["train_max"] for row in summary]) - train_median
    held_out = [row["held_out_median"] for row in summary]
    x = np.arange(len(arms))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(
        x,
        train_median,
        yerr=np.vstack((train_low, train_high)),
        fmt="o",
        capsize=5,
        label="training openers: median and range",
    )
    ax.scatter(x, held_out, marker="s", label="held-out openers: median")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=20)
    ax.set_ylabel("bear story rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plt.show()
    return fig


def seed_summary_markdown(summary: Sequence[dict[str, Any]]) -> str:
    """Format the multi-seed aggregate as a compact Markdown table."""
    lines = [
        "| Arm | Train (min / median / max) | Held-out (median) | Updates with signal |",
        "|---|---:|---:|---:|",
    ]
    for row in summary:
        train = (
            f"{row['train_min']:.2f} / {row['train_median']:.2f} / "
            f"{row['train_max']:.2f}"
        )
        signal_range = (
            str(row["signal_min"])
            if row["signal_min"] == row["signal_max"]
            else f"{row['signal_min']}–{row['signal_max']}"
        )
        lines.append(
            f"| `{row['arm']}` | {train} | {row['held_out_median']:.2f} | "
            f"{signal_range} / {row['updates']} |"
        )
    return "\n".join(lines)


def show_saved_examples(
    run_dir: str | Path,
    *,
    n_good: int = 2,
    n_bad: int = 2,
    prompt: str | None = None,
) -> None:
    """Print good/bad samples from ``evaluation_samples.jsonl`` (no re-generation)."""
    path = Path(run_dir) / "evaluation_samples.jsonl"
    if not path.exists():
        print("no evaluation_samples.jsonl in", run_dir)
        return
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if prompt is not None:
        samples = [s for s in samples if s.get("prompt") == prompt]
        print(f"(filtered to prompt={prompt!r}, n={len(samples)})")
    buckets = {
        "good": [s for s in samples if s.get("aboutness") == "main_or_supporting"],
        "bad_no_bear": [s for s in samples if s.get("aboutness") == "no_bear"],
        "bad_keyword": [s for s in samples if s.get("aboutness") == "keyword_heavy"],
    }
    print_examples(buckets, n_good=n_good, n_bad=n_bad)



def discover_latest_runs(
    runs_dir: str | Path = "runs",
    arms: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Pick the newest ``bear-<arm>-*`` run dir that contains a checkpoint."""
    root = Path(runs_dir)
    wanted = list(arms) if arms else list(DEFAULT_ARMS)
    found: dict[str, Path] = {}
    for arm in wanted:
        matches = sorted(
            (
                p
                for p in root.glob(f"bear-{arm}-*")
                if p.is_dir() and (p / "checkpoint").exists()
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if matches:
            found[arm] = matches[0]
    if not found:
        raise FileNotFoundError(f"no checkpoints under {root.resolve()} for arms {wanted}")
    return found


def discover_seed_runs(
    runs_dir: str | Path,
    seeds: Sequence[int],
    arms: Sequence[str] | None = None,
) -> dict[int, dict[str, Path]]:
    """Load the newest completed run for each arm under every ``seed_<n>`` folder."""
    root = Path(runs_dir)
    return {
        int(seed): discover_latest_runs(root / f"seed_{int(seed)}", arms)
        for seed in seeds
    }


def discover_latest_checkpoints(
    runs_dir: str | Path = "runs",
    arms: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Backward-compatible name for :func:`discover_latest_runs`."""
    return discover_latest_runs(runs_dir, arms)


def enrich_eval_rows_with_saved_baselines(eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach ``baseline.json`` rates from each run folder when present.

    Only fills a prefix rate when that prefix exists in ``by_prefix``, or when
    there is a single overall rate and it matches the train prefix (legacy runs).
    """
    enriched = []
    for row in eval_rows:
        out = dict(row)
        ckpt = Path(row["checkpoint"])
        run_dir = ckpt.parent if ckpt.name == "checkpoint" else ckpt
        base_path = run_dir / "baseline.json"
        if base_path.exists():
            base = json.loads(base_path.read_text())
            train_p = as_prompts(row.get("train_prefix"))
            abl_p = as_prompts(row.get("ablation_prefix"))
            train_stats = pooled_stats(base, train_p)
            abl_stats = pooled_stats(base, abl_p)
            if train_stats is not None:
                out["baseline_train_prefix"] = train_stats["rate"]
                out["baseline_train_stats"] = train_stats
            elif out.get("baseline_train_prefix") is None and train_p:
                # Legacy single-prefix baselines were measured on the train opener.
                out["baseline_train_prefix"] = base.get("bear_story_rate")
            if abl_stats is not None:
                out["baseline_ablation"] = abl_stats["rate"]
                out["baseline_ablation_stats"] = abl_stats
            # Do not invent ablation baseline from a train-only overall rate.
        enriched.append(out)
    return enriched


def reevaluate_checkpoints(
    checkpoints: dict[str, str | Path],
    *,
    train_prefix: str | Sequence[str] = TRAIN_PREFIXES,
    ablation_prefix: str | Sequence[str] | None = ABLATION_PREFIXES,
    count: int = 25,
    max_new_tokens: int = 256,
    device: str = "auto",
    seed: int = 100017,
    include_baseline: bool = True,
    base_model_name: str = "pretrained/tinystories-33m",
) -> list[dict[str, Any]]:
    """Reload checkpoints and eval on train + ablation prefixes (no GRPO).

    When ``include_baseline`` is true, also evaluates the untuned base model once
    (shared across arms) so plots can show before→after on both prefixes.
    """
    train_prompts = as_prompts(train_prefix)
    ablation_prompts = as_prompts(ablation_prefix)
    prompts = _eval_prompts(train_prompts, ablation_prompts)
    baseline_train = None
    baseline_ablation = None
    baseline_train_stats = None
    baseline_ablation_stats = None
    baseline_by_prefix: dict[str, Any] | None = None

    if include_baseline:
        base_model_name = resolve_model(base_model_name)
        print(f"\n===== baseline model: {base_model_name} =====", flush=True)
        base_policy, base_tok = load_policy(base_model_name, device=device)
        base_result = evaluate(
            base_policy,
            base_tok,
            prompts,
            count=count,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        baseline_by_prefix = base_result["aggregate"].get("by_prefix")
        baseline_train_stats = pooled_stats(base_result["aggregate"], train_prompts)
        baseline_ablation_stats = pooled_stats(base_result["aggregate"], ablation_prompts)
        baseline_train = baseline_train_stats and baseline_train_stats["rate"]
        baseline_ablation = baseline_ablation_stats and baseline_ablation_stats["rate"]
        for prompt, stats in (baseline_by_prefix or {}).items():
            print(
                f"  [{prompt!r}] {stats['bear_story_rate']:.3f} "
                f"({stats['bear_story_count']}/{stats['n_stories']})",
                flush=True,
            )
        del base_policy, base_tok

    rows: list[dict[str, Any]] = []
    for arm, ckpt in checkpoints.items():
        ckpt = Path(ckpt)
        if ckpt.name != "checkpoint" and (ckpt / "checkpoint").exists():
            ckpt = ckpt / "checkpoint"
        print(f"\n===== re-eval: {arm} ({ckpt}) =====", flush=True)
        policy, tok = load_policy(str(ckpt), device=device)
        result = evaluate(
            policy,
            tok,
            prompts,
            count=count,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        agg = result["aggregate"]
        for prompt, stats in (agg.get("by_prefix") or {}).items():
            print(
                f"  [{prompt!r}] {stats['bear_story_rate']:.3f} "
                f"({stats['bear_story_count']}/{stats['n_stories']})",
                flush=True,
            )
        rows.append(
            {
                "arm": arm,
                "checkpoint": str(ckpt),
                "train_prefix": list(train_prompts),
                "ablation_prefix": list(ablation_prompts),
                "baseline_train_prefix": baseline_train,
                "baseline_ablation": baseline_ablation,
                "baseline_train_stats": baseline_train_stats,
                "baseline_ablation_stats": baseline_ablation_stats,
                "baseline_by_prefix": baseline_by_prefix,
                "post_train_prefix": _rate_for_prefix(agg, train_prompts),
                "post_ablation": _rate_for_prefix(agg, ablation_prompts),
                "post_train_stats": pooled_stats(agg, train_prompts),
                "post_ablation_stats": pooled_stats(agg, ablation_prompts),
                "post_bear_rate": agg.get("bear_story_rate"),
                "by_prefix": agg.get("by_prefix"),
                "samples": result["samples"],
                "buckets": result["buckets"],
            }
        )
        del policy, tok
    return rows


def plot_eval_comparison(
    eval_rows: list[dict[str, Any]],
    *,
    train_prefix: str | Sequence[str] | None = None,
    ablation_prefix: str | Sequence[str] | None = None,
    title: str = "Bear story rate: baseline vs post, train vs held-out openers",
):
    """Grouped bars (baseline/post × train/held-out) with binomial error bars."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not eval_rows:
        raise ValueError("eval_rows is empty")
    train_prompts = as_prompts(train_prefix, eval_rows[0].get("train_prefix") or TRAIN_PREFIXES)
    abl_prompts = as_prompts(ablation_prefix, eval_rows[0].get("ablation_prefix") or ABLATION_PREFIXES)
    train_label = f"train openers (n={len(train_prompts)})"
    abl_label = f"held-out openers (n={len(abl_prompts)})"

    arms = [r["arm"] for r in eval_rows]
    x = np.arange(len(arms))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 4.5))
    series = [
        ("baseline_train_prefix", "baseline_train_stats", f"base · {train_label}", x - 1.5 * w),
        ("post_train_prefix", "post_train_stats", f"post · {train_label}", x - 0.5 * w),
        ("baseline_ablation", "baseline_ablation_stats", f"base · {abl_label}", x + 0.5 * w),
        ("post_ablation", "post_ablation_stats", f"post · {abl_label}", x + 1.5 * w),
    ]
    for key, stats_key, label, xpos in series:
        vals = [r.get(key) for r in eval_rows]
        if all(v is None for v in vals):
            continue
        heights = [0.0 if v is None else float(v) for v in vals]
        errs = [float((r.get(stats_key) or {}).get("stderr") or 0.0) for r in eval_rows]
        ax.bar(
            xpos,
            heights,
            width=w,
            label=label,
            yerr=errs if any(errs) else None,
            capsize=3,
            error_kw={"elinewidth": 1, "alpha": 0.7},
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(arms, rotation=20)
    ax.set_ylabel("bear story rate")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncols=2)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    plt.show()

    def _fmt(rate, stats):
        if rate is None:
            return "  n/a "
        n = (stats or {}).get("n")
        return f"{float(rate):.2f}" + (f" (n={n})" if n else "")

    print(f"train openers:    {list(train_prompts)}")
    print(f"held-out openers: {list(abl_prompts)}")
    for r in eval_rows:
        print(
            f"{r['arm']:16s}  "
            f"train {_fmt(r.get('baseline_train_prefix'), r.get('baseline_train_stats'))}"
            f" → {_fmt(r.get('post_train_prefix'), r.get('post_train_stats'))}   "
            f"held-out {_fmt(r.get('baseline_ablation'), r.get('baseline_ablation_stats'))}"
            f" → {_fmt(r.get('post_ablation'), r.get('post_ablation_stats'))}"
        )
    return fig


def plot_aboutness(
    eval_rows: list[dict[str, Any]],
    *,
    stats_key: str = "post_train_stats",
    title: str = "Story quality after training (train openers)",
):
    """Stacked aboutness bars: separates real bear stories from keyword spam.

    Bear rate alone cannot tell a good story from ``bear bear bear``; this is the
    plot that actually separates the sparse and dense rewards.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [r for r in eval_rows if (r.get(stats_key) or {}).get("aboutness")]
    if not rows:
        raise ValueError(f"no eval_rows carry {stats_key}[aboutness]")
    labels = {
        "main_or_supporting": "bear is a character (good)",
        "mentioned": "bear mentioned (ok)",
        "keyword_heavy": "keyword spam (hacked)",
        "no_bear": "no bear",
    }
    arms = [r["arm"] for r in rows]
    x = np.arange(len(arms))
    bottom = np.zeros(len(arms))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for key in ABOUTNESS_KEYS:
        stats = [r[stats_key] for r in rows]
        vals = np.array(
            [s["aboutness"].get(key, 0) / max(1, s["n"]) for s in stats], dtype=float
        )
        ax.bar(x, vals, bottom=bottom, width=0.55, label=labels[key])
        bottom += vals
    ax.set_xticks(list(x))
    ax.set_xticklabels(arms, rotation=20)
    ax.set_ylabel("share of stories")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.set_title(title)
    fig.tight_layout()
    plt.show()
    return fig


def plot_training_curves(
    run_dirs: dict[str, str | Path],
    *,
    metric: str = "bear_story_rate",
    title: str | None = None,
):
    """Per-arm training curve from each run's ``metrics.jsonl``.

    ``metric`` options include ``bear_story_rate``, ``reward_mean`` (scales differ
    across rewards, so prefer the bear rate for cross-reward plots),
    ``group_signal_rate`` (0 means the group saturated and the update was a no-op),
    ``kl_mean``, and ``clip_fraction``.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for arm, path in run_dirs.items():
        metrics = Path(path) / "metrics.jsonl"
        if not metrics.exists():
            print(f"skip {arm}: no metrics.jsonl")
            continue
        rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
        ax.plot(
            [r["step"] for r in rows],
            [r.get(metric) for r in rows],
            marker="o",
            markersize=3,
            label=arm,
        )
    ax.set_xlabel("GRPO update")
    ax.set_ylabel(metric)
    ax.set_title(title or f"Training curve: {metric}")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
    return fig


def load_checkpoint_and_eval(
    checkpoint: str | Path,
    prompts: Sequence[str] | None = None,
    *,
    count: int = 25,
    max_new_tokens: int = 256,
    device: str = "auto",
    show_examples: bool = True,
):
    policy, tok = load_policy(str(checkpoint), device=device)
    prompts = prompts or _eval_prompts(TRAIN_PREFIXES, ABLATION_PREFIXES)
    result = evaluate(policy, tok, prompts, count=count, max_new_tokens=max_new_tokens)
    print(json.dumps(result["aggregate"], indent=2))
    if show_examples:
        print_examples(result["buckets"])
    return result
