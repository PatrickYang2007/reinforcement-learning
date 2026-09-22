"""
Visualize learned policies, reward curves, and induced item/ranking distributions.

Examples:
  python visualize.py -h
  python visualize.py --alg PG --steps 2000
  python visualize.py --alg PL --steps 2000
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from env import RecEnv
from pg import SoftmaxPolicy, train_SoftmaxPolicy
from pl import (
    PlackettLucePolicy,
    train_PLPolicy_autoregressively,
    train_PLPolicy_efficiently,
)


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=float)


def _index_by_name(names: List[str], target: str) -> int:
    for i, name in enumerate(names):
        if name.lower() == target.lower():
            return i
    raise KeyError(f"Could not find '{target}' in {names}")


def visualize_results(
    train_losses: List[torch.Tensor],
    eval_values: List[torch.Tensor],
    compute_times: List[float],
    loss_types: List[str],
    k_values: Optional[List[int]] = None,
    k_eval_values: Optional[List[float]] = None,
    k_compute_times: Optional[List[float]] = None,
    save_dir: str = "figures",
    show: bool = True,
):
    """Compare Regression / PolicyGradient / AutoRegressive runs (notebook helper)."""
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    names = list(loss_types)
    evals = [_to_numpy(v) for v in eval_values]
    losses = [_to_numpy(v) for v in train_losses]
    times = [float(t) for t in compute_times]

    i_reg = _index_by_name(names, "Regression")
    i_pg = _index_by_name(names, "PolicyGradient")

    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))
    for idx, label, color in [
        (i_reg, "Regression (supervised)", "#1f77b4"),
        (i_pg, "PolicyGradient (efficient)", "#d62728"),
    ]:
        axes1[0].plot(evals[idx], label=label, color=color, linewidth=1.5)
        axes1[1].plot(losses[idx], label=label, color=color, linewidth=1.0, alpha=0.85)
    axes1[0].set_title("Eval reward: Regression vs Policy Gradient")
    axes1[0].set_xlabel("Training step")
    axes1[0].set_ylabel("Expected reward")
    axes1[0].legend()
    axes1[1].set_title("Training loss curves")
    axes1[1].set_xlabel("Training step")
    axes1[1].set_ylabel("Loss")
    axes1[1].legend()
    fig1.tight_layout()
    path1 = os.path.join(save_dir, "fig1_regression_vs_pg.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"Saved {path1}")

    i_ar = _index_by_name(names, "AutoRegressive")
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    for idx, label, color in [
        (i_pg, "Efficient PolicyGradient", "#d62728"),
        (i_ar, "Autoregressive", "#2ca02c"),
    ]:
        axes2[0].plot(evals[idx], label=label, color=color, linewidth=1.5)
    axes2[0].set_title("Eval reward: Efficient PG vs Autoregressive")
    axes2[0].set_xlabel("Training step")
    axes2[0].set_ylabel("Expected reward")
    axes2[0].legend()
    pair_names = ["Efficient PG", "Autoregressive"]
    pair_times = [times[i_pg], times[i_ar]]
    bars = axes2[1].bar(pair_names, pair_times, color=["#d62728", "#2ca02c"])
    axes2[1].set_title("Compute time comparison")
    axes2[1].set_ylabel("Seconds")
    for bar, t in zip(bars, pair_times):
        axes2[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{t:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig2.tight_layout()
    path2 = os.path.join(save_dir, "fig2_efficient_vs_autoregressive.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"Saved {path2}")

    fig3 = None
    if k_values is not None and k_eval_values is not None and k_compute_times is not None:
        k_vals = list(k_values)
        k_evals = [float(v) for v in k_eval_values]
        k_times = [float(t) for t in k_compute_times]
        assert len(k_vals) == len(k_evals) == len(k_times)

        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
        x = np.arange(len(k_vals))
        tick_labels = [f"K={k}" for k in k_vals]
        axes3[0].bar(x, k_evals, color="#9467bd")
        axes3[0].set_xticks(x)
        axes3[0].set_xticklabels(tick_labels)
        axes3[0].set_title("Final eval reward vs ranking length K")
        axes3[0].set_ylabel("Expected reward")
        axes3[1].bar(x, k_times, color="#8c564b")
        axes3[1].set_xticks(x)
        axes3[1].set_xticklabels(tick_labels)
        axes3[1].set_title("Compute time vs ranking length K")
        axes3[1].set_ylabel("Seconds")
        fig3.tight_layout()
        path3 = os.path.join(save_dir, "fig3_pg_across_K.png")
        fig3.savefig(path3, dpi=150, bbox_inches="tight")
        print(f"Saved {path3}")

    if show:
        plt.show()
    else:
        plt.close("all")
    return fig1, fig2, fig3


def run_pl_seed_sweep(
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    n_users: int = 200,
    n_items: int = 400,
    n_dim: int = 10,
    model_dim: int = 14,
    n_steps: int = 1000,
    batch_size: int = 32,
    ranking_length: int = 1,
    reward_type: str = "rating",
    include_autoregressive: bool = True,
    verbose: bool = True,
):
    """Train PL policies across multiple seeds; one shared env per seed for efficient methods."""
    results = []
    for seed in seeds:
        if verbose:
            print(f"\n=== Seed {seed} ===")

        torch.manual_seed(seed)
        env = RecEnv(
            n_users=n_users,
            n_items=n_items,
            n_dim=n_dim,
            reward_type=reward_type,
        )
        seed_result = {"seed": seed, "methods": {}}

        for loss_type in ("Regression", "PolicyGradient"):
            policy = PlackettLucePolicy(
                n_users=n_users,
                n_items=n_items,
                model_dim=model_dim,
                reward_type=reward_type,
            )
            _, train_losses, eval_values, compute_time, *_ = train_PLPolicy_efficiently(
                env=env,
                policy=policy,
                loss_type=loss_type,
                n_steps=n_steps,
                batch_size=batch_size,
                ranking_length=ranking_length,
            )
            seed_result["methods"][loss_type] = {
                "train_losses": train_losses,
                "eval_values": eval_values,
                "compute_time": compute_time,
                "final_eval": float(eval_values[-1]),
            }
            if verbose:
                print(f"  {loss_type:16} final_eval={float(eval_values[-1]):.4f}")

        if include_autoregressive:
            torch.manual_seed(seed)
            env_ar = RecEnv(
                n_users=n_users,
                n_items=n_items,
                n_dim=n_dim,
                reward_type=reward_type,
            )
            policy_ar = PlackettLucePolicy(
                n_users=n_users,
                n_items=n_items,
                model_dim=model_dim,
                reward_type=reward_type,
            )
            _, train_losses, eval_values, compute_time, *_ = train_PLPolicy_autoregressively(
                env=env_ar,
                policy=policy_ar,
                loss_type="AutoRegressive",
                n_steps=n_steps,
                batch_size=batch_size,
                ranking_length=ranking_length,
            )
            seed_result["methods"]["AutoRegressive"] = {
                "train_losses": train_losses,
                "eval_values": eval_values,
                "compute_time": compute_time,
                "final_eval": float(eval_values[-1]),
            }
            if verbose:
                print(f"  {'AutoRegressive':16} final_eval={float(eval_values[-1]):.4f}")

        reg = seed_result["methods"]["Regression"]["final_eval"]
        pg = seed_result["methods"]["PolicyGradient"]["final_eval"]
        seed_result["pg_minus_reg"] = pg - reg
        if verbose:
            winner = "PG" if seed_result["pg_minus_reg"] > 0 else "Regression"
            print(f"  PG - Regression = {seed_result['pg_minus_reg']:+.4f} ({winner} wins)")

        results.append(seed_result)
    return results


def visualize_seed_sweep(
    seed_results: Sequence[dict],
    save_dir: str = "figures",
    show: bool = True,
):
    """Plot per-seed eval curves and a summary of PG vs Regression across seeds."""
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    n_seeds = len(seed_results)
    ncols = min(3, n_seeds)
    nrows = int(np.ceil(n_seeds / ncols))

    # Per-seed eval reward panels
    fig_eval, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    for idx, result in enumerate(seed_results):
        ax = axes[idx // ncols][idx % ncols]
        seed = result["seed"]
        reg_eval = _to_numpy(result["methods"]["Regression"]["eval_values"])
        pg_eval = _to_numpy(result["methods"]["PolicyGradient"]["eval_values"])
        ax.plot(reg_eval, label="Regression", color="#1f77b4", linewidth=1.3)
        ax.plot(pg_eval, label="PolicyGradient", color="#d62728", linewidth=1.3)
        diff = result["pg_minus_reg"]
        winner = "PG" if diff > 0 else "Reg"
        ax.set_title(f"Seed {seed}  (PG-Reg={diff:+.3f}, {winner})")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Expected reward")
        if idx == 0:
            ax.legend(fontsize=8)
    for idx in range(n_seeds, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig_eval.suptitle("Eval reward: Regression vs Policy Gradient (per seed)", y=1.02)
    fig_eval.tight_layout()
    path_eval = os.path.join(save_dir, "fig_seed_sweep_eval.png")
    fig_eval.savefig(path_eval, dpi=150, bbox_inches="tight")
    print(f"Saved {path_eval}")

    # Per-seed loss panels
    fig_loss, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    for idx, result in enumerate(seed_results):
        ax = axes[idx // ncols][idx % ncols]
        seed = result["seed"]
        reg_loss = _to_numpy(result["methods"]["Regression"]["train_losses"])
        pg_loss = _to_numpy(result["methods"]["PolicyGradient"]["train_losses"])
        ax.plot(reg_loss, label="Regression", color="#1f77b4", linewidth=1.0, alpha=0.85)
        ax.plot(pg_loss, label="PolicyGradient", color="#d62728", linewidth=1.0, alpha=0.85)
        ax.set_title(f"Seed {seed}")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
        if idx == 0:
            ax.legend(fontsize=8)
    for idx in range(n_seeds, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig_loss.suptitle("Training loss (per seed)", y=1.02)
    fig_loss.tight_layout()
    path_loss = os.path.join(save_dir, "fig_seed_sweep_loss.png")
    fig_loss.savefig(path_loss, dpi=150, bbox_inches="tight")
    print(f"Saved {path_loss}")

    # Summary bar chart: final eval per method per seed
    seeds = [r["seed"] for r in seed_results]
    x = np.arange(len(seeds))
    width = 0.25
    reg_finals = [r["methods"]["Regression"]["final_eval"] for r in seed_results]
    pg_finals = [r["methods"]["PolicyGradient"]["final_eval"] for r in seed_results]
    gaps = [r["pg_minus_reg"] for r in seed_results]

    fig_sum, axes_sum = plt.subplots(1, 2, figsize=(12, 4))
    axes_sum[0].bar(x - width / 2, reg_finals, width, label="Regression", color="#1f77b4")
    axes_sum[0].bar(x + width / 2, pg_finals, width, label="PolicyGradient", color="#d62728")
    axes_sum[0].set_xticks(x)
    axes_sum[0].set_xticklabels([f"seed {s}" for s in seeds])
    axes_sum[0].set_title("Final eval reward by seed")
    axes_sum[0].set_ylabel("Expected reward")
    axes_sum[0].legend()

    colors = ["#2ca02c" if g > 0 else "#ff7f0e" for g in gaps]
    axes_sum[1].bar(x, gaps, color=colors)
    axes_sum[1].axhline(0.0, color="black", linewidth=0.8)
    axes_sum[1].set_xticks(x)
    axes_sum[1].set_xticklabels([f"seed {s}" for s in seeds])
    axes_sum[1].set_title("PG - Regression (final eval)")
    axes_sum[1].set_ylabel("Reward gap")
    fig_sum.tight_layout()
    path_sum = os.path.join(save_dir, "fig_seed_sweep_summary.png")
    fig_sum.savefig(path_sum, dpi=150, bbox_inches="tight")
    print(f"Saved {path_sum}")

    if show:
        plt.show()
    else:
        plt.close("all")
    return fig_eval, fig_loss, fig_sum


def plot_reward_curve(
    eval_steps: Sequence[int],
    eval_values: Union[torch.Tensor, np.ndarray],
    train_losses: Union[torch.Tensor, np.ndarray],
    title: str,
    save_path: str,
    show: bool = True,
):
    """Plot reward curve during training (+ loss)."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(eval_steps, _to_numpy(eval_values), color="#d62728", linewidth=1.8)
    axes[0].set_title(f"{title}: reward curve")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Expected reward")

    axes[1].plot(_to_numpy(train_losses), color="#1f77b4", linewidth=1.0, alpha=0.85)
    axes[1].set_title(f"{title}: training loss")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Loss")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_learned_policy(
    policy,
    n_users_show: int = 20,
    save_path: str = "figures/learned_policy.png",
    title: str = "Learned policy scores",
    show: bool = True,
):
    """Heatmap of user-item logits for a subset of users (learned policy)."""
    sns.set_theme(style="whitegrid", context="notebook")
    user_ids = torch.arange(min(n_users_show, policy.n_users))
    with torch.no_grad():
        logits = policy.base_model(user_ids, requires_grad=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(_to_numpy(logits), cmap="viridis", ax=ax, cbar_kws={"label": "logit"})
    ax.set_title(title)
    ax.set_xlabel("Item id")
    ax.set_ylabel("User id")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_item_ranking_distribution(
    policy,
    ranking_length: int = 1,
    n_samples: int = 5000,
    top_n: int = 30,
    save_path: str = "figures/item_distribution.png",
    title: str = "Induced item / ranking distribution",
    show: bool = True,
):
    """Histogram of items selected by the (stochastic) policy."""
    sns.set_theme(style="whitegrid", context="notebook")
    user_ids = torch.randint(0, policy.n_users, (n_samples,))
    with torch.no_grad():
        item_ids = policy.sample_action(user_ids, ranking_length=ranking_length, is_deterministic=False)
    flat = item_ids.reshape(-1).cpu().numpy()

    counts = np.bincount(flat, minlength=policy.n_items)
    top_idx = np.argsort(counts)[::-1][:top_n]
    top_counts = counts[top_idx]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(top_idx)), top_counts, color="#9467bd")
    ax.set_xticks(np.arange(len(top_idx)))
    ax.set_xticklabels(top_idx, rotation=90, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Item id (top selected)")
    ax.set_ylabel("Selection count")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def run_pg(
    steps: int,
    n_users: int,
    n_items: int,
    n_dim: int,
    model_dim: int,
    batch_size: int,
    save_dir: str,
    show: bool,
    seed: int,
    use_baseline: bool = False,
    loss_type: str = "PolicyGradient",
    learning_rate: float = 1e-3,
):
    torch.manual_seed(seed)
    env = RecEnv(n_users=n_users, n_items=n_items, n_dim=n_dim, reward_type="rating")
    policy = SoftmaxPolicy(
        n_users=n_users, n_items=n_items, model_dim=model_dim, reward_type="rating"
    )
    if use_baseline:
        print(
            "Note: --baseline requested, but train_SoftmaxPolicy has no use_baseline "
            "param in this pg.py; training without baseline."
        )
    print(
        f"Training SoftmaxPolicy (PG, {loss_type}, lr={learning_rate}) "
        f"for {steps} steps..."
    )
    policy, losses, evals, _ = train_SoftmaxPolicy(
        env=env,
        policy=policy,
        loss_type=loss_type,
        n_steps=steps,
        batch_size=batch_size,
        ranking_length=1,
        learning_rate=learning_rate,
    )
    # train_SoftmaxPolicy logs eval every 100 steps
    eval_steps = list(range(0, steps, 100))[: len(_to_numpy(evals))]
    os.makedirs(save_dir, exist_ok=True)
    plot_reward_curve(
        eval_steps,
        evals,
        losses,
        title=f"PG ({loss_type})",
        save_path=os.path.join(save_dir, "pg_reward_curve.png"),
        show=show,
    )
    plot_learned_policy(
        policy,
        save_path=os.path.join(save_dir, "pg_learned_policy.png"),
        title=f"PG learned policy ({loss_type})",
        show=show,
    )
    plot_item_ranking_distribution(
        policy,
        ranking_length=1,
        save_path=os.path.join(save_dir, "pg_item_distribution.png"),
        title=f"PG induced item distribution ({loss_type})",
        show=show,
    )


def run_pl(
    steps: int,
    n_users: int,
    n_items: int,
    n_dim: int,
    model_dim: int,
    batch_size: int,
    ranking_length: int,
    save_dir: str,
    show: bool,
    seed: int,
    sampler: str = "gumbel",
    learning_rate: float = 1e-3,
):
    torch.manual_seed(seed)
    env = RecEnv(n_users=n_users, n_items=n_items, n_dim=n_dim, reward_type="rating")
    policy = PlackettLucePolicy(
        n_users=n_users, n_items=n_items, model_dim=model_dim, reward_type="rating"
    )
    print(
        f"Training PlackettLucePolicy (PL, sampler={sampler}, K={ranking_length}, "
        f"lr={learning_rate}) for {steps} steps..."
    )
    train_kwargs = dict(
        env=env,
        policy=policy,
        n_steps=steps,
        batch_size=batch_size,
        ranking_length=ranking_length,
        learning_rate=learning_rate,
    )
    if sampler == "gumbel":
        policy, losses, evals, _ = train_PLPolicy_efficiently(
            loss_type="PolicyGradient",
            **train_kwargs,
        )
    else:
        policy, losses, evals, _ = train_PLPolicy_autoregressively(
            loss_type="AutoRegressive",
            **train_kwargs,
        )
    # pl.py logs eval every training step
    eval_steps = list(range(len(_to_numpy(evals))))
    os.makedirs(save_dir, exist_ok=True)
    plot_reward_curve(
        eval_steps,
        evals,
        losses,
        title=f"PL ({sampler}, K={ranking_length})",
        save_path=os.path.join(save_dir, "pl_reward_curve.png"),
        show=show,
    )
    plot_learned_policy(
        policy,
        save_path=os.path.join(save_dir, "pl_learned_policy.png"),
        title="PL learned policy (user-item logits)",
        show=show,
    )
    plot_item_ranking_distribution(
        policy,
        ranking_length=ranking_length,
        save_path=os.path.join(save_dir, "pl_item_distribution.png"),
        title=f"PL induced ranking distribution (K={ranking_length})",
        show=show,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a recommendation policy and visualize the learned policy, "
            "reward curve during training, and induced item/ranking distributions."
        )
    )
    parser.add_argument(
        "--alg",
        type=str,
        choices=["PG", "PL"],
        required=False,
        help="Algorithm: PG (one-step softmax bandit) or PL (Plackett-Luce ranking).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3000,
        help="Number of training steps (default: 3000).",
    )
    parser.add_argument("--n-users", type=int, default=300, help="Number of users.")
    parser.add_argument("--n-items", type=int, default=800, help="Number of items.")
    parser.add_argument(
        "--n-dim",
        type=int,
        default=10,
        help="Ground-truth embedding dimension in RecEnv.",
    )
    parser.add_argument(
        "--model-dim",
        type=int,
        default=14,
        help="Learned two-tower embedding dimension.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="PG only: use a batch-mean baseline in REINFORCE.",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["PolicyGradient", "Regression"],
        default="PolicyGradient",
        help="PG only: PolicyGradient (REINFORCE) or Regression (supervised).",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        dest="learning_rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate for PG or PL training (default: 1e-3).",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["gumbel", "naive"],
        default="gumbel",
        help="PL only: gumbel = efficient top-k; naive = sequential/autoregressive.",
    )
    parser.add_argument(
        "-k",
        "--k",
        "--ranking-length",
        dest="ranking_length",
        type=int,
        default=3,
        help="PL only: ranking depth K (ignored for PG).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="figures",
        help="Directory to save PNG plots.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save plots without opening interactive windows.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.alg is None:
        parser.print_help()
        parser.error("--alg is required (PG or PL)")

    show = not args.no_show
    # Headless-friendly default when no display is available.
    if os.environ.get("DISPLAY") is None and os.environ.get("MPLBACKEND") is None:
        import matplotlib

        matplotlib.use("Agg")
        show = False

    if args.alg == "PG":
        if args.sampler != "gumbel":
            print("Note: --sampler is only used with --alg PL; ignoring for PG.")
        run_pg(
            steps=args.steps,
            n_users=args.n_users,
            n_items=args.n_items,
            n_dim=args.n_dim,
            model_dim=args.model_dim,
            batch_size=args.batch_size,
            save_dir=args.save_dir,
            show=show,
            seed=args.seed,
            use_baseline=args.baseline,
            loss_type=args.loss_type,
            learning_rate=args.learning_rate,
        )
    else:
        if args.baseline:
            print("Note: --baseline is only used with --alg PG; ignoring for PL.")
        if args.loss_type != "PolicyGradient":
            print("Note: --loss-type is only used with --alg PG; ignoring for PL.")
        run_pl(
            steps=args.steps,
            n_users=args.n_users,
            n_items=args.n_items,
            n_dim=args.n_dim,
            model_dim=args.model_dim,
            batch_size=args.batch_size,
            ranking_length=args.ranking_length,
            save_dir=args.save_dir,
            show=show,
            seed=args.seed,
            sampler=args.sampler,
            learning_rate=args.learning_rate,
        )


if __name__ == "__main__":
    main()
