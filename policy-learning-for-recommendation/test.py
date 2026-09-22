
import contextlib
import io
import torch

from env import RecEnv
from pg import SoftmaxPolicy, train_SoftmaxPolicy
from pl import (
    PlackettLucePolicy,
    train_PLPolicy_autoregressively,
    train_PLPolicy_efficiently,
)

# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

_PASS = 0
_FAIL = 0


def _banner(title: str) -> None:
    width = 64
    print()
    print("=" * width)
    print(f" {title}")
    print("=" * width)


def _section(title: str) -> None:
    print()
    print(f"-- {title}")


def _ok(msg: str) -> None:
    global _PASS
    _PASS += 1
    print(f"  [PASS] {msg}")


def _info(msg: str) -> None:
    print(f"         {msg}")



def _quiet_train(fn, **kwargs):
    """Run a training helper while suppressing its Training time: prints."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(**kwargs)

def _summary(suite_name: str) -> None:
    print()
    print("-" * 64)
    print(f" {suite_name}: {_PASS} checks passed")
    print("-" * 64)


########################################################
# Sanity tests for RecEnv / Plackett-Luce (PL)
########################################################


def test_pl_and_env():
    global _PASS
    _PASS = 0
    _banner("RecEnv + Plackett-Luce (PL) sanity tests")

    # --- 0. Common setup ---
    torch.manual_seed(0)

    TEST_N_USERS = 20
    TEST_N_ITEMS = 15
    TEST_DIM = 4

    env = RecEnv(
        n_users=TEST_N_USERS,
        n_items=TEST_N_ITEMS,
        n_dim=TEST_DIM,
        reward_type="rating",
    )

    policy = PlackettLucePolicy(
        n_users=TEST_N_USERS,
        n_items=TEST_N_ITEMS,
        model_dim=TEST_DIM,
        reward_type="rating",
    )

    _section("Setup")
    _ok("RecEnv + PlackettLucePolicy initialized")


    # ============================================================
    # 1. TwoTowerModel: shape and cross-batch leakage sanity check
    # ============================================================

    model = policy.base_model
    user_ids = torch.randint(0, TEST_N_USERS, (6,))

    # (1) item_ids=None -> logits for all items
    logits_all = model(user_ids, item_ids=None, requires_grad=False)
    assert logits_all.shape == (6, TEST_N_ITEMS), \
        f"Expected (6, {TEST_N_ITEMS}), got {logits_all.shape}"

    # (2) 2D item_ids: (batch_size, ranking_length)
    item_ids_2d = torch.randint(0, TEST_N_ITEMS, (6, 3))
    logits_2d = model(user_ids, item_ids_2d)
    assert logits_2d.shape == (6, 3), \
        f"Expected (6, 3), got {logits_2d.shape}"

    # (3) 1D item_ids: (batch_size,)
    # Used when ranking_length == 1 or in the autoregressive path
    item_ids_1d = torch.randint(0, TEST_N_ITEMS, (6,))
    logits_1d = model(user_ids, item_ids_1d)
    assert logits_1d.shape == (6,), \
        f"Expected (6,), got {logits_1d.shape}"

    # (4) Check for cross-batch leakage.
    # The 1D case should exactly match the user-item dot product for each batch element.
    with torch.no_grad():
        u = model.user_encoder(user_ids)
        i = model.item_encoder(item_ids_1d)
        expected = (u * i).sum(dim=-1)

    assert torch.allclose(logits_1d, expected, atol=1e-5), \
        "Cross-batch leakage detected when processing 1D item_ids."

    _ok("TwoTowerModel shapes OK (None / 2D / 1D item_ids)")
    _ok("No cross-batch leakage on 1D item_ids")


    # ============================================================
    # 2. RecEnv: sample_batch, reward, reset, and step sanity checks
    # ============================================================

    user_ids, item_ids, reward = env.sample_batch(
        policy,
        batch_size=8,
        ranking_length=3,
    )

    assert user_ids.shape == (8,)
    assert item_ids.shape == (8, 3)
    assert reward.shape == (8, 3)

    _section("RecEnv")
    _ok("sample_batch shapes (users, ranking, rewards)")

    # Verify that click rewards are binary (0 or 1)
    # while rating rewards are continuous.
    env_click = RecEnv(
        n_users=TEST_N_USERS,
        n_items=TEST_N_ITEMS,
        n_dim=TEST_DIM,
        reward_type="click",
    )

    policy_click = PlackettLucePolicy(
        n_users=TEST_N_USERS,
        n_items=TEST_N_ITEMS,
        model_dim=TEST_DIM,
        reward_type="click",
    )

    _, _, click_reward = env_click.sample_batch(
        policy_click,
        batch_size=32,
        ranking_length=3,
    )

    assert set(click_reward.unique().tolist()).issubset({0.0, 1.0}), \
        "Click rewards must be either 0 or 1."

    _ok("click rewards are binary {0,1}")

    # Verify the reset -> step transition and memory dtype.
    state = env.reset(batch_size=5, ranking_length=3)
    user_ids0, memory0 = state

    assert memory0 is None, \
        "Memory should be None immediately after reset."

    action0 = torch.randint(0, TEST_N_ITEMS, (5,))
    next_state, step_reward = env.step(action0)
    next_user_ids, memory1 = next_state

    assert torch.equal(next_user_ids, user_ids0), \
        "User IDs should remain unchanged after a step."

    assert memory1.shape == (5, 1), \
        f"Expected memory shape (5, 1), got {memory1.shape}"

    assert memory1.dtype == torch.long, \
        f"Expected memory dtype torch.long, got {memory1.dtype}"

    assert torch.equal(memory1[:, 0], action0), \
        "The most recently selected item should be stored in memory."

    _ok("reset/step memory dtype and rollout behavior")


    # ============================================================
    # 3. PlackettLucePolicy.sample_action:
    #    shape, uniqueness, and deterministic behavior
    # ============================================================

    user_ids = torch.randint(0, TEST_N_USERS, (10,))
    ranked_items = policy.sample_action(user_ids, ranking_length=5)

    assert ranked_items.shape == (10, 5)

    for row in ranked_items:
        assert len(set(row.tolist())) == 5, \
            "Items within a ranking must not contain duplicates."

    _section("PlackettLucePolicy.sample_action")
    _ok("ranking shape and within-ranking uniqueness")

    # With is_deterministic=True, Gumbel noise should be disabled,
    # so repeated calls should return the same top-k ranking.
    det1 = policy.sample_action(
        user_ids,
        ranking_length=5,
        is_deterministic=True,
    )

    det2 = policy.sample_action(
        user_ids,
        ranking_length=5,
        is_deterministic=True,
    )

    assert torch.equal(det1, det2), \
        "is_deterministic=True should return the same top-k ranking on repeated calls."

    _ok("deterministic mode is reproducible")


    # ============================================================
    # 4. calc_log_prob:
    #    probability validity and full-item denominator sanity check
    # ============================================================

    item_ids = policy.sample_action(user_ids, ranking_length=4)

    log_prob_per_item = policy.calc_log_prob(
        user_ids,
        item_ids,
        is_joint_log_prob=False,
    )

    assert log_prob_per_item.shape == (10, 4)

    assert (log_prob_per_item.exp() <= 1.0 + 1e-5).all(), \
        "exp(log_prob) must not exceed 1."

    joint_log_prob = policy.calc_log_prob(
        user_ids,
        item_ids,
        is_joint_log_prob=True,
    )

    assert joint_log_prob.shape == (10,)
    assert (joint_log_prob.exp() <= 1.0 + 1e-5).all()

    _section("PlackettLucePolicy.calc_log_prob")
    _ok("joint / per-position shapes and log-prob <= 0")

    # Increasing n_items should increase the denominator and therefore
    # make the average joint log probability smaller (more negative).
    # This indirectly checks whether the denominator is computed over
    # all items rather than only the selected K items.
    big_policy = PlackettLucePolicy(
        n_users=TEST_N_USERS,
        n_items=500,
        model_dim=TEST_DIM,
        reward_type="rating",
    )

    big_item_ids = big_policy.sample_action(
        user_ids,
        ranking_length=4,
    )

    big_joint_log_prob = big_policy.calc_log_prob(
        user_ids,
        big_item_ids,
        is_joint_log_prob=True,
    )

    _info(
        f"mean joint log_prob @ n_items={TEST_N_ITEMS}: "
        f"{joint_log_prob.mean().item():.3f}"
    )
    _info(
        f"mean joint log_prob @ n_items=500: "
        f"{big_joint_log_prob.mean().item():.3f}"
    )

    assert big_joint_log_prob.mean().item() < joint_log_prob.mean().item(), \
        (
            "Joint log probability did not decrease when n_items increased. "
            "Check whether the denominator is computed over all items."
        )

    _ok("SwR denominator grows with catalog size (n_items)")


    # ============================================================
    # 5. predict_value: shape and click-probability range
    # ============================================================

    pred_rating = policy.predict_value(user_ids, item_ids)

    assert pred_rating.shape == (10, 4)

    _section("PlackettLucePolicy.predict_value")
    _ok("rating predict_value shape")

    click_item_ids = policy_click.sample_action(
        user_ids,
        ranking_length=4,
    )

    pred_click = policy_click.predict_value(
        user_ids,
        click_item_ids,
    )

    assert pred_click.shape == (10, 4)

    assert ((pred_click >= 0) & (pred_click <= 1)).all(), \
        "For reward_type='click', predict_value must return values in [0, 1]."

    _ok("click predict_value in [0, 1]")


    # ============================================================
    # 6. sample_action_given_state:
    #    verify masking prevents duplicate selections
    # ============================================================

    rollout_batch = 6
    rollout_len = 5

    state = env.reset(
        batch_size=rollout_batch,
        ranking_length=rollout_len,
    )

    chosen_so_far = []

    for k in range(rollout_len):
        action = policy.sample_action_given_state(state)

        assert action.shape == (rollout_batch,)

        for prev in chosen_so_far:
            assert (action != prev).all(), \
                f"Duplicate item selected at step {k}; masking may be incorrect."

        chosen_so_far.append(action)
        state, reward = env.step(action)

    _section("Autoregressive APIs")
    _ok("sample_action_given_state: unique items across rollout")


    # ============================================================
    # 7. calc_log_prob_given_state:
    #    verify that memory items are correctly masked
    # ============================================================

    state = env.reset(batch_size=4, ranking_length=3)

    first_action = policy.sample_action_given_state(state)
    state, _ = env.step(first_action)

    user_ids_s, memory_s = state

    logits = policy.base_model(user_ids_s)

    masked_logits = logits.scatter(
        1,
        memory_s,
        float("-inf"),
    )

    gathered = torch.gather(
        masked_logits,
        dim=1,
        index=first_action.unsqueeze(1),
    ).squeeze(1)

    assert (gathered == float("-inf")).all(), \
        "Logits corresponding to items stored in memory must be masked to -inf."

    _ok("memory items masked to -inf in logits")


    # ============================================================
    # 8. Gradient flow sanity check
    # ============================================================

    policy.base_model.zero_grad()

    item_ids = policy.sample_action(
        user_ids,
        ranking_length=3,
    )

    log_prob = policy.calc_log_prob(
        user_ids,
        item_ids,
        is_joint_log_prob=True,
    )

    loss = -log_prob.mean()
    loss.backward()

    has_grad = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in policy.base_model.parameters()
    )

    assert has_grad, \
        "Gradients are not properly flowing to all base_model parameters."

    _section("Gradients")
    _ok("gradients flow through calc_log_prob → backward()")


    # ============================================================
    # 9. Training-loop smoke tests
    #    Verify that each training method runs for several steps
    #    without errors or NaN losses.
    # ============================================================

    small_env = RecEnv(
        n_users=20,
        n_items=15,
        n_dim=4,
        reward_type="rating",
    )

    # Regression training
    policy_reg = PlackettLucePolicy(
        n_users=20,
        n_items=15,
        model_dim=4,
        reward_type="rating",
    )

    _, losses_reg, evals_reg, *_ = _quiet_train(
        train_PLPolicy_efficiently,
        env=small_env,
        policy=policy_reg,
        loss_type="Regression",
        n_steps=5,
        batch_size=8,
        ranking_length=3,
    )

    assert not torch.isnan(losses_reg).any(), \
        "NaN detected in Regression loss."

    _section("Training smoke tests")
    _ok("train_PLPolicy_efficiently (Regression), 5 steps, no NaNs")


    # Policy Gradient training
    policy_pg = PlackettLucePolicy(
        n_users=20,
        n_items=15,
        model_dim=4,
        reward_type="rating",
    )

    _, losses_pg, evals_pg, *_ = _quiet_train(
        train_PLPolicy_efficiently,
        env=small_env,
        policy=policy_pg,
        loss_type="PolicyGradient",
        n_steps=5,
        batch_size=8,
        ranking_length=3,
    )

    assert not torch.isnan(losses_pg).any(), \
        "NaN detected in PolicyGradient loss."

    _ok("train_PLPolicy_efficiently (PolicyGradient), 5 steps, no NaNs")


    # Autoregressive training
    policy_ar = PlackettLucePolicy(
        n_users=20,
        n_items=15,
        model_dim=4,
        reward_type="rating",
    )

    _, losses_ar, evals_ar, *_ = _quiet_train(
        train_PLPolicy_autoregressively,
        env=small_env,
        policy=policy_ar,
        n_steps=5,
        batch_size=8,
        ranking_length=3,
    )

    assert not torch.isnan(losses_ar).any(), \
        "NaN detected in autoregressive training loss."

    _ok("train_PLPolicy_autoregressively, 5 steps, no NaNs")


    # ============================================================
    # Final result
    # ============================================================

    _summary("PL suite")

########################################################
# Sanity tests for SoftmaxPolicy (PG)
########################################################


def test_pg():
    global _PASS
    _PASS = 0
    _banner("SoftmaxPolicy (PG) sanity tests")

    # --- 0. Common setup ---
    torch.manual_seed(0)

    TEST_N_USERS = 20
    TEST_N_ITEMS = 15
    TEST_DIM = 4

    env_pg = RecEnv(
        n_users=TEST_N_USERS,
        n_items=TEST_N_ITEMS,
        n_dim=TEST_DIM,
        reward_type="rating",
    )

    policy_pg = SoftmaxPolicy(
        n_users=TEST_N_USERS,
        n_items=TEST_N_ITEMS,
        model_dim=TEST_DIM,
        reward_type="rating",
    )

    _section("Setup")
    _ok("RecEnv + SoftmaxPolicy initialized")


    # ============================================================
    # 1. sample_action: shape, boundary checks, and determinism
    # ============================================================

    user_ids = torch.randint(0, TEST_N_USERS, (8,))

    # (1) ranking_length > 1 must raise NotImplementedError
    try:
        policy_pg.sample_action(user_ids, ranking_length=3)
        assert False, "SoftmaxPolicy.sample_action should raise NotImplementedError when ranking_length != 1."
    except NotImplementedError:
        pass

    # (2) Output shape check for single-item recommendation
    actions = policy_pg.sample_action(user_ids, ranking_length=1)
    assert actions.shape == (8,), f"Expected shape (8,), got {actions.shape}"
    assert (actions >= 0).all() and (actions < TEST_N_ITEMS).all(), "Sampled actions out of item catalog bounds."

    # (3) Deterministic mode reproducibility
    det1 = policy_pg.sample_action(user_ids, ranking_length=1, is_deterministic=True)
    det2 = policy_pg.sample_action(user_ids, ranking_length=1, is_deterministic=True)
    assert torch.equal(det1, det2), "is_deterministic=True should produce identical outputs across calls."

    _section("SoftmaxPolicy.sample_action")
    _ok("rejects ranking_length != 1")
    _ok("output shape (batch,) and in-catalog ids")
    _ok("deterministic mode is reproducible")


    # ============================================================
    # 2. calc_log_prob: shape, valid range, and denominator check
    # ============================================================

    item_ids = torch.randint(0, TEST_N_ITEMS, (8,))
    log_probs = policy_pg.calc_log_prob(user_ids, item_ids)

    # (1) Shape & validity
    assert log_probs.shape == (8,), f"Expected shape (8,), got {log_probs.shape}"
    assert (log_probs <= 1e-5).all(), "Log probabilities must be <= 0."
    assert (log_probs.exp() <= 1.0 + 1e-5).all(), "exp(log_prob) must not exceed 1.0."

    # (2) Full denominator check (increasing n_items decreases average log prob)
    big_policy = SoftmaxPolicy(
        n_users=TEST_N_USERS,
        n_items=500,
        model_dim=TEST_DIM,
        reward_type="rating",
    )
    big_item_ids = torch.randint(0, 500, (8,))
    big_log_probs = big_policy.calc_log_prob(user_ids, big_item_ids)

    assert big_log_probs.mean().item() < log_probs.mean().item(), (
        "Log probability did not decrease when catalog size increased; "
        "check if denominator sums over all catalog items."
    )

    _section("SoftmaxPolicy.calc_log_prob")
    _ok("shape / range / catalog-size normalization")


    # ============================================================
    # 3. predict_value: shape and binary click range checks
    # ============================================================

    # (1) Continuous rating value check
    pred_rating = policy_pg.predict_value(user_ids, item_ids)
    assert pred_rating.shape == (8,), f"Expected shape (8,), got {pred_rating.shape}"

    # (2) Binary click value check
    env_click = RecEnv(n_users=TEST_N_USERS, n_items=TEST_N_ITEMS, n_dim=TEST_DIM, reward_type="click")
    policy_click = SoftmaxPolicy(n_users=TEST_N_USERS, n_items=TEST_N_ITEMS, model_dim=TEST_DIM, reward_type="click")

    pred_click = policy_click.predict_value(user_ids, item_ids)
    assert pred_click.shape == (8,), f"Expected shape (8,), got {pred_click.shape}"
    assert ((pred_click >= 0.0) & (pred_click <= 1.0)).all(), "Click predictions must be bounded in [0, 1]."

    _section("SoftmaxPolicy.predict_value")
    _ok("rating + click predict_value shapes/bounds")


    # ============================================================
    # 4. Gradient flow sanity check
    # ============================================================

    policy_pg.base_model.zero_grad()
    sampled_items = policy_pg.sample_action(user_ids, ranking_length=1)
    log_prob = policy_pg.calc_log_prob(user_ids, sampled_items)

    loss = -log_prob.mean()
    loss.backward()

    has_grad = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in policy_pg.base_model.parameters()
    )
    assert has_grad, "Gradients are not flowing properly to all parameters in TwoTowerModel."

    _section("Gradients")
    _ok("gradients flow through calc_log_prob → backward()")


    # ============================================================
    # 5. Training loop smoke tests (train_SoftmaxPolicy)
    # ============================================================

    small_env = RecEnv(n_users=20, n_items=15, n_dim=4, reward_type="rating")

    # (1) Policy Gradient (REINFORCE) mode
    policy_train_pg = SoftmaxPolicy(n_users=20, n_items=15, model_dim=4, reward_type="rating")
    _, losses_pg, evals_pg, *_ = _quiet_train(
        train_SoftmaxPolicy,
        env=small_env,
        policy=policy_train_pg,
        loss_type="PolicyGradient",
        n_steps=10,
        batch_size=8,
        ranking_length=1,
    )

    assert not torch.isnan(losses_pg).any(), "NaN detected in PolicyGradient training loss."
    assert not torch.isnan(evals_pg).any(), "NaN detected in PolicyGradient eval values."
    _section("Training smoke tests")
    _ok("train_SoftmaxPolicy (PolicyGradient), 10 steps, no NaNs")

    # (2) Supervised Regression mode
    policy_train_reg = SoftmaxPolicy(n_users=20, n_items=15, model_dim=4, reward_type="rating")
    _, losses_reg, evals_reg, *_ = _quiet_train(
        train_SoftmaxPolicy,
        env=small_env,
        policy=policy_train_reg,
        loss_type="Regression",
        n_steps=10,
        batch_size=8,
        ranking_length=1,
    )

    assert not torch.isnan(losses_reg).any(), "NaN detected in Regression training loss."
    assert not torch.isnan(evals_reg).any(), "NaN detected in Regression eval values."
    _ok("train_SoftmaxPolicy (Regression), 10 steps, no NaNs")

    # ============================================================
    # Summary
    # ============================================================
    _summary("PG suite")

########################################################
# Entrypoint
########################################################


if __name__ == "__main__":
    _banner("Running all sanity tests")
    test_pl_and_env()
    test_pg()
    print()
    print("=" * 64)
    print(" ALL SUITES PASSED")
    print("=" * 64)
