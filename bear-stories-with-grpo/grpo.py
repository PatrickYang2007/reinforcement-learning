"""GRPO math: completion log-probs, group-relative advantages, clipped surrogate loss, and k3 KL penalty."""

from __future__ import annotations

import inspect

import torch


def sequence_log_probs(model, prompt_ids, prompt_mask, completion_ids, action_mask):
    """Return completion token log probabilities [B, T], including EOS.

    Prompt tensors have shape [B, P]; completion IDs and action_mask are [B, T].
    Masked positions return zero. Gradients flow only through the model.
    """
    all_logits, target_ids, valid_tokens = _forward_full_sequence(
        model, prompt_ids, prompt_mask, completion_ids, action_mask
    )
    prompt_length = prompt_ids.shape[1]
    completion_length = target_ids.shape[1]

    # all_logits[:, k] predicts the token at position k + 1.
    # target_ids has shape [batch, completion_length].
    # Positions P-1 .. P+T-2 predict completion tokens 0 .. T-1.
    logits = all_logits[:, prompt_length - 1 : prompt_length - 1 + completion_length, :]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    selected_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    selected_log_probs = selected_log_probs.masked_fill(~valid_tokens, 0.0)

    return selected_log_probs


def _forward_full_sequence(model, prompt_ids, prompt_mask, completion_ids, action_mask):
    """Provided model plumbing for :func:`sequence_log_probs`.

    Returns the model logits for ``prompt + completion``, the completion token
    IDs, and a Boolean mask selecting valid completion tokens. All returned
    tensors are on the model's device.
    """
    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    prompt_mask = prompt_mask.to(device)
    completion_ids = completion_ids.to(device)
    valid_tokens = action_mask.to(device, dtype=torch.bool)

    full_ids = torch.cat((prompt_ids, completion_ids), dim=1)
    full_attention_mask = torch.cat(
        (prompt_mask, valid_tokens.to(prompt_mask.dtype)), dim=1
    )
    position_ids = full_attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(full_attention_mask == 0, 1)

    model_inputs = {}
    if "position_ids" in inspect.signature(model.forward).parameters:
        model_inputs["position_ids"] = position_ids
    logits = model(
        input_ids=full_ids,
        attention_mask=full_attention_mask,
        use_cache=False,
        **model_inputs,
    ).logits
    return logits, completion_ids, valid_tokens


def group_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Normalize rewards [B] within each prompt group using population std."""
    rewards = rewards.detach().float()
    group_ids = group_ids.to(rewards.device)
    advantages = torch.zeros_like(rewards)
    for group in torch.unique(group_ids):
        members = group_ids == group
        group_rewards = rewards[members]
        baseline = group_rewards.mean()
        std = group_rewards.std(correction=0)  # population std: divide by G
        advantages[members] = (group_rewards - baseline) / (std + eps)
    return advantages


def grpo_loss(
    current_log_probs,
    old_log_probs,
    action_mask,
    advantages,
    *,
    clip_epsilon: float | None,
):
    """Clipped or unclipped GRPO policy loss (no KL term)."""
    mask = action_mask.to(dtype=torch.bool)
    old = old_log_probs.detach()
    zeros = torch.zeros_like(current_log_probs)
    cur = torch.where(mask, current_log_probs, zeros)
    old = torch.where(mask, old, zeros)

    ratio = torch.exp(cur - old)
    # One advantage per story, broadcast over its tokens.
    adv = advantages.detach().to(ratio.device, ratio.dtype).unsqueeze(1)
    if clip_epsilon is None:
        surrogate = ratio * adv
    else:
        clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
        surrogate = torch.minimum(ratio * adv, clipped * adv)

    if clip_epsilon is None:
        clip_frac = torch.zeros((), device=ratio.device)
    else:
        outside = ratio.ne(ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)).to(ratio.dtype)
        counts = mask.sum(1).clamp_min(1).to(ratio.dtype)
        clip_frac = ((outside * mask.to(ratio.dtype)).sum(1) / counts).mean()
    counts = mask.sum(1).clamp_min(1).to(surrogate.dtype)
    per = (surrogate * mask.to(surrogate.dtype)).sum(1) / counts
    return -per.mean(), clip_frac


def kl_penalty(current_log_probs, reference_log_probs, action_mask):
    """Sampled k3 KL penalty, averaged over tokens then stories.

    Inputs are [B, T]. For d = log pi_ref - log pi, use exp(d) - d - 1.
    This estimates KL(pi || pi_ref) for samples from pi; reused rollouts
    make it a surrogate after the first optimizer update. Reference is detached.
    """
    mask = action_mask.to(dtype=torch.bool)
    zeros = torch.zeros_like(current_log_probs)
    # Zero masked slots before exp() so padding cannot create inf/NaN gradients.
    cur = torch.where(mask, current_log_probs, zeros)
    ref = torch.where(mask, reference_log_probs.detach(), zeros)
    d = ref - cur
    k3 = torch.exp(d) - d - 1.0
    counts = mask.sum(1).clamp_min(1).to(k3.dtype)
    per_story = (k3 * mask.to(k3.dtype)).sum(1) / counts
    return per_story.mean()
