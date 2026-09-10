import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from base import BasePolicy
from env import RecEnv


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim_emb: int = 10,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim_emb = dim_emb

        self.user_encoder = nn.Embedding(n_users, dim_emb)
        self.item_encoder = nn.Embedding(n_items, dim_emb)

    def forward(
        self,
        user_ids: torch.Tensor,  # shape (batch_size, )
        item_ids: Optional[torch.Tensor] = None,  # shape (batch_size, ) or shape (batch_size, ranking_length)
        requires_grad: bool = True,  # should be True when regressing the reward and calculating the log prob, should be False when sampling actions
        **kwargs,
    ):
        ### MAY NEED BATCH PROCESSING HERE (depending on # of items and memory capacity) ###
        is_single_item = item_ids is not None and item_ids.dim() == 1

        if item_ids is None:  # enumerate all actions, when sampling actions, otherwise (when regressing), item_ids should be provided
            item_ids = torch.arange(self.n_items).unsqueeze(0)  # shape (dummy_dim, n_items) -> shape (batch_size, n_items)

        if is_single_item:
            item_ids = item_ids.unsqueeze(1)  # shape (batch_size, ) -> shape (batch_size, 1)

        ### basic logic is: (1) encode users and items, respectively, (2) take inner product between them to score affinity between (user, item) pairs
        if requires_grad:
            learned_user_emb = self.user_encoder(user_ids)
            learned_item_emb = self.item_encoder(item_ids)
        else:
            with torch.no_grad():
                learned_user_emb = self.user_encoder(user_ids)
                learned_item_emb = self.item_encoder(item_ids)

        learned_user_emb = learned_user_emb.unsqueeze(1)  # shape (batch_size, dummy_dim, model_dim) -> shape (batch_size, n_items/ranking_length, model_dim)
        logits = (learned_user_emb * learned_item_emb).sum(dim=-1)  # just taking inner products between two embeddings

        if is_single_item:
            logits = logits.squeeze(1)  # shape (batch_size, 1) -> shape (batch_size, )

        return logits  # shape (batch_size, ) or shape (batch_size, n_items/ranking_length)
#


def gumbel_noise_like(x: torch.Tensor):
    """
    Utility for g_i ~ G(0, 1), Gumbel noise for softmax trick
    ---
    Uses the shape of `x` to make a Gumbel distribution, returns a sample
    """
    return torch.distributions.Gumbel(torch.zeros_like(x), torch.ones_like(x)).sample()


@dataclass
class PlackettLucePolicy(BasePolicy):

    n_users: int = 1000
    n_items: int = 1000
    model_dim: int = 5
    reward_type: str = RecEnv.reward_type #"rating"  # or "click"

    def __post_init__(self):
        self.base_model = TwoTowerModel(self.n_users, self.n_items, dim_emb=self.model_dim)

    def sample_action(
        self,
        user_ids: torch.Tensor,
        ranking_length: int = 1,
        is_deterministic: bool = False,
    ):
        logits = self.base_model(user_ids , requires_grad = False) 
        if is_deterministic: 
            action = torch.argsort(logits , dim = 1 , descending = True)
            action = action[: , :ranking_length]
        else: 
            noise = gumbel_noise_like(logits)
            scores = logits + noise
            action = torch.argsort(scores , dim = 1 , descending = True)
            action = action[: , :ranking_length]

        return action

    def calc_log_prob(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        is_joint_log_prob: bool = False,
    ):

        ranking_length = item_ids.shape[1]

        #gather method you can .gather(dim , and the indecies, so in this case i want to gather indecies item_id)
        logits = self.base_model(user_ids)                   
        gathered = logits.gather(1, item_ids)                 
        denom = torch.logsumexp(logits, dim=1).unsqueeze(1)  
        table = gathered - denom                             
        
        if is_joint_log_prob:            
            log_prob = table.sum(dim = 1)
        else:
            log_prob = table

        return log_prob

    def predict_value(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ):
        logits = self.base_model(user_ids, item_ids)  # need to propagate the gradient here
         # apply some gating function to logit if necessary, e.g., when reward is binary "click"
        if self.reward_type == "click":
          value=torch.sigmoid(logits)
        else:
          value=logits
        return value  # shape (batch_size, ranking_length)

    def sample_action_given_state(
        self,
        state: Tuple[torch.Tensor, Optional[torch.Tensor]],
        is_deterministic: bool = False,
    ):
        user_ids, memory = state

        if memory is None:  # just sample item from softmax without conditioning
            item_ids = self.sample_action(user_ids, ranking_length=1, is_deterministic=is_deterministic).squeeze(1)  # shape (batch_size, )
            return item_ids
        else:
            logits = self.base_model(user_ids , requires_grad = False)
            noise = gumbel_noise_like(logits)    
            if is_deterministic:
                scores = logits.scatter(1 , memory , -torch.inf)
                return torch.argmax(scores , dim = 1)
            
            scores = logits.scatter(1 , memory , -torch.inf) + noise
            action = torch.argmax(scores , dim = 1)
            
        return action

    def calc_log_prob_given_state(
        self,
        state: Tuple[torch.Tensor, Optional[torch.Tensor]],
        action: torch.Tensor,
    ):
        user_ids, memory = state

        if memory is None:  # just sample item from softmax without conditioning
            log_prob = self.calc_log_prob(user_ids, action.unsqueeze(1), is_joint_log_prob=True)  # shape (batch_size, )

        else:
            logits = self.base_model(user_ids)
            chosen = logits[torch.arange(len(user_ids)), action]
            used = logits.scatter(1 , memory , -torch.inf)
            log_prob = chosen - torch.logsumexp(used , dim = 1)

        return log_prob



def train_PLPolicy_efficiently(
    env: RecEnv,
    policy: PlackettLucePolicy,
    loss_type: str = "Regression",  # or "PolicyGradient"
    n_steps: int = 10000,  # hyperparams are tentative, need to be checked
    batch_size: int = 32,
    ranking_length: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
):
    optimizer = torch.optim.Adagrad(
        policy.base_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_losses = torch.zeros((n_steps,))
    eval_values = torch.zeros((n_steps,))

    training_start = time.time()

    for i in range(n_steps):
        user_ids, item_ids, reward = env.sample_batch(
            policy, batch_size=batch_size, ranking_length=ranking_length
        )
        agg_reward = reward.sum(dim=1)  # shape (batch_size,), ranking-wise reward r(c, σ)

        if loss_type == "Regression":
            # TODO 4.5: Implement loss
            pass

        elif loss_type == "PolicyGradient":
            # TODO 4.5: Implement loss
            pass

        else:
            raise NotImplementedError()

        optimizer.zero_grad()

        try:
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()

        except Exception as e:
            print(f"Gradient over/underflow issues at step {i + 1}", e)

        optimizer.step()
        train_losses[i] = loss.item()

        # if it is too slow, maybe we can log eval values per 100 steps each, for example
        eval_values[i] = env.evaluate(policy, n_samples=10000, ranking_length=ranking_length)

    training_end = time.time()
    compute_time = training_end - training_start
    print(f"Training time: {compute_time} seconds")

    return policy, train_losses, eval_values, compute_time


def train_PLPolicy_autoregressively(
    env: RecEnv,
    policy: PlackettLucePolicy,
    loss_type: str = "AutoRegressive",
    n_steps: int = 10000,  # hyperparams are tentative, need to be checked
    batch_size: int = 32,
    ranking_length: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
):
    optimizer = torch.optim.Adagrad(
        policy.base_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_losses = torch.zeros((n_steps,))
    eval_values = torch.zeros((n_steps,))

    training_start = time.time()

    for i in range(n_steps):
        state = env.reset(batch_size=batch_size, ranking_length=ranking_length)

        # this part is step-by-step, computationally inefficient compared to gumble-topk trick
        for k in range(ranking_length):
            action = policy.sample_action_given_state(state)
            next_state, reward = env.step(action)

            # TODO: Implement the autoregressive PolicyGradient loss at step k
            # using policy.calc_log_prob_given_state(state, action) and reward.
            # Accumulate into `loss` and `agg_reward` over the k steps.
            pass

            # update the state (i.e., memory of previously sampled items)
            state = next_state

        optimizer.zero_grad()

        try:
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()

        except Exception as e:
            print(f"Gradient over/underflow issues at step {i + 1}", e)

        optimizer.step()
        train_losses[i] = loss.item()

        # if it is too slow, maybe we can log eval values per 100 steps each, for example
        eval_values[i] = env.evaluate(policy, n_samples=10000, ranking_length=ranking_length)

    training_end = time.time()
    compute_time = training_end - training_start
    print(f"Training time: {compute_time} seconds")

    return policy, train_losses, eval_values, compute_time
