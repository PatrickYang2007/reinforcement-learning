# PG.py is just a simplified version when we have ranking_length == 1.

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from base import BasePolicy
from env import RecEnv
from models import TwoTowerModel, gumbel_noise_like


@dataclass
class SoftmaxPolicy(BasePolicy):

    n_users: int = 1000
    n_items: int = 1000
    model_dim: int = 5
    reward_type: str = "rating"  # or "click"

    def __post_init__(self):
        self.base_model = TwoTowerModel(self.n_users, self.n_items, dim_emb=self.model_dim)

    def sample_action(
        self,
        user_ids: torch.Tensor, # (batch_size, )
        ranking_length: int = 1, # (batch_size, )
        is_deterministic: bool = False,
    ):
        if ranking_length != 1:
            raise NotImplementedError()
        
        logits = self.base_model(user_ids , requires_grad = False)
        if is_deterministic:
            return torch.argmax(logits , dim = 1)
        else: 
            noise = gumbel_noise_like(logits)
            scores = noise + logits

            return torch.argmax(scores, dim = 1)
        
    def calc_log_prob(
        self,
        user_ids: torch.Tensor, # (batch_size, )
        item_ids: torch.Tensor, # (batch_size, )
        is_joint_log_prob: bool = False, # Not used here
    ): 
        logits = self.base_model(user_ids)

        chosen = logits[torch.arange(len(user_ids)), item_ids]   
        return chosen - torch.logsumexp(logits, dim=1)         

    def predict_value(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ):
        logits = self.base_model(user_ids, item_ids, requires_grad=True)

        if self.reward_type == "click":
          value=torch.sigmoid(logits)
        else:
          value=logits
        return value

    ### Not needed for ranking = 1
    def sample_action_given_state(
        self,
        state: Tuple[torch.Tensor, Optional[torch.Tensor]],
        is_deterministic: bool = False,
    ):
        raise NotImplementedError()

    def calc_log_prob_given_state(
        self,
        state: Tuple[torch.Tensor, Optional[torch.Tensor]],
        action: torch.Tensor,
    ):
        raise NotImplementedError()

def train_SoftmaxPolicy(
    env: RecEnv,
    policy: SoftmaxPolicy,
    loss_type: str = "Regression",  # or "PolicyGradient"
    n_steps: int = 10000,
    batch_size: int = 32,
    ranking_length: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5
):
    optimizer = torch.optim.Adagrad(policy.base_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_losses = torch.zeros((n_steps, ))

    eval_envs = 1000
    eval_every = 100
    eval_values = torch.zeros(((n_steps + eval_every - 1) // eval_every, ))

    training_start = time.time()

    for i in range(n_steps):
        user_ids, item_ids, reward = env.sample_batch(policy, batch_size=batch_size, ranking_length=ranking_length)
        reward = reward.squeeze(1)

        if loss_type == "PolicyGradient":
            inter = policy.calc_log_prob(user_ids , item_ids.squeeze(1))
            loss = -(reward * inter).mean()

        elif loss_type == "Regression":
            loss = torch.mean((policy.predict_value(user_ids , item_ids.squeeze(1)) - reward)**2)
            
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
        if i%100 == 0:
            eval_values[i//eval_every] = env.evaluate(policy, n_samples=eval_envs, ranking_length=ranking_length)

    training_end = time.time()
    compute_time = training_end - training_start
    print(f"Training time: {compute_time} seconds")

    return policy, train_losses, eval_values, compute_time


