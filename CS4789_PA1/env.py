# RecEnv class used to represent our contextual bandit recommendation environment.

from dataclasses import dataclass

import torch

from base import BaseEnv, BasePolicy


@dataclass
class RecEnv(BaseEnv):

    n_users: int = 1000
    n_items: int = 1000
    n_dim: int = 5
    reward_type: str = "click"
    reward_noise: float = 1.0

    def __post_init__(self):
        self.ground_truth_user_embs = torch.randn(self.n_users, self.n_dim)  # sample from Normal, shape (n_users, n_dim)
        self.ground_truth_item_embs = torch.randn(self.n_items, self.n_dim)  # sample from Normal, shape (n_items, n_dim)

    def sample_batch(
        self,
        policy: BasePolicy,
        batch_size: int = 32,
        ranking_length: int = 1,
        is_evaluation_mode: bool = False,
    ):
        user_ids = torch.randint(0, self.n_users, (batch_size,))  # sample from randint of self.n_users, shape (batch_size, )
        item_ids = policy.sample_action(user_ids, ranking_length=ranking_length)
        if item_ids.dim() == 1:
            item_ids = item_ids.unsqueeze(1)
        reward = self.sample_reward_given_user_item(user_ids, item_ids, is_expected_reward=is_evaluation_mode)
        return user_ids, item_ids, reward  # shape (batch_size, ), shape (batch_size, ranking_length), shape (batch_size, ranking_length)

    def sample_reward_given_user_item(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        is_expected_reward: bool = False,
    ):
        user_embs = self.ground_truth_user_embs[user_ids]  # shape (batch_size, )
        item_embs = self.ground_truth_item_embs[item_ids]  # shape (batch_size, ranking_length)

        if self.reward_type == "click":
            expected_reward = torch.sigmoid((user_embs.unsqueeze(1) * item_embs).sum(dim=-1))  # sigmoid of (user_embs @ item_embs), shape (batch_size, ranking_length)
        elif self.reward_type == "rating":
            expected_reward = (user_embs.unsqueeze(1) * item_embs).sum(dim=-1)  # (user_embs @ item_embs), shape (batch_size, ranking_length)
        else:
            raise NotImplementedError

        if is_expected_reward:
            reward = expected_reward
        else:
            if self.reward_type == "click":
                reward = torch.bernoulli(expected_reward)  # sample from Bernoulli, shape (batch_size, ranking_length)
            elif self.reward_type == "rating":
                reward = torch.normal(expected_reward, self.reward_noise)  # sample from Normal (use self.reward_noise), shape (batch_size, ranking_length)

        return reward  # shape (batch_size, ranking_length)

    def evaluate(
        self,
        policy: BasePolicy,
        n_samples: int = 1000,
        ranking_length: int = 1,
    ):
        _, _, reward = self.sample_batch(policy, batch_size=n_samples, ranking_length=ranking_length, is_evaluation_mode=True)
        agg_reward = reward.sum(dim=1)  # shape (n_samples,), just taking sum over the ranking
        agg_reward = agg_reward.mean()  # just taking the average of n_samples
        return agg_reward  # scalar

    def reset(
        self,
        batch_size: int = 32,
        ranking_length: int = 1,
    ):
        # YJ : fixed NameError on init_state, specified type on self.memory to prevent silent error
        self.k = 0
        self.memory = torch.zeros((batch_size, ranking_length), dtype=torch.long)
        self.init_state = torch.randint(0, self.n_users, (batch_size,))  # user_ids. sample from randint of self.n_users, shape (batch_size, )

        state = (self.init_state, None)  # None to indicate the empty memory
        return state

    def step(
        self,
        action: torch.Tensor,  # i.e., k'th item, shape (batch_size, )
    ):
        # YJ : cloned self.memory[:, :self.k] to avoid in-place bug
        self.memory[:, self.k] = action
        self.k += 1

        next_state = (self.init_state, self.memory[:, : self.k].clone())  # shape (batch_size, ) and shape (batch_size, k)
        reward = self.sample_reward_given_user_item(self.init_state, action.unsqueeze(1))  # shape (batch_size, )
        return next_state, reward
