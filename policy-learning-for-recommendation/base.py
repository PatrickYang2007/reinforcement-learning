# base class implementation for the environment and policy

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

########################################################
# Policy
########################################################

@dataclass
class BasePolicy(ABC):
    """Class for the polic
    y.

    Input
    ------
    n_users: int
        # of users

    n_items: int
        # of items

    model_dim: int
        dimension of the base model

    """

    @abstractmethod
    def sample_action(
        self,
        user_ids: torch.Tensor,
        ranking_length: int = 1,
        is_deterministic: bool = False,
    ):
        """Sample items for batched user data.

        Input
        ------
        user_ids: torch.Tensor, shape (batch_size,)
            user ids

        ranking_length: int
            length of ranking output

        is_deterministic: bool = False,
            Wether to return deterministic top-k or stochastically sample topk (from Plackett-Luce).

        Return
        ------
        item_ids: torch.Tensor, shape (batch_size, ranking_length)
            item ids

        """
        raise NotImplementedError()

    @abstractmethod
    def calc_log_prob(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        is_joint_log_prob: bool = False,
    ):
        """Calculate log probability of each item peovided previously sampled items.

        Input
        ------
        user_ids: torch.Tensor, shape (batch_size,)
            user ids

        item_ids: torch.Tensor, shape (batch_size, ranking_length)
            item ids

        is_joint_log_prob: bool
            Wether to return joint log probability or not.

        Return
        ------
        log_action_prob: torch.Tensor, shape (batch_size, ranking_length)
            log probability of each item peovided previously sampled items

        joint_log_prob: torch.Tensor, shape (batch_size,)
            joint log probability of sampling the set of ranked items

        """
        raise NotImplementedError()

    @abstractmethod
    def predict_value(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ):
        """Predict the value of each item for the given user.

        Input
        ------
        user_ids: torch.Tensor, shape (batch_size,)
            user ids

        item_ids: torch.Tensor, shape (batch_size, ranking_length)
            item ids

        Return
        ------
        pred: torch.Tensor, shape (batch_size, ranking_length)
            predicted value of each item for the given user

        """
        raise NotImplementedError()

    @abstractmethod
    def sample_action_given_state(
        self,
        state: Tuple[torch.Tensor, Optional[torch.Tensor]],
        is_deterministic: bool = False,
    ):
        """Sample action provided previously sampled items, following interactive interface of openai-gym/gymnasium.

        Input
        ------
        state: Tuple of (torch.Tensor, None) or Tuple of (torch.Tensor, torch.Tensor), shape (batch_size, k-1)
            user ids, and memory of previously sampled item ids (if any)

        is_deterministic: bool = False,
            Wether to return deterministic top-k or stochastically sample topk (from Plackett-Luce).

        Return
        ------
        action: torch.Tensor, shape (batch_size, )
            Sampled item for the k'th position of the ranking.

        """
        raise NotImplementedError()

    @abstractmethod
    def calc_log_prob_given_state(
        self,
        state: Tuple[torch.Tensor, Optional[torch.Tensor]],
        action: torch.Tensor,
    ):
        """Calculate the log-action choice probability of the k'th item, provided previously sampled items.

        Input
        ------
        state: Tuple of (torch.Tensor, None) or Tuple of (torch.Tensor, torch.Tensor), shape (batch_size, k-1)
            user ids, and memory of previously sampled item ids (if any)

        action: torch.Tensor
            k'th item of the ranking

        Return
        ------
        log_prob: torch.Tensor, shape (batch_size, )
            Conditional log prob of sampling k'th item provided previously sampled items.

        """
        raise NotImplementedError()


########################################################
# Environment
########################################################


@dataclass
class BaseEnv(ABC):
    """Class for the synthetic simulation.

    Input
    ------
    n_users: int
        # of users

    n_items: int
        # of items

    n_dims: int
        Dimension of the ground-truth user and item embs

    reward_type: "click", "rating"
        Whether to model binary reward or continuous reward

    reward_noise: float
        Noise level of the reward, when reward_type is "rating"

    """

    @abstractmethod
    def sample_batch(
        self,
        policy: BasePolicy,
        batch_size: int = 32,
        ranking_length: int = 1,
        is_evaluation_mode: bool = False,
    ):
        """Sample batched data.

        Input
        ------
        policy: BasePolicy
            policy instance

        batch_size: int
            # of data samples

        ranking_length: int
            length of ranking output

        is_evaluation_mode: bool = False,
            Wether to return expected reward or not (i.e., sampled reward).

        Return
        ------
        user_ids: torch.Tensor, shape (batch_size,)
            user ids

        item_ids: torch.Tensor, shape (batch_size, ranking_length)
            item ids

        reward: torch.Tensor, shape (batch_size, ranking_length)
            expected or sampled rewards

        """
        raise NotImplementedError()

    @abstractmethod
    def sample_reward_given_user_item(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        is_expected_reward: bool = False,
    ):
        """Sample reward given user and item ids.

        Input
        ------
        user_ids: torch.Tensor, shape (batch_size,)
            user ids

        item_ids: torch.Tensor, shape (batch_size, ranking_length)
            item ids

        is_expected_reward: bool = False,
            Wether to return expected reward or not (i.e., sampled reward).

        Return
        ------
        reward: torch.Tensor, shape (batch_size, ranking_length)
            expected or sampled rewards


        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate(
        self,
        policy: BasePolicy,
        n_samples: int = 1000,
        ranking_length: int = 1,
    ):
        """Evaluate the policy.

        Input
        ------
        policy: BasePolicy
            policy instance

        n_samples: int
            # of data samples

        ranking_length: int
            length of ranking output

        """
        raise NotImplementedError()

    @abstractmethod
    def reset(
        self,
        batch_size: int = 32,
        ranking_length: int = 1,
    ):
        """Interative interface aligning with openai-gym/gymnasium.

        Input
        ------
        batch_size: int
            Batch size (usually openai-gym does not take this config, but for convenience)

        ranking_length: int
            Length of ranking output (usually openai-gym does not take this config, but for convenience)

        Return
        ------
        state: Tuple of (torch.Tensor, None), shape (batch_size, )
            Initial state, i.e., user ids, (and an empty memory of actions)

        """
        raise NotImplementedError()

    @abstractmethod
    def step(
        self,
        action: torch.Tensor,
    ):
        """Interative interface aligning with openai-gym/gymnasium.

        Input
        ------
        action: torch.Tensor, shape (batch_size, )
            k'th item of the ranking

        Return
        ------
        next_state: Tuple of (torch.Tensor, torch.Tensor), shape (batch_size, ) and shape (batch_size, k)
            Next state, i.e., (user ids, 1:k'th ranked items)

        reward: torch.Tensor, shape (batch_size, ranking_length)
            Reward given the current state (user ids) and current action (k'th item).

        """
        raise NotImplementedError()
