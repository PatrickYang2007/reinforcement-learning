# Bear Stories with GRPO

Post-trains the [TinyStories-33M](https://huggingface.co/roneneldan/TinyStories-33M) language model with **Group Relative Policy Optimization (GRPO)** so it writes stories about bears. The experiments compare a sparse keyword reward with a dense embedding-similarity reward, each with and without PPO-style ratio clipping.


## How it works

The language model is the policy: the prompt plus the text generated so far is the state, and each generated token is an action. Each GRPO update:

1. samples a group of G = 16 completions for one story opener,
2. scores each completion with the reward,
3. standardizes the rewards within the group into advantages, A = (r − mean) / (std + ε),
4. takes several optimizer steps on the clipped objective min(w·A, clip(w, 1 − ε, 1 + ε)·A), where w is each token's probability ratio between the current policy and the policy that generated the samples,
5. can add a k3 KL penalty that keeps the policy close to the frozen pretrained model.

**Rewards**

- **Sparse:** 1 if the story contains the whole word "bear" or "bears", otherwise 0.
- **Dense:** cosine similarity between the story's sentence embedding (all-MiniLM-L6-v2) and the embedding of "bear".

| File | Contents |
|---|---|
| `grpo.py` | completion-token log-probs, group-relative advantages, clipped GRPO loss, k3 KL penalty |
| `rewards.py` | sparse and dense rewards |
| `train.py` | grouped rollouts, the GRPO optimizer step, and the training loop |
| `eval.py` | evaluation metrics (bear-story rate, reference perplexity) |
| `utils.py` | experiment runner, summaries, and plots |
| `tests.py` | unit tests |
| `configs/grid_2x2.json` | experiment hyperparameters |
| `bear_stories_grpo.ipynb` | walkthrough: component checks, experiments, figures |

## Experiments

A 2×2 grid with seed 17: sparse or dense reward, with ratio clipping (ε = 0.2) or without. Each run does 20 GRPO updates with 4 optimizer epochs per rollout, learning rate 1e-5, and no KL penalty. Training cycles through five story openers; evaluation uses "Once upon a time", "Deep in the forest", and the held-out opener "Late one night".

## Running it

```bash
pip install -e ".[semantic]"
python tests.py
```

Then open `bear_stories_grpo.ipynb`. A GPU runtime (for example a Colab T4) is recommended for the experiments.
