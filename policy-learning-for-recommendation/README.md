# Policy Learning for Recommendation

Learns recommendation policies in a simulated contextual-bandit environment. A softmax policy recommends one item and a Plackett–Luce policy ranks K items; both are trained with REINFORCE-style policy gradients and compared against a supervised regression baseline.

Course project for Cornell CS 4789/5789 (Introduction to Reinforcement Learning), Fall 2026, built on starter code provided by the course staff.

## How it works

- **Environment:** users and items have hidden embeddings. Recommending item *i* to user *u* returns a noisy reward based on their inner product: a Gaussian rating, or a Bernoulli click with probability sigmoid(u · i).
- **Model:** a two-tower network scores every item for a user.
- **Softmax policy:** samples one item from a softmax over the scores.
- **Plackett–Luce policy:** samples a ranked list with Gumbel-top-k and uses a sampling-with-replacement approximation of the list's log-probability. It can also be trained autoregressively, choosing one position at a time through the environment's `reset`/`step` interface.

| File | Contents |
|---|---|
| `base.py` | abstract policy and environment interfaces |
| `env.py` | the simulated recommendation environment (`RecEnv`) |
| `models.py` | two-tower scoring model and Gumbel noise |
| `pg.py` | softmax policy and its trainer |
| `pl.py` | Plackett–Luce policy and its trainers |
| `visualize.py` | plots and a command-line runner |
| `test.py` | sanity tests |
| `colab_run.ipynb` | experiments and figures |

## Running it

```bash
pip install -e .
python test.py
python visualize.py --alg PG --steps 2000
python visualize.py --alg PL --steps 2000
```
