# VAE for 3D Human Shape Generation in PCA Space

A β-VAE that learns the distribution of registered 3D human body shapes from their global PCA scores and generates new shapes by sampling its latent space.

Each body scan is represented by 35 PCA scores. The model is trained on 19,234 frames, and the script writes the trained model, reconstructions of every frame, and 40,000 generated shapes (as PCA scores) in one run.

## Quick start

```bash
pip install torch numpy pandas scipy matplotlib
python vae_codes.py
```

A run takes about 30 seconds on a laptop CPU. All settings are in the dictionary `C` at the top of `vae_codes.py`; there are no command-line arguments. Results are written to `vae_outputs/`.

Tested with Python 3.14, PyTorch 2.9, NumPy 2.3, pandas 3.0, SciPy 1.18 and Matplotlib 3.10.

## Data

The script expects `real_scores_6motions.mat` next to `vae_codes.py`, containing:

| Variable | Shape | Used by the script | Meaning |
| --- | --- | --- | --- |
| `scores` | 19,234 × 35 | yes | One row per frame, one column per global PCA score |
| `eigenVects` | 196,608 × 35 | no | Orthonormal PCA basis |
| `Mu` | 196,608 × 1 | no | Mean shape |

A surface is rebuilt from a score vector `x` as `Mu + eigenVects @ x`, so generated and reconstructed scores can be turned back into shapes outside this script.

## Method

**Preprocessing.** Frames are split 80/20 at random into training and validation (seed 42). Each score is standardized with its training mean and standard deviation.

**Model.** A fully connected encoder and decoder. With the default settings:

- Encoder: 35 → 128 → 128 → 32, giving 16 means and 16 log-variances
- Latent sample: $z = \mu + \varepsilon \cdot \sigma$, with $\varepsilon \sim \mathcal{N}(0, I)$
- Decoder: 16 → 128 → 128 → 35

**Loss.** Weighted reconstruction error plus β times the KL divergence to $\mathcal{N}(0, I)$:

$$
\mathcal{L} = \frac{1}{B}\sum_{b=1}^{B}\sum_{j=1}^{35} w_j\,(\hat{x}_{bj} - x_{bj})^2 \;+\; \beta \sum_{d=1}^{16} \max(\mathrm{KL}_d,\ \lambda),
\qquad
w_j = \frac{\sigma_j^2}{\frac{1}{35}\sum_k \sigma_k^2}
$$

Here $\sigma_j$ is the training standard deviation of score $j$. Because the PCA basis is orthonormal, the weighted error on standardized scores is proportional to the squared error on the surface itself. Dividing by the average variance makes the weights sum to 35, so the reconstruction term keeps the size of an unweighted loss.

**Training.**

- Adam, batch size 128, 150 epochs
- Learning rate: linear warmup to 1e-3 over 5 epochs, then cosine decay to 1e-5
- β rises linearly from 0.001 to 0.1 over the first 50 epochs (KL annealing)
- Free bits: each latent dimension's KL is not penalized below λ = 0.25 nats, which keeps dimensions from collapsing
- Gradient norm clipped at 1.0
- After every epoch the validation set is scored at β = 0.1 with fixed noise; the epoch with the lowest score from epoch 50 on is kept as the best model

**Generation.** 40,000 latent vectors are drawn from $\mathcal{N}(0, I)$ (seed 44), decoded, and converted back to original score units.

## Configuration

| Key | Default | Meaning |
| --- | --- | --- |
| `batch` | 128 | Batch size |
| `lr`, `min_lr` | 1e-3, 1e-5 | Peak and final learning rate |
| `warmup` | 5 | Warmup epochs |
| `clip` | 1.0 | Maximum gradient norm |
| `epochs` | 150 | Training epochs |
| `lat` | 16 | Latent dimensions |
| `b_min`, `b_max` | 0.001, 0.1 | Start and end of the β ramp |
| `anneal` | 50 | Epochs to reach `b_max` |
| `free_bits` | 0.25 | KL floor per latent dimension, in nats |
| `hidden` | `[128, 128]` | Encoder hidden layers, one entry per layer; the decoder mirrors them |
| `act` | `'relu'` | Activation: `relu`, `leaky_relu`, `tanh`, `silu` or `gelu` |
| `drop` | 0.0 | Dropout after each hidden layer (0 = off) |

## Outputs

Everything is written to `vae_outputs/`:

| File | Contents |
| --- | --- |
| `vae_model.pt` | Best model: weights (`sd`), epoch (`ep`, counted from 0), settings (`C`), `X_mean`, `X_std` |
| `vae_model_last.pt` | Same format, from the last epoch |
| `generated_samples.csv` | 40,000 × 35 generated score vectors, original units, no header |
| `reconstructed.csv` | 19,234 × 35 reconstructions of every frame, decoded from the posterior mean, in file order |
| `X_mean.npy`, `X_std.npy` | Training mean and standard deviation of each score |
| `latent_statistics.csv` | Per latent dimension: aggregate mean and variance, variance of the means, KL in nats, active flag |
| `summary.json` | Settings and results (best epoch, validation loss, active dimensions, reconstruction metrics) |
| `*_history.npy` | Per-epoch training and validation loss, reconstruction error, KL, β and learning rate |
| `elbo.png`, `recon.png`, `kl.png` | Training curves with the best epoch marked |

To reuse a saved model, rebuild it with the `MLP` and `VAE` definitions and the saved `C`, load `sd`, and convert outputs back with `x = output * X_std + X_mean`.

## Results

With the default settings (best epoch 150):

| Metric | Training frames | Validation frames |
| --- | --- | --- |
| Variance explained | 0.991 | 0.991 |
| Mean L2 error per frame | 1.30 | 1.33 |
| Mean absolute error per score | 0.174 | 0.178 |

All 16 latent dimensions are active. Errors are in original score units; the L2 error equals the L2 error on the surface.

The split is random over frames, and neighbouring frames of a motion sequence are nearly identical. The validation numbers therefore show how well the model fits the recorded motions, not how it generalizes to unseen subjects or motions.

## References

- Kingma & Welling (2014). *Auto-Encoding Variational Bayes.* ICLR. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- Higgins et al. (2017). *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.* ICLR.
- Bowman et al. (2016). *Generating Sentences from a Continuous Space.* CoNLL. [arXiv:1511.06349](https://arxiv.org/abs/1511.06349) (KL annealing)
- Kingma et al. (2016). *Improved Variational Inference with Inverse Autoregressive Flow.* NeurIPS. [arXiv:1606.04934](https://arxiv.org/abs/1606.04934) (free bits)
- Kingma & Ba (2015). *Adam: A Method for Stochastic Optimization.* ICLR. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- Loshchilov & Hutter (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983) (cosine learning-rate decay)
- The model structure follows the [official PyTorch VAE example](https://github.com/pytorch/examples/tree/main/vae).
