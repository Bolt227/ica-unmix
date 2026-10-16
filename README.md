# Unmix the Signal: Blind Source Separation with ICA (from scratch)

Recovers clean audio sources from mixed recordings using **Independent Component Analysis**, implemented from scratch with NumPy (no library ICA). It includes PCA whitening via covariance and eigendecomposition, symmetric **FastICA**, and evaluation that accounts for scale and permutation ambiguity.

## How to run

```bash
pip install -r requirements.txt
python ica_unmix.py --sources_dir audio --out results
```

Optional flags: `--nonlin logcosh|kurt|exp`, `--noise_snr 20`, `--seed 0`.

## Repository contents

| Path | Description |
|---|---|
| `ica_unmix.py` | Full pipeline: mix, center, whiten, FastICA, export, evaluate |
| `audio/` | The 3 original source recordings (mono WAV, 22050 Hz) |
| `results/` | Mixtures, separated audio, original audio and plots |
| `WRITEUP.md` | Results, plots and answers to the explanation questions |
| `run_log.txt` | Console output of the run |

## Audio sources

| File | Type | Source / author / license |
|---|---|---|
| `ambient_rain_wind.wav` | Ambient | **TODO: add source, URL, license** |
| `instrumental_music.wav` | Music | **TODO: add source, URL, license** |
| `speech_synthetic_1_.wav` | Speech | **TODO: add source, e.g. TTS tool used** |

## Pipeline

1. **Mix:** the 3 sources are mixed into 3 signals with my own random mixing matrix `A` (condition number 4.59): `X = A S`.
2. **Center:** subtract the mean of each channel.
3. **Whiten:** eigendecomposition of the covariance, `Z = D^(-1/2) Eᵀ X`, so `cov(Z) = I`.
4. **FastICA:** symmetric fixed-point iteration with a `tanh` (logcosh) nonlinearity: `w+ = E[z g(wᵀz)] - E[g'(wᵀz)] w`, followed by symmetric decorrelation `W <- (W Wᵀ)^(-1/2) W`.
5. **Export:** separated signals are written as playable WAV files in `results/`.
6. **Evaluate:** each estimate is matched to an original, then compared with correlation and SDR.

### Handling ICA ambiguities
ICA recovers sources only up to permutation, sign and scale. Before scoring I:
- fix **permutation** with Hungarian matching on the absolute correlation matrix,
- fix **sign and scale** with a least-squares gain of each estimate onto its matched original.

## Results (noiseless mixing)

| Source | ICA corr | ICA SDR (dB) | PCA corr | PCA SDR (dB) |
|---|---|---|---|---|
| Ambient | 0.9999 | 35.45 | 0.64 | 2.27 |
| Music | 0.9996 | 30.90 | 0.80 | 4.47 |
| Speech | 0.9998 | 33.97 | 0.75 | 3.63 |

FastICA converged in 5 iterations. Mean SDR was **33.4 dB for ICA vs 3.5 dB for PCA**.

### Bonus: PCA vs ICA across input SNR

| Input SNR (dB) | inf | 30 | 20 | 10 | 5 | 0 |
|---|---|---|---|---|---|---|
| ICA mean SDR | 33.4 | 23.1 | 13.8 | 5.5 | 2.8 | 1.3 |
| PCA mean SDR | 3.5 | 3.4 | 3.2 | 2.1 | 1.3 | 0.8 |

Plots: `results/sources.png`, `results/mixtures.png`, `results/separated.png`, `results/scatter_stages.png`, `results/snr_sweep.png`.

See `WRITEUP.md` for the discussion.