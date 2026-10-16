"""
Unmix the Signal: Blind Source Separation with ICA implemented from scratch.
Only numpy is used for the maths. scipy is used for wav I/O and the Hungarian matching in evaluation.

Usage:
    python ica_unmix.py                      # uses synthetic demo sources
    python ica_unmix.py --sources_dir audio  # uses your own .wav files (speech, music, ambient...)
    python ica_unmix.py --noise_snr 20       # add sensor noise at 20 dB
"""
import argparse, glob, os
import numpy as np
from scipy.io import wavfile
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 16000


# ---------------------------------------------------------------- sources
def load_sources(folder, seconds=8):
    """Load mono wavs, trim to common length, normalise to zero-mean / unit-variance."""
    files = sorted(glob.glob(os.path.join(folder, "*.wav")))
    assert len(files) >= 2, "Need at least 2 wav files"
    sigs, sr0 = [], None
    for f in files:
        sr, x = wavfile.read(f)
        sr0 = sr0 or sr
        assert sr == sr0, f"Sample-rate mismatch in {f}"
        x = x.astype(np.float64)
        if x.ndim > 1:
            x = x.mean(axis=1)
        sigs.append(x)
    n = min(min(len(s) for s in sigs), int(seconds * sr0))
    S = np.stack([s[:n] for s in sigs])
    S = (S - S.mean(1, keepdims=True)) / S.std(1, keepdims=True)
    return S, sr0, [os.path.splitext(os.path.basename(f))[0] for f in files]


def synth_sources(seconds=8, sr=SR, seed=0):
    """Three synthetic non-Gaussian sources that mimic speech / music / ambient."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    # "speech": harmonic voice with a slowly changing pitch and syllable-like envelope (super-Gaussian)
    f0 = 120 + 30 * np.sin(2 * np.pi * 0.7 * t)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    voice = sum(np.sin(k * ph) / k for k in range(1, 8))
    env = np.clip(np.sin(2 * np.pi * 3 * t), 0, None) ** 2 * (np.sin(2 * np.pi * 0.4 * t) > -0.3)
    speech = voice * env
    # "music": arpeggiated notes with decay
    music = np.zeros_like(t)
    notes = [261.6, 329.6, 392.0, 523.3]
    step = int(0.25 * sr)
    for i in range(0, len(t) - step, step):
        f = notes[(i // step) % 4]
        tt = np.arange(step) / sr
        music[i:i + step] += np.sin(2 * np.pi * f * tt) * np.exp(-6 * tt)
    # "ambient": low-pass filtered noise (rain / wind-like, with sparse crackles)
    noise = rng.laplace(size=len(t))
    kernel = np.ones(40) / 40
    ambient = np.convolve(noise, kernel, mode="same")
    S = np.stack([speech, music, ambient])
    S = (S - S.mean(1, keepdims=True)) / S.std(1, keepdims=True)
    return S, sr, ["speech_synth", "music_synth", "ambient_synth"]


# ---------------------------------------------------------------- preprocessing
def center(X):
    mu = X.mean(axis=1, keepdims=True)
    return X - mu, mu


def pca_whiten(X, eps=1e-10):
    """Whitening via eigendecomposition of covariance: Z = D^-1/2 E^T X, so cov(Z) = I."""
    n = X.shape[1]
    C = (X @ X.T) / (n - 1)
    d, E = np.linalg.eigh(C)  # eigh: symmetric matrix, ascending eigenvalues
    idx = np.argsort(d)[::-1]
    d, E = d[idx], E[:, idx]
    W_white = np.diag(1.0 / np.sqrt(d + eps)) @ E.T
    return W_white @ X, W_white, E, d


# ---------------------------------------------------------------- FastICA
def sym_decorrelate(W):
    """W <- (W W^T)^(-1/2) W  keeps rows orthonormal."""
    s, U = np.linalg.eigh(W @ W.T)
    return U @ np.diag(1.0 / np.sqrt(s)) @ U.T @ W


def fastica(Z, n_components=None, nonlin="logcosh", alpha=1.0, max_iter=500, tol=1e-6, seed=0):
    """
    Symmetric FastICA on whitened data Z (m x n).
    Fixed-point update per row w:
        w+ = E[z g(w^T z)] - E[g'(w^T z)] w
    logcosh: g(u)=tanh(a u),  g'(u)=a (1 - tanh^2(a u))     (robust, general purpose)
    kurt   : g(u)=u^3,        g'(u)=3 u^2                   (kurtosis based)
    exp    : g(u)=u exp(-u^2/2), g'(u)=(1-u^2) exp(-u^2/2)  (super-Gaussian)
    """
    m, n = Z.shape
    k = n_components or m
    rng = np.random.default_rng(seed)
    W = sym_decorrelate(rng.standard_normal((k, m)))
    for it in range(max_iter):
        U = W @ Z
        if nonlin == "logcosh":
            g, gp = np.tanh(alpha * U), alpha * (1 - np.tanh(alpha * U) ** 2)
        elif nonlin == "kurt":
            g, gp = U ** 3, 3 * U ** 2
        elif nonlin == "exp":
            e = np.exp(-U ** 2 / 2)
            g, gp = U * e, (1 - U ** 2) * e
        else:
            raise ValueError(nonlin)
        W_new = (g @ Z.T) / n - np.diag(gp.mean(axis=1)) @ W
        W_new = sym_decorrelate(W_new)
        # convergence: each new row parallel to the old row (|dot| -> 1)
        delta = np.max(np.abs(np.abs(np.sum(W_new * W, axis=1)) - 1))
        W = W_new
        if delta < tol:
            break
    return W, it + 1


def ica_separate(X, **kw):
    Xc, mu = center(X)
    Z, Wwhite, E, d = pca_whiten(Xc)
    W, iters = fastica(Z, **kw)
    S_hat = W @ Z
    unmixing = W @ Wwhite  # S_hat = unmixing @ Xc
    return S_hat, unmixing, Z, iters


# ---------------------------------------------------------------- evaluation
def match_and_scale(S_true, S_est):
    """
    ICA ambiguities: arbitrary order (permutation), sign and scale.
    Resolve by (1) Hungarian matching on |corr|, (2) fixing sign by corr sign,
    (3) least-squares scale of each estimate to its matched original.
    """
    k = S_true.shape[0]
    C = np.corrcoef(S_true, S_est)[:k, k:]
    r, c = linear_sum_assignment(-np.abs(C))
    S_aligned = np.zeros_like(S_true)
    for i, j in zip(r, c):
        est = S_est[j]
        a = (est @ S_true[i]) / (est @ est)  # LS scale (also gives the sign)
        S_aligned[i] = a * est
    return S_aligned, C[r, c], c


def sdr(s, s_hat):
    """Scale-invariant-style SDR in dB: 10 log10(|s|^2 / |s - s_hat|^2) with s_hat already scaled."""
    return 10 * np.log10(np.sum(s ** 2) / (np.sum((s - s_hat) ** 2) + 1e-12))


def evaluate(S_true, S_est, label):
    S_al, corr, perm = match_and_scale(S_true, S_est)
    sdrs = np.array([sdr(S_true[i], S_al[i]) for i in range(S_true.shape[0])])
    print(f"[{label}] |corr| per source: {np.round(np.abs(corr), 4)}  SDR(dB): {np.round(sdrs, 2)}  mean SDR: {sdrs.mean():.2f}")
    return S_al, np.abs(corr), sdrs


# ---------------------------------------------------------------- IO / plots
def write_wav(path, x, sr):
    x = x / (np.max(np.abs(x)) + 1e-12) * 0.9
    wavfile.write(path, sr, (x * 32767).astype(np.int16))


def plot_signals(rows, titles, path, sr, seg=2.0):
    n = int(seg * sr)
    fig, ax = plt.subplots(len(rows), 1, figsize=(10, 1.6 * len(rows)), sharex=True)
    t = np.arange(n) / sr
    for a, x, ti in zip(np.atleast_1d(ax), rows, titles):
        a.plot(t, x[:n], lw=0.6)
        a.set_ylabel(ti, rotation=0, ha="right", va="center", fontsize=8)
    plt.xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def scatter_plot(S, X, Z, Y, path, npts=4000):
    """Joint distributions of first two channels at each stage: shows what whitening/ICA do."""
    stages = [("Sources S", S), ("Mixtures X", X), ("PCA-whitened Z", Z), ("ICA output Y", Y)]
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
    idx = np.random.default_rng(1).choice(S.shape[1], npts, replace=False)
    for a, (name, M) in zip(ax, stages):
        a.scatter(M[0, idx], M[1, idx], s=2, alpha=0.4)
        a.set_title(name)
        a.set_aspect("equal", "datalim")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def kurtosis(x):
    x = (x - x.mean()) / x.std()
    return np.mean(x ** 4) - 3


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources_dir", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--nonlin", default="logcosh", choices=["logcosh", "kurt", "exp"])
    ap.add_argument("--noise_snr", type=float, default=None, help="sensor noise SNR in dB (None = noiseless)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 1. sources + mixing with our own matrix
    S, sr, names = load_sources(args.sources_dir) if args.sources_dir else synth_sources()
    N = S.shape[0]
    A = rng.uniform(0.3, 1.0, size=(N, N))
    while np.linalg.cond(A) > 20:  # avoid ill-conditioned mixing
        A = rng.uniform(0.3, 1.0, size=(N, N))
    X = A @ S
    if args.noise_snr is not None:
        p = np.mean(X ** 2)
        X = X + rng.standard_normal(X.shape) * np.sqrt(p / 10 ** (args.noise_snr / 10))
    print("Mixing matrix A:\n", np.round(A, 3), "\ncond(A) =", round(np.linalg.cond(A), 2))
    print("Source kurtosis (non-Gaussianity):", dict(zip(names, np.round([kurtosis(s) for s in S], 2))))

    # 2-3. center + whiten
    Xc, _ = center(X)
    Z, Wwhite, E, d = pca_whiten(Xc)
    print("cov(Z) ~ I? max |cov - I| =", np.abs(np.cov(Z) - np.eye(N)).max())

    # 4. ICA
    W, iters = fastica(Z, nonlin=args.nonlin)
    Y = W @ Z
    print(f"FastICA ({args.nonlin}) converged in {iters} iterations")

    # PCA-only baseline: whitened components (uncorrelated, but not independent) with no rotation
    # 5. export
    Y_al, corr_ica, sdr_ica = evaluate(S, Y, "ICA")
    _, corr_pca, sdr_pca = evaluate(S, Z, "PCA")
    for i, nme in enumerate(names):
        write_wav(f"{args.out}/mixed_{i}.wav", X[i], sr)
        write_wav(f"{args.out}/original_{i}_{nme}.wav", S[i], sr)
        write_wav(f"{args.out}/separated_{i}_{nme}.wav", Y_al[i], sr)

    # plots
    plot_signals(list(S), [f"src {n}" for n in names], f"{args.out}/sources.png", sr)
    plot_signals(list(X), [f"mix {i}" for i in range(N)], f"{args.out}/mixtures.png", sr)
    plot_signals(list(Y_al), [f"est {n}" for n in names], f"{args.out}/separated.png", sr)
    scatter_plot(S, Xc, Z, Y, f"{args.out}/scatter_stages.png")

    # 6. bonus: PCA vs ICA bar chart + separation quality across SNR levels
    snrs = [None, 30, 20, 10, 5, 0]
    res = []
    for s in snrs:
        Xn = A @ S
        if s is not None:
            Xn = Xn + np.random.default_rng(5).standard_normal(Xn.shape) * np.sqrt(np.mean(Xn ** 2) / 10 ** (s / 10))
        Yn, _, Zn, _ = ica_separate(Xn, nonlin=args.nonlin)
        _, _, sd_i = evaluate(S, Yn, f"ICA snr={s}")
        _, _, sd_p = evaluate(S, Zn, f"PCA snr={s}")
        res.append((sd_i.mean(), sd_p.mean()))
    res = np.array(res)
    labels = ["inf" if s is None else str(s) for s in snrs]
    xs = np.arange(len(snrs))
    plt.figure(figsize=(6, 3.5))
    plt.bar(xs - 0.2, res[:, 0], 0.4, label="ICA")
    plt.bar(xs + 0.2, res[:, 1], 0.4, label="PCA")
    plt.xticks(xs, labels)
    plt.xlabel("input SNR (dB)")
    plt.ylabel("mean SDR (dB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.out}/snr_sweep.png", dpi=130)
    plt.close()
    print("Saved audio + plots to", args.out)


if __name__ == "__main__":
    main()
