# visualize_data.py

from __future__ import annotations

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow_addons.optimizers import AdamW

# =========================
# CVAE config (must match training)
# =========================
SIZE = 52
CHANNELS = 1
LATENT_DIM = 64
INPUT_SHAPE = (SIZE, SIZE, CHANNELS)
BETA_KL = 0.2


class Sampling(layers.Layer):
    def call(self, inputs):
        mu, log_var = inputs
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps


def build_encoder() -> keras.Model:
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = layers.Conv2D(64, 3, activation="gelu", padding="same")(inputs)
    x = layers.Conv2D(32, 3, activation="gelu", padding="same")(x)
    x = layers.Conv2D(16, 3, activation="gelu", padding="same")(x)
    x = layers.Conv2D(8,  3, activation="gelu", padding="same")(x)
    x = layers.Flatten()(x)
    mu = layers.Dense(LATENT_DIM, name="mu")(x)
    log_var = layers.Dense(LATENT_DIM, name="log_var")(x)
    z = Sampling(name="z")([mu, log_var])
    return keras.Model(inputs, [mu, log_var, z], name="encoder")


def build_decoder() -> keras.Model:
    latent_inputs = keras.Input(shape=(LATENT_DIM,))
    x = layers.Dense(8 * SIZE * SIZE, activation="gelu")(latent_inputs)
    x = layers.Reshape((SIZE, SIZE, 8))(x)
    x = layers.Conv2DTranspose(8,  3, activation="gelu", padding="same")(x)
    x = layers.Conv2DTranspose(16, 3, activation="gelu", padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="gelu", padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="gelu", padding="same")(x)
    outputs = layers.Conv2DTranspose(1, 3, activation=None, padding="same")(x)
    return keras.Model(latent_inputs, outputs, name="decoder")


def build_cvae() -> keras.Model:
    encoder = build_encoder()
    decoder = build_decoder()

    inputs = keras.Input(shape=INPUT_SHAPE)
    mu, log_var, z = encoder(inputs)
    outputs = decoder(z)

    cvae = keras.Model(inputs, outputs, name="cvae")

    recon = tf.reduce_mean(tf.reduce_sum(tf.square(inputs - outputs), axis=[1, 2, 3]))
    kl = tf.reduce_mean(
        -0.5 * tf.reduce_sum(1.0 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
    )
    cvae.add_loss(recon + BETA_KL * kl)

    # predict만 해도 되지만, compile은 안전하게 유지
    cvae.compile(optimizer=AdamW(learning_rate=3e-4, weight_decay=0.0))
    return cvae


# =========================
# Data utils
# =========================
def ensure_nhwc(x: np.ndarray) -> np.ndarray:
    """
    Accepts:
    - (N,H,W)   -> (N,H,W,1)
    - (N,1,H,W) -> (N,H,W,1)
    - (N,H,W,1) -> 그대로
    """
    if x.ndim == 3:
        x = x[..., None]
    elif x.ndim == 4 and x.shape[1] == 1 and x.shape[-1] != 1:
        x = np.transpose(x, (0, 2, 3, 1))
    return x.astype(np.float32)


def per_image_standardize_then_rescale_np(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    xt = tf.convert_to_tensor(x, dtype=tf.float32)  # (N,H,W,1)
    xt = tf.map_fn(tf.image.per_image_standardization, xt)
    xt = xt * std + mean
    return xt.numpy()


def parse_classes(s: str, num_classes: int) -> list[int]:
    if not s.strip():
        return list(range(num_classes))
    out = [int(p.strip()) for p in s.split(",") if p.strip() != ""]
    for c in out:
        if c < 0 or c >= num_classes:
            raise ValueError(f"class index out of range: {c} (num_classes={num_classes})")
    return out


def pick_one_per_class(labels: np.ndarray, classes: list[int], seed: int, single_only: bool):
    rng = np.random.default_rng(seed)
    picks = {}
    sums = labels.sum(axis=1) if single_only else None

    for cls in classes:
        if single_only:
            idx = np.where((labels[:, cls] == 1) & (sums == 1))[0]
        else:
            idx = np.where(labels[:, cls] == 1)[0]
        picks[cls] = None if len(idx) == 0 else int(rng.choice(idx, size=1)[0])

    return picks


def build_data_paths(difficulty: str, task: str, data_type: str, split: str) -> tuple[str, str]:
    data_path = os.path.join("data", difficulty, task, data_type, f"{task}_{split}_data.npy")
    label_path = os.path.join("data", difficulty, task, "label", f"{task}_{split}_label.npy")
    return data_path, label_path


def load_split(difficulty: str, task: str, data_type: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    data_path, label_path = build_data_paths(difficulty, task, data_type, split)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data not found: {data_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"label not found: {label_path}")
    return np.load(data_path), np.load(label_path)


def resolve_cvae_weights_path(difficulty: str, cvae_path: str) -> str:
    """
    우선순위:
    1) 사용자가 --cvae_path로 직접 준 경로
    2) 기본: weight/cvae/{difficulty}/best.h5
    """
    if cvae_path.strip():
        return cvae_path

    default_path = os.path.join("weight", "cvae", difficulty, "best.h5")
    return default_path


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()

    # dataset
    ap.add_argument("--difficulty", type=str, default="data_extreme")
    ap.add_argument("--task", type=str, default="mixed")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--data_type", type=str, default="radon", help="CVAE 입력으로 사용할 데이터 타입")
    ap.add_argument("--orig_data_type", type=str, default="radon", help="원본 시각화용 데이터 타입")

    # model (✅ difficulty 반영)
    ap.add_argument(
        "--cvae_path",
        type=str,
        default="",
        help="(옵션) CVAE weights 경로 직접 지정. 비우면 weight/cvae/{difficulty}/best.h5",
    )

    # normalization (CVAE 학습과 동일)
    ap.add_argument("--norm_mean", type=float, default=0.49)
    ap.add_argument("--norm_std", type=float, default=0.23)

    # sampling / plot
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classes", type=str, default="", help="e.g. '0,1,2,3' (empty=all)")
    ap.add_argument("--single_only", action="store_true", help="sum(label)==1인 샘플만 선택")
    ap.add_argument("--thresh", type=float, default=None, help="optional binarize generated (e.g. 0.5)")
    ap.add_argument("--save_path", type=str, default="", help="optional png path")
    ap.add_argument("--show", action="store_true", default=True)

    args = ap.parse_args()

    # -------------------------
    # Load data
    # -------------------------
    orig_x, labels = load_split(args.difficulty, args.task, args.orig_data_type, args.split)
    radon_x, labels2 = load_split(args.difficulty, args.task, args.data_type, args.split)

    if not np.array_equal(labels, labels2):
        raise ValueError("Label mismatch between orig_data_type and data_type. Check folders/files.")

    orig = ensure_nhwc(orig_x)
    radon = ensure_nhwc(radon_x)

    if labels.ndim != 2:
        raise ValueError(f"label must be 2D (N,C). got {labels.shape}")

    n, num_classes = labels.shape
    if orig.shape[0] != n or radon.shape[0] != n:
        raise ValueError(f"N mismatch: orig={orig.shape[0]}, radon={radon.shape[0]}, label={n}")

    classes = parse_classes(args.classes, num_classes)
    picks = pick_one_per_class(labels, classes, seed=args.seed, single_only=args.single_only)

    chosen = [(cls, idx) for cls, idx in picks.items() if idx is not None]
    missing = [cls for cls, idx in picks.items() if idx is None]
    if missing:
        print(f"[WARN] no sample found for classes: {missing}")
    if len(chosen) == 0:
        raise RuntimeError("No samples selected. Check labels or try without --single_only.")

    idxs = [idx for _, idx in chosen]
    orig_sel = orig[idxs]
    radon_sel = radon[idxs]

    # -------------------------
    # Load CVAE weights (✅ difficulty 반영)
    # -------------------------
    cvae_weights = resolve_cvae_weights_path(args.difficulty, args.cvae_path)
    if not os.path.exists(cvae_weights):
        raise FileNotFoundError(
            f"CVAE weights not found: {cvae_weights}\n"
            f"1) --cvae_path로 정확한 경로를 지정하거나\n"
            f"2) weight/cvae/{args.difficulty}/best.h5 위치에 weights를 두세요."
        )

    cvae = build_cvae()
    cvae.load_weights(cvae_weights)
    print(f"✅ Loaded CVAE weights: {cvae_weights}")

    # normalize radon like CVAE training
    radon_norm = per_image_standardize_then_rescale_np(radon_sel, args.norm_mean, args.norm_std)
    gen = cvae.predict(radon_norm, verbose=0)
    gen = ensure_nhwc(gen)

    if args.thresh is not None:
        gen = (gen >= args.thresh).astype(np.float32)

    # -------------------------
    # Plot
    # -------------------------
    cols = len(chosen)
    fig_h = 3 * 2.8
    fig_w = max(8, cols * 2.8)

    fig, axes = plt.subplots(3, cols, figsize=(fig_w, fig_h))
    if cols == 1:
        axes = np.array(axes).reshape(3, 1)

    row_labels = ["Original", f"{args.data_type}", "Generated"]

    for j, (cls, idx) in enumerate(chosen):
        axes[0, j].imshow(orig_sel[j].squeeze())
        axes[0, j].set_title(f"Class {cls} | idx={idx}", fontsize=11)
        axes[0, j].axis("off")

        axes[1, j].imshow(radon_sel[j].squeeze())
        axes[1, j].axis("off")

        axes[2, j].imshow(gen[j].squeeze())
        axes[2, j].axis("off")

    # 라벨이 안 잘리도록 좌측 여백 확보
    plt.tight_layout(rect=(0.12, 0, 1, 1))

    for r, label in enumerate(row_labels):
        pos = axes[r, 0].get_position()
        y_center = (pos.y0 + pos.y1) / 2
        fig.text(
            0.02, y_center, label,
            va="center", ha="left",
            fontsize=14, fontweight="bold",
        )

    if args.save_path.strip():
        out_dir = os.path.dirname(args.save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(args.save_path, dpi=200, bbox_inches="tight")
        print(f"[Saved] {args.save_path}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
