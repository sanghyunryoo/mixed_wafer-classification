# train_cvae.py

from __future__ import annotations

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons.optimizers import AdamW

# =========================
# Defaults
# =========================
SIZE = 52
CHANNELS = 1
INPUT_SHAPE = (SIZE, SIZE, CHANNELS)
LATENT_DIM = 64

BATCH_SIZE = 16
EPOCHS = 300

LR = 3e-4
WEIGHT_DECAY = 0.0
BETA_KL = 0.2


# =========================
# Data
# =========================
def standardize_and_rescale(x: np.ndarray, size: int, channels: int, scale: float = 0.23, shift: float = 0.49) -> np.ndarray:
    x = x.reshape(-1, size, size, channels).astype(np.float32)
    x = tf.image.per_image_standardization(x).numpy()
    return x * scale + shift


def build_data_paths(difficulty: str) -> tuple[str, str]:
    """
    difficulty에 따라 데이터 경로 결정

    - difficulty="data_extreme" 이면: data/data_extreme/mixed/radon/...
    - 그 외에는: data/{difficulty}/mixed/radon/...
      (네 앞 코드들과 동일한 스타일로 쓰고 싶을 때)
    """
    if difficulty == "data_extreme":
        base_dir = os.path.join("data", "data_extreme", "mixed")
    else:
        base_dir = os.path.join("data", difficulty, "mixed")

    train_path = os.path.join(base_dir, "radon", "mixed_train_data.npy")
    val_path = os.path.join(base_dir, "radon", "mixed_val_data.npy")
    return train_path, val_path


def load_train_val(train_path: str, val_path: str, size: int, channels: int) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train data not found: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"val data not found: {val_path}")

    train_x = standardize_and_rescale(np.load(train_path), size=size, channels=channels)
    val_x = standardize_and_rescale(np.load(val_path), size=size, channels=channels)
    return train_x, val_x


# =========================
# Model
# =========================
class Sampling(layers.Layer):
    def call(self, inputs):
        mu, log_var = inputs
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps


def build_encoder(input_shape: tuple[int, int, int], latent_dim: int) -> keras.Model:
    inputs = keras.Input(shape=input_shape, name="encoder_input")

    x = layers.Conv2D(64, (3, 3), activation="gelu", padding="same")(inputs)
    x = layers.Conv2D(32, (3, 3), activation="gelu", padding="same")(x)
    x = layers.Conv2D(16, (3, 3), activation="gelu", padding="same")(x)
    x = layers.Conv2D(8,  (3, 3), activation="gelu", padding="same")(x)

    x = layers.Flatten()(x)
    mu = layers.Dense(latent_dim, name="mu")(x)
    log_var = layers.Dense(latent_dim, name="log_var")(x)
    z = Sampling(name="z")([mu, log_var])

    return keras.Model(inputs, [mu, log_var, z], name="encoder")


def build_decoder(size: int, latent_dim: int, channels: int) -> keras.Model:
    latent_inputs = keras.Input(shape=(latent_dim,), name="decoder_input")

    x = layers.Dense(8 * size * size, activation="gelu")(latent_inputs)
    x = layers.Reshape((size, size, 8))(x)

    x = layers.Conv2DTranspose(8,  (3, 3), activation="gelu", padding="same")(x)
    x = layers.Conv2DTranspose(16, (3, 3), activation="gelu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation="gelu", padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation="gelu", padding="same")(x)

    outputs = layers.Conv2DTranspose(channels, (3, 3), activation=None, padding="same", name="decoded")(x)
    return keras.Model(latent_inputs, outputs, name="decoder")


def build_vae(
    input_shape: tuple[int, int, int],
    size: int,
    channels: int,
    latent_dim: int,
    beta_kl: float,
    lr: float,
    weight_decay: float,
) -> keras.Model:
    encoder = build_encoder(input_shape, latent_dim)
    decoder = build_decoder(size, latent_dim, channels)

    inputs = keras.Input(shape=input_shape, name="vae_input")
    mu, log_var, z = encoder(inputs)
    outputs = decoder(z)

    vae = keras.Model(inputs, outputs, name="cvae")

    recon = tf.reduce_mean(tf.reduce_sum(tf.square(inputs - outputs), axis=[1, 2, 3]))
    kl = tf.reduce_mean(
        -0.5 * tf.reduce_sum(1.0 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
    )
    total = recon + beta_kl * kl

    vae.add_loss(total)
    vae.add_metric(recon, name="recon_loss")
    vae.add_metric(kl, name="kl_loss")

    vae.compile(optimizer=AdamW(learning_rate=lr, weight_decay=weight_decay))
    return vae


# =========================
# Train
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=str, default="data_extreme",
                    help="예: data_extreme(기존), data_hard 등. data/{difficulty}/mixed/... 형태로 로드")
    ap.add_argument("--model_dir", type=str, default="weight/cvae",
                    help="가중치 저장 폴더 (원하면 difficulty별로 저장하도록 바꿀 수도 있음)")
    ap.add_argument("--ckpt_name", type=str, default="best.h5")

    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    ap.add_argument("--beta_kl", type=float, default=BETA_KL)

    ap.add_argument("--seed", type=int, default=1)

    args = ap.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # difficulty 반영한 데이터 경로
    train_path, val_path = build_data_paths(args.difficulty)

    # 저장 경로 (원하면 difficulty별 폴더로도 가능)
    model_dir = os.path.join(args.model_dir, args.difficulty)
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, args.ckpt_name)

    # load data
    train_x, val_x = load_train_val(train_path, val_path, size=SIZE, channels=CHANNELS)

    # build model
    vae = build_vae(
        input_shape=INPUT_SHAPE,
        size=SIZE,
        channels=CHANNELS,
        latent_dim=LATENT_DIM,
        beta_kl=args.beta_kl,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    ckpt = keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    vae.fit(
        train_x,
        epochs=args.epochs,
        batch_size=args.batch,
        validation_data=(val_x, None),
        callbacks=[ckpt],
        shuffle=True,
    )

    print(f"\n✅ Best weights saved to: {ckpt_path}")
    print(f"✅ Data used: train={train_path}, val={val_path}")


if __name__ == "__main__":
    main()
