# train_teacher.py

from __future__ import annotations

import os
import sys
import math
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers
from tensorflow_addons.optimizers import AdamW
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

# =========================
# Path bootstrap
# =========================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))  # /root/wf_cls
PRETRAINED_LEVIT = os.path.join(PROJECT_ROOT, "weight", "pretrained", "levit384_imagenet.h5")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from keras_cv_attention_models import (  # noqa: E402
    levit, maxvit, fastervit, efficientnet, swin_transformer_v2, resnest,
    davit, cotnet, edgenext, cspnext, mobilevit,
)

# =========================
# CVAE (build + load_weights)
# =========================
CVAE_SIZE = 52
CVAE_CHANNELS = 1
CVAE_LATENT_DIM = 64
CVAE_INPUT_SHAPE = (CVAE_SIZE, CVAE_SIZE, CVAE_CHANNELS)
CVAE_BETA_KL = 0.2
CVAE_LR = 3e-4


class Sampling(layers.Layer):
    def call(self, inputs):
        mu, log_var = inputs
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps


def build_cvae() -> keras.Model:
    # encoder
    enc_in = keras.Input(shape=CVAE_INPUT_SHAPE, name="cvae_input")
    x = layers.Conv2D(64, 3, activation="gelu", padding="same")(enc_in)
    x = layers.Conv2D(32, 3, activation="gelu", padding="same")(x)
    x = layers.Conv2D(16, 3, activation="gelu", padding="same")(x)
    x = layers.Conv2D(8, 3, activation="gelu", padding="same")(x)
    x = layers.Flatten()(x)
    mu = layers.Dense(CVAE_LATENT_DIM, name="mu")(x)
    log_var = layers.Dense(CVAE_LATENT_DIM, name="log_var")(x)
    z = Sampling(name="z")([mu, log_var])
    encoder = keras.Model(enc_in, [mu, log_var, z], name="encoder")

    # decoder
    dec_in = keras.Input(shape=(CVAE_LATENT_DIM,), name="decoder_input")
    x = layers.Dense(8 * CVAE_SIZE * CVAE_SIZE, activation="gelu")(dec_in)
    x = layers.Reshape((CVAE_SIZE, CVAE_SIZE, 8))(x)
    x = layers.Conv2DTranspose(8, 3, activation="gelu", padding="same")(x)
    x = layers.Conv2DTranspose(16, 3, activation="gelu", padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="gelu", padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="gelu", padding="same")(x)
    dec_out = layers.Conv2DTranspose(1, 3, activation=None, padding="same", name="decoded")(x)
    decoder = keras.Model(dec_in, dec_out, name="decoder")

    # cvae
    vae_in = keras.Input(shape=CVAE_INPUT_SHAPE, name="vae_input")
    mu, log_var, z = encoder(vae_in)
    vae_out = decoder(z)
    cvae = keras.Model(vae_in, vae_out, name="cvae")

    recon = tf.reduce_mean(tf.reduce_sum(tf.square(vae_in - vae_out), axis=[1, 2, 3]))
    kl = tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + log_var - tf.square(mu) - tf.exp(log_var), axis=1))
    cvae.add_loss(recon + CVAE_BETA_KL * kl)

    cvae.compile(optimizer=AdamW(learning_rate=CVAE_LR, weight_decay=0.0))
    return cvae


# =========================
# Utils
# =========================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def standardize_and_rescale(x: np.ndarray, size: int, scale: float = 0.23, shift: float = 0.49) -> np.ndarray:
    """
    x can be (N,H,W), (N,1,H,W), (N,H,W,1)...
    We'll reshape to (-1,size,size,1) as float32.
    """
    x = x.reshape(-1, size, size, 1).astype(np.float32)
    x = tf.image.per_image_standardization(x).numpy()
    return x * scale + shift


def ensure_monitor_file(path: str, task: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    if task in ("mixed", "multi"):
        df = pd.DataFrame(columns=["accuracy", "macro_f1", "micro_f1"])
    else:
        df = pd.DataFrame(columns=["accuracy", "precision", "recall", "f1"])
    df.to_csv(path, index=False)


def append_metrics(path: str, row: list) -> None:
    df = pd.read_csv(path)
    df.loc[len(df)] = row
    df.to_csv(path, index=False)


# =========================
# Hybrid DataLoader
# =========================
class HybridDataloader(Sequence):
    """
    입력: (B, H, W, 1) 원본
    -> CVAE denoise: (B, H, W, 1)
    -> concat: (B, H, W, 2)
    """

    def __init__(self, cvae_model: keras.Model, x_set: np.ndarray, y_set: np.ndarray,
                 batch_size: int, shuffle: bool = False):
        self.cvae_model = cvae_model
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(len(self.x) / self.batch_size)

    def on_epoch_end(self) -> None:
        self.indices = np.arange(len(self.x))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx: int):
        indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x = self.x[indices].astype(np.float32)  # (B,H,W,1)

        denoised = self.cvae_model.predict(batch_x, verbose=0).astype(np.float32)  # (B,H,W,1)
        hybrid_x = np.concatenate([batch_x, denoised], axis=-1)  # (B,H,W,2)

        batch_y = self.y[indices]
        return hybrid_x, batch_y


# =========================
# Model factory
# =========================
def _safe_pretrained(path: str) -> str | None:
    return path if path and os.path.exists(path) else None


def build_classifier(model_name: str, input_shape: tuple, num_classes: int, classifier_activation: str) -> keras.Model:
    common = dict(
        input_shape=input_shape,
        dropout=0.2,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        pretrained=None,
    )

    if model_name == "levit":
        return levit.LeViT384(
            input_shape=input_shape,
            num_classes=num_classes,
            use_distillation=False,
            classifier_activation=classifier_activation,
            dropout=0.2,
            pretrained=_safe_pretrained(PRETRAINED_LEVIT),
        )
    if model_name == "maxvit":
        return maxvit.MaxViT_Base(**common)
    if model_name == "fastervit":
        return fastervit.FasterViT4(**common)
    if model_name == "efficientnet":
        return efficientnet.EfficientNetV1B5(**common)
    if model_name == "swintransformer":
        return swin_transformer_v2.SwinTransformerV2Base_window16(**common)
    if model_name == "resnet":
        return resnest.ResNest101(**common)
    if model_name == "davit":
        return davit.DaViT_B(**common)
    if model_name == "cotnet":
        return cotnet.CotNetSE152D(**common)
    if model_name == "edgenext":
        return edgenext.EdgeNeXt_Base(**common)
    if model_name == "cspnext":
        return cspnext.CSPNeXtLarge(**common)
    if model_name == "mobilevit":
        return mobilevit.MobileViT_V2_050(**common)

    raise ValueError(f"Unknown --model: {model_name}")


def compile_classifier(model: keras.Model, task: str, lr: float = 3e-4) -> keras.Model:
    opt = AdamW(learning_rate=lr, weight_decay=0.0)
    if task in ("mixed", "multi"):
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    else:
        model.compile(
            optimizer=opt,
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.15),
            metrics=["accuracy"],
        )
    return model


# =========================
# Eval
# =========================
def evaluate(task: str, model: keras.Model, test_loader: Sequence, test_label: np.ndarray,
             monitoring_file: str) -> None:
    if task in ("mixed", "multi"):
        pred = model.predict(test_loader, verbose=0)
        pred = (pred > 0.5).astype(int)

        acc = accuracy_score(test_label, pred)
        macro = f1_score(test_label, pred, average="macro")
        micro = f1_score(test_label, pred, average="micro")

        print("accuracy_score :", acc)
        print("macro f1      :", macro)
        print("micro f1      :", micro)
        print(classification_report(test_label, pred))

        ensure_monitor_file(monitoring_file, task)
        append_metrics(monitoring_file, [acc, macro, micro])

    else:
        pred = model.predict(test_loader, verbose=0)
        pred = np.argmax(pred, axis=1)
        true = np.argmax(test_label, axis=1)

        acc = accuracy_score(true, pred)
        prec = precision_score(true, pred, average="weighted")
        rec = recall_score(true, pred, average="weighted")
        f1 = f1_score(true, pred, average="weighted")

        print("accuracy_score  :", acc)
        print("precision_score :", prec)
        print("recall_score    :", rec)
        print("f1_score        :", f1)

        ensure_monitor_file(monitoring_file, task)
        append_metrics(monitoring_file, [acc, prec, rec, f1])


def evaluate_ensemble_mul(task: str, models: list[keras.Model], test_loader: Sequence,
                         test_label: np.ndarray, monitoring_file: str) -> None:
    """
    mixed/multi: model.predict * model_v2.predict (원소곱) -> threshold
    """
    if task not in ("mixed", "multi"):
        raise ValueError("ensemble_mul은 mixed/multi에서만 사용하도록 설계했습니다.")

    preds = [m.predict(test_loader, verbose=0) for m in models]
    prod = preds[0]
    for p in preds[1:]:
        prod = prod * p

    out = (prod > 0.5).astype(int)

    acc = accuracy_score(test_label, out)
    macro = f1_score(test_label, out, average="macro")
    micro = f1_score(test_label, out, average="micro")

    print("accuracy_score :", acc)
    print("macro f1      :", macro)
    print("micro f1      :", micro)
    print(classification_report(test_label, out))

    ensure_monitor_file(monitoring_file, task)
    append_metrics(monitoring_file, [acc, macro, micro])


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="train: 학습(+평가), test: 가중치 로드 후 평가만")

    parser.add_argument("--difficulty", type=str, default="data_extreme")
    parser.add_argument("--task", type=str, default="mixed")
    parser.add_argument("--data_type", type=str, default="radon")
    parser.add_argument("--model", type=str, default="fastervit")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--size", type=int, default=52)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)

    # hybrid 관련
    parser.add_argument(
        "--cvae_path",
        type=str,
        default="",
        help="(옵션) CVAE weights 경로를 직접 지정. 비우면 weight/cvae/{difficulty}/best.h5 사용",
    )
    parser.add_argument("--in_channels", type=int, default=2, help="원본(1) + denoised(1) = 2")

    # 저장 / 로그
    parser.add_argument("--monitor_suffix", type=str, default="_hybrid")
    parser.add_argument("--weights_ext", type=str, default=".h5")

    # (옵션) 앙상블: 가중치 경로들을 여러 개 주면 곱셈 앙상블로 평가
    parser.add_argument("--ensemble_weights", type=str, nargs="*", default=None,
                        help="예: --ensemble_weights a.h5 b.h5 (mixed/multi에서만)")

    args = parser.parse_args()
    set_seed(args.seed)

    input_shape = (args.size, args.size, args.in_channels)

    # paths
    save_path = f"./weight/{args.model}/{args.task}/{args.data_type}/"
    os.makedirs(save_path, exist_ok=True)

    weights_path = os.path.join(save_path, f"best_{args.seed}_hybrid{args.weights_ext}")
    
    monitor_dir = f"./result/{args.task}/{args.data_type}/{args.model}/"
    monitoring_file = os.path.join(monitor_dir, f"{args.seed}{args.monitor_suffix}.csv")

    # data
    base = f"data/{args.difficulty}/{args.task}/{args.data_type}"
    label_base = f"data/{args.difficulty}/{args.task}/label"

    def load_split(split: str) -> np.ndarray:
        x = np.load(f"{base}/{args.task}_{split}_data.npy")
        return standardize_and_rescale(x, args.size)

    def load_label(split: str) -> np.ndarray:
        return np.load(f"{label_base}/{args.task}_{split}_label.npy")

    # -------------------------
    # CVAE load (difficulty 반영)
    # -------------------------
    if args.cvae_path.strip():
        cvae_path = args.cvae_path
    else:
        cvae_path = os.path.join(PROJECT_ROOT, "weight", "cvae", args.difficulty, "best.h5")

    if not os.path.exists(cvae_path):
        raise FileNotFoundError(
            f"CVAE weights not found: {cvae_path}\n"
            f"1) --cvae_path 로 정확한 경로를 지정하거나\n"
            f"2) weight/cvae/{args.difficulty}/best.h5 위치에 파일을 두세요."
        )

    cvae_model = build_cvae()
    cvae_model.load_weights(cvae_path)
    print(f"✅ CVAE Weights Loaded: {cvae_path}")

    # activation
    classifier_activation = "sigmoid" if args.task in ("mixed", "multi") else "softmax"

    # classifier build/compile
    model = build_classifier(args.model, input_shape, args.num_classes, classifier_activation)
    model = compile_classifier(model, args.task, lr=args.lr)

    # -------------------------
    # TEST ONLY
    # -------------------------
    if args.mode == "test":
        test_x = load_split("test")
        test_y = load_label("test")
        test_loader = HybridDataloader(cvae_model, test_x, test_y, args.batch, shuffle=False)

        # 앙상블이면 그 가중치 목록을 모두 체크/로드
        if args.ensemble_weights:
            if args.task not in ("mixed", "multi"):
                raise ValueError("--ensemble_weights는 mixed/multi에서만 지원합니다.")

            models = []
            for w in args.ensemble_weights:
                if not os.path.exists(w):
                    raise FileNotFoundError(f"[TEST MODE] ensemble weight not found: {w}")
                m = build_classifier(args.model, input_shape, args.num_classes, classifier_activation)
                m = compile_classifier(m, args.task, lr=args.lr)
                m.load_weights(w)
                models.append(m)

            print("Evaluation (ensemble mul) -----------------------------------\n")
            evaluate_ensemble_mul(args.task, models, test_loader, test_y, monitoring_file)
            return

        # 단일 모델 테스트: weights 없으면 에러
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"[TEST MODE] 가중치 파일이 없습니다: {weights_path}\n"
                f"먼저 --mode train 으로 학습해서 가중치를 생성하세요."
            )

        model.load_weights(weights_path)
        print(f"✅ Loaded weights: {weights_path}")

        print("Evaluation -------------------------------------------------\n")
        evaluate(args.task, model, test_loader, test_y, monitoring_file)
        return

    # -------------------------
    # TRAIN (+EVAL)
    # -------------------------
    train_x = load_split("train")
    val_x = load_split("val")
    test_x = load_split("test")

    train_y = load_label("train")
    val_y = load_label("val")
    test_y = load_label("test")

    train_loader = HybridDataloader(cvae_model, train_x, train_y, args.batch, shuffle=True)
    val_loader = HybridDataloader(cvae_model, val_x, val_y, args.batch, shuffle=False)
    test_loader = HybridDataloader(cvae_model, test_x, test_y, args.batch, shuffle=False)

    print("Training -------------------------------------------------\n")
    ckpt = keras.callbacks.ModelCheckpoint(
        filepath=weights_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    model.fit(train_loader, validation_data=val_loader, epochs=args.epochs, callbacks=[ckpt])

    print("Evaluation -------------------------------------------------\n")
    model.load_weights(weights_path)
    evaluate(args.task, model, test_loader, test_y, monitoring_file)

    print(f"\n✅ Best weights: {weights_path}")
    print(f"✅ Monitor csv : {monitoring_file}")


if __name__ == "__main__":
    main()
