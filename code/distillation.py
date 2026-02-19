# distillation.py

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

from keras_cv_attention_models_mlc import (  # noqa: E402
    levit, maxvit, fastervit, efficientnet, swin_transformer_v2, resnest,
    davit, cotnet, edgenext, cspnext, mobilevit,
)

# =========================
# Common utils
# =========================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)

def standardize_and_rescale(x: np.ndarray, size: int, scale: float = 0.23, shift: float = 0.49) -> np.ndarray:
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

def _safe_pretrained(path: str) -> str | None:
    return path if path and os.path.exists(path) else None

def _extract_probs(pred):
    if isinstance(pred, dict):
        for k in ("hard", "cl"):
            if k in pred:
                return pred[k]
        return list(pred.values())[-1]
    if isinstance(pred, (list, tuple)):
        return pred[-1]
    return pred

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
    enc_in = keras.Input(shape=CVAE_INPUT_SHAPE, name="cvae_input")
    x = layers.Conv2D(64, 3, activation="gelu", padding="same")(enc_in)
    x = layers.Conv2D(32, 3, activation="gelu", padding="same")(x)
    x = layers.Conv2D(16, 3, activation="gelu", padding="same")(x)
    x = layers.Conv2D(8,  3, activation="gelu", padding="same")(x)
    x = layers.Flatten()(x)
    mu = layers.Dense(CVAE_LATENT_DIM, name="mu")(x)
    log_var = layers.Dense(CVAE_LATENT_DIM, name="log_var")(x)
    z = Sampling(name="z")([mu, log_var])
    encoder = keras.Model(enc_in, [mu, log_var, z], name="encoder")

    dec_in = keras.Input(shape=(CVAE_LATENT_DIM,), name="decoder_input")
    x = layers.Dense(8 * CVAE_SIZE * CVAE_SIZE, activation="gelu")(dec_in)
    x = layers.Reshape((CVAE_SIZE, CVAE_SIZE, 8))(x)
    x = layers.Conv2DTranspose(8,  3, activation="gelu", padding="same")(x)
    x = layers.Conv2DTranspose(16, 3, activation="gelu", padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="gelu", padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="gelu", padding="same")(x)
    dec_out = layers.Conv2DTranspose(1, 3, activation=None, padding="same", name="decoded")(x)
    decoder = keras.Model(dec_in, dec_out, name="decoder")

    vae_in = keras.Input(shape=CVAE_INPUT_SHAPE, name="vae_input")
    mu, log_var, z = encoder(vae_in)
    vae_out = decoder(z)
    cvae = keras.Model(vae_in, vae_out, name="cvae")

    recon = tf.reduce_mean(tf.reduce_sum(tf.square(vae_in - vae_out), axis=[1, 2, 3]))
    kl = tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + log_var - tf.square(mu) - tf.exp(log_var), axis=1))
    cvae.add_loss(recon + CVAE_BETA_KL * kl)
    cvae.compile(optimizer=AdamW(learning_rate=CVAE_LR, weight_decay=0.0))
    return cvae

def load_cvae_weights(difficulty: str, override_path: str = "") -> keras.Model:
    if override_path.strip():
        cvae_path = override_path
    else:
        cvae_path = os.path.join(PROJECT_ROOT, "weight", "cvae", difficulty, "best.h5")

    if not os.path.exists(cvae_path):
        raise FileNotFoundError(
            f"CVAE weights not found: {cvae_path}\n"
            f"1) --cvae_path로 정확한 경로를 지정하거나\n"
            f"2) weight/cvae/{difficulty}/best.h5 위치에 파일을 두세요."
        )

    cvae = build_cvae()
    cvae.load_weights(cvae_path)
    print(f"✅ CVAE Weights Loaded: {cvae_path}")
    return cvae

# =========================
# Distillation loss
# =========================
def bernoulli_kl(y_true, y_pred, eps: float = 1e-7):
    y_true = tf.clip_by_value(y_true, eps, 1.0 - eps)
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    kl = y_true * tf.math.log(y_true / y_pred) + (1.0 - y_true) * tf.math.log((1.0 - y_true) / (1.0 - y_pred))
    return tf.reduce_mean(tf.reduce_sum(kl, axis=-1))

# =========================
# DataLoaders
# =========================
class HybridDataloader(Sequence):
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

class DistillDataloader(Sequence):
    """
    return: hybrid_x, {"soft": teacher_prob, "hard": label}
    """
    def __init__(self,
                 teacher_model: keras.Model,
                 cvae_model: keras.Model,
                 x_set: np.ndarray,
                 y_set: np.ndarray,
                 batch_size: int,
                 shuffle: bool = False):
        self.teacher_model = teacher_model
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
        denoised = self.cvae_model.predict(batch_x, verbose=0).astype(np.float32)
        hybrid_x = np.concatenate([batch_x, denoised], axis=-1)  # (B,H,W,2)

        t_pred = self.teacher_model.predict(hybrid_x, verbose=0)
        t_prob = _extract_probs(t_pred).astype(np.float32)  # (B,C)

        hard = self.y[indices]
        return hybrid_x, {"soft": t_prob, "hard": hard}

# =========================
# Model factory
# =========================
def build_classifier_backbone(model_name: str, input_shape: tuple, num_classes: int, classifier_activation: str) -> keras.Model:
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

def build_student_with_two_heads(backbone: keras.Model) -> keras.Model:
    y = backbone.output
    return keras.Model(backbone.input, {"soft": y, "hard": y}, name=f"{backbone.name}_distill")

def compile_student(model: keras.Model, task: str, lr: float, soft_w: float, hard_w: float) -> keras.Model:
    opt = AdamW(learning_rate=lr, weight_decay=0.0)

    if task in ("mixed", "multi"):
        model.compile(
            optimizer=opt,
            loss={"soft": bernoulli_kl, "hard": "binary_crossentropy"},
            loss_weights={"soft": soft_w, "hard": hard_w},
            metrics={"hard": ["accuracy"]},
        )
    else:
        model.compile(
            optimizer=opt,
            loss={
                "soft": tf.keras.losses.KLDivergence(),
                "hard": tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.15),
            },
            loss_weights={"soft": soft_w, "hard": hard_w},
            metrics={"hard": ["accuracy"]},
        )
    return model

# =========================
# Teacher loader (framework rule)
# =========================
def load_teacher_model(
    teacher_path: str,
    teacher_model_name: str,
    input_shape: tuple,
    num_classes: int,
    classifier_activation: str,
    lr: float,
) -> keras.Model:
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f"Teacher model not found: {teacher_path}")

    if teacher_path.endswith(".h5"):
        teacher = build_classifier_backbone(teacher_model_name, input_shape, num_classes, classifier_activation)
        # predict만 사용: compile 필요는 없지만 호환성 위해 try
        try:
            opt = AdamW(learning_rate=lr, weight_decay=0.0)
            if classifier_activation == "sigmoid":
                teacher.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
            else:
                teacher.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
        except Exception:
            pass

        teacher.load_weights(teacher_path)
        print(f"✅ Teacher weights loaded: {teacher_path}")
        return teacher

    teacher = keras.models.load_model(teacher_path, compile=False)
    print(f"✅ Teacher full model loaded: {teacher_path}")
    return teacher

# =========================
# Eval
# =========================
def evaluate(task: str, model: keras.Model, test_loader: Sequence, test_label: np.ndarray, monitoring_file: str) -> None:
    pred = model.predict(test_loader, verbose=0)
    pred = _extract_probs(pred)

    if task in ("mixed", "multi"):
        out = (pred > 0.5).astype(int)
        acc = accuracy_score(test_label, out)
        macro = f1_score(test_label, out, average="macro")
        micro = f1_score(test_label, out, average="micro")

        print("accuracy_score :", acc)
        print("macro f1      :", macro)
        print("micro f1      :", micro)
        print(classification_report(test_label, out))

        ensure_monitor_file(monitoring_file, task)
        append_metrics(monitoring_file, [acc, macro, micro])
    else:
        out = np.argmax(pred, axis=1)
        true = np.argmax(test_label, axis=1)

        acc = accuracy_score(true, out)
        prec = precision_score(true, out, average="weighted")
        rec = recall_score(true, out, average="weighted")
        f1 = f1_score(true, out, average="weighted")

        print("accuracy_score  :", acc)
        print("precision_score :", prec)
        print("recall_score    :", rec)
        print("f1_score        :", f1)

        ensure_monitor_file(monitoring_file, task)
        append_metrics(monitoring_file, [acc, prec, rec, f1])

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
    parser.add_argument("--model", type=str, default="levit")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--size", type=int, default=52)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)

    # cvae
    parser.add_argument("--cvae_path", type=str, default="",
                        help="(옵션) CVAE weights 경로 직접 지정. 비우면 weight/cvae/{difficulty}/best.h5")

    # distillation
    parser.add_argument("--soft_w", type=float, default=1.0)
    parser.add_argument("--hard_w", type=float, default=1.0)

    # teacher: teacher_suffix 없음. seed가 suffix로 붙는 규칙.
    parser.add_argument("--teacher_path", type=str, default="",
                        help="(옵션) teacher 경로 직접 지정. 비우면 규칙으로 자동 생성")
    parser.add_argument("--teacher_model", type=str, default="fastervit",
                        help="teacher 아키텍처 이름(teacher_path가 weights-only일 때 필요)")
    parser.add_argument("--teacher_seed", type=int, default=1)
    parser.add_argument("--teacher_weights_ext", type=str, default=".h5",
                        help="자동 생성 시 teacher 파일 확장자 (.h5)")

    # save/log
    parser.add_argument("--monitor_suffix", type=str, default="_distill")
    parser.add_argument("--weights_ext", type=str, default=".h5")

    args = parser.parse_args()
    set_seed(args.seed)

    # -------------------------
    # paths
    # -------------------------
    save_path = f"./weight/{args.model}/{args.task}/{args.data_type}/"
    os.makedirs(save_path, exist_ok=True)

    weights_path = os.path.join(save_path, f"best_{args.seed}_distill{args.weights_ext}")

    monitor_dir = f"./result/{args.task}/{args.data_type}/{args.model}/"
    monitoring_file = os.path.join(monitor_dir, f"{args.seed}{args.monitor_suffix}.csv")

    # -------------------------
    # data
    # -------------------------
    base = f"data/{args.difficulty}/{args.task}/{args.data_type}"
    label_base = f"data/{args.difficulty}/{args.task}/label"

    def load_split(split: str) -> np.ndarray:
        x = np.load(f"{base}/{args.task}_{split}_data.npy")
        return standardize_and_rescale(x, args.size)

    def load_label(split: str) -> np.ndarray:
        return np.load(f"{label_base}/{args.task}_{split}_label.npy")

    # -------------------------
    # load CVAE
    # -------------------------
    cvae_model = load_cvae_weights(args.difficulty, args.cvae_path)

    # -------------------------
    # activation
    # -------------------------
    classifier_activation = "sigmoid" if args.task in ("mixed", "multi") else "softmax"

    # -------------------------
    # teacher path (AUTO: best_{seed}.h5)
    # -------------------------
    if args.teacher_path.strip():
        teacher_path = args.teacher_path
    else:
        teacher_path = os.path.join(
            "weight",
            args.teacher_model,
            args.task,
            args.data_type,
            f"best_{args.teacher_seed}_hybrid{args.teacher_weights_ext}",
        )

    # teacher/student input shape: hybrid (2ch)
    input_shape = (args.size, args.size, 2)

    teacher_model = load_teacher_model(
        teacher_path=teacher_path,
        teacher_model_name=args.teacher_model,
        input_shape=input_shape,
        num_classes=args.num_classes,
        classifier_activation=classifier_activation,
        lr=args.lr,
    )

    # student
    backbone = build_classifier_backbone(args.model, input_shape, args.num_classes, classifier_activation)
    student = build_student_with_two_heads(backbone)
    student = compile_student(student, args.task, lr=args.lr, soft_w=args.soft_w, hard_w=args.hard_w)

    # -------------------------
    # TEST ONLY
    # -------------------------
    if args.mode == "test":
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"[TEST MODE] 가중치 파일이 없습니다: {weights_path}\n"
                f"먼저 --mode train 으로 학습해서 가중치를 생성하세요."
            )

        student.load_weights(weights_path)
        print(f"✅ Loaded student weights: {weights_path}")

        test_x = load_split("test")
        test_y = load_label("test")
        test_loader = HybridDataloader(cvae_model, test_x, test_y, args.batch, shuffle=False)

        print("Evaluation -------------------------------------------------\n")
        evaluate(args.task, student, test_loader, test_y, monitoring_file)
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

    train_loader = DistillDataloader(teacher_model, cvae_model, train_x, train_y, args.batch, shuffle=True)
    val_loader = DistillDataloader(teacher_model, cvae_model, val_x, val_y, args.batch, shuffle=False)
    test_loader = HybridDataloader(cvae_model, test_x, test_y, args.batch, shuffle=False)

    print("Training -------------------------------------------------\n")
    ckpt = keras.callbacks.ModelCheckpoint(
        filepath=weights_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    student.fit(train_loader, validation_data=val_loader, epochs=args.epochs, callbacks=[ckpt])

    print("Evaluation -------------------------------------------------\n")
    student.load_weights(weights_path)
    evaluate(args.task, student, test_loader, test_y, monitoring_file)

    print(f"\n✅ Best student weights: {weights_path}")
    print(f"✅ Monitor csv         : {monitoring_file}")

if __name__ == "__main__":
    main()
