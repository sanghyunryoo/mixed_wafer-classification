# teacher_selection.py

from __future__ import annotations

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))  # /root/wf_cls
PRETRAINED_LEVIT = os.path.join(PROJECT_ROOT, "weight/pretrained", "levit384_imagenet.h5")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import math
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence
from tensorflow_addons.optimizers import AdamW
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from keras_cv_attention_models import (
    levit, maxvit, fastervit, efficientnet, swin_transformer_v2, resnest,
    davit, cotnet, edgenext, cspnext, mobilevit,
)

# =========================
# Data utils
# =========================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)

def standardize_and_rescale(x: np.ndarray, size: int, scale: float = 0.23, shift: float = 0.49) -> np.ndarray:
    """원 코드: tf.image.per_image_standardization * 0.23 + 0.49"""
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
# Dataloader
# =========================
class Dataloader(Sequence):
    def __init__(self, x_set: np.ndarray, y_set: np.ndarray, batch_size: int, shuffle: bool = False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx: int):
        indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = self.x[indices].astype(np.float32)  # (B, H, W, 1)
        batch_y = self.y[indices]
        return batch_x, batch_y

    def on_epoch_end(self) -> None:
        self.indices = np.arange(len(self.x))
        if self.shuffle:
            np.random.shuffle(self.indices)

# =========================
# Model factory
# =========================
def build_classifier(model_name: str, input_shape: tuple, num_classes: int, classifier_activation: str):
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
            pretrained=PRETRAINED_LEVIT,
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

def compile_classifier(model: keras.Model, task: str) -> keras.Model:
    opt = AdamW(learning_rate=3e-4, weight_decay=0.0)
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_classes", type=int, default=8)

    args = parser.parse_args()

    set_seed(args.seed)

    size = args.size
    input_shape = (size, size, 1)

    # save / monitor paths
    save_path = f"./weight/{args.model}/{args.task}/{args.data_type}/"
    os.makedirs(save_path, exist_ok=True)
    weights_path = os.path.join(save_path, f"best_{args.seed}.weights.h5")

    save_dir = f"./result/{args.task}/{args.data_type}/{args.model}/"
    monitoring_file = os.path.join(save_dir, f"{args.seed}.csv")

    # activation
    classifier_activation = "sigmoid" if args.task in ("mixed", "multi") else "softmax"

    # build model (train/test 공통)
    model = build_classifier(
        model_name=args.model,
        input_shape=input_shape,
        num_classes=args.num_classes,
        classifier_activation=classifier_activation,
    )
    model = compile_classifier(model, args.task)

    # -------------------------
    # TEST ONLY
    # -------------------------
    if args.mode == "test":
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"[TEST MODE] 가중치 파일이 없습니다: {weights_path}\n"
                f"먼저 train 모드로 학습해서 가중치를 생성하세요."
            )

        base = f"data/{args.difficulty}/{args.task}/{args.data_type}"
        test_x = np.load(f"{base}/{args.task}_test_data.npy")
        test_x = standardize_and_rescale(test_x, size)

        label_base = f"data/{args.difficulty}/{args.task}/label"
        test_y = np.load(f"{label_base}/{args.task}_test_label.npy")

        test_loader = Dataloader(test_x, test_y, args.batch, shuffle=False)

        print("Load weights -------------------------------------------------\n")
        model.load_weights(weights_path)

        print("Evaluation -------------------------------------------------\n")
        evaluate(args.task, model, test_loader, test_y, monitoring_file)
        return

    # -------------------------
    # TRAIN (+EVAL)
    # -------------------------
    base = f"data/{args.difficulty}/{args.task}/{args.data_type}"
    train_x = np.load(f"{base}/{args.task}_train_data.npy")
    val_x   = np.load(f"{base}/{args.task}_val_data.npy")
    test_x  = np.load(f"{base}/{args.task}_test_data.npy")

    train_x = standardize_and_rescale(train_x, size)
    val_x   = standardize_and_rescale(val_x, size)
    test_x  = standardize_and_rescale(test_x, size)

    label_base = f"data/{args.difficulty}/{args.task}/label"
    train_y = np.load(f"{label_base}/{args.task}_train_label.npy")
    val_y   = np.load(f"{label_base}/{args.task}_val_label.npy")
    test_y  = np.load(f"{label_base}/{args.task}_test_label.npy")

    train_loader = Dataloader(train_x, train_y, args.batch, shuffle=True)
    val_loader   = Dataloader(val_x, val_y, args.batch, shuffle=False)
    test_loader  = Dataloader(test_x, test_y, args.batch, shuffle=False)

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
