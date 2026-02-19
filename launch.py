# launch.py
# GUI launcher for scripts under ./code/
#  1) code/train_cvae.py
#  2) code/visualize_data.py
#  3) code/teacher_selection.py
#  4) code/train_teacher.py
#  5) code/distillation.py

import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(PROJECT_DIR, "code")

# ---- scripts (inside code/) ----
CVAE_SCRIPT = os.path.join("code", "train_cvae.py")
VIS_SCRIPT = os.path.join("code", "visualize_data.py")
TEACHER_SEL_SCRIPT = os.path.join("code", "teacher_selection.py")
TEACHER_HYBRID_SCRIPT = os.path.join("code", "train_teacher.py")
DISTILL_SCRIPT = os.path.join("code", "distillation.py")


class LauncherApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # ---------- Window ----------
        self.title("WF/WBM Launcher (CVAE + Viz + Teacher + Distill)")
        self.geometry("1040x860")
        self.minsize(920, 700)
        self.configure(bg="#121212")

        self.current_process = None

        self._setup_style()
        self._create_widgets()
        self._center_main_window()

        self.append_log("WF/WBM GUI Launcher\n")
        self.append_log(f"Project directory: {PROJECT_DIR}\n")
        self.append_log(f"Code directory   : {CODE_DIR}\n")

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- UI / style ----------
    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        bg = "#121212"
        card_bg = "#1E1E1E"
        accent = "#3D8DFF"
        accent_hover = "#5DA0FF"
        text_main = "#FFFFFF"
        text_sub = "#C0C0C0"

        style.configure("TFrame", background=bg)
        style.configure("Card.TFrame", background=card_bg, relief="flat")

        style.configure(
            "Title.TLabel",
            background=bg,
            foreground=text_main,
            font=("Helvetica", 18, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=bg,
            foreground=text_sub,
            font=("Helvetica", 11),
        )
        style.configure(
            "CardTitle.TLabel",
            background=card_bg,
            foreground=text_main,
            font=("Helvetica", 13, "bold"),
        )
        style.configure(
            "Status.TLabel",
            background=bg,
            foreground=text_sub,
            font=("Helvetica", 10),
        )

        style.configure(
            "Accent.TButton",
            font=("Helvetica", 11, "bold"),
            padding=8,
            background=accent,
            foreground=text_main,
            borderwidth=0,
        )
        style.map(
            "Accent.TButton",
            background=[("active", accent_hover)],
            foreground=[("active", text_main)],
        )

        style.configure(
            "Secondary.TButton",
            font=("Helvetica", 11),
            padding=8,
            background="#2A2A2A",
            foreground=text_main,
            borderwidth=0,
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#3A3A3A")],
            foreground=[("active", text_main)],
        )

        style.configure(
            "Danger.TButton",
            font=("Helvetica", 11, "bold"),
            padding=8,
            background="#D9534F",
            foreground=text_main,
            borderwidth=0,
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#E06663")],
            foreground=[("active", text_main)],
        )

        # --- Custom Checkbutton indicator ---
        self._chk_off = tk.PhotoImage(
            data="iVBORw0KGgoAAAANSUhEUgAAABIAAAASCAYAAABWzo5XAAAAR0lEQVR4nGNgGGyAEV3gwIED/4nR6ODggKKXiRxD8Ko9cODAf1INQlbPhE8xKWDUoFGDqAFQ8gspKZuBATW/MeGSIMWQwQkAOJgdFRG2EegAAAAASUVORK5CYII="
        )
        self._chk_on = tk.PhotoImage(
            data="iVBORw0KGgoAAAANSUhEUgAAABIAAAASCAYAAABWzo5XAAAAeElEQVR4nM2TQRIAEAhFy8V09JwsKzNEUSt/Yyb1+kMB/CbUAWaWl0IiWmpLBuLmMrNEQXN+8ZIjSoFEZHMeBg1IrTUPOjkJgzSktfYG8roj4jZ/R9CA6NOCuI5enF1BVlcrDqB2TU/2/MX6cQHWfSvWxVx8g/ypDhwWS4eUFUI2AAAAAElFTkSuQmCC"
        )

        try:
            style.element_create(
                "VCheckbutton.indicator",
                "image",
                self._chk_off,
                ("selected", self._chk_on),
                ("disabled", self._chk_off),
                ("disabled", "selected", self._chk_on),
            )
        except tk.TclError:
            pass

        style.layout(
            "V.TCheckbutton",
            [
                ("Checkbutton.padding", {"sticky": "nswe", "children": [
                    ("VCheckbutton.indicator", {"side": "left", "sticky": ""}),
                    ("Checkbutton.focus", {"side": "left", "sticky": "w", "children": [
                        ("Checkbutton.label", {"side": "left", "sticky": ""}),
                    ]}),
                ]}),
            ],
        )
        style.configure("V.TCheckbutton", background=bg, foreground=text_main)

    def _create_widgets(self):
        # ---------- Header ----------
        header = ttk.Frame(self)
        header.pack(fill="x", padx=20, pady=(15, 10))

        title_lbl = ttk.Label(header, text="WF/WBM Launcher", style="Title.TLabel")
        title_lbl.pack(anchor="w")

        subtitle_text = (
            "  1) Activate your conda env before running this app.\n"
            "  2) Runs scripts under ./code/ with project-root as working directory.\n"
        )
        subtitle_lbl = ttk.Label(header, text=subtitle_text, style="Subtitle.TLabel", justify="left")
        subtitle_lbl.pack(anchor="w", pady=(3, 0))

        # =====================================================
        # Global Stop / Status card
        # =====================================================
        top_card = ttk.Frame(self, style="Card.TFrame")
        top_card.pack(fill="x", padx=20, pady=(0, 10), ipady=10)

        top_title = ttk.Label(top_card, text="Job Control", style="CardTitle.TLabel")
        top_title.grid(row=0, column=0, columnspan=3, sticky="w", padx=15, pady=(8, 4))

        self.btn_stop = ttk.Button(
            top_card,
            text="Stop current job",
            style="Danger.TButton",
            command=self.stop_current_process,
        )
        self.btn_stop.grid(row=1, column=2, sticky="nsew", padx=(5, 15), pady=(6, 6))

        status_frame = ttk.Frame(top_card, style="Card.TFrame")
        status_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=15, pady=(6, 6))

        self.status_label = ttk.Label(status_frame, text="Status: Idle", style="Status.TLabel")
        self.status_label.pack(side="left")

        self.progress = ttk.Progressbar(status_frame, mode="indeterminate", length=180)
        self.progress.pack(side="right")

        top_card.columnconfigure(0, weight=1)
        top_card.columnconfigure(1, weight=1)
        top_card.columnconfigure(2, weight=0)

        # =====================================================
        # Card 1) CVAE Training
        # =====================================================
        cvae_card = ttk.Frame(self, style="Card.TFrame")
        cvae_card.pack(fill="x", padx=20, pady=(0, 10), ipady=10)

        cvae_title = ttk.Label(
            cvae_card,
            text="1) Train CVAE (code/train_cvae.py)",
            style="CardTitle.TLabel",
        )
        cvae_title.grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(8, 4))

        self.btn_cvae = ttk.Button(
            cvae_card,
            text="Open CVAE config",
            style="Accent.TButton",
            command=self.open_cvae_config,
        )
        self.btn_cvae.grid(row=1, column=0, columnspan=2, sticky="ew", padx=15, pady=(6, 10))

        cvae_card.columnconfigure(0, weight=1)
        cvae_card.columnconfigure(1, weight=1)

        # =====================================================
        # Card 2) Visualization
        # =====================================================
        vis_card = ttk.Frame(self, style="Card.TFrame")
        vis_card.pack(fill="x", padx=20, pady=(0, 10), ipady=10)

        vis_title = ttk.Label(
            vis_card,
            text="2) Visualize (code/visualize_data.py)",
            style="CardTitle.TLabel",
        )
        vis_title.grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(8, 4))

        self.btn_vis = ttk.Button(
            vis_card,
            text="Open Visualization config",
            style="Accent.TButton",
            command=self.open_visualize_config,
        )
        self.btn_vis.grid(row=1, column=0, columnspan=2, sticky="ew", padx=15, pady=(6, 10))

        vis_card.columnconfigure(0, weight=1)
        vis_card.columnconfigure(1, weight=1)

        # =====================================================
        # Card 3) Teacher Selection (single-channel classifier)
        # =====================================================
        ts_card = ttk.Frame(self, style="Card.TFrame")
        ts_card.pack(fill="x", padx=20, pady=(0, 10), ipady=10)

        ts_title = ttk.Label(
            ts_card,
            text="3) Teacher Selection (code/teacher_selection.py)",
            style="CardTitle.TLabel",
        )
        ts_title.grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(8, 4))

        self.btn_ts = ttk.Button(
            ts_card,
            text="Open Teacher-Selection config",
            style="Accent.TButton",
            command=self.open_teacher_selection_config,
        )
        self.btn_ts.grid(row=1, column=0, columnspan=2, sticky="ew", padx=15, pady=(6, 10))

        ts_card.columnconfigure(0, weight=1)
        ts_card.columnconfigure(1, weight=1)

        # =====================================================
        # Card 4) Train Teacher (hybrid + CVAE)
        # =====================================================
        th_card = ttk.Frame(self, style="Card.TFrame")
        th_card.pack(fill="x", padx=20, pady=(0, 10), ipady=10)

        th_title = ttk.Label(
            th_card,
            text="4) Train Teacher Hybrid (code/train_teacher.py)",
            style="CardTitle.TLabel",
        )
        th_title.grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(8, 4))

        self.btn_th = ttk.Button(
            th_card,
            text="Open Teacher-Hybrid config",
            style="Accent.TButton",
            command=self.open_train_teacher_config,
        )
        self.btn_th.grid(row=1, column=0, columnspan=2, sticky="ew", padx=15, pady=(6, 10))

        th_card.columnconfigure(0, weight=1)
        th_card.columnconfigure(1, weight=1)

        # =====================================================
        # Card 5) Distillation
        # =====================================================
        kd_card = ttk.Frame(self, style="Card.TFrame")
        kd_card.pack(fill="x", padx=20, pady=(0, 10), ipady=10)

        kd_title = ttk.Label(
            kd_card,
            text="5) Distillation (code/distillation.py)",
            style="CardTitle.TLabel",
        )
        kd_title.grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(8, 4))

        self.btn_kd = ttk.Button(
            kd_card,
            text="Open Distillation config",
            style="Accent.TButton",
            command=self.open_distillation_config,
        )
        self.btn_kd.grid(row=1, column=0, columnspan=2, sticky="ew", padx=15, pady=(6, 10))

        kd_card.columnconfigure(0, weight=1)
        kd_card.columnconfigure(1, weight=1)

        # ---------- Log area ----------
        log_frame = ttk.Frame(self)
        log_frame.pack(fill="both", expand=True, padx=20, pady=(5, 15))

        self.log_widget = ScrolledText(
            log_frame,
            height=18,
            font=("Courier", 10),
            bg="#1E1E1E",
            fg="#F0F0F0",
            insertbackground="#FFFFFF",
            borderwidth=0,
            relief="flat",
        )
        self.log_widget.pack(fill="both", expand=True)
        self.log_widget.configure(state="disabled")

    def _center_main_window(self):
        self.update_idletasks()
        w = self.winfo_width()
        h = self.winfo_height()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = int((sw - w) / 2)
        y = int((sh - h) / 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _center_child_window(self, window, width=520, height=420):
        self.update_idletasks()
        px = self.winfo_rootx()
        py = self.winfo_rooty()
        pw = self.winfo_width()
        ph = self.winfo_height()
        x = px + (pw - width) // 2
        y = py + (ph - height) // 2
        window.geometry(f"{width}x{height}+{x}+{y}")

    # ---------- log / status ----------
    def _sanitize_log_text(self, text: str) -> str:
        text = text.replace("\b", "").replace("\x08", "")
        text = text.replace("\r", "")
        return text

    def _append_log_raw(self, text: str):
        self.log_widget.configure(state="normal")
        self.log_widget.insert(tk.END, text)
        self.log_widget.see(tk.END)
        self.log_widget.configure(state="disabled")

    def append_log(self, text: str):
        clean = self._sanitize_log_text(text)
        self.log_widget.after(0, self._append_log_raw, clean)

    def set_status(self, text: str):
        def _set():
            self.status_label.config(text=text)
        self.status_label.after(0, _set)

    # =====================================================
    # Dialog 1) CVAE training config (code/train_cvae.py)
    # =====================================================
    def open_cvae_config(self):
        dialog = tk.Toplevel(self)
        dialog.title("CVAE options (code/train_cvae.py)")
        dialog.transient(self)
        dialog.resizable(False, False)

        difficulty = tk.StringVar(value="data_extreme")
        model_dir = tk.StringVar(value="weight/cvae")
        ckpt_name = tk.StringVar(value="best.h5")

        epochs = tk.StringVar(value="300")
        batch = tk.StringVar(value="16")
        lr = tk.StringVar(value="0.0003")
        weight_decay = tk.StringVar(value="0.0")
        beta_kl = tk.StringVar(value="0.2")
        seed = tk.StringVar(value="1")

        frame = ttk.Frame(dialog)
        frame.pack(fill="both", expand=True, padx=15, pady=15)

        r = 0

        def browse_dir(var):
            path = filedialog.askdirectory(initialdir=PROJECT_DIR, title="Select directory")
            if path:
                var.set(os.path.relpath(path, PROJECT_DIR))

        def row_combo(label, var, values):
            nonlocal r
            ttk.Label(frame, text=label, style="Status.TLabel").grid(row=r, column=0, sticky="w")
            cb = ttk.Combobox(frame, textvariable=var, state="readonly", values=values, width=22)
            cb.grid(row=r, column=1, sticky="w", padx=(8, 0), pady=3)
            r += 1

        def row_entry(label, var, width=24):
            nonlocal r
            ttk.Label(frame, text=label, style="Status.TLabel").grid(row=r, column=0, sticky="w")
            ttk.Entry(frame, textvariable=var, width=width).grid(row=r, column=1, sticky="w", padx=(8, 0), pady=3)
            r += 1

        row_combo("difficulty", difficulty, ["data_easy", "data_hard", "data_extreme"])
        ttk.Label(frame, text="model_dir (base)", style="Status.TLabel").grid(row=r, column=0, sticky="w")
        ttk.Entry(frame, textvariable=model_dir, width=44).grid(row=r, column=1, sticky="w", padx=(8, 0), pady=3)
        ttk.Button(frame, text="Browse", style="Secondary.TButton", command=lambda: browse_dir(model_dir)).grid(
            row=r, column=2, sticky="w", padx=(6, 0), pady=3
        )
        r += 1

        row_entry("ckpt_name", ckpt_name, width=22)
        row_entry("epochs", epochs, width=10)
        row_entry("batch", batch, width=10)
        row_entry("lr", lr, width=12)
        row_entry("weight_decay", weight_decay, width=12)
        row_entry("beta_kl", beta_kl, width=12)
        row_entry("seed", seed, width=10)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=r, column=0, columnspan=3, sticky="e", pady=(12, 0))

        def on_run():
            try:
                int(epochs.get().strip())
                int(batch.get().strip())
                float(lr.get().strip())
                float(weight_decay.get().strip())
                float(beta_kl.get().strip())
                int(seed.get().strip())
            except ValueError:
                messagebox.showerror("Error", "epochs/batch/seed=int, lr/weight_decay/beta_kl=float")
                return

            extra = [
                "--difficulty", difficulty.get().strip(),
                "--model_dir", model_dir.get().strip(),
                "--ckpt_name", ckpt_name.get().strip(),
                "--epochs", epochs.get().strip(),
                "--batch", batch.get().strip(),
                "--lr", lr.get().strip(),
                "--weight_decay", weight_decay.get().strip(),
                "--beta_kl", beta_kl.get().strip(),
                "--seed", seed.get().strip(),
            ]

            self.append_log(f"\n[INFO] Run train_cvae.py with args:\n  {extra}\n")
            self.run_script(CVAE_SCRIPT, "CVAE Training", extra_args=extra)
            dialog.destroy()

        ttk.Button(btn_frame, text="Run", style="Accent.TButton", command=on_run).pack(side="right", padx=(6, 0))
        ttk.Button(btn_frame, text="Cancel", style="Secondary.TButton", command=dialog.destroy).pack(side="right")

        frame.columnconfigure(1, weight=1)
        self._center_child_window(dialog, width=860, height=420)

    # =====================================================
    # Dialog 2) Visualization config (code/visualize_data.py)
    # =====================================================
    def open_visualize_config(self):
        dialog = tk.Toplevel(self)
        dialog.title("Visualization options (code/visualize_data.py)")
        dialog.transient(self)
        dialog.resizable(False, False)

        difficulty = tk.StringVar(value="data_extreme")
        task = tk.StringVar(value="mixed")
        split = tk.StringVar(value="test")
        data_type = tk.StringVar(value="radon")
        orig_data_type = tk.StringVar(value="orig")

        cvae_path = tk.StringVar(value="")  # optional override

        seed = tk.StringVar(value="0")
        classes = tk.StringVar(value="")  # empty => all
        single_only = tk.BooleanVar(value=False)

        norm_mean = tk.StringVar(value="0.49")
        norm_std = tk.StringVar(value="0.23")

        thresh = tk.StringVar(value="")  # optional
        save_path = tk.StringVar(value="")  # optional png path
        show = tk.BooleanVar(value=True)

        frame = ttk.Frame(dialog)
        frame.pack(fill="both", expand=True, padx=15, pady=15)

        r = 0

        def browse_weights_or_h5(var):
            path = filedialog.askopenfilename(
                initialdir=PROJECT_DIR,
                title="Select CVAE weights file",
                filetypes=[("H5 files", "*.h5"), ("All files", "*.*")],
            )
            if path:
                var.set(os.path.relpath(path, PROJECT_DIR))

        def browse_png(var):
            path = filedialog.asksaveasfilename(
                initialdir=PROJECT_DIR,
                title="Save figure as PNG (optional)",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            )
            if path:
                var.set(os.path.relpath(path, PROJECT_DIR))

        def row_combo(label, var, values):
            nonlocal r
            ttk.Label(frame, text=label, style="Status.TLabel").grid(row=r, column=0, sticky="w")
            cb = ttk.Combobox(frame, textvariable=var, state="readonly", values=values, width=24)
            cb.grid(row=r, column=1, sticky="w", padx=(8, 0), pady=3)
            r += 1

        def row_entry(label, var, width=28):
            nonlocal r
            ttk.Label(frame, text=label, style="Status.TLabel").grid(row=r, column=0, sticky="w")
            ttk.Entry(frame, textvariable=var, width=width).grid(row=r, column=1, sticky="w", padx=(8, 0), pady=3)
            r += 1

        row_combo("difficulty", difficulty, ["data_easy", "data_hard", "data_extreme"])
        row_combo("task", task, ["mixed", "multi", "single"])
        row_combo("split", split, ["train", "val", "test"])
        row_combo("data_type (input)", data_type, ["radon", "orig", "original"])
        row_combo("orig_data_type (row1)", orig_data_type, ["radon", "orig", "original"])

        ttk.Label(frame, text="cvae_path (optional override)", style="Status.TLabel").grid(row=r, column=0, sticky="w")
        ttk.Entry(frame, textvariable=cvae_path, width=58).grid(row=r, column=1, sticky="ew", padx=(8, 0), pady=3)
        ttk.Button(frame, text="Browse", style="Secondary.TButton", command=lambda: browse_weights_or_h5(cvae_path)).grid(
            row=r, column=2, sticky="w", padx=(6, 0), pady=3
        )
        r += 1

        sep = ttk.Separator(frame, orient="horizontal")
        sep.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(10, 8))
        r += 1

        row_entry("seed", seed, width=10)
        row_entry("classes (e.g. 0,1,2) empty=all", classes, width=36)

        ttk.Checkbutton(frame, text="single_only (sum(label)==1)", variable=single_only, style="V.TCheckbutton").grid(
            row=r, column=0, sticky="w", pady=(6, 0)
        )
        r += 1

        row_entry("norm_mean", norm_mean, width=10)
        row_entry("norm_std", norm_std, width=10)
        row_entry("thresh (optional, e.g. 0.5)", thresh, width=10)

        ttk.Label(frame, text="save_path (optional .png)", style="Status.TLabel").grid(row=r, column=0, sticky="w")
        ttk.Entry(frame, textvariable=save_path, width=58).grid(row=r, column=1, sticky="ew", padx=(8, 0), pady=3)
        ttk.Button(frame, text="Browse", style="Secondary.TButton", command=lambda: browse_png(save_path)).grid(
            row=r, column=2, sticky="w", padx=(6, 0), pady=3
        )
        r += 1

        ttk.Checkbutton(frame, text="show figure (--show)", variable=show, style="V.TCheckbutton").grid(
            row=r, column=0, sticky="w", pady=(6, 0)
        )
        r += 1

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=r, column=0, columnspan=3, sticky="e", pady=(12, 0))

        def on_run():
            try:
                int(seed.get().strip())
                float(norm_mean.get().strip())
                float(norm_std.get().strip())
                if thresh.get().strip():
                    float(thresh.get().strip())
            except ValueError:
                messagebox.showerror("Error", "seed=int, norm_mean/std=float, thresh=float(optional).")
                return

            extra = [
                "--difficulty", difficulty.get().strip(),
                "--task", task.get().strip(),
                "--split", split.get().strip(),
                "--data_type", data_type.get().strip(),
                "--orig_data_type", orig_data_type.get().strip(),
                "--seed", seed.get().strip(),
                "--norm_mean", norm_mean.get().strip(),
                "--norm_std", norm_std.get().strip(),
            ]

            cv = cvae_path.get().strip()
            if cv:
                extra += ["--cvae_path", cv]

            cls_str = classes.get().strip()
            if cls_str:
                extra += ["--classes", cls_str]

            if single_only.get():
                extra += ["--single_only"]

            th = thresh.get().strip()
            if th:
                extra += ["--thresh", th]

            sp = save_path.get().strip()
            if sp:
                extra += ["--save_path", sp]

            if show.get():
                extra += ["--show"]

            self.append_log(f"\n[INFO] Run visualize_data.py with args:\n  {extra}\n")
            self.run_script(VIS_SCRIPT, "Visualization", extra_args=extra)
            dialog.destroy()

        ttk.Button(btn_frame, text="Run", style="Accent.TButton", command=on_run).pack(side="right", padx=(6, 0))
        ttk.Button(btn_frame, text="Cancel", style="Secondary.TButton", command=dialog.destroy).pack(side="right")

        frame.columnconfigure(1, weight=1)
        self._center_child_window(dialog, width=940, height=620)

    # =====================================================
    # Dialog 3) Teacher Selection (code/teacher_selection.py)
    # =====================================================
    def open_teacher_selection_config(self):
        dialog = tk.Toplevel(self)
        dialog.title("Teacher-Selection options (code/teacher_selection.py)")
        dialog.transient(self)
        dialog.resizable(False, False)

        mode = tk.StringVar(value="train")
        difficulty = tk.StringVar(value="data_extreme")
        task = tk.StringVar(value="mixed")
        data_type = tk.StringVar(value="radon")
        model = tk.StringVar(value="levit")
        seed = tk.StringVar(value="1")

        size = tk.StringVar(value="52")
        batch = tk.StringVar(value="16")
        epochs = tk.StringVar(value="30")
        num_classes = tk.StringVar(value="8")

        frame = ttk.Frame(dialog)
        frame.pack(fill="both", expand=True, padx=15, pady=15)
        r = 0

        def row_combo(label, var, values, w=24):
            nonlocal r
            ttk.Label(frame, text=label, style="Status.TLabel").grid(row=r, column=0, sticky="w")
            cb = ttk.Combobox(frame, textvariable=var, state="readonly", values=values, width=w)
            cb.grid(row=r, column=1, sticky="w", padx=(8, 0), pady=3)
            r += 1

        def row_entry(label, var, width=18):
            nonlocal r
            ttk.Label(frame, text=label, style="Status.TLabel").grid(row=r, column=0, sticky="w")
            ttk.Entry(frame, textvariable=var, width=width).grid(row=r, column=1, sticky="w", padx=(8, 0), pady=3)
            r += 1

        row_combo("mode", mode, ["train", "test"])
        row_combo("difficulty", difficulty, ["data_easy", "data_hard", "data_extreme"])
        row_combo("task", task, ["mixed", "multi", "single"])
        row_combo("data_type", data_type, ["radon", "orig", "original"])
        row_combo(
            "model",
            model,
            ["levit", "maxvit", "fastervit", "efficientnet", "swintransformer",
             "resnet", "davit", "cotnet", "edgenext", "cspnext", "mobilevit"],
            w=28
        )

        row_entry("seed", seed, width=10)
        row_entry("size", size, width=10)
        row_entry("batch", batch, width=10)
        row_entry("epochs", epochs, width=10)
        row_entry("num_classes", num_classes, width=10)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=r, column=0, columnspan=3, sticky="e", pady=(12, 0))

        def on_run():
            try:
                int(seed.get().strip())
                int(size.get().strip())
                int(batch.get().strip())
                int(epochs.get().strip())
                int(num_classes.get().strip())
            except ValueError:
                messagebox.showerror("Error", "seed/size/batch/epochs/num_classes must be int.")
                return

            extra = [
                "--mode", mode.get().strip(),
                "--difficulty", difficulty.get().strip(),
                "--task", task.get().strip(),
                "--data_type", data_type.get().strip(),
                "--model", model.get().strip(),
                "--seed", seed.get().strip(),
                "--size", size.get().strip(),
                "--batch", batch.get().strip(),
                "--epochs", epochs.get().strip(),
                "--num_classes", num_classes.get().strip(),
            ]

            self.append_log(f"\n[INFO] Run teacher_selection.py with args:\n  {extra}\n")
            self.run_script(TEACHER_SEL_SCRIPT, "Teacher Selection", extra_args=extra)
            dialog.destroy()

        ttk.Button(btn_frame, text="Run", style="Accent.TButton", command=on_run).pack(side="right", padx=(6, 0))
        ttk.Button(btn_frame, text="Cancel", style="Secondary.TButton", command=dialog.destroy).pack(side="right")

        frame.columnconfigure(1, weight=1)
        self._center_child_window(dialog, width=900, height=520)

    # =====================================================
    # Dialog 4) Train Teacher Hybrid (code/train_teacher.py)
    # =====================================================
    def open_train_teacher_config(self):
        dialog = tk.Toplevel(self)
        dialog.title("Teacher-Hybrid options (code/train_teacher.py)")
        dialog.transient(self)
        dialog.resizable(False, False)

        mode = tk.StringVar(value="train")
        difficulty = tk.StringVar(value="data_extreme")
        task = tk.StringVar(value="mixed")
        data_type = tk.StringVar(value="radon")
        model = tk.StringVar(value="fastervit")
        seed = tk.StringVar(value="1")

        size = tk.StringVar(value="52")
        batch = tk.StringVar(value="8")
        epochs = tk.StringVar(value="50")
        num_classes = tk.StringVar(value="8")
        lr = tk.StringVar(value="0.0003")

        cvae_path = tk.StringVar(value="")  # optional override
        in_channels = tk.StringVar(value="2")

        monitor_suffix = tk.StringVar(value="_hybrid")
        weights_ext = tk.StringVar(value=".h5")

        ensemble_weights = tk.StringVar(value="")  # optional, space-separated

        frame = ttk.Frame(dialog)
        frame.pack(fill="both", expand=True, padx=15, pady=15)
        r = 0

        def browse_h5(var):
            path = filedialog.askopenfilename(
                initialdir=PROJECT_DIR,
                title="Select file",
                filetypes=[("H5 files", "*.h5"), ("All files", "*.*")],
            )
            if path:
                var.set(os.path.relpath(path, PROJECT_DIR))

        def row_combo(label, var, values, w=24):
            nonlocal r
            ttk.Label(frame, text=label, style="Status.TLabel").grid(row=r, column=0, sticky="w")
            cb = ttk.Combobox(frame, textvariable=var, state="readonly", values=values, width=w)
            cb.grid(row=r, column=1, sticky="w", padx=(8, 0), pady=3)
            r += 1

        def row_entry(label, var, width=22):
            nonlocal r
            ttk.Label(frame, text=label, style="Status.TLabel").grid(row=r, column=0, sticky="w")
            ttk.Entry(frame, textvariable=var, width=width).grid(row=r, column=1, sticky="w", padx=(8, 0), pady=3)
            r += 1

        row_combo("mode", mode, ["train", "test"])
        row_combo("difficulty", difficulty, ["data_easy", "data_hard", "data_extreme"])
        row_combo("task", task, ["mixed", "multi", "single"])
        row_combo("data_type", data_type, ["radon", "orig", "original"])
        row_combo(
            "model",
            model,
            ["levit", "maxvit", "fastervit", "efficientnet", "swintransformer",
             "resnet", "davit", "cotnet", "edgenext", "cspnext", "mobilevit"],
            w=28
        )

        row_entry("seed", seed, width=10)
        row_entry("size", size, width=10)
        row_entry("batch", batch, width=10)
        row_entry("epochs", epochs, width=10)
        row_entry("num_classes", num_classes, width=10)
        row_entry("lr", lr, width=12)

        ttk.Label(frame, text="cvae_path (optional override)", style="Status.TLabel").grid(row=r, column=0, sticky="w")
        ttk.Entry(frame, textvariable=cvae_path, width=58).grid(row=r, column=1, sticky="ew", padx=(8, 0), pady=3)
        ttk.Button(frame, text="Browse", style="Secondary.TButton", command=lambda: browse_h5(cvae_path)).grid(
            row=r, column=2, sticky="w", padx=(6, 0), pady=3
        )
        r += 1

        row_entry("in_channels", in_channels, width=10)
        row_entry("monitor_suffix", monitor_suffix, width=18)
        row_entry("weights_ext", weights_ext, width=10)

        ttk.Label(frame, text="ensemble_weights (optional, space-separated)", style="Status.TLabel").grid(
            row=r, column=0, sticky="w"
        )
        ttk.Entry(frame, textvariable=ensemble_weights, width=58).grid(
            row=r, column=1, sticky="ew", padx=(8, 0), pady=3
        )
        r += 1

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=r, column=0, columnspan=3, sticky="e", pady=(12, 0))

        def on_run():
            try:
                int(seed.get().strip())
                int(size.get().strip())
                int(batch.get().strip())
                int(epochs.get().strip())
                int(num_classes.get().strip())
                float(lr.get().strip())
                int(in_channels.get().strip())
            except ValueError:
                messagebox.showerror("Error", "Check numeric fields.")
                return

            extra = [
                "--mode", mode.get().strip(),
                "--difficulty", difficulty.get().strip(),
                "--task", task.get().strip(),
                "--data_type", data_type.get().strip(),
                "--model", model.get().strip(),
                "--seed", seed.get().strip(),
                "--size", size.get().strip(),
                "--batch", batch.get().strip(),
                "--epochs", epochs.get().strip(),
                "--num_classes", num_classes.get().strip(),
                "--lr", lr.get().strip(),
                "--in_channels", in_channels.get().strip(),
                "--monitor_suffix", monitor_suffix.get().strip(),
                "--weights_ext", weights_ext.get().strip(),
            ]

            cv = cvae_path.get().strip()
            if cv:
                extra += ["--cvae_path", cv]

            ens = ensemble_weights.get().strip()
            if ens:
                # nargs="*" 형태라서 토큰으로 분리해서 전달
                extra += ["--ensemble_weights"] + ens.split()

            self.append_log(f"\n[INFO] Run train_teacher.py with args:\n  {extra}\n")
            self.run_script(TEACHER_HYBRID_SCRIPT, "Teacher Hybrid", extra_args=extra)
            dialog.destroy()

        ttk.Button(btn_frame, text="Run", style="Accent.TButton", command=on_run).pack(side="right", padx=(6, 0))
        ttk.Button(btn_frame, text="Cancel", style="Secondary.TButton", command=dialog.destroy).pack(side="right")

        frame.columnconfigure(1, weight=1)
        self._center_child_window(dialog, width=980, height=640)

    # =====================================================
    # Dialog 5) Distillation (code/distillation.py)
    # =====================================================
    def open_distillation_config(self):
        dialog = tk.Toplevel(self)
        dialog.title("Distillation options (code/distillation.py)")
        dialog.transient(self)
        dialog.resizable(False, False)

        mode = tk.StringVar(value="train")
        difficulty = tk.StringVar(value="data_extreme")
        task = tk.StringVar(value="mixed")
        data_type = tk.StringVar(value="radon")

        student_model = tk.StringVar(value="levit")
        seed = tk.StringVar(value="1")

        size = tk.StringVar(value="52")
        batch = tk.StringVar(value="16")
        epochs = tk.StringVar(value="100")
        num_classes = tk.StringVar(value="8")
        lr = tk.StringVar(value="0.0003")

        cvae_path = tk.StringVar(value="")  # optional override

        soft_w = tk.StringVar(value="1.0")
        hard_w = tk.StringVar(value="1.0")

        teacher_path = tk.StringVar(value="")  # optional override
        teacher_model = tk.StringVar(value="fastervit")
        teacher_seed = tk.StringVar(value="1")
        teacher_weights_ext = tk.StringVar(value=".h5")

        monitor_suffix = tk.StringVar(value="_distill")
        weights_ext = tk.StringVar(value=".h5")

        frame = ttk.Frame(dialog)
        frame.pack(fill="both", expand=True, padx=15, pady=15)
        r = 0

        def browse_file(var):
            path = filedialog.askopenfilename(
                initialdir=PROJECT_DIR,
                title="Select file",
                filetypes=[("H5 files", "*.h5"), ("All files", "*.*")],
            )
            if path:
                var.set(os.path.relpath(path, PROJECT_DIR))

        def row_combo(label, var, values, w=24):
            nonlocal r
            ttk.Label(frame, text=label, style="Status.TLabel").grid(row=r, column=0, sticky="w")
            cb = ttk.Combobox(frame, textvariable=var, state="readonly", values=values, width=w)
            cb.grid(row=r, column=1, sticky="w", padx=(8, 0), pady=3)
            r += 1

        def row_entry(label, var, width=22):
            nonlocal r
            ttk.Label(frame, text=label, style="Status.TLabel").grid(row=r, column=0, sticky="w")
            ttk.Entry(frame, textvariable=var, width=width).grid(row=r, column=1, sticky="w", padx=(8, 0), pady=3)
            r += 1

        row_combo("mode", mode, ["train", "test"])
        row_combo("difficulty", difficulty, ["data_easy", "data_hard", "data_extreme"])
        row_combo("task", task, ["mixed", "multi", "single"])
        row_combo("data_type", data_type, ["radon", "orig", "original"])

        row_combo(
            "student model",
            student_model,
            ["levit", "maxvit", "fastervit", "efficientnet", "swintransformer",
             "resnet", "davit", "cotnet", "edgenext", "cspnext", "mobilevit"],
            w=28
        )

        row_entry("seed", seed, width=10)
        row_entry("size", size, width=10)
        row_entry("batch", batch, width=10)
        row_entry("epochs", epochs, width=10)
        row_entry("num_classes", num_classes, width=10)
        row_entry("lr", lr, width=12)

        ttk.Label(frame, text="cvae_path (optional override)", style="Status.TLabel").grid(row=r, column=0, sticky="w")
        ttk.Entry(frame, textvariable=cvae_path, width=58).grid(row=r, column=1, sticky="ew", padx=(8, 0), pady=3)
        ttk.Button(frame, text="Browse", style="Secondary.TButton", command=lambda: browse_file(cvae_path)).grid(
            row=r, column=2, sticky="w", padx=(6, 0), pady=3
        )
        r += 1

        sep = ttk.Separator(frame, orient="horizontal")
        sep.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(10, 8))
        r += 1

        row_entry("soft_w", soft_w, width=12)
        row_entry("hard_w", hard_w, width=12)

        ttk.Label(frame, text="teacher_path (optional override)", style="Status.TLabel").grid(row=r, column=0, sticky="w")
        ttk.Entry(frame, textvariable=teacher_path, width=58).grid(row=r, column=1, sticky="ew", padx=(8, 0), pady=3)
        ttk.Button(frame, text="Browse", style="Secondary.TButton", command=lambda: browse_file(teacher_path)).grid(
            row=r, column=2, sticky="w", padx=(6, 0), pady=3
        )
        r += 1

        row_combo(
            "teacher_model (used when teacher_path is weights-only)",
            teacher_model,
            ["levit", "maxvit", "fastervit", "efficientnet", "swintransformer",
             "resnet", "davit", "cotnet", "edgenext", "cspnext", "mobilevit"],
            w=36
        )

        row_entry("teacher_seed (suffix rule)", teacher_seed, width=10)
        row_entry("teacher_weights_ext", teacher_weights_ext, width=10)

        row_entry("monitor_suffix", monitor_suffix, width=18)
        row_entry("weights_ext", weights_ext, width=10)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=r, column=0, columnspan=3, sticky="e", pady=(12, 0))

        def on_run():
            try:
                int(seed.get().strip())
                int(size.get().strip())
                int(batch.get().strip())
                int(epochs.get().strip())
                int(num_classes.get().strip())
                float(lr.get().strip())
                float(soft_w.get().strip())
                float(hard_w.get().strip())
                int(teacher_seed.get().strip())
            except ValueError:
                messagebox.showerror("Error", "Check numeric fields.")
                return

            extra = [
                "--mode", mode.get().strip(),
                "--difficulty", difficulty.get().strip(),
                "--task", task.get().strip(),
                "--data_type", data_type.get().strip(),
                "--model", student_model.get().strip(),
                "--seed", seed.get().strip(),
                "--size", size.get().strip(),
                "--batch", batch.get().strip(),
                "--epochs", epochs.get().strip(),
                "--num_classes", num_classes.get().strip(),
                "--lr", lr.get().strip(),
                "--soft_w", soft_w.get().strip(),
                "--hard_w", hard_w.get().strip(),
                "--teacher_model", teacher_model.get().strip(),
                "--teacher_seed", teacher_seed.get().strip(),
                "--teacher_weights_ext", teacher_weights_ext.get().strip(),
                "--monitor_suffix", monitor_suffix.get().strip(),
                "--weights_ext", weights_ext.get().strip(),
            ]

            cv = cvae_path.get().strip()
            if cv:
                extra += ["--cvae_path", cv]

            tp = teacher_path.get().strip()
            if tp:
                extra += ["--teacher_path", tp]

            self.append_log(f"\n[INFO] Run distillation.py with args:\n  {extra}\n")
            self.run_script(DISTILL_SCRIPT, "Distillation", extra_args=extra)
            dialog.destroy()

        ttk.Button(btn_frame, text="Run", style="Accent.TButton", command=on_run).pack(side="right", padx=(6, 0))
        ttk.Button(btn_frame, text="Cancel", style="Secondary.TButton", command=dialog.destroy).pack(side="right")

        frame.columnconfigure(1, weight=1)
        self._center_child_window(dialog, width=1020, height=760)

    # ---------- run / stop scripts ----------
    def run_script(self, script_name: str, friendly_name: str, extra_args=None):
        if self.current_process is not None and self.current_process.poll() is None:
            messagebox.showwarning(
                "Already running",
                "Another script is currently running.\n"
                "Wait until it finishes, or click 'Stop current job' first.",
            )
            return

        script_path = os.path.join(PROJECT_DIR, script_name)
        if not os.path.exists(script_path):
            messagebox.showerror("Error", f"Script not found:\n{script_path}")
            return

        def worker():
            self.set_status(f"Status: Running ({friendly_name})")
            self.progress.start(10)

            self.append_log(f"\n========== Start {script_name} ({friendly_name}) ==========\n")
            try:
                cmd = [sys.executable, script_path]
                if extra_args:
                    cmd += extra_args

                self.current_process = subprocess.Popen(
                    cmd,
                    cwd=PROJECT_DIR,                 # IMPORTANT: project-root
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                for line in self.current_process.stdout:
                    self.append_log(line)

                ret = self.current_process.wait()
                self.append_log(f"\n========== End {script_name} (return code: {ret}) ==========\n")
            except Exception as e:
                self.append_log(f"\n[ERROR] {e}\n")
            finally:
                self.current_process = None
                self.progress.stop()
                self.set_status("Status: Idle")

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def stop_current_process(self):
        if self.current_process is not None and self.current_process.poll() is None:
            if messagebox.askyesno("Stop job", "A script is running.\nTerminate it?"):
                try:
                    self.current_process.terminate()
                    self.append_log("\n[INFO] Process terminated by the user.\n")
                except Exception as e:
                    self.append_log(f"\n[ERROR] Failed to terminate process: {e}\n")
        else:
            messagebox.showinfo("Info", "There is no running process.")

    # ---------- closing ----------
    def on_close(self):
        if self.current_process is not None and self.current_process.poll() is None:
            if not messagebox.askyesno(
                "Quit", "A script is still running.\nAre you sure you want to quit?"
            ):
                return
            try:
                self.current_process.terminate()
            except Exception:
                pass
        self.destroy()


if __name__ == "__main__":
    app = LauncherApp()
    app.mainloop()
