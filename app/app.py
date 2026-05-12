"""
app.py
======
Gradio application — Gradient Descent and Optimizers Interactive Demo.

Three tabs:
  1. Loss Landscape    — Pick a surface + optimizer, watch the trajectory
  2. Optimizer Compare — Run two optimizers head-to-head on a classification task
  3. Scheduler Explorer — Visualize and compare LR schedule curves

Deployed to Hugging Face Spaces.
Run locally: python app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the src/ package is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# Add app/ itself to path for component imports
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from components.landscape_tab import (
    OPTIMIZER_MAP,
    SURFACE_MAP,
    run_landscape,
    run_landscape_comparison,
)
from components.scheduler_tab import SCHEDULER_MAP, plot_scheduler_curve
from components.training_tab import OPTIMIZER_BUILDERS, run_training_comparison

# ---------------------------------------------------------------------------
# Tab 1 — Loss Landscape
# ---------------------------------------------------------------------------


def build_landscape_tab() -> gr.Tab:
    with gr.Tab("Loss Landscape") as tab:
        gr.Markdown(
            """
            ## Loss Landscape Visualization

            Select a loss surface and an optimizer.  
            The trajectory shows where gradient descent goes from the starting point.

            **Surfaces:**
            - **Quadratic (ill-conditioned):** Elongated bowl — exposes oscillation problems
            - **Rosenbrock (banana):** Narrow curved valley — the classic optimizer benchmark
            - **Beale:** Multiple local minima
            - **Himmelblau:** Four equal global minima — initialization determines which is found
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                surface_dd = gr.Dropdown(
                    choices=list(SURFACE_MAP.keys()),
                    value="Rosenbrock (banana)",
                    label="Loss Surface",
                )
                opt_dd = gr.Dropdown(
                    choices=list(OPTIMIZER_MAP.keys()),
                    value="Adam",
                    label="Optimizer",
                )
                lr_slider = gr.Slider(
                    minimum=1e-4,
                    maximum=0.5,
                    value=0.01,
                    step=1e-4,
                    label="Learning Rate",
                )
                steps_slider = gr.Slider(
                    minimum=50,
                    maximum=1000,
                    value=300,
                    step=50,
                    label="Number of Steps",
                )
                run_btn = gr.Button("Run", variant="primary")

            with gr.Column(scale=2):
                landscape_plot = gr.Plot(label="Optimizer Trajectory")

        run_btn.click(
            fn=run_landscape,
            inputs=[surface_dd, opt_dd, lr_slider, steps_slider],
            outputs=landscape_plot,
        )

        gr.Markdown("### Multi-Optimizer Comparison")
        with gr.Row():
            with gr.Column(scale=1):
                multi_surface_dd = gr.Dropdown(
                    choices=list(SURFACE_MAP.keys()),
                    value="Quadratic (ill-conditioned)",
                    label="Surface",
                )
                multi_opts = gr.CheckboxGroup(
                    choices=list(OPTIMIZER_MAP.keys()),
                    value=["Batch GD", "SGD + Momentum", "Adam"],
                    label="Select Optimizers to Compare",
                )
                multi_lr = gr.Slider(
                    minimum=1e-4,
                    maximum=0.3,
                    value=0.01,
                    step=1e-4,
                    label="Learning Rate (shared)",
                )
                multi_steps = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=200,
                    step=50,
                    label="Steps",
                )
                multi_btn = gr.Button("Compare", variant="primary")

            with gr.Column(scale=2):
                multi_plot = gr.Plot(label="Multi-Optimizer Trajectory")

        multi_btn.click(
            fn=run_landscape_comparison,
            inputs=[multi_surface_dd, multi_opts, multi_lr, multi_steps],
            outputs=multi_plot,
        )

    return tab


# ---------------------------------------------------------------------------
# Tab 2 — Optimizer Comparison
# ---------------------------------------------------------------------------


def build_training_tab() -> gr.Tab:
    with gr.Tab("Optimizer Comparison") as tab:
        gr.Markdown(
            """
            ## Head-to-Head Optimizer Comparison

            Both optimizers train the same tiny MLP on a 2-class spiral dataset.  
            Same random seed, same model architecture, same data — only the optimizer differs.

            **What to try:**
            - Adam vs SGD (same LR) → Adam converges much faster
            - Adam vs AdamW → similar curves, AdamW generalizes better on larger models
            - SGD+Momentum vs Adam → on simple tasks, momentum SGD can match Adam
            - Lion → try a ~10x smaller LR than Adam (e.g. lr=0.0001 vs Adam lr=0.001)
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                opt1_dd = gr.Dropdown(
                    choices=list(OPTIMIZER_BUILDERS.keys()),
                    value="Adam",
                    label="Optimizer 1",
                )
                opt2_dd = gr.Dropdown(
                    choices=list(OPTIMIZER_BUILDERS.keys()),
                    value="SGD + Momentum",
                    label="Optimizer 2",
                )
                lr_in = gr.Number(value=0.01, label="Learning Rate", precision=6)
                steps_in = gr.Slider(
                    minimum=100,
                    maximum=2000,
                    value=500,
                    step=100,
                    label="Training Steps",
                )
                compare_btn = gr.Button("Compare", variant="primary")

            with gr.Column(scale=2):
                training_plot = gr.Plot(label="Loss Curves")

        compare_btn.click(
            fn=run_training_comparison,
            inputs=[opt1_dd, opt2_dd, lr_in, steps_in],
            outputs=training_plot,
        )

    return tab


# ---------------------------------------------------------------------------
# Tab 3 — Scheduler Explorer
# ---------------------------------------------------------------------------


def build_scheduler_tab() -> gr.Tab:
    with gr.Tab("Scheduler Explorer") as tab:
        gr.Markdown(
            """
            ## Learning Rate Schedule Visualizer

            Select one or more schedulers to see what LR curve they produce  
            over the chosen number of epochs.

            **Key observations:**
            - **StepLR:** Sudden drops cause training instability around step boundaries
            - **Cosine:** Smooth decay — the most widely used default
            - **OneCycleLR:** Warmup → high LR → decay. Fast.ai training recipe
            - **Warmup + Cosine:** Standard for Transformers (BERT, GPT, ViT)
            - **CyclicalLR:** Periodic high LR helps escape sharp minima
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                sched_checks = gr.CheckboxGroup(
                    choices=list(SCHEDULER_MAP.keys()),
                    value=["None (constant)", "CosineAnnealingLR", "OneCycleLR", "Warmup + Cosine"],
                    label="Schedulers to Compare",
                )
                base_lr_in = gr.Number(value=0.001, label="Base Learning Rate", precision=6)
                epochs_in = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=30,
                    step=5,
                    label="Total Epochs",
                )
                sched_btn = gr.Button("Plot Schedules", variant="primary")

            with gr.Column(scale=2):
                sched_plot = gr.Plot(label="LR Schedule Curves")

        sched_btn.click(
            fn=plot_scheduler_curve,
            inputs=[sched_checks, base_lr_in, epochs_in],
            outputs=sched_plot,
        )

    return tab


# ---------------------------------------------------------------------------
# App assembly
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Gradient Descent and Optimizers",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # Gradient Descent and Optimizers
            **Interactive optimizer and learning rate scheduler benchmark.**

            Built with [`gdo`](https://github.com/siddharth-bokka/Gradient-Descent-and-Optimizers) —
            a production-grade Python library implementing all optimizers from scratch in NumPy,
            with PyTorch training infrastructure and MLflow experiment tracking.

            ---
            """
        )
        build_landscape_tab()
        build_training_tab()
        build_scheduler_tab()

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
