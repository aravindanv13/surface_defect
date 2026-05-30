"""
main.py
-------
CLI entry-point for the Surface Defect Detection project.

Commands:
    train     — Train Baseline CNN and/or Transfer Learning model
    evaluate  — Evaluate saved models on the test set + generate report
    gradcam   — Run Grad-CAM on images in a directory
    preprocess — Verify dataset structure and plot class distribution

Examples:
    # Train both models
    python main.py train --data_dir data/ --epochs 30 --model both

    # Train only transfer model with ResNet18
    python main.py train --data_dir data/ --model transfer --backbone resnet18

    # Evaluate saved models
    python main.py evaluate --data_dir data/ --model both

    # Grad-CAM on a folder of images with the best model
    python main.py gradcam --image_dir data/test/crack/ --model_path models/best_model.pth --model_type transfer

    # Verify dataset and plot distributions
    python main.py preprocess --data_dir data/ --compute_stats
"""

import argparse
import sys
import os
from pathlib import Path

import torch

# ── Local imports ──────────────────────────────────────────────────────
from src.data_loader import build_dataloaders, load_single_image
from src.model_baseline import build_baseline_cnn
from src.model_transfer import TransferModel
from src.train import train_model
from src.evaluate import evaluate_model, generate_comparison_report, plot_training_curves
from src.gradcam import visualize_gradcam_batch
from src.preprocess import verify_dataset_structure, plot_class_distribution
from src.utils import (
    get_device,
    get_logger,
    set_seed,
    TrainingHistory,
    load_checkpoint,
)

logger = get_logger()


# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Surface Defect Detection — Main CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    # ── train ─────────────────────────────────────────────────────────────
    train_p = subparsers.add_parser("train", help="Train model(s)")
    train_p.add_argument("--data_dir",    type=str, default="data/",
                         help="Root dataset directory (train/val/test subfolders)")
    train_p.add_argument("--model",       type=str, default="both",
                         choices=["baseline", "transfer", "both"],
                         help="Which model to train")
    train_p.add_argument("--backbone",    type=str, default="resnet18",
                         choices=["resnet18", "resnet50", "vgg16"],
                         help="Backbone for transfer learning model")
    train_p.add_argument("--epochs",      type=int, default=30)
    train_p.add_argument("--batch_size",  type=int, default=32)
    train_p.add_argument("--lr",          type=float, default=1e-3)
    train_p.add_argument("--weight_decay",type=float, default=1e-4)
    train_p.add_argument("--scheduler",   type=str, default="cosine",
                         choices=["cosine", "step", "plateau", "onecycle"])
    train_p.add_argument("--img_size",    type=int, default=224)
    train_p.add_argument("--num_workers", type=int, default=4)
    train_p.add_argument("--model_dir",   type=str, default="models/")
    train_p.add_argument("--seed",        type=int, default=42)
    train_p.add_argument("--unfreeze_epoch", type=int, default=10,
                         help="Epoch at which to unfreeze transfer model backbone")
    train_p.add_argument("--freeze_backbone", action="store_true", default=True)
    train_p.add_argument("--no_gpu",      action="store_true",
                         help="Force CPU training")

    # ── evaluate ──────────────────────────────────────────────────────────
    eval_p = subparsers.add_parser("evaluate", help="Evaluate saved model(s)")
    eval_p.add_argument("--data_dir",    type=str, default="data/")
    eval_p.add_argument("--model",       type=str, default="both",
                        choices=["baseline", "transfer", "both"])
    eval_p.add_argument("--model_dir",   type=str, default="models/")
    eval_p.add_argument("--backbone",    type=str, default="resnet18",
                        choices=["resnet18", "resnet50", "vgg16"])
    eval_p.add_argument("--batch_size",  type=int, default=32)
    eval_p.add_argument("--img_size",    type=int, default=224)
    eval_p.add_argument("--reports_dir", type=str, default="reports/")
    eval_p.add_argument("--outputs_dir", type=str, default="outputs/")
    eval_p.add_argument("--no_gpu",      action="store_true")

    # ── gradcam ───────────────────────────────────────────────────────────
    gcam_p = subparsers.add_parser("gradcam", help="Run Grad-CAM visualisation")
    gcam_p.add_argument("--image_dir",   type=str, required=True,
                        help="Directory of images to process")
    gcam_p.add_argument("--model_path",  type=str, required=True,
                        help="Path to saved .pth checkpoint")
    gcam_p.add_argument("--model_type",  type=str, default="transfer",
                        choices=["baseline", "transfer"])
    gcam_p.add_argument("--backbone",    type=str, default="resnet18",
                        choices=["resnet18", "resnet50", "vgg16"])
    gcam_p.add_argument("--num_classes", type=int, default=4)
    gcam_p.add_argument("--class_names", nargs="+",
                        default=["crack", "dent", "no_defect", "scratch"])
    gcam_p.add_argument("--output_dir",  type=str, default="outputs/gradcam/")
    gcam_p.add_argument("--img_size",    type=int, default=224)
    gcam_p.add_argument("--no_gpu",      action="store_true")

    # ── preprocess ────────────────────────────────────────────────────────
    pre_p = subparsers.add_parser("preprocess", help="Verify dataset and plot stats")
    pre_p.add_argument("--data_dir",     type=str, default="data/")
    pre_p.add_argument("--output_dir",   type=str, default="outputs/")
    pre_p.add_argument("--compute_stats",action="store_true")

    return parser


# ─────────────────────────────────────────────
# Command handlers
# ─────────────────────────────────────────────

def cmd_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device(use_gpu=not args.no_gpu)

    # Build data loaders
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
    )
    num_classes = len(class_names)

    # Shared training config
    config = {
        "epochs"         : args.epochs,
        "lr"             : args.lr,
        "weight_decay"   : args.weight_decay,
        "scheduler"      : args.scheduler,
        "grad_clip"      : 5.0,
        "model_dir"      : args.model_dir,
        "log_dir"        : "outputs/logs/",
    }

    # ── Baseline CNN ──────────────────────────────────────────────────
    if args.model in ("baseline", "both"):
        logger.info("\n" + "═" * 60)
        logger.info("  TRAINING: Baseline CNN")
        logger.info("═" * 60)
        baseline = build_baseline_cnn(num_classes=num_classes)
        config["history_path"] = "outputs/baseline_history.json"
        config["unfreeze_epoch"] = None
        history_baseline = train_model(
            model=baseline,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            model_name="baseline_cnn",
        )

    # ── Transfer Learning ─────────────────────────────────────────────
    if args.model in ("transfer", "both"):
        logger.info("\n" + "═" * 60)
        logger.info(f"  TRAINING: Transfer Model ({args.backbone})")
        logger.info("═" * 60)
        transfer = TransferModel(
            backbone_name=args.backbone,
            num_classes=num_classes,
            freeze_backbone=True,
        )
        config["history_path"] = f"outputs/{args.backbone}_history.json"
        config["unfreeze_epoch"] = args.unfreeze_epoch
        history_transfer = train_model(
            model=transfer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            model_name=f"transfer_{args.backbone}",
        )


def cmd_evaluate(args: argparse.Namespace) -> None:
    device = get_device(use_gpu=not args.no_gpu)

    _, _, test_loader, class_names = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=2,
    )
    num_classes = len(class_names)
    all_results = {}

    # ── Baseline ──────────────────────────────────────────────────────
    if args.model in ("baseline", "both"):
        baseline_ckpt = os.path.join(args.model_dir, "baseline_cnn_best_model.pth")
        if not Path(baseline_ckpt).exists():
            # Try last checkpoint
            baseline_ckpt = os.path.join(args.model_dir, "baseline_cnn_last.pth")

        if Path(baseline_ckpt).exists():
            model = build_baseline_cnn(num_classes=num_classes)
            model, _, _ = load_checkpoint(model, baseline_ckpt, device=device)

            # Load history if available
            hist_path = "outputs/baseline_history.json"
            history = TrainingHistory.load(hist_path) if Path(hist_path).exists() else None

            metrics = evaluate_model(
                model=model,
                test_loader=test_loader,
                class_names=class_names,
                device=device,
                model_name="baseline_cnn",
                reports_dir=args.reports_dir,
                outputs_dir=args.outputs_dir,
                history=history,
            )
            all_results["Baseline CNN"] = metrics
        else:
            logger.warning(f"Baseline checkpoint not found: {baseline_ckpt}")

    # ── Transfer ──────────────────────────────────────────────────────
    if args.model in ("transfer", "both"):
        transfer_ckpt = os.path.join(
            args.model_dir, f"transfer_{args.backbone}_best_model.pth"
        )
        if not Path(transfer_ckpt).exists():
            transfer_ckpt = os.path.join(
                args.model_dir, f"transfer_{args.backbone}_last.pth"
            )

        if Path(transfer_ckpt).exists():
            model = TransferModel(
                backbone_name=args.backbone,
                num_classes=num_classes,
                freeze_backbone=False,
            )
            model, _, _ = load_checkpoint(model, transfer_ckpt, device=device)

            hist_path = f"outputs/{args.backbone}_history.json"
            history = TrainingHistory.load(hist_path) if Path(hist_path).exists() else None

            metrics = evaluate_model(
                model=model,
                test_loader=test_loader,
                class_names=class_names,
                device=device,
                model_name=f"transfer_{args.backbone}",
                reports_dir=args.reports_dir,
                outputs_dir=args.outputs_dir,
                history=history,
            )
            all_results[f"Transfer ({args.backbone})"] = metrics
        else:
            logger.warning(f"Transfer checkpoint not found: {transfer_ckpt}")

    # ── Comparison report ─────────────────────────────────────────────
    if len(all_results) > 1:
        generate_comparison_report(
            results=all_results,
            save_path=os.path.join(args.reports_dir, "model_comparison.json"),
            plot_path=os.path.join(args.reports_dir, "model_comparison.png"),
        )


def cmd_gradcam(args: argparse.Namespace) -> None:
    device = get_device(use_gpu=not args.no_gpu)

    # Gather images
    image_dir = Path(args.image_dir)
    image_paths = sorted([
        str(p) for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ])

    if not image_paths:
        logger.error(f"No images found in: {image_dir}")
        sys.exit(1)

    logger.info(f"Found {len(image_paths)} images for Grad-CAM")

    # Load model
    if args.model_type == "baseline":
        model = build_baseline_cnn(num_classes=args.num_classes)
    else:
        model = TransferModel(
            backbone_name=args.backbone,
            num_classes=args.num_classes,
            freeze_backbone=False,
        )

    model, _, _ = load_checkpoint(model, args.model_path, device=device)
    model.eval()

    visualize_gradcam_batch(
        model=model,
        image_paths=image_paths,
        class_names=args.class_names,
        device=device,
        output_dir=args.output_dir,
        img_size=args.img_size,
    )


def cmd_preprocess(args: argparse.Namespace) -> None:
    from src.preprocess import compute_dataset_stats

    summary = verify_dataset_structure(args.data_dir)
    plot_class_distribution(summary, args.output_dir)

    if args.compute_stats:
        train_dir = str(Path(args.data_dir) / "train")
        compute_dataset_stats(train_dir)


# ─────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "train"     : cmd_train,
        "evaluate"  : cmd_evaluate,
        "gradcam"   : cmd_gradcam,
        "preprocess": cmd_preprocess,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
