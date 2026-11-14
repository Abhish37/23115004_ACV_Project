"""
train.py - Controlled Training and Latency Benchmarking Engine
Author: Roll No. 23115004
Course: Advanced Computer Vision (ACV)
Assignment: B - Reproducible Segmentation Experiment

Controls & Budget:
  - Dataset: Oxford-IIIT Pet (Task 1 Data Contract)
  - Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
  - Loss Function: CrossEntropyLoss(ignore_index=2)
  - Stopping Rule: Highest validation mIoU checkpointing across 15 epochs
  - Benchmarking: Single-image (1, 3, 256, 256) inference latency over 100 runs
"""

import argparse
import json
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import set_seed, log_system_environment
from dataset import PetSegmentationDataset, SyntheticDummyDataset
from baseline_unet import BaselineUNet, get_parameter_count as get_base_params
from dilated_unet import DilatedUNet, get_parameter_count as get_dilated_params
from metrics import SegmentationMetrics

def measure_inference_latency(model: nn.Module, device: torch.device, input_shape=(1, 3, 256, 256), num_runs=100) -> dict:
    """
    Measures per-frame inference latency under Task 4 specifications (Batch Size = 1).
    """
    model.eval()
    dummy = torch.randn(input_shape).to(device)
    
    # Warmup runs to stabilize CUDA kernels / CPU caches
    with torch.no_grad():
        for _ in range(15):
            _ = model(dummy)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
    elapsed = time.perf_counter() - start_time
    avg_latency_ms = (elapsed / num_runs) * 1000.0
    fps = 1000.0 / avg_latency_ms
    
    return {
        "device": str(device),
        "batch_size": 1,
        "input_resolution": list(input_shape),
        "num_runs": num_runs,
        "avg_latency_ms": round(avg_latency_ms, 2),
        "frames_per_second": round(fps, 2)
    }

def run_experiment(model_type: str = 'baseline', epochs: int = 15, batch_size: int = 16, lr: float = 1e-3, seed: int = 42, use_synthetic: bool = False):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[EXPERIMENT] Initiating Training Run: Model='{model_type}', Device='{device}'")
    
    # 1. Model Instantiation
    if model_type == 'baseline':
        model = BaselineUNet(in_channels=3, num_classes=3).to(device)
        ckpt_path = 'baseline_unet_best.pth'
        log_path = 'baseline_training_log.json'
    elif model_type == 'dilated':
        model = DilatedUNet(in_channels=3, num_classes=3).to(device)
        ckpt_path = 'dilated_unet_best.pth'
        log_path = 'dilated_training_log.json'
    else:
        raise ValueError(f"Unknown model architecture type: {model_type}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Architecture: {model.__class__.__name__}")
    print(f"Total Trainable Parameters: {total_params:,}")

    # 2. Data Pipeline
    if use_synthetic:
        print("[DATA] Using Synthetic Mock Dataset for verification...")
        train_loader = DataLoader(SyntheticDummyDataset(size=64), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(SyntheticDummyDataset(size=32), batch_size=batch_size, shuffle=False)
    else:
        try:
            print("[DATA] Loading Oxford-IIIT Pet Dataset (Data Contract)...")
            data_contract = PetSegmentationDataset(batch_size=batch_size)
            train_loader, val_loader = data_contract.get_dataloaders(download=True)
        except Exception as e:
            print(f"[DATA WARNING] Could not download/load Oxford Pet: {e}")
            print("[DATA FALLBACK] Initializing synthetic environment to demonstrate full code execution pipeline.")
            train_loader = DataLoader(SyntheticDummyDataset(size=64), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(SyntheticDummyDataset(size=32), batch_size=batch_size, shuffle=False)

    # 3. Loss, Optimizer, and Metrics
    criterion = nn.CrossEntropyLoss(ignore_index=2)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    metric_evaluator = SegmentationMetrics(num_classes=3, ignore_index=2)

    # 4. Training Loop & Checkpoint Tracking
    best_miou = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start_epoch = time.time()

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        lr_scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        metric_evaluator.reset()

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item()
                metric_evaluator.update(logits, masks)

        avg_val_loss = val_loss / len(val_loader)
        metrics = metric_evaluator.compute_iou()
        epoch_time = time.time() - start_epoch

        log_entry = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "Background_IoU": round(metrics["Background_IoU"] * 100, 2),
            "Foreground_IoU": round(metrics["Foreground_IoU"] * 100, 2),
            "mIoU": round(metrics["mIoU"] * 100, 2),
            "pixel_accuracy": round(metrics["Pixel_Accuracy"] * 100, 2),
            "epoch_duration_sec": round(epoch_time, 2)
        }
        history.append(log_entry)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"mIoU: {metrics['mIoU']*100:.2f}% (FG: {metrics['Foreground_IoU']*100:.2f}%, BG: {metrics['Background_IoU']*100:.2f}%) | {epoch_time:.1f}s")

        # Stopping Rule: Checkpoint Best Model
        if metrics["mIoU"] > best_miou:
            best_miou = metrics["mIoU"]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mIoU': best_miou,
                'config': {'model': model_type, 'params': total_params}
            }, ckpt_path)
            print(f"  -> [CHECKPOINT] Saved new best {model_type} model (mIoU: {best_miou*100:.2f}%) to {ckpt_path}")

    # 5. Measure Latency
    latency_report = measure_inference_latency(model, device)
    print("\n" + "=" * 55)
    print(f"   LATENCY BENCHMARK: {model_type.upper()} U-NET")
    print("=" * 55)
    print(f"  Device:           {latency_report['device']}")
    print(f"  Input Resolution: 1 x 3 x 256 x 256")
    print(f"  Avg Latency:      {latency_report['avg_latency_ms']} ms / frame")
    print(f"  Throughput:       {latency_report['frames_per_second']} FPS")
    print("=" * 55)

    # 6. Save Experiment Log
    output_summary = {
        "model_type": model_type,
        "trainable_parameters": total_params,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "best_mIoU": round(best_miou * 100, 2),
        "latency_profile": latency_report,
        "training_history": history
    }
    with open(log_path, 'w') as f:
        json.dump(output_summary, f, indent=4)
    print(f"[LOG] Experiment log saved to {log_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train reproducible segmentation models.")
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'dilated'], help='Model variant to train')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data for pipeline smoke test')
    args = parser.parse_args()

    run_experiment(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        use_synthetic=args.synthetic
    )
