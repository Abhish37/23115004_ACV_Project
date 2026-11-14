"""
metrics.py - Quantitative Evaluation Metrics for Semantic Segmentation
Author: Roll No. 23115004
Course: Advanced Computer Vision (ACV)
Assignment: B - Reproducible Segmentation Experiment

Metrics Implemented:
  - Confusion Matrix Accumulator
  - Per-Class Intersection-over-Union (IoU)
  - Mean Intersection-over-Union (mIoU) excluding ignore_index=2
  - Overall Pixel Accuracy
"""

import numpy as np
import torch

class SegmentationMetrics:
    """
    Maintains an accumulated confusion matrix across evaluation batches
    to calculate mathematically exact global segmentation metrics.
    """
    def __init__(self, num_classes: int = 3, ignore_index: int = 2):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = {0: "Background", 1: "Pet Foreground", 2: "Border (Ignored)"}
        self.reset()

    def reset(self):
        """Zeroes the confusion matrix."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Accumulates batch predictions into the confusion matrix.
        
        Args:
            preds (torch.Tensor): Logits (B, C, H, W) or class predictions (B, H, W)
            targets (torch.Tensor): Ground truth labels (B, H, W)
        """
        if preds.dim() == 4:
            preds = torch.argmax(preds, dim=1)
            
        p = preds.view(-1).cpu().numpy().astype(np.int64)
        t = targets.view(-1).cpu().numpy().astype(np.int64)

        # Mask out pixels belonging to the ignored boundary class
        valid_mask = (t != self.ignore_index)
        p_valid = p[valid_mask]
        t_valid = t[valid_mask]

        # Accumulate 2D histogram
        bincount = np.bincount(
            self.num_classes * t_valid + p_valid,
            minlength=self.num_classes ** 2
        )
        self.confusion_matrix += bincount.reshape((self.num_classes, self.num_classes))

    def compute_iou(self) -> dict:
        """
        Computes per-class IoU, Mean IoU, and Pixel Accuracy.
        """
        tp = np.diag(self.confusion_matrix)
        gt_sum = self.confusion_matrix.sum(axis=1)
        pred_sum = self.confusion_matrix.sum(axis=0)
        union = gt_sum + pred_sum - tp

        # Avoid zero division
        iou = np.where(union > 0, tp / np.maximum(union, 1e-12), 0.0)

        # Extract evaluated classes (0: Background, 1: Foreground)
        bg_iou = float(iou[0])
        fg_iou = float(iou[1])
        miou = float((bg_iou + fg_iou) / 2.0)

        total_evaluated_pixels = gt_sum[0] + gt_sum[1]
        pixel_acc = float((tp[0] + tp[1]) / np.maximum(total_evaluated_pixels, 1e-12))

        return {
            "Background_IoU": bg_iou,
            "Foreground_IoU": fg_iou,
            "mIoU": miou,
            "Pixel_Accuracy": pixel_acc,
            "Evaluated_Classes": ["Background (0)", "Foreground (1)"],
            "Ignored_Class": "Border/Trimap (2)"
        }

    def print_summary(self) -> None:
        """Prints a structured ASCII metrics table."""
        res = self.compute_iou()
        print("-" * 50)
        print("          EVALUATION METRIC BREAKDOWN             ")
        print("-" * 50)
        print(f"  Background IoU    : {res['Background_IoU'] * 100:.2f}%")
        print(f"  Foreground IoU    : {res['Foreground_IoU'] * 100:.2f}%")
        print(f"  Mean IoU (mIoU)   : {res['mIoU'] * 100:.2f}%")
        print(f"  Pixel Accuracy    : {res['Pixel_Accuracy'] * 100:.2f}%")
        print("-" * 50)
