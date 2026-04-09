# Checkpoints Summary

Best checkpoints for each experiment, evaluated on both **validation set (149 images)** and **test set (146 images)**.

> Note: val mIoU is used to select the best checkpoint during training. Test set results represent the true generalization performance.

---

## Validation Set Results (149 images)

| Experiment | Best Checkpoint | mIoU | Board IoU | Board Acc | mDice |
|---|---|---|---|---|---|
| Exp0 - B0 Baseline | best_mIoU_iter_18000.pth | **88.84%** | **79.95%** | 92.30% | **93.86%** |
| Exp1 - B1 Standard | best_mIoU_iter_10000.pth | 88.36% | 79.09% | 92.01% | 93.56% |
| **Exp2 - B1 Augmented** | best_mIoU_iter_14000.pth | 87.95% | 78.33% | 90.57% | 93.31% |
| Exp3 - Optimized | best_mIoU_iter_14000.pth | 86.65% | 76.15% | 92.79% | 92.51% |
| Exp3 - Optimized2 | best_mIoU_iter_12000.pth | 86.91% | 76.55% | 91.17% | 92.67% |
| Exp3 - Optimized3 | best_mIoU_iter_18000.pth | 88.20% | 78.87% | **94.18%** | 93.47% |

---

## Val → Test Gap Analysis

| Experiment | Val mIoU | Test mIoU | Gap |
|---|---|---|---|
| Exp0 - B0 Baseline | **88.84%** | 87.15% | -1.69% |
| Exp1 - B1 Standard | 88.36% | 84.29% | -4.07% |
| **Exp2 - B1 Augmented** ⭐ | 87.95% | **87.26%** | **-0.69%** |
| Exp3 - Optimized | 86.65% | 84.58% | -2.07% |
| Exp3 - Optimized2 | 86.91% | 84.92% | -1.99% |
| Exp3 - Optimized3 | 88.20% | 86.42% | -1.78% |

> **Key insight**: Exp2 - B1 Augmented has the smallest Val→Test gap (-0.69%), confirming it generalizes best despite not having the highest val mIoU. Exp1 - B1 Standard shows the largest gap (-4.07%), indicating overfitting without augmentation.

---

## Checkpoint Storage

All `.pth` files are stored on Google Drive and Hugging Face:
**[https://huggingface.co/ilMassy/advertising-panel-segmentation](https://huggingface.co/ilMassy/advertising-panel-segmentation)**

Checkpoints are not tracked in this GitHub repository due to file size constraints.
