## Segmentazione Avanzata di Pannelli Pubblicitari (High-Res Transformer Approach)

### 🇮🇹 Descrizione del Progetto
Sviluppo di un sistema di Semantic Segmentation ad alta precisione per
l'individuazione di pannelli pubblicitari in ambito broadcast sportivo.
A differenza degli approcci standard (CNN-based), il progetto adotta
l'architettura SegFormer, sfruttando un Hierarchical Transformer Encoder
per gestire la coerenza spaziale e la variabilità di risoluzione.
L'obiettivo è superare i limiti della bassa risoluzione (640x640)
utilizzando un dataset custom a 1080p (967 immagini annotate manualmente
su CVAT.ai) per garantire la massima precisione sui bordi e nelle
condizioni di occlusione critica. Il progetto include un confronto
sperimentale tra SegFormer-B0 (baseline) e SegFormer-B1 con
augmentazioni sport-specific e ottimizzazioni architetturali.

### 🇬🇧 Project Description
Development of a high-precision Semantic Segmentation system for
advertising board detection in sports broadcasting. Unlike standard
CNN-based approaches, this project employs the SegFormer architecture,
utilizing a Hierarchical Transformer Encoder to manage spatial coherence
and resolution variability. The goal is to overcome low-resolution
limitations (640x640) by using a custom 1080p dataset (967 manually
annotated images via CVAT.ai) to ensure maximum edge precision and
robustness under critical occlusion conditions. The project includes
an experimental comparison between SegFormer-B0 (baseline) and
SegFormer-B1 with sport-specific augmentations and architectural
optimizations.

---

## 🛠️ Background e Tecniche Utilizzate

- **Architettura**: SegFormer (Mix Transformer Encoder + Lightweight MLP Decoder)
- **Framework**: PyTorch 2.4.1, MMSegmentation 1.2.2, mmcv 2.2.0
- **Dataset**: Dataset custom di broadcast sportivo, 967 immagini annotate manualmente a 1920x1080p
- **Task**: Semantic Segmentation binaria (background vs pannello pubblicitario)
- **Approccio**: Fine-tuning su pesi pre-addestrati ImageNet (MIT-B0 e MIT-B1)

---

## ⚙️ Installazione

### ☁️ Google Colab (Consigliato)

#### Phase 2-3: Baseline e SegFormer-B1 Standard
1. Apri `notebooks/Segformer_training.ipynb` su Google Colab
2. Vai su `Opzioni di connessione aggiuntive → Cambia tipo di runtime → T4 GPU`
3. Esegui le celle in ordine dalla 1 alla 11

#### Phase 4: Augmentazione Sport-Specific
1. Apri `notebooks/Segformer_augmented.ipynb` su Google Colab
2. Vai su `Opzioni di connessione aggiuntive → Cambia tipo di runtime → T4 GPU`
3. Esegui le celle in ordine dalla 1 alla 11

#### Phase 5: Architectural Optimization
1. Apri `notebooks/Segformer_optimized*.ipynb` su Google Colab
2. Vai su `Opzioni di connessione aggiuntive → Cambia tipo di runtime → T4 GPU`
3. Esegui le celle in ordine dalla 1 alla 11

#### Phase 6: Critical Analysis & Benchmarking
1. Apri `notebooks/Segformer_analysis.ipynb` su Google Colab
2. Vai su `Opzioni di connessione aggiuntive → Cambia tipo di runtime → T4 GPU`
3. Esegui le celle in ordine dalla 1 alla 10

### 💻 Installazione Locale (CPU/GPU)
```bash
# 1. Clone the repository
git clone https://github.com/ilMassy/advertising-panel-segmentation
cd advertising-panel-segmentation

# 2. Create conda environment
conda create -n boards python=3.10 -y
conda activate boards

# 3. Install dependencies
pip install -r requirements.txt

# 4. Fix mmcv version check
sed -i 's/MMCV_MAX = .2.2.0./MMCV_MAX = "2.3.0"/' \
    $(python -c "import mmseg, os; print(os.path.dirname(mmseg.__file__))")/__init__.py
```

---

## 📁 Struttura del Repository
```
advertising-panel-segmentation/
├── configs/
│   ├── segformer_b0_baseline.py          # SegFormer-B0 baseline config (Phase 2)
│   ├── segformer_b1_augmented.py         # SegFormer-B1 augmented config (Phase 4)
│   ├── segformer_b1_optimized.py         # SegFormer-B1 optimized config (Phase 5 - opt1)
│   ├── segformer_b1_optimized2.py        # SegFormer-B1 optimized config (Phase 5 - opt2)
│   ├── segformer_b1_optimized3.py        # SegFormer-B1 optimized config (Phase 5 - opt3)
│   └── segformer_b1_standard.py          # SegFormer-B1 standard config (Phase 3)
├── models/
│   ├── checkpoints/
│   │   └── README.md                     # Val + Test results and Val→Test gap analysis
│   └── README.md                         # Performance summary of trained models
├── notebooks/
│   ├── Segformer_analysis.ipynb          # Critical analysis notebook Phase 6
│   ├── Segformer_augmented.ipynb         # Training notebook Phase 4
│   ├── Segformer_optimized.ipynb         # Training notebook Phase 5 - opt1
│   ├── Segformer_optimized2.ipynb        # Training notebook Phase 5 - opt2
│   ├── Segformer_optimized3.ipynb        # Training notebook Phase 5 - opt3
│   ├── Segformer_training.ipynb          # Training notebook Phase 2-3
│   └── Upload_to_HuggingFace.ipynb       # Upload models and dataset to Hugging Face
├── results/
│   ├── dataset_previews/
│   │   └── mask_check_preview.png        # Green overlay confirming high-resolution mask precision (see src/check_masks.py)
│   ├── exp0_segformer_b0_baseline/       # SegFormer-B0 baseline training logs
│   ├── exp1_segformer_b1_standard/       # SegFormer-B1 standard training logs
│   ├── exp2_segformer_b1_augmented/      # SegFormer-B1 augmented training logs
│   ├── exp3_segformer_b1_optimized/      # SegFormer-B1 optimized training logs (opt1)
│   ├── exp3_segformer_b1_optimized2/     # SegFormer-B1 optimized training logs (opt2)
│   ├── exp3_segformer_b1_optimized3/     # SegFormer-B1 optimized training logs (opt3)
│   ├── attention_maps_comparison.png     # Attention maps: B0 vs B1 Standard vs B1 Augmented (Phase 6)
│   ├── benchmarking_complete.png         # All 6 models Val vs Test mIoU, Val→Test gap, Precision vs Recall (Phase 6)
│   ├── best_cases.png                    # Top-5 best predictions of B1 Augmented on test set (Board IoU 0.945–0.962) (Phase 6)
│   ├── error_analysis.png                # TP/FP/FN distribution, IoU/Precision/Recall boxplots: B0 vs B1 Standard vs B1 Augmented (Phase 6)
│   └── worst_cases.png                   # Top-5 worst predictions of B1 Augmented on test set (Board IoU 0.291–0.435) (Phase 6)
├── src/
│   ├── CVAT_preparation.py               # COCO JSON → binary PNG masks
│   ├── check_masks.py                    # Visual mask verification
│   ├── extract_frames.py                 # Frame extraction from video
│   └── reorder_by_prefix.py              # Sequential frame renaming
├── .gitignore                            # Git ignore rules (checkpoints, pycache, DS_Store)
├── README.md                             # Project overview, installation, results, references
└── requirements.txt                      # Python dependencies with pinned versions
```

---

## 🧪 Esperimenti
| Esperimento | Modello | Augmentation | mIoU | Board IoU | Dice | Precision | Recall |
|---|---|---|---|---|---|---|---|
| Exp0 - Baseline | SegFormer-B0 | Standard | 87.15% | 76.28% | 86.55% | 84.13% | 89.10% |
| Exp1 - Standard | SegFormer-B1 | Standard | 84.29% | 71.12% | 83.12% | 79.26% | 87.39% |
| **Exp2 - Augmented** ⭐ | **SegFormer-B1** | **Sport-specific** | **87.26%** | **76.45%** | **86.66%** | **85.76%** | **87.57%** |
| Exp3 - Optimized1 | SegFormer-B1 | Sport-specific reduced + LR 3e-5 + ch=512 | 84.58% | 71.68% | 83.51% | 78.66% | 88.99% |
| Exp3 - Optimized2 | SegFormer-B1 | Sport-specific reduced + LR 3e-5 + ch=256 | 84.92% | 72.26% | 83.90% | 80.40% | 87.71% |
| Exp3 - Optimized3 | SegFormer-B1 | Sport-specific + drop_path=0.15 | 86.42% | 74.99% | 85.71% | 82.25% | 89.47% |

> **Nota**: È stato effettuato un ulteriore tentativo con augmentazioni ancora più aggressive
> (CoarseDropout max_holes=10 p=0.6, GridDistortion, OpticalDistortion, ImageCompression),
> interrotto anticipatamente a causa di prestazioni sul validation set significativamente
> inferiori rispetto all'Augmented standard. I risultati parziali hanno confermato il trend
> già osservato: augmentazioni troppo aggressive degradano la generalizzazione su dataset
> di dimensioni ridotte.

> **Best model**: Exp2 - SegFormer-B1 Augmented con augmentazioni sport-specific bilanciate.
> Modelli e dataset disponibili su 🤗 [Hugging Face](https://huggingface.co/ilMassy/advertising-panel-segmentation)

---

## 🤗 Hugging Face

Modelli addestrati e dataset sono disponibili pubblicamente su Hugging Face:

**[https://huggingface.co/ilMassy/advertising-panel-segmentation](https://huggingface.co/ilMassy/advertising-panel-segmentation)**

Il repository include:
- Tutti i checkpoint dei modelli addestrati (`best_mIoU_iter_*.pth`)
- Il dataset completo (`processed.zip` — 967 immagini + maschere, split train/val/test)

---

## 🚀 Roadmap

- [x] **Project Structure & Repository Setup**: Configurazione ambiente Python 3.10+ e MMSegmentation stack.

- [x] **Phase 1 - Dataset Curation**: Estrazione frame a 1920x1080 da highlights HD, annotazione poligonale su CVAT.ai, conversione COCO JSON → maschere binarie PNG (967 immagini, split train/val/test).

- [x] **Phase 2 - Baseline SegFormer-B0**: Addestramento del modello leggero B0 per stabilire il benchmark di riferimento (IoU, Dice, Precision, Recall).

- [x] **Phase 3 - SegFormer-B1 Standard**: Fine-tuning di B1 sul dataset custom e confronto quantitativo con la baseline B0. Best checkpoint a iter 10000 (mIoU 84.29%).

- [x] **Phase 4 - Domain-Specific Augmentation**: Implementazione di tecniche sport-specific (Motion Blur, Occlusion Cutout, Color Jitter) per migliorare la robustezza del modello. Best checkpoint a iter 14000 (mIoU 87.26%).

- [x] **Phase 5 - Architectural Optimization**: Sperimentazione sistematica di ottimizzazioni architetturali (LR, decoder channels, stochastic depth dropout, early stopping). Nessuna configurazione ha superato Exp2 — analisi documentata nel repository.

- [x] **Phase 6 - Critical Analysis & Benchmarking**: Confronto qualitativo e quantitativo tra tutti gli esperimenti con visualizzazione attention maps (B0 vs B1 Standard vs B1 Augmented), analisi top-5 best/worst cases sul test set, distribuzione errori TP/FP/FN. Modello ottimale identificato: Exp2 - B1 Augmented (mIoU 87.26%, gap Val→Test -0.69%).

- [ ] **Phase 7 - Final Documentation**: Relazione tecnica dettagliata su architettura, training log, analisi dei casi d'uso difficili e sviluppi futuri.

---

## 📖 Literature & References

- **SegFormer**: Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*. [arXiv:2105.15203](https://arxiv.org/abs/2105.15203)

- **Vision Transformer (ViT)**: Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

- **MMSegmentation**: Contributors of MMSegmentation (2020). *OpenMMLab Semantic Segmentation Toolbox and Benchmark*. [GitHub](https://github.com/open-mmlab/mmsegmentation)

- **Albumentations**: Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Kurakin, A., & Kalinin, A. A. (2020). *Albumentations: Fast and Flexible Image Augmentations*. [Information 2020](https://www.mdpi.com/2078-2489/11/2/125)

- **ImageNet Pretraining**: Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009). *ImageNet: A Large-Scale Hierarchical Image Database*. CVPR 2009. [Paper](https://ieeexplore.ieee.org/document/5206848)
