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

## 🎬 Inferenza su Video

Il modello migliore (Exp2 — SegFormer-B1 Augmented) può essere esteso all’inferenza su video sportivi elaborando i frame individualmente. La pipeline proposta opera alla **risoluzione originale di 1920×1080**, preservando il dettaglio spaziale senza introdurre downscaling esplicito, grazie all’encoder gerarchico MiT, intrinsecamente agnostico alla risoluzione di input.

> **Nota implementativa**: il codice seguente rappresenta una proposta architetturale per l’inferenza su sequenze video. Sebbene sia coerente con le API di MMSegmentation e con il flusso di processamento previsto dal framework, non è ancora stato validato estensivamente su flussi video reali e potrebbe generare errori. La pipeline è stata sviluppata e testata in ambiente **Google Colab con GPU NVIDIA Tesla T4**.

> **Nota sulla risoluzione**: risoluzioni inferiori (ad es. 720p o 480p) sono supportate, ma possono comportare una riduzione delle prestazioni sui dettagli sottili, in quanto il modello è stato addestrato su immagini a 1080p. Risoluzioni superiori (ad es. 4K) sono teoricamente gestibili, ma richiedono una quantità significativamente maggiore di memoria GPU e possono necessitare di un ridimensionamento preventivo per rientrare nei limiti hardware disponibili.

### Step 1 — Scarica il checkpoint da Hugging Face

```python
from huggingface_hub import hf_hub_download
import os

# --- Configuration ---
REPO_ID = "ilMassy/advertising-panel-segmentation"
CHECKPOINT_FILENAME = "models/exp2_segformer_b1_augmented/best_mIoU_iter_14000.pth"
LOCAL_DIR = "./checkpoints"

# --- Download checkpoint from Hugging Face Hub ---
checkpoint_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=CHECKPOINT_FILENAME,
    local_dir=LOCAL_DIR
)

# --- Sanity check ---
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError("Checkpoint download failed")

print(f"[INFO] Checkpoint available at: {checkpoint_path}")
```

### Step 2 — Carica il modello

```python
from mmseg.apis import init_model
import torch
import os

# --- Configuration ---
CONFIG_PATH = "configs/segformer_b1_augmented.py"

# --- Device selection ---
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# --- Validate config path ---
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

# --- Initialize model ---
model = init_model(
    CONFIG_PATH,
    checkpoint_path,  # use path from Step 1
    device=device
)

# --- Ensure test pipeline exists (fallback fix) ---
if not hasattr(model.cfg, "test_pipeline"):
    print("[WARNING] test_pipeline not found, falling back to val_pipeline")
    model.cfg.test_pipeline = model.cfg.val_pipeline

# --- Set evaluation mode ---
model.eval()

print("[INFO] Model initialized and ready")
```

### Step 3 — Inferenza frame per frame

```python
import cv2
import torch
import numpy as np
from mmseg.apis import inference_model

# --- Configuration ---
INPUT_VIDEO = "video_input.mp4"
OUTPUT_VIDEO = "video_output.mp4"
TARGET_CLASS = 1

# Optional: speed/VRAM optimization
USE_RESIZE = False
RESIZE_TO = (1280, 720)  # (width, height)

# --- Open video ---
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError("Error opening input video")

fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"[INFO] Video: {width}x{height} @ {fps} FPS, {total} frames")

# --- Video writer (more compatible codec) ---
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# --- Inference loop ---
with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device != "cpu")):

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        original_frame = frame

        # --- Optional resize for performance ---
        if USE_RESIZE:
            frame = cv2.resize(frame, RESIZE_TO)

        # --- Convert BGR to RGB ---
        rgb = frame[..., ::-1]

        # --- Run inference ---
        result = inference_model(model, rgb)

        # --- Extract segmentation mask ---
        mask = result.pred_sem_seg.squeeze().cpu().numpy().astype(np.uint8)

        # --- Restore original resolution if resized ---
        if USE_RESIZE:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            frame = original_frame

        # --- Apply overlay (highlight target class in red) ---
        overlay = frame.copy()
        overlay[mask == TARGET_CLASS] = (0, 0, 255)

        blended = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

        out.write(blended)

        # --- Progress logging ---
        if (i + 1) % 50 == 0:
            print(f"[INFO] Processed {i+1}/{total} frames")

# --- Release resources ---
cap.release()
out.release()

print("[INFO] Processing completed successfully")
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
│   │   └── README.md                     # Val set results and Val→Test gap analysis
│   └── README.md                         # Performance summary of trained models on Test set
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
├── README.md                             # Project overview, installation, results, future directions, references
└── requirements.txt                      # Python dependencies with pinned versions
```

---

## 🧪 Esperimenti

> Metriche calcolate sul **test set** (146 immagini). Dice, Precision e Recall si riferiscono alla classe **board** (pannello pubblicitario), non quindi alla classe **background** ne alla media tra i valori delle due classi.

| Esperimento | Modello | Augmentation | mIoU | Board IoU | Dice | Precision | Recall |
|---|---|---|---|---|---|---|---|
| Exp0 - Baseline | SegFormer-B0 | Standard | 87.15% | 76.28% | 86.55% | 84.13% | 89.10% |
| Exp1 - Standard | SegFormer-B1 | Standard | 84.29% | 71.12% | 83.12% | 79.26% | 87.39% |
| **Exp2 - Augmented** ⭐ | **SegFormer-B1** | **Sport-specific** | **87.26%** | **76.45%** | **86.66%** | **85.76%** | **87.57%** |
| Exp3 - Optimized | SegFormer-B1 | Sport-specific reduced + LR 3e-5 + ch=512 + early stopping (patience=5) | 84.58% | 71.68% | 83.51% | 78.66% | 88.99% |
| Exp3 - Optimized2 | SegFormer-B1 | Sport-specific reduced + LR 3e-5 + ch=256 + early stopping (patience=5) | 84.92% | 72.26% | 83.90% | 80.40% | 87.71% |
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

## 🔭 Sviluppi Futuri

### Miglioramenti a breve termine (senza nuovo training)
- **Test Time Augmentation (TTA)**: applicazione di flip orizzontale e multi-scale inference durante il test sul modello Exp2 - B1 Augmented per un potenziale guadagno di +1-2% mIoU senza riaddestrare il modello.

### Miglioramenti sul dataset
- **Espansione del dataset**: la soglia critica stimata è 2000-3000 immagini per rendere efficaci le ottimizzazioni architetturali testate nella Phase 5. Vale la pena sottolineare che le immagini utilizzate sono di altissima qualità (Full HD 1920x1080), una caratteristica rara nei dataset pubblici di segmentazione sportiva che tipicamente operano a risoluzioni molto inferiori (640x640). Questa qualità è proprio ciò che rende il task potenzialmente eccellente per SegFormer: l'encoder gerarchico MiT sfrutta la ricchezza di dettaglio ad alta risoluzione per costruire rappresentazioni multi-scala che i modelli CNN-based non riescono a sfruttare altrettanto efficacemente. Un'espansione mantenendo la stessa qualità 1080p permetterebbe di sbloccare architetture più potenti come SegFormer-B2 o B3 e raggiungere performance potenzialmente molto superiori.
- **Etichettatura più precisa**: annotazioni più accurate sui bordi dei pannelli, in particolare nei casi di occlusione parziale da giocatori, ridurrebbero i falsi negativi identificati nell'analisi dei worst cases.
- **Augmentazioni per la distinzione giocatori/pannelli**: tecniche specifiche per insegnare al modello a separare i giocatori dai pannelli, come ad esempio instance-aware dropout che rimuova selettivamente le regioni corrispondenti ai giocatori durante il training, o l'uso di segmentazioni delle istanze dei giocatori come segnale ausiliario.

### Miglioramenti architetturali (con dataset espanso)
- **SegFormer-B2 o B3**: con 2000-3000 immagini sarebbe possibile sfruttare backbone più potenti (`embed_dims=128` per B2) senza incorrere nell'overfitting osservato nella Phase 5.
- **Pseudo-labeling**: uso del modello Augmented per generare maschere automatiche su video non annotati, con validazione manuale sopra soglia di confidence, per espandere il dataset a costo ridotto.
- **Domain adaptation**: pretraining su dataset pubblici di segmentazione sportiva prima del fine-tuning sul dataset custom, per ridurre il domain gap tra ImageNet e il broadcast sportivo.
- **Architetture alternative**: valutazione di Mask2Former che usa attention cross-modale per separare istanze diverse — potrebbe gestire meglio la distinzione giocatori/pannelli rispetto alla segmentazione semantica pura.

---

## 📖 Literature & References

- **SegFormer**: Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*. NeurIPS 2021. [arXiv:2105.15203](https://arxiv.org/abs/2105.15203)

- **Vision Transformer (ViT)**: Dosovitskiy, A. et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

- **MMSegmentation**: OpenMMLab Contributors (2020). *OpenMMLab Semantic Segmentation Toolbox and Benchmark*. [GitHub](https://github.com/open-mmlab/mmsegmentation)

- **Albumentations**: Buslaev, A. et al. (2020). *Albumentations: Fast and Flexible Image Augmentations*. Information 2020. [DOI](https://www.mdpi.com/2078-2489/11/2/125)

- **DeepLabV3**: Chen, L. C., Papandreou, G., Schroff, F., & Adam, H. (2017). *Rethinking Atrous Convolution for Semantic Image Segmentation*. [arXiv:1706.05587](https://arxiv.org/abs/1706.05587)

- **SETR**: Zheng, S. et al. (2021). *Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers*. CVPR 2021. [arXiv:2012.15840](https://arxiv.org/abs/2012.15840)

- **ImageNet**: Deng, J. et al. (2009). *ImageNet: A Large-Scale Hierarchical Image Database*. CVPR 2009. [DOI](https://ieeexplore.ieee.org/document/5206848)
