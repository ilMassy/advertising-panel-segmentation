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

## 🛠️ Background e Tecniche Utilizzate

- **Architettura**: SegFormer (Mix Transformer Encoder + Lightweight MLP Decoder)
- **Framework**: PyTorch 2.4.1, MMSegmentation 1.2.2, mmcv 2.2.0
- **Dataset**: Dataset custom di broadcast sportivo, 967 immagini annotate manualmente a 1920x1080p
- **Task**: Semantic Segmentation binaria (background vs pannello pubblicitario)
- **Approccio**: Fine-tuning su pesi pre-addestrati ImageNet (MIT-B0 e MIT-B1)

---

## ⚙️ Installazione

### ☁️ Google Colab (Consigliato)
1. Apri `notebooks/Segformer_training.ipynb` su Google Colab
2. Vai su `Opzioni di connessione aggiuntive → Cambia tipo di runtime → T4 GPU`
3. Esegui le celle in ordine dalla 1 alla 6

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
│   ├── segformer_b0_baseline.py    # SegFormer-B0 baseline config
│   └── segformer_b1_standard.py    # SegFormer-B1 standard config
├── models/                         
│   └── README.md                   # Performance summary of SegFormer-B0 and B1 models
├── notebooks/
│   └── Segformer_training.ipynb    # Google Colab training notebook
├── results/                        
│   ├── exp0_segformer_b0_baseline  # SegFormer-B0 baseline training logs
│   └── exp1_segformer_b1_standard  # SegFormer-B0 standard training logs
├── src/
│   ├── CVAT_preparation.py         # COCO JSON → binary PNG masks
│   ├── check_masks.py              # Visual mask verification
│   ├── extract_frames.py           # Frame extraction from video
│   └── reorder_by_prefix.py        # Sequential frame renaming
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🧪 Esperimenti

| Esperimento | Modello | Augmentation | mIoU | Board IoU | Dice | Precision | Recall |
|---|---|---|---|---|---|---|---|
| Exp0 - Baseline | SegFormer-B0 | Standard | 87.77% | 78.09% | 87.70% | 83.15% | 92.77% |
| Exp1 - Standard | SegFormer-B1 | Standard | 88.90% | 80.01% | 88.89% | 87.67% | 90.15% |
| Exp2 - Augmented | SegFormer-B1 | Sport-specific | - | - | - | - | - |
| Exp3 - Optimized | SegFormer-B1 | Sport-specific + custom | - | - | - | - | - |

---

## 🚀 Roadmap

- [x] **Project Structure & Repository Setup**: Configurazione ambiente Python 3.10+ e MMSegmentation stack.

- [x] **Phase 1 - Dataset Curation**: Estrazione frame a 1920x1080 da highlights HD, annotazione poligonale su CVAT.ai, conversione COCO JSON → maschere binarie PNG (967 immagini, split train/val/test).

- [x] **Phase 2 - Baseline SegFormer-B0**: Addestramento del modello leggero B0 per stabilire il benchmark di riferimento (IoU, Dice, Precision, Recall).

- [x] **Phase 3 - SegFormer-B1 Standard**: Fine-tuning di B1 sul dataset custom e confronto quantitativo con la baseline B0.

- [ ] **Phase 4 - Domain-Specific Augmentation**: Implementazione di tecniche sport-specific per simulare Motion Blur, Occlusion Cutout e Color Jitter per migliorare la robustezza del modello.

- [ ] **Phase 5 - Architectural Optimization**: Modifica del Decoder Head e degli iper-parametri di Self-Attention per ottimizzare il rilevamento su pannelli in prospettiva.

- [ ] **Phase 6 - Critical Analysis & Benchmarking**: Confronto qualitativo e quantitativo tra tutti gli esperimenti (mIoU, Dice, Precision, Recall) e analisi dei casi difficili (occlusioni, motion blur, pannelli piccoli).

- [ ] **Phase 7 - Final Documentation**: Relazione tecnica dettagliata su architettura, training log, analisi dei casi d'uso difficili e sviluppi futuri.
