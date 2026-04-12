# Segmentazione Avanzata di Pannelli Pubblicitari LED
## High-Resolution Transformer Approach per Broadcast Sportivo

---

### Abstract

Questo progetto sviluppa un sistema di **Semantic Segmentation ad alta precisione** per l’individuazione di pannelli pubblicitari LED in contesti di broadcast sportivo. A differenza degli approcci CNN-based tradizionali, spesso addestrati su immagini ridimensionate (es. 640×640) per vincoli computazionali, il progetto adotta l’architettura **SegFormer**, basata su un encoder gerarchico Mix Transformer (MiT), in grado di catturare efficacemente sia informazioni locali sia dipendenze a lungo raggio.

L’addestramento è condotto su un dataset custom ad alta risoluzione (**1920×1080**, 967 immagini annotate manualmente tramite CVAT.ai), con l’obiettivo di preservare il dettaglio spaziale e migliorare la precisione dei bordi anche in presenza di occlusioni e variabilità prospettica.

Il lavoro include un confronto sperimentale sistematico tra **sei configurazioni** — SegFormer-B0 baseline, SegFormer-B1 standard, B1 con augmentazioni sport-specific e tre varianti di ottimizzazione architetturale — supportato da un’analisi quantitativa e qualitativa completa (attention maps, error analysis, best/worst case visualization).

Il modello ottimale identificato è **Exp2 – B1 Augmented**, che raggiunge prestazioni di rilievo (mIoU 87.26%, Board IoU 76.45%, Precision 85.76%) con un gap di generalizzazione contenuto (Val→Test −0.69%), indicando un buon equilibrio tra capacità rappresentativa e robustezza.

---

## Sommario

1. [Descrizione del Problema e Contesto Applicativo](#1-descrizione-del-problema-e-contesto-applicativo)
   - 1.1 [Contesto Applicativo](#11-contesto-applicativo)
   - 1.2 [Formalizzazione del Problema](#12-formalizzazione-del-problema)
   - 1.3 [Sfide Specifiche del Dominio](#13-sfide-specifiche-del-dominio)
   - 1.4 [Obiettivi del Progetto](#14-obiettivi-del-progetto)
2. [Il Dataset](#2-il-dataset)
   - 2.1 [Acquisizione e Caratteristiche](#21-acquisizione-e-caratteristiche)
   - 2.2 [Pipeline di Preprocessing](#22-pipeline-di-preprocessing)
   - 2.3 [Qualità e Integrità dei Dati](#23-qualità-e-integrità-dei-dati)
3. [Soluzioni in Letteratura](#3-soluzioni-in-letteratura)
   - 3.1 [Approcci CNN-based](#31-approcci-cnn-based)
   - 3.2 [Vision Transformers](#32-vision-transformers)
   - 3.3 [Perché SegFormer](#33-perché-segformer)
4. [Architettura SegFormer](#4-architettura-segformer)
   - 4.1 [Mix Transformer Encoder (MiT)](#41-mix-transformer-encoder-mit)
   - 4.2 [Lightweight MLP Decoder](#42-lightweight-mlp-decoder)
   - 4.3 [Perché B1 invece di B0 o B2](#43-perché-b1-invece-di-b0-o-b2)
   - 4.4 [Sinergia con Dataset 1080p](#44-sinergia-con-dataset-1080p)
5. [Framework Software](#5-framework-software)
   - 5.1 [Stack Tecnologico](#51-stack-tecnologico)
   - 5.2 [Pipeline di Training](#52-pipeline-di-training)
   - 5.3 [Pipeline di Inferenza](#53-pipeline-di-inferenza)
   - 5.4 [Struttura del Repository](#54-struttura-del-repository)
6. [Esperimenti e Risultati](#6-esperimenti-e-risultati)
   - 6.1 [Configurazioni Sperimentali](#61-configurazioni-sperimentali)
   - 6.2 [Risultati Quantitativi](#62-risultati-quantitativi)
   - 6.3 [Phase 4 — Domain-Specific Augmentation](#63-phase-4--domain-specific-augmentation)
   - 6.4 [Phase 5 — Architectural Optimization](#64-phase-5--architectural-optimization)
   - 6.5 [Analisi del Gap Val→Test](#65-analisi-del-gap-valtest)
7. [Analisi Critica dei Risultati](#7-analisi-critica-dei-risultati)
   - 7.1 [Attention Maps](#71-attention-maps)
   - 7.2 [Best Cases](#72-best-cases)
   - 7.3 [Worst Cases e Categorie di Errore](#73-worst-cases-e-categorie-di-errore)
   - 7.4 [Error Analysis Quantitativa](#74-error-analysis-quantitativa)
8. [Sviluppi Futuri](#8-sviluppi-futuri)
9. [Conclusioni](#9-conclusioni)
10. [Bibliografia](#10-bibliografia)

---

## 1. Descrizione del Problema e Contesto Applicativo

### 1.1 Contesto Applicativo

Il rilevamento automatico di pannelli pubblicitari LED in trasmissioni sportive rappresenta un task di crescente rilevanza commerciale. Le applicazioni principali includono:

- **Virtual advertising**: sostituzione digitale dei pannelli in tempo reale per diversi mercati geografici
- **Brand monitoring**: misurazione automatica del tempo di esposizione di un brand durante le trasmissioni
- **Analytics sportive**: analisi della visibilità pubblicitaria in funzione delle azioni di gioco

Il task richiede un sistema in grado di operare su immagini ad alta risoluzione tipiche del broadcast professionale (1920×1080p), con robustezza a condizioni variabili di illuminazione, occlusione e prospettiva.

### 1.2 Formalizzazione del Problema

Il problema è formalizzato come **Semantic Segmentation binaria**: dato un frame $I \in \mathbb{R}^{H \times W \times 3}$, il modello produce una maschera $\hat{M} \in \{0, 1\}^{H \times W}$ dove:

$$
\hat{M}(i,j) =
\begin{cases}
1 & \text{se il pixel } (i,j) \text{ appartiene a un pannello pubblicitario} \\
0 & \text{altrimenti (background)}
\end{cases}
$$

Le metriche di valutazione adottate sono:

$$\text{IoU} = \frac{|M \cap \hat{M}|}{|M \cup \hat{M}|}$$

$$\text{Dice} = \frac{2|M \cap \hat{M}|}{|M| + |\hat{M}|}$$

$$\text{Precision} = \frac{TP}{TP + FP} \quad \text{Recall} = \frac{TP}{TP + FN}$$

dove $M$ è la maschera ground truth e $\hat{M}$ quella predetta.

### 1.3 Sfide Specifiche del Dominio

Il task presenta sfide peculiari rispetto alla segmentazione generica:

- **Occlusione dinamica**: i giocatori attraversano continuamente i pannelli, generando occlusioni parziali imprevedibili
- **Variabilità prospettica**: i pannelli appaiono con diverse angolazioni e distorsioni geometriche a seconda della posizione della telecamera
- **Condizioni di illuminazione variabili**: le luci dello stadio, i riflessi e le diverse temperature di colore delle telecamere broadcast creano variazioni cromatiche significative
- **Motion blur**: il rapido movimento della telecamera durante le azioni di gioco introduce sfocatura
- **Compressione video**: gli artefatti di compressione JPEG/broadcast degradano la qualità percepita dei bordi

### 1.4 Obiettivi del Progetto

Il progetto si propone di:

1. Costruire un dataset custom annotato manualmente a risoluzione 1080p
2. Confrontare sistematicamente SegFormer-B0 e B1 su questo task specifico
3. Valutare l'efficacia di augmentazioni sport-specific per migliorare la robustezza
4. Esplorare ottimizzazioni architetturali motivate teoricamente
5. Identificare empiricamente il punto di ottimo tra capacità architetturale e dimensione del dataset

---

## 2. Il Dataset

### 2.1 Acquisizione e Caratteristiche

Il dataset è stato costruito integralmente per questo progetto, non essendo disponibili dataset pubblici di alta qualità specifici per la segmentazione di pannelli pubblicitari nel contesto del broadcast sportivo.

| Caratteristica | Valore |
|---|---|
| Immagini totali | 967 |
| Risoluzione | 1920×1080 (Full HD) |
| Formato immagini | JPEG (qualità 95) |
| Formato maschere | PNG (lossless) |
| Training set | 672 immagini |
| Validation set | 149 immagini |
| Test set | 146 immagini |
| Classi | 2 (background=0, board=1) |
| Sorgente | Highlights sportivi HD |

Le immagini sono state estratte da highlights sportivi in Full HD, garantendo la massima qualità possibile. La scelta del formato JPEG per le immagini e PNG per le maschere è motivata dalla necessità di bilanciare qualità e occupazione di storage: JPEG con qualità 95 è percettivamente lossless per le immagini RGB, mentre PNG è rigorosamente lossless per le maschere binarie dove qualsiasi artefatto di compressione comprometterebbe l'integrità delle annotazioni.

### 2.2 Pipeline di Preprocessing

La pipeline di preprocessing si articola in quattro fasi:

**Fase 1 — Estrazione frame** (`extract_frames.py`):
```python
cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
# Qualità 95/100 — perdita percettivamente impercettibile
```

**Fase 2 — Annotazione manuale** (CVAT.ai):
Le immagini sono state annotate tramite annotazioni poligonali su CVAT.ai. La piattaforma non modifica le immagini originali — esporta esclusivamente le coordinate dei poligoni in formato COCO JSON, lasciando intatte le immagini originali su disco.

**Fase 3 — Generazione maschere** (`CVAT_preparation.py`):
```python
# La maschera viene creata con le dimensioni originali dell'immagine
mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
# Label ID: 0=background, 1=board
mask[binary_mask > 0] = 1
# Salvataggio PNG lossless
cv2.imwrite(os.path.join(mask_dir, mask_filename), mask)
```

**Fase 4 — Verifica** (`check_masks.py`):
Script di validazione visiva che sovrappone maschere e immagini originali per verificare la correttezza delle annotazioni.

### 2.3 Qualità e Integrità dei Dati

Un aspetto critico del progetto è la preservazione della qualità 1080p lungo tutta la pipeline:

| Componente | Risoluzione | Note |
|---|---|---|
| Immagini originali | 1920×1080 | JPEG qualità 95 |
| Maschere | 1920×1080 | PNG lossless |
| Val/Test pipeline | 1920×1080 | Resize alla risoluzione originale |
| Train pipeline | Crop 640×640 da 1080p | Dettaglio nativo preservato |

In fase di training, il `RandomCrop(640×640)` estrae patch da immagini 1080p **senza downscaling preventivo** — ogni pixel del crop contiene informazione reale ad alta risoluzione, non pixel interpolati. Questo è fondamentalmente diverso dall'approccio CNN standard che prima ridimensiona l'intera immagine a 640×640 perdendo il dettaglio originale.

---

## 3. Soluzioni in Letteratura

### 3.1 Approcci CNN-based

Gli approcci tradizionali per la semantic segmentation si basano su architetture CNN come:

- **FCN (Fully Convolutional Networks)**: prime reti interamente convoluzionali per segmentazione, limitata risoluzione dell'output
- **DeepLab V3+**: usa Atrous Spatial Pyramid Pooling (ASPP) per catturare feature a scale diverse, ma con convoluzioni locali
- **U-Net**: architettura encoder-decoder con skip connections, ottima per segmentazione medica ma con campo recettivo limitato

**Limiti fondamentali delle CNN per questo task**:

1. Le convoluzioni locali hanno campo recettivo fisso — non catturano dipendenze a lungo raggio tra pixel distanti. Per pannelli che si estendono orizzontalmente per l'intera larghezza del frame (1920 pixel), la CNN vede solo porzioni locali alla volta.

2. Richiedono downscaling a 640×640 per gestire la memoria GPU — perdendo il dettaglio critico per la precisione dei bordi.

3. Non hanno un meccanismo nativo per catturare feature a scale diverse simultaneamente senza moduli aggiuntivi (es. FPN, ASPP).

### 3.2 Vision Transformers

L'introduzione di ViT (Vision Transformer, Dosovitskiy et al., 2020) ha rivoluzionato la computer vision applicando il meccanismo di Self-Attention alle immagini. ViT suddivide l'immagine in patch 16×16 e applica Self-Attention globale su tutti i patch simultaneamente.

**Vantaggi rispetto alle CNN**:
- Self-Attention cattura dipendenze a lungo raggio in un singolo layer
- Nessun bias induttivo di località — il modello impara quali patch sono correlati

**Limitazioni di ViT puro per la segmentazione**:
- Usa **positional encoding fisso** — vincolato alla risoluzione di training. Se si addestra a 640×640 e si inferisce a 1080p, le posizioni assolute dei patch cambiano e il positional encoding non generalizza
- Produce feature map a singola risoluzione — non adatto per task che richiedono output ad alta risoluzione come la segmentazione
- Alto costo computazionale: complessità O(n²) dove n è il numero di patch

**SETR (Segmentation Transformer)** tenta di risolvere queste limitazioni ma mantiene il ViT come backbone con le stesse limitazioni sul positional encoding.

### 3.3 Perché SegFormer

SegFormer (Xie et al., 2021) risolve tutti i problemi sopra citati con due innovazioni chiave:

**Innovazione 1 — Mix-FFN al posto del Positional Encoding**:

ViT usa positional encoding fisso $PE \in \mathbb{R}^{N \times C}$ che dipende dalla risoluzione di training. SegFormer sostituisce il MLP-FFN standard con un **Mix-FFN** che incorpora una convoluzione 3×3:

$$x_{out} = MLP(GELU(Conv_{3\times3}(MLP(x_{in}))))$$

La convoluzione 3×3 introduce implicitamente informazione posizionale locale in modo **agnostico alla risoluzione** — funziona a qualsiasi scala senza vincoli di training.

**Innovazione 2 — Encoder Gerarchico a 4 Stage**:

Il MiT (Mix Transformer) processa l'immagine a 4 risoluzioni decrescenti, visibile direttamente nelle configurazioni:

```python
backbone=dict(
    embed_dims=64,
    num_heads=[1, 2, 5, 8],    # 4 stage, heads crescenti
    num_layers=[2, 2, 2, 2],   # 2 layer per stage
)
```

| Stage | Risoluzione (da 1080p) | Heads | Funzione |
|---|---|---|---|
| 1 | H/4 × W/4 (270×480) | 1 | Feature locali, bordi fini |
| 2 | H/8 × W/8 (135×240) | 2 | Feature intermedie |
| 3 | H/16 × W/16 (67×120) | 5 | Feature semantiche |
| 4 | H/32 × W/32 (33×60) | 8 | Contesto globale |

Ogni stage usa **Efficient Self-Attention** con Sequence Reduction Ratio (`sr_ratios=[8,4,2,1]`) che riduce la complessità computazionale da O(n²) a O(n²/R²).

---

## 4. Architettura SegFormer

### 4.1 Mix Transformer Encoder (MiT)

L'encoder MiT elabora l'immagine attraverso 4 stage gerarchici, ciascuno con:

1. **Overlap Patch Embedding**: suddivisione dell'immagine in patch sovrapposti (diversamente da ViT che usa patch non sovrapposti)
2. **Efficient Self-Attention**: Self-Attention con riduzione della sequenza tramite `sr_ratio`
3. **Mix-FFN**: feed-forward network con convoluzione 3×3 incorporata

Il processo per ogni stage può essere formalizzato come:

$$\hat{x}^l = \text{EfficientSelfAttn}(LN(x^{l-1})) + x^{l-1}$$
$$x^l = \text{MixFFN}(LN(\hat{x}^l)) + \hat{x}^l$$

dove $LN$ è la Layer Normalization.

### 4.2 Lightweight MLP Decoder

Il decoder di SegFormer è sorprendentemente semplice rispetto ai decoder CNN-based:

```python
decode_head=dict(
    in_channels=[64, 128, 320, 512],  # feature dai 4 stage del backbone B1
    num_classes=2,
    channels=256,                      # dimensione di fusione
)
```

Il decoder unifica le feature dei 4 stage con semplici proiezioni MLP, sfruttando il fatto che il Self-Attention dell'encoder ha già catturato il contesto globale — non serve un decoder pesante come in DeepLab o U-Net.

### 4.3 Perché B1 invece di B0 o B2

La scelta di B1 come backbone ottimale è motivata sia teoricamente che empiricamente.

**Differenza fondamentale: embed_dims**

`embed_dims` è la **dimensionalità dello spazio di embedding** — quante caratteristiche il modello può estrarre da ogni patch:

| Modello | embed_dims | Parametri totali | Rappresentazione per patch |
|---|---|---|---|
| B0 | 32 | ~3.7M | 32 dimensioni |
| **B1** | **64** | **~13.7M** | **64 dimensioni** |
| B2 | 128 | ~25.4M | 128 dimensioni |

**Perché non B0**:
B0 con `embed_dims=32` produce rappresentazioni a 32 dimensioni per patch — sufficienti per scene semplici ma limitanti per la variabilità del broadcast sportivo (occlusioni, prospettive diverse, illuminazione variabile). Questo è confermato empiricamente dalle **attention maps**: B0 mostra attivazioni significativamente più basse (colorazione blu nella heatmap jet) rispetto a B1 (colorazione rossa), indicando una capacità discriminativa spaziale inferiore.

**Perché non B2**:
B2 con `embed_dims=128` richiederebbe almeno 2000-3000 immagini per evitare overfitting. Gli esperimenti di Phase 5 con `decoder channels=512` (aumento analogo di capacità) hanno dimostrato empiricamente che modelli con maggiore capacità peggiorano le prestazioni su 672 immagini di training — il decoder più grande impara pattern specifici del training set invece di generalizzare.

**Conferma empirica di B1**:
Il gap Val→Test di solo -0.69% per B1 Augmented (vs -4.07% per B1 Standard senza augmentazioni) è la conferma quantitativa più forte che B1 con le augmentazioni corrette è il punto di equilibrio ottimale tra capacità architetturale e dimensione del dataset.

### 4.4 Sinergia con Dataset 1080p

La combinazione SegFormer + dataset 1080p non è casuale — è una sinergia architetturale:

**CNN standard**: richiede downscaling a 640×640 → perdita di dettaglio → bordi imprecisi → IoU più basso

**SegFormer con Mix-FFN**: agnostico alla risoluzione → inferenza a 1920×1080 → dettaglio originale preservato → bordi precisi

In fase di inferenza, il modello vede l'immagine completa 1920×1080. Lo Stage 1 del MiT opera a H/4×W/4 = 270×480 — già superiore alla risoluzione finale di molti modelli CNN. Questo significa che le feature locali estratte dallo Stage 1 contengono più informazione di quelle estratte da una CNN che opera su immagini ridimensionate a 640×640.

Con un dataset espanso a 2000-3000 immagini mantenendo la stessa qualità 1080p, questa sinergia permetterebbe di sbloccare B2 o B3 e raggiungere performance irraggiungibili da qualsiasi modello CNN su questo task, indipendentemente dalla quantità di dati.

---

## 5. Framework Software

### 5.1 Stack Tecnologico

| Componente | Versione | Ruolo |
|---|---|---|
| PyTorch | 2.4.1 | Deep learning framework |
| MMSegmentation | 1.2.2 | Training/evaluation pipeline |
| mmcv | 2.2.0 | Computer vision utilities |
| mmengine | 0.10.7 | Training engine |
| Albumentations | latest | Augmentation library |
| Google Colab (T4 GPU) | — | Training environment |
| Google Drive | — | Checkpoint storage durante training |
| GitHub | — | Repository e versionamento |
| Hugging Face | — | Model e dataset hosting pubblico |

### 5.2 Pipeline di Training

La pipeline di training è gestita interamente da MMSegmentation con configurazioni Python:

```
Input 1920×1080
       ↓
RandomResize (0.5x–2.0x della scala originale)
       ↓
RandomCrop 640×640 (patch ad alta risoluzione da 1080p)
       ↓
RandomFlip (p=0.5)
       ↓
PhotoMetricDistortion
       ↓
[Albumentations augmentations — solo Phase 4/5]
       ↓
PackSegInputs → MiT Encoder → MLP Decoder
       ↓
CrossEntropyLoss
       ↓
AdamW (lr=6e-5, weight_decay=0.01)
       ↓
PolyLR scheduler (warmup 1500 iter)
```

I checkpoint vengono salvati **direttamente su Google Drive** durante il training (`--work-dir` punta a Drive), garantendo la persistenza dei risultati in caso di disconnessione di Colab.

### 5.3 Pipeline di Inferenza

```
Input 1920×1080 (risoluzione originale completa)
       ↓
Resize scale=(1920, 1080), keep_ratio=True
       ↓
MiT Encoder (4 stage gerarchici)
       ↓
MLP Decoder (fusione feature multi-scala)
       ↓
Output maschera 1920×1080 (0=background, 1=board)
```

In inferenza non c'è crop — il modello vede l'immagine completa alla risoluzione originale, sfruttando appieno la gerarchia multi-scala dell'encoder MiT.

### 5.4 Struttura del Repository

```
advertising-panel-segmentation/
├── configs/
│   ├── segformer_b0_baseline.py          # Configurazione baseline SegFormer-B0 (Fase 2)
│   ├── segformer_b1_augmented.py         # Configurazione SegFormer-B1 con data augmentation (Fase 4)
│   ├── segformer_b1_optimized.py         # Configurazione SegFormer-B1 ottimizzata (Fase 5 - opt1)
│   ├── segformer_b1_optimized2.py        # Configurazione SegFormer-B1 ottimizzata (Fase 5 - opt2)
│   ├── segformer_b1_optimized3.py        # Configurazione SegFormer-B1 ottimizzata (Fase 5 - opt3)
│   └── segformer_b1_standard.py          # Configurazione SegFormer-B1 standard (Fase 3)
├── docs/
│   ├── Final_Report_ITA.md               # Relazione tecnica finale e analisi dei risultati in italiano
│   └── Presentation_ITA.pdf              # Presentazione powerpoint in italiano
├── models/
│   ├── checkpoints/
│   │   └── README.md                     # Risultati Val set e analisi del gap Val→Test
│   └── README.md                         # Riepilogo delle prestazioni dei modelli addestrati sul Test set
├── notebooks/
│   ├── Segformer_analysis.ipynb          # Notebook per l'analisi critica (Fase 6)
│   ├── Segformer_augmented.ipynb         # Notebook di addestramento Fase 4
│   ├── Segformer_optimized.ipynb         # Notebook di addestramento Fase 5 - opt1
│   ├── Segformer_optimized2.ipynb        # Notebook di addestramento Fase 5 - opt2
│   ├── Segformer_optimized3.ipynb        # Notebook di addestramento Fase 5 - opt3
│   ├── Segformer_training.ipynb          # Notebook di addestramento Fasi 2-3
│   └── Upload_to_HuggingFace.ipynb       # Caricamento dei modelli e del dataset su Hugging Face
├── results/
│   ├── dataset_previews/
│   │   └── mask_check_preview.png        # Overlay verde per confermare la precisione delle maschere ad alta risoluzione
│   ├── exp0_segformer_b0_baseline/       # Log di addestramento SegFormer-B0 baseline
│   ├── exp1_segformer_b1_standard/       # Log di addestramento SegFormer-B1 standard
│   ├── exp2_segformer_b1_augmented/      # Log di addestramento SegFormer-B1 con augmentation
│   ├── exp3_segformer_b1_optimized/      # Log di addestramento SegFormer-B1 ottimizzato (opt1)
│   ├── exp3_segformer_b1_optimized2/     # Log di addestramento SegFormer-B1 ottimizzato (opt2)
│   ├── exp3_segformer_b1_optimized3/     # Log di addestramento SegFormer-B1 ottimizzato (opt3)
│   ├── attention_maps_comparison.png     # Mappe di attenzione: confronto B0 vs B1 Standard vs B1 Augmented (Fase 6)
│   ├── benchmarking_complete.png         # Confronto dei 6 modelli: mIoU Val vs Test, gap Val→Test, Precision vs Recall
│   ├── best_cases.png                    # Top-5 migliori predizioni di B1 Augmented sul test set (Board IoU 0.945–0.962)
│   ├── error_analysis.png                # Distribuzione TP/FP/FN e boxplot IoU/Precision/Recall: B0 vs B1 Standard vs B1 Augmented
│   └── worst_cases.png                   # Top-5 peggiori predizioni di B1 Augmented sul test set (Board IoU 0.291–0.435)
├── src/
│   ├── CVAT_preparation.py               # Conversione COCO JSON → maschere PNG binarie
│   ├── check_masks.py                    # Verifica visiva delle maschere
│   ├── extract_frames.py                 # Estrazione dei frame dal video
│   └── reorder_by_prefix.py              # Rinominazione sequenziale dei frame
├── .gitignore                            # Regole di esclusione Git (checkpoint, cache, file di sistema)
├── README.md                             # Panoramica del progetto, installazione, inferenza su video, risultati e riferimenti
└── requirements.txt                      # Dipendenze Python con versioni specifiche
```

Modelli addestrati e dataset sono disponibili pubblicamente su Hugging Face:
**https://huggingface.co/ilMassy/advertising-panel-segmentation**

---

## 6. Esperimenti e Risultati

### 6.1 Configurazioni Sperimentali

Tutti gli esperimenti condividono:
- **Ottimizzatore**: AdamW (betas=(0.9, 0.999), weight_decay=0.01)
- **LR Scheduler**: LinearLR warmup (0→1500 iter) + PolyLR decay (1500→20000 iter)
- **Checkpoint**: salvato ogni 2000 iter, mantenuto il best mIoU su validation set
- **Max iterations**: 20000
- **Batch size**: 4
- **GPU**: NVIDIA T4 (Google Colab)

### 6.2 Risultati Quantitativi

**Validation Set (149 immagini)**:

| Esperimento | Best Checkpoint | Val mIoU | Val Board IoU | Val Board Acc | Val mDice |
|---|---|---|---|---|---|
| Exp0 - B0 Baseline | iter_18000 | 88.84% | 79.95% | 92.30% | 93.86% |
| Exp1 - B1 Standard | iter_10000 | 88.36% | 79.09% | 92.01% | 93.56% |
| **Exp2 - B1 Augmented** | **iter_14000** | **87.95%** | **78.33%** | **90.57%** | **93.31%** |
| Exp3 - Optimized1 | iter_14000 | 86.65% | 76.15% | 92.79% | 92.51% |
| Exp3 - Optimized2 | iter_12000 | 86.91% | 76.55% | 91.17% | 92.67% |
| Exp3 - Optimized3 | iter_18000 | 88.20% | 78.87% | 94.18% | 93.47% |

**Test Set (146 immagini)**:

| Esperimento | mIoU | Board IoU | Dice | Precision | Recall |
|---|---|---|---|---|---|
| Exp0 - B0 Baseline | 87.15% | 76.28% | 86.55% | 84.13% | 89.10% |
| Exp1 - B1 Standard | 84.29% | 71.12% | 83.12% | 79.26% | 87.39% |
| **Exp2 - B1 Augmented** ⭐ | **87.26%** | **76.45%** | **86.66%** | **85.76%** | **87.57%** |
| Exp3 - Optimized1 | 84.58% | 71.68% | 83.51% | 78.66% | 88.99% |
| Exp3 - Optimized2 | 84.92% | 72.26% | 83.90% | 80.40% | 87.71% |
| Exp3 - Optimized3 | 86.42% | 74.99% | 85.71% | 82.25% | 89.47% |

Dice, Precision e Recall si riferiscono alla classe board (pannello pubblicitario), non quindi alla classe background ne alla media tra i valori delle due classi.

### 6.3 Phase 4 — Domain-Specific Augmentation

Le augmentazioni sport-specific sono state implementate tramite Albumentations, integrate nella pipeline MMSeg tramite il wrapper `Albu`:

| Augmentazione | Parametri | Motivazione |
|---|---|---|
| MotionBlur | blur_limit=9, p=0.4 | Simula movimento telecamera durante azioni |
| GaussNoise | var_limit=(10,50), p=0.3 | Simula compressione segnale broadcast |
| RandomBrightnessContrast | ±0.3, p=0.4 | Simula variazioni luci stadio |
| HueSaturationValue | hue±20, sat±30, val±20, p=0.3 | Simula differenze white balance tra telecamere |
| CoarseDropout | holes=6, 80×80px, p=0.3 | Simula occlusione pannelli da giocatori |

**Risultati Phase 4**: Exp2 supera Exp1 di +2.97% mIoU e migliora la Precision da 79.26% a 85.76% — le augmentazioni hanno insegnato al modello a non fare affidamento su pattern visivi ingannevoli come le maglie colorate dei giocatori.

È stato effettuato un ulteriore tentativo con augmentazioni ancora più aggressive (CoarseDropout max_holes=10 p=0.6, GridDistortion, OpticalDistortion, ImageCompression), interrotto anticipatamente a causa di prestazioni sul validation set significativamente inferiori rispetto all'Augmented standard. I risultati parziali hanno confermato il trend già osservato: augmentazioni troppo aggressive degradano la generalizzazione su dataset di dimensioni ridotte.

### 6.4 Phase 5 — Architectural Optimization

Sono state testate sistematicamente le seguenti modifiche architetturali:

| Modifica | Motivazione teorica | Risultato |
|---|---|---|
| LR 3e-5 invece di 6e-5 | Convergenza più stabile su dataset piccolo | Peggiora — su 20000 iter non converge completamente |
| decoder channels 512 | Maggiore capacità per pannelli in prospettiva | Peggiora — overfitting su 672 immagini |
| drop_path_rate=0.15 da 0.1 | Regolarizzazione stochastic depth | Neutro/lieve peggioramento |
| sr_ratios=[4,2,1,1] | Attention più locale per bordi obliqui | Non utilizzata — incompatibile con pesi pre-addestrati |
| Dice Loss + CE | Bilancio background/board | Causa CUDA index out of bounds error con class_weight in questa versione di MMSeg — non utilizzata; senza class_weight spinge verso alto Recall / bassa Precision peggiorando la generalizzazione |
| Early Stopping (patience=5) | Evitare overfitting finale | Concettualmente corretto ma non compensa gli altri limiti |

**Pattern emerso**: ogni modifica ha sistematicamente aumentato il Recall (fino a 89.47%) a scapito della Precision (fino a 78.66%), spostando il modello verso una segmentazione "generosa" che trova più pannelli ma con bordi imprecisi e falsi positivi.

### 6.5 Analisi del Gap Val→Test

| Esperimento | Val mIoU | Test mIoU | Gap Val→Test |
|---|---|---|---|
| Exp0 - B0 Baseline | 88.84% | 87.15% | -1.69% |
| Exp1 - B1 Standard | 88.36% | 84.29% | -4.07% |
| **Exp2 - B1 Augmented** | **87.95%** | **87.26%** | **-0.69%** ⭐ |
| Exp3 - Optimized1 | 86.65% | 84.58% | -2.07% |
| Exp3 - Optimized2 | 86.91% | 84.92% | -1.99% |
| Exp3 - Optimized3 | 88.20% | 86.42% | -1.78% |

Il gap Val→Test è la metrica più rilevante per valutare la capacità di generalizzazione reale. Exp2 ha il gap più basso (-0.69%) nonostante non abbia il val mIoU più alto — questo dimostra che le augmentazioni sport-specific hanno prodotto un modello che generalizza meglio su dati mai visti, invece di adattarsi specificamente al validation set.

---

## 7. Analisi Critica dei Risultati

### 7.1 Attention Maps

Le attention maps visualizzano dove ogni modello "guarda" durante la segmentazione, estratte dagli hook sui layer di attention del backbone.

**Osservazione chiave**: le attention maps mostrano che B1 (Standard e Augmented) attiva feature significativamente più intense rispetto a B0 (colorazione rossa vs blu nella heatmap jet). Questo è la conferma visiva diretta della differenza in `embed_dims`: B0 con `embed_dims=32` produce rappresentazioni a 32 dimensioni per patch — numericamente inferiori nella magnitudine delle attivazioni. B1 con `embed_dims=64` raddoppia la ricchezza rappresentativa, estraendo feature più discriminative per ogni patch dell'immagine.

Questa differenza nelle attention maps supporta la scelta di B1 come backbone ottimale: anche se i risultati numerici di B0 e B1 Augmented sono simili (87.15% vs 87.26% mIoU), la maggiore capacità di attivazione di B1 lo renderebbe significativamente più robusto su casi difficili con dataset espanso.

### 7.2 Best Cases

Le 5 immagini con Board IoU più alto sul test set (0.945–0.962) corrispondono a condizioni ideali:
- Pannelli frontali senza deformazioni prospettiche significative
- Buona illuminazione uniforme
- Nessuna occlusione da giocatori
- Colori dei pannelli ben distinti dallo sfondo

In queste condizioni il modello raggiunge IoU quasi perfetto, confermando che l'architettura è fondamentalmente corretta per il task.

### 7.3 Worst Cases e Categorie di Errore

Le 5 immagini con Board IoU più basso (0.291–0.435) rivelano tre categorie di errore sistematiche:

**Categoria A — Occlusione da giocatori**:
Il modello genera **falsi positivi** sul corpo del giocatore davanti al pannello. Il modello ha imparato ad associare la forma allungata orizzontale (tipica dei pannelli) con la classe "board", ma quando un giocatore occlude parte del pannello il modello "completa" la forma includendo parte del giocatore. Il CoarseDropout nelle augmentazioni mitiga parzialmente questo problema ma non lo risolve completamente.

**Categoria B — Prospettiva estrema**:
Pannelli visti quasi lateralmente diventano strisce molto sottili (pochi pixel di altezza). Il modello genera **falsi negativi** — non riesce a identificare oggetti così sottili come pannelli. Questo è un limite fondamentale che richiederebbe augmentazioni geometriche più aggressive (PerspectiveTransform) per essere affrontato efficacemente.

**Categoria C — Confusione cromatica**:
Pannelli con colori simili alle maglie dei giocatori o al fondo del campo generano errori di bordo. Un caso ricorrente nel dataset è la pubblicità Emirates (sfondo rosso) che può essere confusa con le maglie rosse di alcune squadre. Le augmentazioni HueSaturationValue e RandomBrightnessContrast affrontano parzialmente questo problema.

### 7.4 Error Analysis Quantitativa

L'analisi degli errori a livello pixel su tutto il test set (146 immagini) conferma i pattern qualitativi:

- **Exp2 - B1 Augmented** presenta il miglior tasso di falsi positivi (FP ~1.05%) e l'equilibrio ottimale tra FP e FN — l'introduzione delle strategie di data augmentation ha svolto un ruolo cruciale nella regolarizzazione del modello, riducendo drasticamente le attivazioni errate sugli elementi del background (falsi positivi), a fronte di un marginale incremento dei falsi negativi
- **Exp1 - B1 Standard** ha il peggior FP rate (~1.65%) — senza augmentazioni il modello sovrasegmenta, classificando come board elementi simili ai pannelli nel background
- **Exp0 - B0 Baseline** mostra il miglior FN rate (~10.9%) ma con IoU distribution più variabile — meno parametri riducono i falsi positivi rispetto alla versione standard, ma aumentano i casi difficili (6 immagini con IoU < 0.5)

La distribuzione IoU mostra che Exp2 — B1 Augmented possiede la varianza più contenuta e il boxplot più compatto, garantendo le prestazioni più costanti e affidabili su tutto il test set, nonostante una mediana leggermente inferiore alla Baseline B0.

---

## 8. Sviluppi Futuri

### 8.1 Miglioramenti a Breve Termine (senza nuovo training)

**Test Time Augmentation (TTA)**: applicazione di flip orizzontale e multi-scale inference durante il test sul modello Exp2 per un potenziale guadagno di +1-2% mIoU senza riaddestrare. MMSegmentation supporta TTA nativamente.

### 8.2 Miglioramenti sul Dataset

**Espansione del dataset**: la soglia critica stimata è 2000-3000 immagini per rendere efficaci le ottimizzazioni architetturali testate nella Phase 5. Vale la pena sottolineare che le immagini utilizzate sono di altissima qualità (Full HD 1920×1080), una caratteristica rara nei dataset pubblici di segmentazione sportiva. Questa qualità è ciò che rende il task potenzialmente eccellente per SegFormer: il Mix-FFN agnostico alla risoluzione e l'encoder gerarchico MiT sfruttano nativamente il dettaglio 1080p che CNN e ViT non possono gestire altrettanto efficacemente. Un'espansione mantenendo la stessa qualità 1080p permetterebbe di sbloccare SegFormer-B2 o B3 e raggiungere performance potenzialmente molto superiori.

**Etichettatura più precisa**: annotazioni più accurate dei bordi dei pannelli, in particolare nei casi di occlusione parziale da parte dei giocatori, ridurrebbero i falsi negativi osservati nell’analisi dei worst cases; inoltre, una migliore distinzione tra pannelli e giocatori contribuirebbe a limitare i falsi positivi dovuti a confusione visiva.

**Augmentazioni per la distinzione giocatori/pannelli**: tecniche specifiche per insegnare al modello a separare giocatori dai pannelli, come instance-aware dropout che rimuova selettivamente le regioni corrispondenti ai giocatori durante il training, o l'uso di segmentazioni delle istanze dei giocatori come segnale ausiliario.

### 8.3 Miglioramenti Architetturali (con dataset espanso)

**SegFormer-B2 o B3**: con 2000-3000 immagini sarebbe possibile sfruttare backbone più potenti (`embed_dims=128` per B2) senza overfitting.

**Pseudo-labeling**: uso del modello Augmented per generare maschere automatiche su video non annotati, con validazione manuale sopra soglia di confidence.

**Domain adaptation**: pretraining su dataset pubblici di segmentazione sportiva prima del fine-tuning sul dataset custom, per ridurre il domain gap tra ImageNet e il broadcast sportivo.

**Architetture alternative**: valutazione di Mask2Former che usa attention cross-modale per separare istanze diverse — potrebbe gestire meglio la distinzione giocatori/pannelli rispetto alla segmentazione semantica pura.

---

## 9. Conclusioni

Il progetto ha raggiunto il suo obiettivo principale: identificare empiricamente la configurazione ottimale di SegFormer per la segmentazione di pannelli pubblicitari in contesto broadcast sportivo su dataset di dimensioni ridotte.

**Risultato principale**: L'esperimento Exp2 (SegFormer-B1 Augmented) è stato identificato come la configurazione di benchmark ottimale, presentando il miglior bilanciamento tra capacità rappresentativa e resilienza all'overfitting con mIoU 87.26%, Board IoU 76.45%, Precision 85.76% e gap Val→Test di soli -0.69%.

**Contributi principali**:

1. **Dataset custom 1080p**: 967 immagini annotate manualmente in Full HD — significativamente superiore ai dataset pubblici che operano tipicamente a 640×640

2. **Analisi empirica sistematica**: 6 configurazioni sperimentali con metriche complete su validation e test set separati, che hanno identificato tre leggi empiriche per questo dominio:
   - Con 672 immagini, le augmentazioni sport-specific bilanciate battono le ottimizzazioni architetturali
   - Il LR ottimale per dataset piccoli con budget fisso di iterazioni è 6e-5, non 3e-5
   - Aumentare la capacità del decoder (256→512 channels) peggiora la generalizzazione sotto la soglia critica di ~2000 immagini

3. **Analisi architetturale**: le attention maps dimostrano che B1 (`embed_dims=64`) estrae rappresentazioni significativamente più ricche di B0 (`embed_dims=32`), confermando la scelta del backbone ottimale con evidenza visiva diretta

4. **Sinergia SegFormer + 1080p**: il Mix-FFN agnostico alla risoluzione e l'encoder gerarchico MiT rendono SegFormer architetturalmente superiore a CNN e ViT puro per dataset ad alta risoluzione — una combinazione che con più dati potrebbe permettere di sbloccare performance irraggiungibili dagli approcci tradizionali

---

## 10. Bibliografia

1. Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*. NeurIPS 2021. [arXiv:2105.15203](https://arxiv.org/abs/2105.15203)

2. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

3. Contributors of MMSegmentation (2020). *OpenMMLab Semantic Segmentation Toolbox and Benchmark*. [GitHub](https://github.com/open-mmlab/mmsegmentation)

4. Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Kurakin, A., & Kalinin, A. A. (2020). *Albumentations: Fast and Flexible Image Augmentations*. Information 2020, 11(2), 125. [DOI](https://www.mdpi.com/2078-2489/11/2/125)

5. Chen, L. C., Papandreou, G., Schroff, F., & Adam, H. (2017). *Rethinking Atrous Convolution for Semantic Image Segmentation (DeepLab V3)*. [arXiv:1706.05587](https://arxiv.org/abs/1706.05587)

6. Zheng, S., et al. (2021). *Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers (SETR)*. CVPR 2021. [arXiv:2012.15840](https://arxiv.org/abs/2012.15840)

7. Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009). *ImageNet: A Large-Scale Hierarchical Image Database*. CVPR 2009. [DOI](https://ieeexplore.ieee.org/document/5206848)

---

*Repository GitHub*: https://github.com/ilMassy/advertising-panel-segmentation

*Dataset e Modelli (Hugging Face)*: https://huggingface.co/ilMassy/advertising-panel-segmentation
