## Segmentazione Avanzata di Pannelli Pubblicitari (High-Res Transformer Approach)
🇮🇹 Descrizione del Progetto
Sviluppo di un sistema di Semantic Segmentation ad alta precisione per l'individuazione di pannelli pubblicitari in ambito broadcast sportivo.
A differenza degli approcci standard (CNN-based), il progetto adotta l'architettura SegFormer, sfruttando un Hierarchical Transformer Encoder per gestire la coerenza spaziale e la variabilità di risoluzione. L'obiettivo è superare i limiti della bassa risoluzione (640x640) utilizzando un dataset custom a 1080p per garantire la massima precisione sui bordi e nelle condizioni di occlusione critica.

🇬🇧 Project Description
Development of a high-precision Semantic Segmentation system for advertising board detection in sports broadcasting.
Unlike standard CNN-based approaches, this project employs the SegFormer architecture, utilizing a Hierarchical Transformer Encoder to manage spatial coherence and resolution variability. The goal is to overcome low-resolution limitations (640x640) by using a custom 1080p dataset to ensure maximum edge precision and robustness under critical occlusion conditions.

## 🚀 Roadmap

- [x] **Project Structure & Repository Setup**: Configurazione ambiente Python 3.10+ e MMSegmentation stack.

- [x] **Phase 1 - Dataset Curation**: Estrazione frame a 1920x1080 da highlights HD, annotazione poligonale su CVAT.ai, conversione COCO JSON → maschere binarie PNG (967 immagini, split train/val/test).

- [ ] **Phase 2 - Baseline SegFormer-B0**: Addestramento del modello leggero B0 per stabilire il benchmark di riferimento (IoU, Dice, Precision, Recall).

- [ ] **Phase 3 - SegFormer-B1 Standard**: Fine-tuning di B1 sul dataset custom e confronto quantitativo con la baseline B0.

- [ ] **Phase 4 - Domain-Specific Augmentation**: Implementazione di tecniche sport-specific per simulare Motion Blur, Occlusion Cutout e Color Jitter per migliorare la robustezza del modello.

- [ ] **Phase 5 - Architectural Optimization**: Modifica del Decoder Head e degli iper-parametri di Self-Attention per ottimizzare il rilevamento su pannelli in prospettiva.

- [ ] **Phase 6 - Critical Analysis & Benchmarking**: Confronto qualitativo e quantitativo tra tutti gli esperimenti (mIoU, Dice, Precision, Recall) e analisi dei casi difficili (occlusioni, motion blur, pannelli piccoli).

- [ ] **Phase 7 - Final Documentation**: Relazione tecnica dettagliata su architettura, training log, analisi dei casi d'uso difficili e sviluppi futuri.
