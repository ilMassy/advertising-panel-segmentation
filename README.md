## E1: Segmentazione Avanzata di Pannelli Pubblicitari (High-Res Transformer Approach)
🇮🇹 Descrizione del Progetto
Sviluppo di un sistema di Semantic Segmentation ad alta precisione per l'individuazione di pannelli pubblicitari in ambito broadcast sportivo.
A differenza degli approcci standard (CNN-based), il progetto adotta l'architettura SegFormer, sfruttando un Hierarchical Transformer Encoder per gestire la coerenza spaziale e la variabilità di risoluzione. L'obiettivo è superare i limiti della bassa risoluzione (640x640) utilizzando un dataset custom a 1080p per garantire la massima precisione sui bordi e nelle condizioni di occlusione critica.

🇬🇧 Project Description
Development of a high-precision Semantic Segmentation system for advertising board detection in sports broadcasting.
Unlike standard CNN-based approaches, this project employs the SegFormer architecture, utilizing a Hierarchical Transformer Encoder to manage spatial coherence and resolution variability. The goal is to overcome low-resolution limitations (640x640) by using a custom 1080p dataset to ensure maximum edge precision and robustness under critical occlusion conditions.

🚀 Roadmap

[x] Project Structure & Repository Setup: Configurazione ambiente Python 3.12 e MMSegmentation.

[ ] Phase 1: High-Res Dataset Curation: Estrazione frame a 1920x1080 da highlights HD e annotazione poligonale chirurgica su CVAT.ai.

[ ] Phase 2: Baseline SegFormer-B1: Addestramento del modello base per stabilire il benchmark di IoU (Intersection over Union).

[ ] Phase 3: Domain-Specific Augmentation: Implementazione di tecniche per simulare Motion Blur e Occlusioni (come richiesto dalle slide del progetto E1).

[ ] Phase 4: Architectural Optimization: Modifica del Decoder Head o degli iper-parametri di Self-Attention per ottimizzare il rilevamento su pannelli in prospettiva.

[ ] Phase 5: Critical Analysis & Benchmarking: Confronto qualitativo e quantitativo tra il dataset standard (bassa res) e l'approccio Full HD.

[ ] Phase 6: Final Documentation: Relazione tecnica dettagliata su architettura, training log e analisi dei casi d'uso difficili.
