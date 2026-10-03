# End-to-End Genomic Pipeline and Deep Learning Benchmark (PoC)
### ゲノムデータ処理パイプラインとディープラーニングベンチマーク

## Project Overview / プロジェクト概要
This Proof of Concept (PoC) demonstrates an end-to-end data engineering and machine learning pipeline designed for genomic sequence validation and binary classification. The architecture simulates raw clinical environments, executes pipeline stress testing through structured anomaly injection, implements constraint-based data cleansing using Python (Pandas), normalizes data into a relational database schema (3NF SQLite), and trains deep neural networks (MLP) to evaluate categorical classification convergence based on a deterministic evolutionary signal (the TATA-box consensus motif).

---

## Pipeline Stages / パイプラインの各ステージ

### 1. Data Simulation and Stress Testing / データシミュレーションとストレステスト
This stage simulates raw multiplex clinical-demographic matrices paired with unstructured genomic sequences (10,000 baseline samples). Background nucleotide base compositions are biased across a fixed length of 200 base pairs. To evaluate the mathematical resilience of downstream validation layers, the pipeline injects structured clinical outliers and missing fields:
- Demographic noise: 20% out-of-bounds metrics (outlier clinical indicators) and 10% missing values (NaN) injected into the Age distribution.
- Categorical target noise: 20% invalid classification codes outside the operational binary range and 10% missing indicators injected into the Label vector.

### 2. Cleansing and Feature Engineering / クレンジングと特徴量エンジニアリング
Executes strict data validation gates using a centralized processing layer to isolate clean states and balance target classes:
- Relational deduplication across multi-source entry keys (PatientID, RefTag).
- Categorical target validation and structural casting, removing missing values and restricting entries to operational ranges (0.0 and 1.0).
- Demographic boundary enforcement restricting demographics to allowed clinical protocol parameters (ages 20–65).
- Stratified under-sampling execution to guarantee an absolute balanced operational dataset (1,000 records per class).
- Feature Engineering: Vectorized sequence substitution injecting a deterministic functional motif (TATAAA consensus sequence) at locus position 30 exclusively inside positive class target vectors.

### 3. Relational Ingestion (3NF Schema) / リレーショナルインジェクション（3NF正規化）
Validates enterprise data storage scalability using an automated local database engine:
- Instantiates a high-performance staging environment table (staging_source) utilizing low-latency bulk insertions (executemany).
- Executes automatic reset blocks, relational constraints mapping (PRAGMA foreign_keys = ON), and multi-layered database integrity audits.
- Achieves structural Third Normal Form (3NF) decomposition by decoupling demographic indices (patients parent schema) from technical analytical layers (biology child schema) linked via foreign keys and protected with high-performance B-Tree clustering indexes (idx_biology_label).

### 4. Deep Learning Benchmarking / ディープラーニングベンチマーク
Transforms categorical genetic profiles into input feature spaces via equidistant One-Hot sequence matrices (800 continuous dimensions per record based on a 4-base mapping layout). Executes a stratified split (80% Train, 20% Test) and benchmarks deep multi-layer representations (MLP Classifier Architectures) to map loss optimization and out-of-sample prediction profiles:
- Architecture A: Standard Multi-layer Perceptron (32 × 16 hidden topography with Adam Optimization, ReLU activations, and max_iter=15).
- Architecture B: Focused Sequential Architecture (64 hidden units in a single layer configuration with Adam Optimization, ReLU activations, and max_iter=15).

---

## Analytical Artifacts / 生成される成果物
Upon execution, the automated runtime synchronizes and exports the following deployment assets directly within the local workspace:
- raw_genomic_dataset.[xlsx/parquet/fasta] (Initial unvalidated data assets containing structural noise)
- clean_genomic_dataset_sql.[xlsx/parquet/fasta] (Clean validated enterprise execution assets matching strict clinical criteria)
- genomic_production.db (Normalized relational production database layer)
- benchmark_learning_curves.png (Backpropagation loss convergence plots)
- benchmark_accuracy_barplot.png (Out-of-sample performance bar chart evaluating testing accuracy)
- benchmark_confusion_matrices.png (Side-by-side classification error heatmaps showing true vs. predicted labels)

---

## Setup and Execution / 実行手順
1. Clone this repository to your production workspace environment.
2. Verify you have your system dependencies ready and run the package manifest installation:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute the centralized orchestrator runtime:
   ```bash
   python main.py
   ```



