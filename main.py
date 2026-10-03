# ==============================================================================
# SYSTEM & STANDARD LIBRARY IMPORTS (PEP 8 COMPLIANT)
# ==============================================================================
import os
import random
import sqlite3
import subprocess
import sys
import warnings  # Added to suppress diagnostic noise
from pathlib import Path

# Silence non-critical library and framework warning diagnostics
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==============================================================================
# THIRD-PARTY / ECOSYSTEM IMPORTS (PEP 8 COMPLIANT)
# ==============================================================================
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Enforce non-interactive visualization backend for headless/CI pipelines
matplotlib.use('Agg')


# ==============================================================================
# CENTRALIZED DYNAMIC PATH CONFIGURATION
# ==============================================================================
# ==============================================================================
# CENTRALIZED DYNAMIC PATH CONFIGURATION (INTERNATIONAL PRODUCTION STANDARD)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent

EXCEL_RAW_PATH = BASE_DIR / "raw_genomic_dataset.xlsx"
PARQUET_RAW_PATH = BASE_DIR / "raw_genomic_dataset.parquet"
FASTA_RAW_PATH = BASE_DIR / "raw_genomic_dataset.fasta"

EXCEL_READY_PATH = BASE_DIR / "clean_genomic_dataset_sql.xlsx"
PARQUET_READY_PATH = BASE_DIR / "clean_genomic_dataset_sql.parquet"
FASTA_READY_PATH = BASE_DIR / "clean_genomic_dataset_sql.fasta"

DB_PATH = BASE_DIR / "genomic_production.db"

# Seed initialization for operational reproducibility
np.random.seed(26)

# ==============================================================================
# PIPELINE STAGE FUNCTIONS: DATA GENERATION, ANOMALIES & CLEANSING
# ==============================================================================

def generate_genomic_dataset(n_samples: int = 10000) -> pd.DataFrame:
    """Simulates raw patient clinical records paired with synthetic genomic sequences.

    Args:
        n_samples (int): Total number of patient records to generate.

    Returns:
        pd.DataFrame: Simulated raw genomic and clinical data workspace.
    """
    dataset = []
    dna_bases = ['A', 'C', 'G', 'T']
    
    for _ in range(n_samples):
        # Biased distribution simulating specific genomic background noise
        dna_sequence = "".join(np.random.choice(dna_bases, p=[0.1, 0.4, 0.4, 0.1], size=200))
            
        clinical_records = {
            "PatientID": np.random.randint(1, n_samples),
            "Age": random.randrange(20, 66),
            "RefTag": np.random.randint(1, n_samples),
            "Label": np.random.randint(0, 2),
            "DnaSequence": dna_sequence
        }
        dataset.append(clinical_records)
        
    return pd.DataFrame(dataset)


def export_initial_formats(df: pd.DataFrame) -> None:
    """Synchronizes the simulated raw data layer across multi-standard extensions."""
    df.to_excel(EXCEL_RAW_PATH, index=False)
    # Using 'fastparquet' engine to ensure robust cross-platform compatibility
    # and prevent OS-level execution blocks on PyArrow compiled binaries.
    df.to_parquet(PARQUET_RAW_PATH, index=False, engine='fastparquet')
    fasta_file = open(FASTA_RAW_PATH, "w", encoding="utf-8")
    for _, r in df.iterrows():
        fasta_file.write(f">Patient_{r['PatientID']}|Label_{r['Label']}\n{r['DnaSequence']}\n")
    fasta_file.close()


def inject_pipeline_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Injects structured clinical outliers and missing fields for stress testing.

    Args:
        df (pd.DataFrame): Clean input data workspace.

    Returns:
        pd.DataFrame: Modified dataframe containing boundary violations and NaN states.
    """
    df_corrupted = df.copy()
    forbidden_ages = list(range(1, 20)) + list(range(66, 120)) + [250, -5]
    forbidden_labels = [-1, 2, 99, 999]

    # Inject demographic constraint violations (20% structural noise, 10% missing values)
    idx_age_noise = df_corrupted.sample(frac=0.20, random_state=1).index
    df_corrupted.loc[idx_age_noise, 'Age'] = np.random.choice(forbidden_ages, size=len(idx_age_noise), replace=True)
    idx_age_nan = df_corrupted.sample(frac=0.10, random_state=2).index
    df_corrupted.loc[idx_age_nan, 'Age'] = np.nan

    # Inject categorical target anomalies (20% categorical noise, 10% missing values)
    idx_label_noise = df_corrupted.sample(frac=0.20, random_state=3).index
    df_corrupted.loc[idx_label_noise, 'Label'] = np.random.choice(forbidden_labels, size=len(idx_label_noise), replace=True)
    idx_label_nan = df_corrupted.sample(frac=0.10, random_state=4).index
    df_corrupted.loc[idx_label_nan, 'Label'] = np.nan

    return df_corrupted


def clean_and_balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Executes data validation rules, isolates clean states, and balances target classes.

    Args:
        df (pd.DataFrame): Corrupted input dataframe.

    Returns:
        pd.DataFrame: Cleaned data workspace down-sampled to exact target counts.
    """
    # 1. Deduplication using database relational keys
    df_cleaned = df.drop_duplicates(subset=["PatientID"])
    df_cleaned = df_cleaned.drop_duplicates(subset=["RefTag"])

    # 2. Categorical target validation and structural casting
    df_cleaned = df_cleaned.dropna(subset=["Label"])
    df_cleaned = df_cleaned[df_cleaned["Label"].isin([0.0, 1.0])]
    df_cleaned["Label"] = df_cleaned["Label"].astype(int)

    # 3. Demographic clinical boundary enforcement
    df_cleaned = df_cleaned[(df_cleaned["Age"] >= 20) & (df_cleaned["Age"] <= 65)]

    # 4. Stratified under-sampling execution (1000 records per operational class)
    df_balanced = pd.concat([
        df_cleaned[df_cleaned["Label"] == 0].sample(1000, random_state=26),
        df_cleaned[df_cleaned["Label"] == 1].sample(1000, random_state=26)
    ]).sample(frac=1, random_state=26)

    return df_balanced


def execute_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Injects a deterministic biological motif (TATA-box consensus) into positive records.

    Args:
        df (pd.DataFrame): Balanced input dataframe.

    Returns:
        pd.DataFrame: Dataframe containing functional genomic target signals.
    """
    df_engineered = df.copy()
    idx_tata = df_engineered[df_engineered['Label'] == 1].index

    # Vectorized substitution of the TATA-box motif at locus position 30
    df_engineered.loc[idx_tata, 'DnaSequence'] = df_engineered.loc[idx_tata, 'DnaSequence'].apply(
        lambda seq: seq[:30] + 'TATAAA' + seq[36:]
    )
    return df_engineered

# ==============================================================================
# PIPELINE STAGE FUNCTIONS: SQLITE ENGINE & INTEGRITY AUDITS
# ==============================================================================

def run_database_pipeline() -> None:
    """Initializes the relational engine, builds a 3NF schema, and validates data ingestion."""
    connection = sqlite3.connect(str(DB_PATH))
    cursor = connection.cursor()

    # Enforce relational constraints and wipe stale table assets
    cursor.execute("PRAGMA foreign_keys = ON;")
    cursor.execute("DROP TABLE IF EXISTS staging_source")
    cursor.execute("DROP TABLE IF EXISTS biology")
    cursor.execute("DROP TABLE IF EXISTS patients")
    connection.commit()

    # DDL: Ingestion Stage Staging Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS staging_source (
        PatientID INTEGER, Age INTEGER, RefTag INTEGER, DnaSequence TEXT, Label INTEGER
    )''')
    connection.commit()

    # High-performance bulk insertion using validation sheet layer
    df_source = pd.read_excel(EXCEL_READY_PATH)
    data_to_insert = list(df_source[["PatientID", "Age", "RefTag", "DnaSequence", "Label"]].itertuples(index=False, name=None))
    cursor.executemany("INSERT INTO staging_source VALUES (?, ?, ?, ?, ?)", data_to_insert)
    connection.commit()
    print(f"INFO - SQL Engine: {len(data_to_insert)} records loaded into staging environment.\n")

    print("==================================================")
    print("=== PHASE A: SOURCE DATABASE INTEGRITY AUDIT   ===")
    print("==================================================")
    
    cursor.execute("SELECT COUNT(*) FROM staging_source")
    print(f"[Query 1] Total records in Source Table: {cursor.fetchone()}") 

    cursor.execute("SELECT COUNT(DISTINCT PatientID) FROM staging_source")
    print(f"[Query 2] Unique patients identified: {cursor.fetchone()}") 

    cursor.execute("SELECT Label, COUNT(*) FROM staging_source GROUP BY Label")
    print("[Query 3] Class Target Distribution:")
    for row in cursor.fetchall():
        print(f"          -> Class {row}: {row} samples")

    cursor.execute("SELECT AVG(Age) FROM staging_source")
    average_age_row = cursor.fetchone()
    print(f"[Query 4] Population Average Age: {average_age_row[0]:.2f} years")

    cursor.execute("SELECT COUNT(DISTINCT RefTag), MAX(cnt) FROM (SELECT RefTag, COUNT(*) as cnt FROM staging_source GROUP BY RefTag)")
    batch_stats = cursor.fetchone()
    print(f"[Query 5] Unique batches (RefTag): {batch_stats} | Max sample density: {batch_stats}")

    cursor.execute("SELECT MIN(LENGTH(DnaSequence)), MAX(LENGTH(DnaSequence)) FROM staging_source")
    dna_lengths = cursor.fetchone() 
    print(f"[Query 6] Genomic Integrity Check: Min Length {dna_lengths} bp, Max Length {dna_lengths} bp")
    print("==================================================")

    print("\n==================================================")
    print("=== PHASE B: RELATIONAL SCHEMA DECOMPOSITION   ===")
    print("==================================================")
    
    # 3NF Normalization: Parent Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patients (
        PatientID INTEGER PRIMARY KEY, Age INTEGER NOT NULL, RefTag INTEGER NOT NULL
    )''')

    # 3NF Normalization: Child Table tied via Foreign Key
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS biology (
        SequenceID INTEGER PRIMARY KEY AUTOINCREMENT,
        PatientID INTEGER NOT NULL, DnaSequence TEXT NOT NULL, Label INTEGER NOT NULL,
        FOREIGN KEY (PatientID) REFERENCES patients(PatientID)
    )''')
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_biology_label ON biology(Label);")
    connection.commit()

    # Lossless relational decomposition mapping
    cursor.execute("INSERT OR IGNORE INTO patients (PatientID, Age, RefTag) SELECT DISTINCT PatientID, Age, RefTag FROM staging_source")
    cursor.execute("INSERT INTO biology (PatientID, DnaSequence, Label) SELECT PatientID, DnaSequence, Label FROM staging_source")
    connection.commit()

    print("INFO - Relational normalization finalized: 'patients' and 'biology' schemas deployed.")
    print("==================================================")
    connection.close()

# ==============================================================================
# PIPELINE STAGE FUNCTIONS: DEEP NEURAL NETWORK INTEL & BENCHMARKS
# ==============================================================================

def encode_dna_sequences(sequences, mapping):
    """Flattens 2D One-Hot categorical sequence representations into standalone 1D arrays."""
    encoded_matrix = []
    for seq in sequences:
        one_hot_seq = []
        for base in seq:
            one_hot_seq.extend(mapping[base])
        encoded_matrix.append(one_hot_seq)
    return np.array(encoded_matrix, dtype=np.float32)


def run_machine_learning_pipeline(df: pd.DataFrame) -> tuple:
    """Executes feature tensor encoding, cross-validation splitting, and model training benchmarks."""
    mapping = {'A': [1.0, 0.0, 0.0, 0.0], 'C': [0.0, 1.0, 0.0, 0.0], 'G': [0.0, 0.0, 1.0, 0.0], 'T': [0.0, 0.0, 0.0, 1.0]}
    
    X_raw = df['DnaSequence'].values
    y = df['Label'].values

    X = encode_dna_sequences(X_raw, mapping)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=26, stratify=y)

    print(f"Input Feature Matrix Shape: {X.shape} | Training Dimensions: {X_train.shape}")

    # Model A Initialization (Standard MLP Topology)
    print("\n[TRAINING] Initializing Network Architecture A (32x16 Layer Config)...")
    model_nn_a = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', max_iter=15, random_state=26, verbose=True)
    model_nn_a.fit(X_train, y_train)

    # Model B Initialization (Deep Linear Map Topology)
    print("\n[TRAINING] Initializing Network Architecture B (64x Single Layer Config)...")
    model_nn_b = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam', max_iter=15, random_state=26, verbose=True)
    model_nn_b.fit(X_train, y_train)

    y_pred_a = model_nn_a.predict(X_test)
    y_pred_b = model_nn_b.predict(X_test)

    return y_test, y_pred_a, y_pred_b, model_nn_a.loss_curve_, model_nn_b.loss_curve_


def generate_pipeline_benchmarks(y_test, y_pred_a, y_pred_b, loss_a, loss_b):
    """Orchestrates computational convergence evaluation plots and saves results to disk."""
    acc_a = accuracy_score(y_test, y_pred_a)
    acc_b = accuracy_score(y_test, y_pred_b)

    # Plot 1: Backpropagation Loss Convergence Curves
    plt.figure(figsize=(7, 4))
    plt.plot(loss_a, label='Neural Network Architecture A', color='blue')
    plt.plot(loss_b, label='Neural Network Architecture B', color='green')
    plt.title('Loss Curve Comparison (Backpropagation Convergence)')
    plt.xlabel('Iterations (Epochs)'); plt.ylabel('Loss Value')
    plt.legend(); plt.grid(True)
    plt.savefig("benchmark_learning_curves.png", dpi=300); plt.close()
    print("INFO - Learning curves successfully saved as 'benchmark_learning_curves.png'")

    # Plot 2: Categorical Model Out-of-Sample Accuracy Bars
    plt.figure(figsize=(6, 4))
    sns.barplot(x=['Network Arch A (32x16)', 'Network Arch B (64,)'], y=[acc_a, acc_b], palette=['blue', 'green'])
    plt.title('Final Accuracy on Unseen Test Set')
    plt.ylabel('Accuracy (0.0 - 1.0)'); plt.ylim(0, 1.05)
    for i, v in enumerate([acc_a, acc_b]):
        plt.text(i, v + 0.02, f"{v:.2%}", ha='center', fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("benchmark_accuracy_barplot.png", dpi=300); plt.close()
    print("INFO - Accuracy barplot successfully saved as 'benchmark_accuracy_barplot.png'")

    # Plot 3: Side-by-Side Confusion Matrices
    cm_a = confusion_matrix(y_test, y_pred_a)
    cm_b = confusion_matrix(y_test, y_pred_b)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(cm_a, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False)
    ax1.set_title('Confusion Matrix: Architecture A')
    ax1.set_xlabel('Predicted Label'); ax1.set_ylabel('True Label')
    ax1.set_xticklabels(['No TATA', 'TATA-Box']); ax1.set_yticklabels(['No TATA', 'TATA-Box'])

    sns.heatmap(cm_b, annot=True, fmt='d', cmap='Greens', ax=ax2, cbar=False)
    ax2.set_title('Confusion Matrix: Architecture B')
    ax2.set_xlabel('Predicted Label'); ax2.set_ylabel('True Label')
    ax2.set_xticklabels(['No TATA', 'TATA-Box']); ax2.set_yticklabels(['No TATA', 'TATA-Box'])
    
    plt.tight_layout()
    plt.savefig("benchmark_confusion_matrices.png", dpi=300); plt.close()
    print("INFO - Confusion matrices successfully saved as 'benchmark_confusion_matrices.png'")

# ==============================================================================
# PIPELINE ORCHESTRATION ENGINE (MAIN RUNTIME)
# ==============================================================================

def main():
    print("Executing Stage 1: Data Simulation and Multiclass Staging...")
    df_raw = generate_genomic_dataset(n_samples=10000)
    export_initial_formats(df_raw)

    print("Executing Stage 2: Stress Testing via Structural Anomaly Injection...")
    df_corrupted = inject_pipeline_anomalies(df_raw)

    print("Executing Stage 3: Constraint Wrangling and Down-sampling Alignment...")
    df_cleaned = clean_and_balance_dataset(df_corrupted)
    df_final = execute_feature_engineering(df_cleaned)

    print("\n==================================================")
    print("POST-CLEANING FINAL PIPELINE INTEGRITY AUDIT")
    print("==================================================")
    df_final.info()
    
    tata_pos1 = df_final.loc[df_final['Label'] == 1, 'DnaSequence'].apply(lambda s: s[30:36] == 'TATAAA').all()
    tata_pos0 = df_final.loc[df_final['Label'] == 0, 'DnaSequence'].apply(lambda s: s[30:36] == 'TATAAA').any()
    print(f"\nVerification: Consensus motif correctly isolated on targets: {tata_pos1 and not tata_pos0}")

    df_final.to_excel(EXCEL_READY_PATH, index=False)
    df_final.to_parquet(PARQUET_READY_PATH, index=False)
    fasta_ready_file = open(FASTA_READY_PATH, "w", encoding="utf-8")
    for _, r in df_final.iterrows():
        fasta_ready_file.write(f">Patient_{r['PatientID']}|Label_{r['Label']}|Age_{r['Age']}\n{r['DnaSequence']}\n")
    fasta_ready_file.close()

    print("\nExecuting Stage 4: Relational Ingestion Engine Pipeline...")
    run_database_pipeline()

    print("\nExecuting Stage 5: Tensor Sequence Classification (MLP Inference Evaluation)...")
    y_test, y_pred_a, y_pred_b, loss_a, loss_b = run_machine_learning_pipeline(df_final)

    print("\nExecuting Stage 6: Graphical Performance Benchmark Synchronization...")
    generate_pipeline_benchmarks(y_test, y_pred_a, y_pred_b, loss_a, loss_b)

    print("\n==================================================")
    print("COMPUTATIONAL PRODUCTION PIPELINE EXECUTED SUCCESSFULLY")
    print("==================================================")


if __name__ == "__main__":
    main()

