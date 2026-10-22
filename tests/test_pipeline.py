import os
import sys
import unittest
import numpy as np
import pandas as pd

# Automatically inject the parent directory into the Python path resolution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import updated functions and the central CONFIG
from main import (
    generate_genomic_dataset,
    inject_pipeline_anomalies,
    clean_and_balance_dataset,
    execute_feature_engineering,
    encode_dna_sequences,
    CONFIG
)

class TestGenomicPipeline(unittest.TestCase):
    """Automated unit tests focused strictly on pure memory computations and pipeline constraints."""

    def setUp(self):
        """Executed before each test to ensure operational reproducibility."""
        np.random.seed(26) # Forces identical random extraction for all tests

    def test_stage1_generation_properties(self):
        """[Stage 1] Verifies row dimensions and key features of simulated data."""
        n_samples = 100
        df = generate_genomic_dataset(n_samples=n_samples)
        self.assertEqual(len(df), n_samples)
        self.assertIn("DnaSequence", df.columns)
        self.assertIn("Age", df.columns)

    def test_stage2_anomaly_injection(self):
        """[Stage 2] Verifies that anomaly injection introduces boundary out-of-bounds or NaNs."""
        df_raw = generate_genomic_dataset(n_samples=200)
        df_corrupted = inject_pipeline_anomalies(df_raw)
       
        # Check if NaNs or structural noise were successfully injected
        has_nans = df_corrupted["Age"].isna().any() or df_corrupted["Label"].isna().any()
        self.assertTrue(has_nans, "Anomaly injector failed to introduce expected NaN states.")

    def test_stage3_cleansing_constraints(self):
        """[Stage 3] Validates strict age boundaries and global configuration class balancing."""
        df_raw = generate_genomic_dataset(n_samples=10000) # Increased to guarantee enough sampling density
        df_corrupted = inject_pipeline_anomalies(df_raw)
        df_cleaned = clean_and_balance_dataset(df_corrupted)
       
        # Verify age constraints are locked dynamically against central CONFIG parameters
        min_age = CONFIG["clinical_constraints"]["min_age"]
        max_age = CONFIG["clinical_constraints"]["max_age"]
        self.assertTrue((df_cleaned["Age"] >= min_age).all())
        self.assertTrue((df_cleaned["Age"] <= max_age).all())
       
        # Verify target down-sampling distribution matches CONFIG requirements by accessing labels surgically
        expected_samples = CONFIG["clinical_constraints"]["samples_per_class"]
        class_counts = df_cleaned["Label"].value_counts()
        
        self.assertEqual(int(class_counts.at[0]), expected_samples)
        self.assertEqual(int(class_counts.at[1]), expected_samples)

    def test_stage4_feature_engineering_motif(self):
        """[Stage 4] Ensures the biological consensus motif is correctly targeted only to positive labels."""
        df_raw = generate_genomic_dataset(n_samples=10000)
        df_cleaned = clean_and_balance_dataset(df_raw)
        df_engineered = execute_feature_engineering(df_cleaned)
       
        motif = CONFIG["genomic_engineering"]["motif"]
        pos = CONFIG["genomic_engineering"]["locus_position"]
        cut = CONFIG["genomic_engineering"]["cut_position"]
       
        # Verify that ALL positive samples contain the motif at the designated locus position
        positive_sequences = df_engineered.loc[df_engineered['Label'] == 1, 'DnaSequence']
        motif_isolated = positive_sequences.apply(lambda s: s[pos:cut] == motif).all()
        self.assertTrue(motif_isolated, "Feature engineering failed to securely inject the biological motif.")

    def test_stage5_tensor_encoding_shape(self):
        """[Stage 5] Ensures categorical nucleotide sequences convert to clean tensor matrices."""
        mapping = {'A': [1.0, 0.0, 0.0, 0.0], 'C': [0.0, 1.0, 0.0, 0.0], 'G': [0.0, 0.0, 1.0, 0.0], 'T': [0.0, 0.0, 0.0, 1.0]}
        mock_sequences = ["ACGT", "TGCA"]
        encoded = encode_dna_sequences(mock_sequences, mapping)
       
        # 2 samples, 4 bases * 4 dimensions = 16 features
        self.assertEqual(encoded.shape, (2, 16))
        self.assertEqual(encoded.dtype, np.float32)

if __name__ == "__main__":
    unittest.main()

