import os
import sys
import unittest
import numpy as np
import pandas as pd

# Automatically inject the parent directory into the Python path resolution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    generate_genomic_dataset,
    clean_and_balance_dataset,
    encode_dna_sequences
)

class TestGenomicPipeline(unittest.TestCase):
    """Automated unit tests focused strictly on pure memory computations."""

    def test_stage1_generation_properties(self):
        """[Stage 1] Verifies row dimensions and key features of simulated data."""
        n_samples = 100
        df = generate_genomic_dataset(n_samples=n_samples)
        self.assertEqual(len(df), n_samples)
        self.assertIn("DnaSequence", df.columns)

    def test_stage2_cleansing_constraints(self):
        """[Stage 2] Validates strict age boundaries and class balancing properties."""
        df_raw = generate_genomic_dataset(n_samples=5000)
        df_cleaned = clean_and_balance_dataset(df_raw)
        
        # Verify age constraints are locked between 20 and 65
        self.assertTrue((df_cleaned["Age"] >= 20).all())
        self.assertTrue((df_cleaned["Age"] <= 65).all())
        
        # Verify perfect down-sampling distribution by accessing specific class counts
        class_counts = df_cleaned["Label"].value_counts()
        self.assertEqual(class_counts[0], 1000)
        self.assertEqual(class_counts[1], 1000)

    def test_stage3_tensor_encoding_shape(self):
        """[Stage 3] Ensures categorical nucleotide sequences convert to clean tensor matrices."""
        mapping = {'A': [1.0, 0.0, 0.0, 0.0], 'C': [0.0, 1.0, 0.0, 0.0], 'G': [0.0, 0.0, 1.0, 0.0], 'T': [0.0, 0.0, 0.0, 1.0]}
        mock_sequences = ["ACGT", "TGCA"]
        encoded = encode_dna_sequences(mock_sequences, mapping)
        
        # 2 samples, 4 bases * 4 dimensions = 16 features
        self.assertEqual(encoded.shape, (2, 16))
        self.assertEqual(encoded.dtype, np.float32)

if __name__ == "__main__":
    unittest.main()
