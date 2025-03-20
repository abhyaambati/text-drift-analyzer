import unittest
import numpy as np
from datetime import datetime, timedelta
from drift_analyzer import DriftAnalyzer
from sample_data_generator import create_sample_dataset

class TestDriftAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.analyzer = DriftAnalyzer()
        self.sample_data = create_sample_dataset(n_samples=100)
    
    def test_calculate_embeddings(self):
        """Test embedding calculation."""
        texts = ["Test text 1", "Test text 2"]
        embeddings = self.analyzer.calculate_embeddings(texts)
        self.assertEqual(embeddings.shape[0], 2)
        self.assertEqual(embeddings.shape[1], 384)  # MiniLM-L6-v2 embedding size
    
    def test_calculate_coherence_loss(self):
        """Test coherence loss calculation."""
        ref_texts = ["The quick brown fox", "A quick brown fox"]
        curr_texts = ["The quick brown fox", "A quick brown fox"]
        loss = self.analyzer.calculate_coherence_loss(ref_texts, curr_texts)
        self.assertGreaterEqual(loss, 0)
        self.assertLessEqual(loss, 1)
    
    def test_calculate_statistical_drift(self):
        """Test statistical drift calculation."""
        ref_texts = ["The quick brown fox", "A quick brown fox"]
        curr_texts = ["The quick brown fox", "A quick brown fox"]
        drift = self.analyzer.calculate_statistical_drift(ref_texts, curr_texts)
        self.assertIn('mean_drift', drift)
        self.assertIn('std_drift', drift)
        self.assertIn('kl_divergence', drift)
    
    def test_analyze_drift(self):
        """Test drift analysis on sample data."""
        results = self.analyzer.analyze_drift(
            self.sample_data['texts'].tolist(),
            self.sample_data['timestamps'].tolist(),
            window_size=20
        )
        self.assertIn('timestamps', results)
        self.assertIn('coherence_loss', results)
        self.assertIn('drift_score', results)
        self.assertIn('mean_drift', results)
        self.assertIn('std_drift', results)
        self.assertIn('kl_divergence', results)
        self.assertIn('anomaly_scores', results)
    
    def test_detect_significant_drift(self):
        """Test significant drift detection."""
        drift_scores = [1.0, 2.0, 5.1, 3.0, 4.0]
        significant = self.analyzer.detect_significant_drift(drift_scores, threshold=5.0)
        self.assertEqual(sum(significant), 1)
    
    def test_detect_anomalies(self):
        """Test anomaly detection."""
        anomaly_scores = [1.0, 2.0, 3.0, 10.0, 4.0]
        anomalies = self.analyzer.detect_anomalies(anomaly_scores, threshold=2.0)
        self.assertEqual(sum(anomalies), 1)
    
    def test_generate_report(self):
        """Test report generation."""
        results = self.analyzer.analyze_drift(
            self.sample_data['texts'].tolist(),
            self.sample_data['timestamps'].tolist(),
            window_size=20
        )
        report = self.analyzer.generate_report(results)
        self.assertIn('total_samples', report)
        self.assertIn('average_drift', report)
        self.assertIn('max_drift', report)
        self.assertIn('drift_trend', report)
        self.assertIn('significant_drift_events', report)
        self.assertIn('mean_drift_trend', report)
        self.assertIn('std_drift_trend', report)
        self.assertIn('kl_divergence_trend', report)
        self.assertIn('anomaly_count', report)
        self.assertIn('drift_volatility', report)

if __name__ == '__main__':
    unittest.main() 