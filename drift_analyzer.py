import numpy as np
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import torch
from scipy import stats
from tqdm import tqdm
import logging

class DriftAnalyzer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 32):
        """Initialize the drift analyzer with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.batch_size = batch_size
        logging.info(f"Using device: {self.device}")
        
    def calculate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Calculate embeddings for a list of texts with batching for memory efficiency."""
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Calculating embeddings"):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)
    
    def calculate_coherence_loss(self, 
                               reference_texts: List[str],
                               current_texts: List[str]) -> float:
        """Calculate coherence loss between reference and current texts."""
        ref_embeddings = self.calculate_embeddings(reference_texts)
        curr_embeddings = self.calculate_embeddings(current_texts)
        
        similarities = cosine_similarity(ref_embeddings, curr_embeddings)
        coherence_loss = 1 - np.mean(similarities)
        return coherence_loss
    
    def calculate_statistical_drift(self, 
                                  reference_texts: List[str],
                                  current_texts: List[str]) -> Dict[str, float]:
        """Calculate statistical measures of drift between reference and current texts."""
        ref_embeddings = self.calculate_embeddings(reference_texts)
        curr_embeddings = self.calculate_embeddings(current_texts)
        
        # Calculate mean and std of embeddings
        ref_mean = np.mean(ref_embeddings, axis=0)
        ref_std = np.std(ref_embeddings, axis=0)
        curr_mean = np.mean(curr_embeddings, axis=0)
        curr_std = np.std(curr_embeddings, axis=0)
        
        # Calculate statistical distances
        mean_drift = np.linalg.norm(ref_mean - curr_mean)
        std_drift = np.linalg.norm(ref_std - curr_std)
        
        # Calculate KL divergence between distributions
        kl_div = stats.entropy(ref_std, curr_std)
        
        return {
            'mean_drift': mean_drift,
            'std_drift': std_drift,
            'kl_divergence': kl_div
        }
    
    def analyze_drift(self,
                     texts: List[str],
                     timestamps: List[datetime],
                     window_size: int = 100) -> Dict[str, List]:
        """Analyze drift in texts over time using sliding windows with advanced metrics."""
        results = {
            'timestamps': [],
            'coherence_loss': [],
            'drift_score': [],
            'mean_drift': [],
            'std_drift': [],
            'kl_divergence': [],
            'anomaly_scores': []
        }
        
        # Sort texts by timestamp
        sorted_indices = np.argsort(timestamps)
        texts = [texts[i] for i in sorted_indices]
        timestamps = [timestamps[i] for i in sorted_indices]
        
        for i in tqdm(range(0, len(texts) - window_size + 1), desc="Analyzing drift"):
            window_texts = texts[i:i + window_size]
            window_timestamps = timestamps[i:i + window_size]
            
            # Use first half as reference
            ref_texts = window_texts[:window_size//2]
            curr_texts = window_texts[window_size//2:]
            
            # Calculate various drift metrics
            coherence_loss = self.calculate_coherence_loss(ref_texts, curr_texts)
            statistical_drift = self.calculate_statistical_drift(ref_texts, curr_texts)
            
            # Calculate anomaly score using statistical measures
            anomaly_score = (statistical_drift['mean_drift'] + 
                           statistical_drift['std_drift'] + 
                           statistical_drift['kl_divergence']) / 3
            
            results['timestamps'].append(window_timestamps[-1])
            results['coherence_loss'].append(coherence_loss)
            results['drift_score'].append(coherence_loss * 100)
            results['mean_drift'].append(statistical_drift['mean_drift'])
            results['std_drift'].append(statistical_drift['std_drift'])
            results['kl_divergence'].append(statistical_drift['kl_divergence'])
            results['anomaly_scores'].append(anomaly_score)
            
        return results
    
    def detect_significant_drift(self,
                               drift_scores: List[float],
                               threshold: float = 5.0) -> List[bool]:
        """Detect significant drift events based on a threshold."""
        return [score > threshold for score in drift_scores]
    
    def detect_anomalies(self,
                        anomaly_scores: List[float],
                        threshold: float = 2.0) -> List[bool]:
        """Detect anomalies using statistical methods."""
        scores = np.array(anomaly_scores)
        # Use median and MAD (Median Absolute Deviation) for more robust anomaly detection
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        # Convert MAD to standard deviation approximation
        std = mad * 1.4826
        return [score > (median + threshold * std) for score in scores]
    
    def generate_report(self, results: Dict[str, List]) -> Dict:
        """Generate a comprehensive drift analysis report with advanced metrics."""
        drift_scores = results['drift_score']
        anomaly_scores = results['anomaly_scores']
        
        # Calculate basic statistics
        basic_stats = {
            'total_samples': len(drift_scores),
            'average_drift': np.mean(drift_scores),
            'max_drift': np.max(drift_scores),
            'drift_trend': np.polyfit(range(len(drift_scores)), drift_scores, 1)[0],
            'significant_drift_events': sum(self.detect_significant_drift(drift_scores))
        }
        
        # Calculate advanced statistics
        advanced_stats = {
            'mean_drift_trend': np.polyfit(range(len(results['mean_drift'])), 
                                         results['mean_drift'], 1)[0],
            'std_drift_trend': np.polyfit(range(len(results['std_drift'])), 
                                        results['std_drift'], 1)[0],
            'kl_divergence_trend': np.polyfit(range(len(results['kl_divergence'])), 
                                            results['kl_divergence'], 1)[0],
            'anomaly_count': sum(self.detect_anomalies(anomaly_scores)),
            'drift_volatility': np.std(drift_scores)
        }
        
        # Combine all statistics
        return {**basic_stats, **advanced_stats} 