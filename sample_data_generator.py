import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_texts(n_samples: int = 1000) -> list:
    """Generate sample texts with controlled drift patterns."""
    base_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A quick brown fox jumps over the lazy dog.",
        "The brown fox quickly jumps over the lazy dog.",
        "A brown fox quickly jumps over the lazy dog.",
        "The lazy dog is jumped over by the quick brown fox.",
        "A lazy dog is jumped over by the quick brown fox."
    ]
    
    texts = []
    drift_factor = 0.0
    drift_increment = 0.1 / n_samples
    
    for _ in range(n_samples):
        # Add controlled drift by modifying words
        text = random.choice(base_texts)
        if random.random() < drift_factor:
            # Introduce drift by replacing words
            words = text.split()
            if len(words) > 2:
                idx = random.randint(0, len(words)-2)
                words[idx] = random.choice(['fast', 'swift', 'rapid', 'speedy'])
                text = ' '.join(words)
        
        texts.append(text)
        drift_factor += drift_increment
    
    return texts

def generate_timestamps(n_samples: int, start_date: datetime = None) -> list:
    """Generate timestamps for the sample data."""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    
    timestamps = []
    for i in range(n_samples):
        timestamp = start_date + timedelta(hours=i)
        timestamps.append(timestamp)
    
    return timestamps

def create_sample_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """Create a sample dataset with controlled drift patterns."""
    texts = generate_sample_texts(n_samples)
    timestamps = generate_timestamps(n_samples)
    
    df = pd.DataFrame({
        'texts': texts,
        'timestamps': timestamps
    })
    
    return df

def save_sample_dataset(filename: str = 'sample_drift_data.csv', n_samples: int = 1000):
    """Save a sample dataset to a CSV file."""
    df = create_sample_dataset(n_samples)
    df.to_csv(filename, index=False)
    print(f"Sample dataset saved to {filename}")
    print(f"Number of samples: {n_samples}")
    print(f"Date range: {df['timestamps'].min()} to {df['timestamps'].max()}")

if __name__ == '__main__':
    save_sample_dataset() 