# Part 2: Feature Extraction for HMM-Based Emotion Recognition

**Objective:** Extract acoustic features from speech signals that will serve as observations for the Hidden Markov Model.

## 1. Setup

Import necessary libraries.

```python
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
```

## 2. Feature Extraction

Implement functions to extract relevant acoustic features from audio files.

```python
def extract_mfcc_features(audio_file, n_mfcc=13):
    """
    Extract Mel-Frequency Cepstral Coefficients from an audio file.
    
    Args:
        audio_file: Path to the audio file
        n_mfcc: Number of MFCCs to extract
        
    Returns:
        Array of MFCC features
    """
    # YOUR CODE HERE
    # 1. Load the audio file using librosa
    # 2. Extract MFCCs from the audio signal
    # 3. Calculate statistics of MFCCs (mean, std, etc.)
    
    return np.array([])  # Replace with actual implementation
```

```python
def extract_prosodic_features(audio_file):
    """
    Extract prosodic features (pitch, energy) from an audio file.
    
    Args:
        audio_file: Path to the audio file
        
    Returns:
        Dictionary containing prosodic features
    """
    # YOUR CODE HERE
    # 1. Load the audio file using librosa
    # 2. Extract fundamental frequency (F0) using librosa.pyin
    # 3. Calculate energy/intensity features
    # 4. Calculate statistics of F0 and energy (mean, std, range, etc.)
    
    return {}  # Replace with actual implementation
```

## 3. Feature Vector Creation

Combine acoustic features into observation sequences for HMM training.

```python
def create_observation_sequences(file_paths, frame_length=0.025, frame_shift=0.010):
    """
    Process audio files into sequences of observation vectors for HMM training.
    
    Args:
        file_paths: List of paths to audio files
        frame_length: Length of each frame in seconds
        frame_shift: Shift between consecutive frames in seconds
        
    Returns:
        List of observation sequences, each being a 2D array (frames x features)
    """
    # YOUR CODE HERE
    # 1. For each audio file:
    #    a. Extract MFCC features per frame
    #    b. Extract prosodic features per frame
    #    c. Combine features into frame-level observation vectors
    # 2. Return list of observation sequences
    
    return []  # Replace with actual implementation
```

## 4. Feature Visualization

Visualize extracted features to understand their distribution across emotions.

```python
def visualize_features(features, emotions, feature_names):
    """
    Create visualizations of extracted features across different emotions.
    
    Args:
        features: Array of features (n_samples x n_features)
        emotions: Array of emotion labels
        feature_names: List of feature names
        
    Returns:
        None (displays plots)
    """
    # YOUR CODE HERE
    # 1. Create box plots of features by emotion
    # 2. Create scatter plots for pairs of important features
    # 3. Optional: dimensionality reduction visualization (PCA, t-SNE)
    
    pass
```
