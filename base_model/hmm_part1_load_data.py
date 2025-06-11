import os
import pandas as pd
import librosa
import numpy as np
from pathlib import Path

def parse_ravdess_filename(filename):
    """
    Parse RAVDESS filename to extract emotion label
    Format: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
    """
    parts = filename.split('.')[0].split('-')
    
    emotion_map = {
        '01': 'neutral',
        '02': 'calm', 
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    emotion_code = parts[2]  # Third position is emotion
    actor_id = parts[6]      # Seventh position is actor
    
    return {
        'emotion': emotion_map.get(emotion_code, 'unknown'),
        'emotion_code': int(emotion_code),
        'actor': int(actor_id),
        'filename': filename
    }

def load_ravdess_dataset(data_path):
    """
    Load RAVDESS dataset and extract basic audio features
    """
    audio_files = []
    
    # Walk through all Actor folders
    for actor_folder in Path(data_path).glob('Actor_*'):
        for wav_file in actor_folder.glob('*.wav'):
            file_info = parse_ravdess_filename(wav_file.name)
            file_info['filepath'] = str(wav_file)
            audio_files.append(file_info)
    
    return pd.DataFrame(audio_files)

def extract_audio_features(filepath, sr=16000, n_mfcc=13):
    """
    Extract audio features for HMM observations
    """
    try:
        # Load audio
        y, sr = librosa.load(filepath, sr=sr)
        
        # Extract features
        features = {
            'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc),
            'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr),
            'spectral_rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr),
            'zero_crossing_rate': librosa.feature.zero_crossing_rate(y),
            'chroma': librosa.feature.chroma_stft(y=y, sr=sr)
        }
        
        # Concatenate all features
        feature_vector = np.concatenate([
            features['mfcc'],
            features['spectral_centroid'],
            features['spectral_rolloff'], 
            features['zero_crossing_rate'],
            features['chroma']
        ], axis=0)
        
        return feature_vector.T  # Return as (time_steps, features)
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def load_ravdess_features(data_dir=None, feature_type='mfcc'):
    """
    Load and prepare RAVDESS features for HMM processing.
    This function loads the dataset, extracts features, and organizes them
    by emotion class for the HMM pipeline.
    
    Args:
        data_dir: Directory containing RAVDESS audio files or precomputed features
        feature_type: Type of feature to extract ('mfcc', 'spectral', 'prosodic', 'chroma', 'all')
        
    Returns:
        observation_sequences: List of feature sequences for HMM
        emotion_labels: List of emotion labels for each sequence
        emotion_classes: List of unique emotion classes
    """
    try:
        # If data_dir is None, use example data
        if data_dir is None:
            # Generate some synthetic data for testing
            print("No data directory specified, using synthetic data for demonstration")
            return generate_synthetic_data()
            
        # Check if we have precomputed features
        feature_path = os.path.join(data_dir, f'{feature_type}')
        if os.path.exists(feature_path) and os.path.isdir(feature_path):
            print(f"Loading precomputed {feature_type} features from {feature_path}")
            return load_precomputed_features(feature_path)
        
        # Otherwise extract features from audio files
        print(f"Extracting {feature_type} features from audio in {data_dir}")
        df = load_ravdess_dataset(data_dir)
        
        # Group by emotion for processing
        emotion_groups = df.groupby('emotion')
        emotion_classes = sorted(df['emotion'].unique())
        
        observation_sequences = []
        emotion_labels = []
        
        # Process each file
        print(f"Processing {len(df)} audio files...")
        for idx, row in df.iterrows():
            features = extract_audio_features(row['filepath'])
            
            # Skip if feature extraction failed
            if features is None:
                continue
                
            # Select the requested feature type
            if feature_type == 'mfcc':
                # MFCCs are the first 13 dimensions
                feature_subset = features[:, :13]
            elif feature_type == 'spectral':
                # Spectral features (centroid + rolloff)
                feature_subset = features[:, 13:15]
            elif feature_type == 'prosodic':
                # Zero crossing rate as a prosodic feature
                feature_subset = features[:, 15:16]
            elif feature_type == 'chroma':
                # Chroma features 
                feature_subset = features[:, 16:]
            else:  # 'all' or any other value
                feature_subset = features
            
            observation_sequences.append(feature_subset)
            emotion_labels.append(row['emotion'])
            
        print(f"Extracted {len(observation_sequences)} feature sequences")
        print(f"Feature dimensionality: {observation_sequences[0].shape[1]}")
        
        return observation_sequences, emotion_labels, emotion_classes
        
    except Exception as e:
        print(f"Error loading RAVDESS features: {str(e)}")
        # Return synthetic data as fallback
        print("Falling back to synthetic data for demonstration")
        return generate_synthetic_data()


def generate_synthetic_data(n_sequences=120, seq_length=100, n_features=13):
    """
    Generate synthetic data for testing the HMM pipeline.
    
    Returns:
        observation_sequences: List of synthetic feature sequences
        emotion_labels: List of synthetic emotion labels
        emotion_classes: List of emotion classes
    """
    print("Generating synthetic data for demonstration purposes")
    np.random.seed(42)  # For reproducibility
    
    # Define emotion classes
    emotion_classes = ['angry', 'happy', 'sad', 'neutral']
    n_classes = len(emotion_classes)
    
    # Generate balanced synthetic data
    observation_sequences = []
    emotion_labels = []
    
    for emotion_idx, emotion in enumerate(emotion_classes):
        # Create class-specific mean to simulate different emotional patterns
        emotion_mean = np.zeros(n_features)
        emotion_mean[emotion_idx:emotion_idx+3] = 0.5  # Give each emotion a distinctive pattern
        
        # Generate sequences for this emotion
        for i in range(n_sequences // n_classes):
            # Add some randomness to sequence length (80-120 frames)
            actual_length = seq_length + np.random.randint(-20, 21)
            
            # Create sequence with emotion-specific characteristics
            sequence = np.random.randn(actual_length, n_features) * 0.3  # Base noise
            sequence += emotion_mean  # Add emotion-specific pattern
            
            # Add some temporal structure (simple trends)
            trend = np.linspace(0, 0.5, actual_length).reshape(-1, 1)
            sequence += trend * emotion_mean  # Emotion-specific trend
            
            observation_sequences.append(sequence)
            emotion_labels.append(emotion)
    
    print(f"Generated {len(observation_sequences)} synthetic sequences")
    print(f"Feature dimensionality: {n_features}")
    
    return observation_sequences, emotion_labels, emotion_classes


def load_precomputed_features(feature_dir):
    """
    Load precomputed features from a directory.
    
    Args:
        feature_dir: Directory containing precomputed features
        
    Returns:
        observation_sequences: List of feature sequences
        emotion_labels: List of emotion labels
        emotion_classes: List of unique emotion classes
    """
    # Check for NPZ files (compressed NumPy arrays)
    feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.npz')]
    
    if not feature_files:
        raise ValueError(f"No feature files found in {feature_dir}")
    
    observation_sequences = []
    emotion_labels = []
    
    for file in feature_files:
        file_path = os.path.join(feature_dir, file)
        data = np.load(file_path)
        
        # Extract features and label
        observation_sequences.append(data['features'])
        emotion_labels.append(data['emotion'].item())  # Convert from numpy string to Python string
    
    # Get unique emotion classes
    emotion_classes = sorted(list(set(emotion_labels)))
    
    print(f"Loaded {len(observation_sequences)} precomputed feature sequences")
    print(f"Feature dimensionality: {observation_sequences[0].shape[1]}")
    
    return observation_sequences, emotion_labels, emotion_classes


# Example usage:
if __name__ == "__main__":
    # Load dataset metadata
    data_path = "/Users/anyadecarlo/datasci224-final/ravdess-emotional-speech-audio"
    
    # Example 1: Load the dataset
    df = load_ravdess_dataset(data_path)
    print("Dataset loaded!")
    print(f"Total files: {len(df)}")
    print(f"Emotions: {df['emotion'].value_counts()}")
    print(f"Actors: {df['actor'].nunique()}")
    
    # Example 2: Extract features from one file
    first_file = df.iloc[0]
    features = extract_audio_features(first_file['filepath'])
    print(f"Feature shape for {first_file['filename']}: {features.shape}")
    print(f"Emotion: {first_file['emotion']}")
    
    # Example 3: Use the pipeline function with synthetic data
    obs, labels, classes = load_ravdess_features()
    print(f"\nSynthetic data: {len(obs)} sequences, {len(classes)} emotion classes")
    print(f"Classes: {classes}")
    print(f"First sequence shape: {obs[0].shape}")
    print(f"First 5 labels: {labels[:5]}")
    