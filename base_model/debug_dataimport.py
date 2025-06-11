import os
import numpy as np
from pathlib import Path
import librosa
import matplotlib.pyplot as plt

def debug_ravdess_structure(data_path):
    """Debug the structure of the RAVDESS dataset folders and files"""
    
    print("=== DEBUGGING RAVDESS DIRECTORY STRUCTURE ===\n")
    
    # Check if path exists
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"Error: Directory {data_path} does not exist!")
        return
        
    # Look for Actor folders
    actor_folders = list(data_dir.glob('Actor_*'))
    print(f"Found {len(actor_folders)} Actor folders")
    
    # List first few actor folders
    for i, actor_folder in enumerate(sorted(actor_folders)[:3]):
        print(f"Actor folder {i+1}: {actor_folder.name}")
    print()
    
    # If no actor folders, check if files are directly in the data_path
    if not actor_folders:
        wav_files = list(data_dir.glob('*.wav'))
        print(f"Found {len(wav_files)} wav files directly in folder")
    
    # Get file counts for a sample actor
    if actor_folders:
        first_actor = actor_folders[0]
        wav_files = list(first_actor.glob('*.wav'))
        print(f"In {first_actor.name}, found {len(wav_files)} files")
        
        # Try to load a sample file with librosa
        if wav_files:
            try:
                sample_file = wav_files[0]
                print(f"\nTrying to load sample file: {sample_file}")
                y, sr = librosa.load(sample_file)
                duration = librosa.get_duration(y=y, sr=sr)
                print(f"Successfully loaded audio file: {sample_file.name}")
                print(f"Sample rate: {sr} Hz, Duration: {duration:.2f} seconds")
            except Exception as e:
                print(f"Error loading audio file: {e}")

def debug_filenames(data_path):
    """Debug what the actual filenames look like and parse their structure"""
    
    print("\n=== DEBUGGING RAVDESS FILENAMES ===\n")
    
    # Look for Actor folders
    actor_folders = list(Path(data_path).glob('Actor_*'))
    
    if not actor_folders:
        # Maybe files are directly in the folder?
        wav_files = list(Path(data_path).glob('*.wav'))
        print(f"Found {len(wav_files)} wav files directly in folder")
        
        for i, wav_file in enumerate(wav_files[:5]):  # Show first 5
            print(f"File {i+1}: {wav_file.name}")
            parts = wav_file.name.split('.')[0].split('-')
            print(f"  Parts: {parts}")
            print(f"  Length: {len(parts)}")
            if len(parts) >= 3:
                print(f"  Emotion position (index 2): '{parts[2]}'")
            print()
    else:
        # Files are in Actor folders
        first_actor = actor_folders[0]
        wav_files = list(first_actor.glob('*.wav'))
        print(f"In {first_actor.name}, found {len(wav_files)} files")
        
        for i, wav_file in enumerate(wav_files[:5]):  # Show first 5
            print(f"File {i+1}: {wav_file.name}")
            parts = wav_file.name.split('.')[0].split('-')
            print(f"  Parts: {parts}")
            print(f"  Length: {len(parts)}")
            if len(parts) >= 3:
                print(f"  Emotion position (index 2): '{parts[2]}'")
                # Map emotion code to label according to RAVDESS convention
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
                emotion_code = parts[2]
                emotion_label = emotion_map.get(emotion_code, 'unknown')
                print(f"  Emotion: {emotion_label} (code: {emotion_code})")
                
                # Check intensity (if available)
                if len(parts) >= 4:
                    intensity_map = {'01': 'normal', '02': 'strong'}
                    intensity = intensity_map.get(parts[3], 'unknown')
                    print(f"  Intensity: {intensity} (code: {parts[3]})")
            print()
        # Files are in Actor folders
        first_actor = actor_folders[0]
        wav_files = list(first_actor.glob('*.wav'))
        print(f"In {first_actor.name}, found {len(wav_files)} files")
        
        for i, wav_file in enumerate(wav_files[:5]):  # Show first 5
            print(f"File {i+1}: {wav_file.name}")
            parts = wav_file.name.split('.')[0].split('-')
            print(f"  Parts: {parts}")
            print(f"  Length: {len(parts)}")
            if len(parts) >= 3:
                print(f"  Emotion position (index 2): '{parts[2]}'")
            print()

# -----------------------------------------------------------------------------
# Primary loader used by both classic and hybrid pipelines
# -----------------------------------------------------------------------------
def load_ravdess_features(data_dir, feature_type: str = 'mfcc', *, return_file_paths: bool = False):
    """
    Load RAVDESS features for the HMM pipeline.
    Simple interface function to extract acoustic features from RAVDESS audio files.
    
    Args:
        data_dir: Directory containing RAVDESS audio files
        feature_type: Type of feature to extract ('mfcc', 'spectral', 'prosodic', 'chroma')
        return_file_paths: Optional parameter to return file paths list
        
    Returns:
        observation_sequences: List of feature sequences
        emotion_labels: List of emotion labels 
        emotion_classes: List of unique emotion classes
        file_paths: List of file paths (if return_file_paths is True)
    """
    print(f"Loading {feature_type} features from {data_dir}...")
    
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    # Find all audio files in the dataset
    all_wav_files = []
    actor_folders = list(Path(data_dir).glob('Actor_*'))
    
    if actor_folders:
        for actor_folder in actor_folders:
            wav_files = list(actor_folder.glob('*.wav'))
            all_wav_files.extend(wav_files)
    else:
        # Try direct wav files if no Actor folders
        all_wav_files = list(Path(data_dir).glob('*.wav'))
    
    if not all_wav_files:
        raise ValueError(f"No audio files found in {data_dir}")
    
    print(f"Found {len(all_wav_files)} audio files")
    
    # Emotion mapping (RAVDESS standard)
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
    
    # Extract features and labels
    observation_sequences: list[np.ndarray] = []
    emotion_labels: list[str] = []
    file_paths: list[str] = []
    
    for wav_file in all_wav_files:
        try:
            # Extract emotion from filename
            parts = wav_file.name.split('.')[0].split('-')
            if len(parts) < 3:
                continue
                
            emotion_code = parts[2]
            emotion = emotion_map.get(emotion_code, 'unknown')
            if emotion == 'unknown':
                continue
            
            # Load audio
            y, sr = librosa.load(str(wav_file), sr=16000)
            
            # Extract features based on type
            if feature_type == 'mfcc':
                # 13-dimensional MFCC coefficients
                features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
                
            elif feature_type == 'spectral':
                # Spectral features: centroid and rolloff
                centroid = librosa.feature.spectral_centroid(y=y, sr=sr).T
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).T
                bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).T
                features = np.hstack((centroid, rolloff, bandwidth))
                
            elif feature_type == 'prosodic':
                # Prosodic features: zero-crossing rate and RMS energy
                zcr = librosa.feature.zero_crossing_rate(y).T
                rms = librosa.feature.rms(y=y).T
                features = np.hstack((zcr, rms))
                
            elif feature_type == 'chroma':
                # Chroma features
                features = librosa.feature.chroma_stft(y=y, sr=sr).T
                
            elif feature_type == 'combined':
                # Combine all features for a comprehensive representation
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
                centroid = librosa.feature.spectral_centroid(y=y, sr=sr).T
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).T
                zcr = librosa.feature.zero_crossing_rate(y).T
                rms = librosa.feature.rms(y=y).T
                features = np.hstack((mfcc, centroid, rolloff, zcr, rms))
            
            else:
                # Default to MFCCs if unknown feature type requested
                print(f"Unknown feature type '{feature_type}', defaulting to MFCC")
                features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
            
            # Add to dataset
            observation_sequences.append(features)
            emotion_labels.append(emotion)
            file_paths.append(str(wav_file))
            
        except Exception as e:
            print(f"Error processing {wav_file.name}: {e}")
    
    # Get unique emotion classes
    emotion_classes = sorted(list(set(emotion_labels)))
    
    print(f"Successfully loaded {len(observation_sequences)} sequences")
    print(f"Found {len(emotion_classes)} emotion classes: {emotion_classes}")
    
    if return_file_paths:
        return observation_sequences, emotion_labels, emotion_classes, file_paths
    return observation_sequences, emotion_labels, emotion_classes


# Testing and debug code
data_path = "/Users/anyadecarlo/datasci224-final/ravdess-emotional-speech-audio"

# Execute the function to debug the data import
if __name__ == "__main__":
    # Run debug functions
    debug_ravdess_structure(data_path)
    debug_filenames(data_path)
    
    # Test feature extraction
    import numpy as np
    print("\nTesting feature extraction...")
    obs, labels, classes = load_ravdess_features(data_path, 'mfcc')
    print(f"First sequence shape: {obs[0].shape if obs else 'No data'}")