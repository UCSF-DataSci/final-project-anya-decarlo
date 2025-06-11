import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
from pathlib import Path
from tabulate import tabulate
from scipy.stats import gmean
import soundfile as sf

# Set publication-quality style
plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

def parse_ravdess_filename(filename):
    """
    Parse RAVDESS filename to extract emotion label and metadata
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
    
    intensity_map = {
        '01': 'normal',
        '02': 'strong'
    }
    
    statement_map = {
        '01': 'Kids are talking by the door',
        '02': 'Dogs are sitting by the door'
    }
    
    return {
        'emotion': emotion_map.get(parts[2], 'unknown'),
        'intensity': intensity_map.get(parts[3], 'unknown'),
        'statement': statement_map.get(parts[4], 'unknown'),
        'repetition': int(parts[5]),
        'actor': int(parts[6]),
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
            try:
                # Get file info from filename
                file_info = parse_ravdess_filename(wav_file.name)
                file_info['filepath'] = str(wav_file)
                
                # Load audio to get duration
                y, sr = librosa.load(str(wav_file), sr=None)
                file_info['duration'] = librosa.get_duration(y=y, sr=sr)
                
                audio_files.append(file_info)
            except Exception as e:
                print(f"Error processing {wav_file.name}: {e}")
                continue
    
    return pd.DataFrame(audio_files)

def generate_publication_figures(data_dir, output_dir='ravdess_analysis'):
    """
    Generate publication-quality figures and tables for the RAVDESS dataset.
    
    Args:
        data_dir: Directory containing RAVDESS audio files
        output_dir: Directory to save output files (default: 'ravdess_analysis')
    """
    # Load dataset
    print("Loading RAVDESS dataset...")
    df = load_ravdess_dataset(data_dir)
    print(f"Loaded {len(df)} audio files")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving outputs to {output_dir}")
    
    # Figure 1: Emotion Distribution
    print("Generating emotion distribution plot...")
    plt.figure(figsize=(12, 6))
    emotion_counts = df['emotion'].value_counts().sort_index()
    ax = sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
    plt.title('Distribution of Emotions in RAVDESS Dataset', pad=20)
    plt.xlabel('Emotion')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Duration Distribution by Emotion
    print("Generating duration distribution plot...")
    plt.figure(figsize=(12, 6))
    # Use a colorblind-friendly palette
    sns.boxplot(data=df, x='emotion', y='duration', palette='colorblind')
    plt.title('Audio Duration Distribution by Emotion', pad=20, fontsize=18)
    plt.xlabel('Emotion', fontsize=15)
    plt.ylabel('Duration (seconds)', fontsize=15)
    plt.xticks(rotation=45, ha='right', fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # Save as both PNG and PDF for publication
    plt.savefig(os.path.join(output_dir, 'duration_by_emotion.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'duration_by_emotion.pdf'), bbox_inches='tight')
    plt.close()
    
    # DURATION EXPLANATION:
    # "Duration" is the length of each audio file in seconds. This plot shows, for each emotion, the spread and central tendency of how long the recordings are. The box shows the middle 50% of durations, the line is the median, and whiskers/outliers show the range. This helps you see if some emotions are spoken/sung for longer or shorter times, and how consistent the durations are within each emotion category.
    
    # Figure 3: Intensity Distribution by Emotion
    print("Generating intensity distribution plot...")
    plt.figure(figsize=(12, 6))
    intensity_by_emotion = pd.crosstab(df['emotion'], df['intensity'])
    intensity_by_emotion.plot(kind='bar', stacked=True)
    plt.title('Emotional Intensity Distribution', pad=20)
    plt.xlabel('Emotion')
    plt.ylabel('Number of Samples')
    plt.legend(title='Intensity')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'intensity_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate summary statistics tables
    print("Generating summary statistics...")
    # Table 1: Overall Dataset Statistics
    summary_stats = pd.DataFrame({
        'Metric': [
            'Total Samples',
            'Number of Actors',
            'Number of Emotions',
            'Mean Duration (s)',
            'Std Duration (s)',
            'Min Duration (s)',
            'Max Duration (s)'
        ],
        'Value': [
            len(df),
            df['actor'].nunique(),
            df['emotion'].nunique(),
            f"{df['duration'].mean():.2f}",
            f"{df['duration'].std():.2f}",
            f"{df['duration'].min():.2f}",
            f"{df['duration'].max():.2f}"
        ]
    })
    
    # Table 2: Emotion-specific Statistics
    emotion_stats = df.groupby('emotion').agg({
        'duration': ['mean', 'std', 'min', 'max'],
        'intensity': lambda x: x.value_counts().to_dict()
    }).round(2)
    
    # Save tables as LaTeX
    print("Saving LaTeX tables...")
    with open(os.path.join(output_dir, 'dataset_summary.tex'), 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Overall RAVDESS Dataset Statistics}\n")
        f.write(tabulate(summary_stats, headers='keys', tablefmt='latex', showindex=False))
        f.write("\n\\end{table}\n")
    
    with open(os.path.join(output_dir, 'emotion_statistics.tex'), 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Emotion-specific Statistics}\n")
        f.write(tabulate(emotion_stats, headers='keys', tablefmt='latex'))
        f.write("\n\\end{table}\n")
    
    print("Analysis complete!")
    return {
        'summary_stats': summary_stats,
        'emotion_stats': emotion_stats
    }

def compute_audio_summary_table(data_dir, output_dir='ravdess_analysis'):
    """
    Compute summary statistics for audio properties: duration, RMS amplitude, pitch (F0), spectral centroid.
    Save as a LaTeX table for publication.
    """
    df = load_ravdess_dataset(data_dir)
    stats = {
        'Duration (s)': [],
        'RMS Amplitude': [],
        'Pitch (Hz)': [],
        'Spectral Centroid (Hz)': []
    }
    for filepath in df['filepath']:
        try:
            y, sr = librosa.load(filepath, sr=None)
            # Duration
            duration = librosa.get_duration(y=y, sr=sr)
            # RMS amplitude
            rms = np.mean(librosa.feature.rms(y=y))
            # Pitch (F0)
            f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            pitch = np.mean(f0)
            # Spectral centroid
            centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            # Append
            stats['Duration (s)'].append(duration)
            stats['RMS Amplitude'].append(rms)
            stats['Pitch (Hz)'].append(pitch)
            stats['Spectral Centroid (Hz)'].append(centroid)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    # Compute summary statistics
    summary = pd.DataFrame({
        'Property': ['Duration (s)', 'RMS Amplitude', 'Pitch (Hz)', 'Spectral Centroid (Hz)'],
        'Mean': [np.mean(stats[k]) for k in stats],
        'Std': [np.std(stats[k]) for k in stats],
        'Min': [np.min(stats[k]) for k in stats],
        'Max': [np.max(stats[k]) for k in stats]
    })
    # Save as LaTeX
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'audio_summary_table.tex'), 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Summary statistics for audio properties in the RAVDESS dataset. Mean, standard deviation, minimum, and maximum are reported for duration, RMS amplitude, pitch (F0), and spectral centroid.}\n")
        f.write(tabulate(summary, headers='keys', tablefmt='latex', showindex=False, floatfmt=".2f"))
        f.write("\n\\end{table}\n")
    print("Audio summary table saved as LaTeX.")
    return summary

def compute_combined_feature_table(data_dir, output_dir='ravdess_analysis'):
    """
    Compute summary statistics for all extracted features and create a combined table (LaTeX and PNG).
    """
    import librosa
    import pandas as pd
    import numpy as np
    import os
    from tabulate import tabulate

    df = load_ravdess_dataset(data_dir)
    stats = {
        'Duration (s)': [],
        'RMS Amplitude': [],
        'Pitch (Hz)': [],
        'Spectral Centroid (Hz)': [],
        'Spectral Bandwidth (Hz)': [],
        'Spectral Rolloff (Hz)': [],
        'MFCC': [],
        'Chroma': [],
        'Zero-Crossing Rate': []
    }
    for filepath in df['filepath']:
        try:
            y, sr = librosa.load(filepath, sr=None)
            # Duration
            duration = librosa.get_duration(y=y, sr=sr)
            # RMS amplitude
            rms = np.mean(librosa.feature.rms(y=y))
            # Pitch (F0)
            f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            pitch = np.mean(f0)
            # Spectral centroid
            centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            # Spectral bandwidth
            bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            # Spectral rolloff
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            # MFCC (mean of means across 13 coefficients)
            mfcc = np.mean(np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1))
            # Chroma (mean of means across 12 coefficients)
            chroma = np.mean(np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1))
            # Zero-crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            # Append
            stats['Duration (s)'].append(duration)
            stats['RMS Amplitude'].append(rms)
            stats['Pitch (Hz)'].append(pitch)
            stats['Spectral Centroid (Hz)'].append(centroid)
            stats['Spectral Bandwidth (Hz)'].append(bandwidth)
            stats['Spectral Rolloff (Hz)'].append(rolloff)
            stats['MFCC'].append(mfcc)
            stats['Chroma'].append(chroma)
            stats['Zero-Crossing Rate'].append(zcr)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    # Table metadata with brief, human-readable calculation descriptions
    feature_rows = [
        [
            'Duration (s)',
            'Length of audio file in seconds.',
            'Total time from start to end',
            1
        ],
        [
            'RMS Amplitude',
            'Average signal energy.',
            'Mean RMS amplitude',
            1
        ],
        [
            'Pitch (Hz)',
            'Fundamental frequency (F0) in Hz.',
            'Mean pitch (Hz)',
            1
        ],
        [
            'Spectral Centroid (Hz)',
            'Center of mass of the spectrum.',
            'Mean spectral centroid',
            1
        ],
        [
            'Spectral Bandwidth (Hz)',
            'Spread of the spectrum.',
            'Mean spectral bandwidth',
            1
        ],
        [
            'Spectral Rolloff (Hz)',
            'Frequency below which 85% of energy is contained.',
            'Mean spectral rolloff',
            1
        ],
        [
            'MFCC',
            'Cepstral coefficients (mean of 13).',
            'Mean of 13 MFCCs',
            13
        ],
        [
            'Chroma',
            'Pitch class energy (mean of 12).',
            'Mean of 12 chroma features',
            12
        ],
        [
            'Zero-Crossing Rate',
            'Rate of sign changes in waveform.',
            'Mean zero-crossing rate',
            1
        ]
    ]
    # Compute summary statistics (rounded)
    means = [round(np.mean(stats[row[0]]), 2) for row in feature_rows]
    stds = [round(np.std(stats[row[0]]), 2) for row in feature_rows]
    mins = [round(np.min(stats[row[0]]), 2) for row in feature_rows]
    maxs = [round(np.max(stats[row[0]]), 2) for row in feature_rows]
    # Build DataFrame
    summary = pd.DataFrame([
        row + [means[i], stds[i], mins[i], maxs[i]]
        for i, row in enumerate(feature_rows)
    ], columns=[
        'Feature Name', 'Description', 'Calculation', 'Dimensionality',
        'Mean', 'Std', 'Min', 'Max'
    ])
    # Round and format all numbers as strings
    for col in ['Mean', 'Std', 'Min', 'Max']:
        summary[col] = summary[col].apply(lambda x: f"{float(x):.2f}")
    # Force line breaks every 20 characters for Description and Calculation
    def wrap_text(text, width=20):
        return '\n'.join([text[i:i+width] for i in range(0, len(text), width)])
    summary['Description'] = summary['Description'].apply(lambda x: wrap_text(x, 20))
    summary['Calculation'] = summary['Calculation'].apply(lambda x: wrap_text(x, 20))
    # Save as PDF and PNG using matplotlib classic table
    plt.figure(figsize=(18, 1.2 + 0.8*len(summary)))
    table = plt.table(
        cellText=summary.values,
        colLabels=summary.columns,
        cellLoc='center',
        loc='center',
        colLoc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.25, 1.4)
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.7)
        if key[0] == 0:
            cell.set_fontsize(12)
            cell.set_text_props(weight='bold')
    plt.axis('off')
    plt.tight_layout(pad=0.2)
    plt.savefig(os.path.join(output_dir, 'combined_audio_feature_table.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'combined_audio_feature_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Combined audio feature table saved as LaTeX, PDF, and PNG (classic table, forced wrapping, no overlap).")
    return summary

if __name__ == "__main__":
    data_dir = "ravdess-emotional-speech-audio"
    results = generate_publication_figures(data_dir)
    
    # Print summary statistics
    print("\nDataset Summary Statistics:")
    print(tabulate(results['summary_stats'], headers='keys', tablefmt='grid', showindex=False))
    
    print("\nEmotion-specific Statistics:")
    print(tabulate(results['emotion_stats'], headers='keys', tablefmt='grid'))
    
    # Add audio summary table
    audio_summary = compute_audio_summary_table(data_dir)
    print("\nAudio Data Summary Table:")
    print(tabulate(audio_summary, headers='keys', tablefmt='grid', showindex=False, floatfmt=".2f"))
    
    # Add combined feature table
    combined_summary = compute_combined_feature_table(data_dir)
    print("\nCombined Audio Feature Table:")
    print(tabulate(combined_summary, headers='keys', tablefmt='grid', showindex=False, floatfmt=".2f")) 