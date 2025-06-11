"""
HMM Evaluation Module for Speech Emotion Recognition
This module implements evaluation metrics, cross-validation, and results visualization
for Hidden Markov Models applied to speech emotion recognition.

Based on Rabiner's tutorial and comprehensive HMM implementation guide.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import entropy
from joblib import Parallel, delayed
from pathlib import Path

# Import our HMM implementation
import hmm_part2_model_implementation as hmm_model
import hmm_part1_load_data as data_loader

def evaluate_hmm_performance(true_labels, predicted_labels, emotion_classes):
    """
    Calculate performance metrics for the HMM emotion classifier.
    
    Args:
        true_labels: Ground truth emotion labels
        predicted_labels: Predicted emotion labels from the HMM
        emotion_classes: List of emotion class names
        
    Returns:
        Dictionary of performance metrics (accuracy, precision, recall, F1)
    """
    # Calculate overall accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Calculate per-class precision, recall, and F1 score
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    
    # Generate per-class metrics
    class_report = classification_report(true_labels, predicted_labels, 
                                        target_names=emotion_classes, 
                                        zero_division=0, 
                                        output_dict=True)
    
    # Compile metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'class_report': class_report
    }
    
    return metrics


def train_emotion_specific_hmms(training_data, emotion_classes, n_states=5, n_components=1, n_iter=100, n_jobs=-1):
    """
    Train a separate HMM for each emotion class.
    
    Args:
        training_data: Dictionary mapping emotion labels to lists of observation sequences
        emotion_classes: List of all emotion class names
        n_states: Number of hidden states for each HMM
        n_components: Number of Gaussian mixture components
        n_iter: Maximum number of Baum-Welch iterations
        n_jobs: Number of CPU jobs for parallel training
        
    Returns:
        Dictionary mapping emotion labels to their trained HMM parameters
    """
    def _train_single(emotion):
        if emotion not in training_data or len(training_data[emotion]) == 0:
            print(f"No training data for {emotion}, skipping.")
            return emotion, None
        observation_sequences = training_data[emotion]
        combined_sequence = np.vstack(observation_sequences)
        A, B_params, pi, log_likelihood = hmm_model.baum_welch_algorithm(
            combined_sequence,
            N=n_states,
            max_iter=n_iter,
            n_components=n_components
        )
        params = {
            'A': A,
            'B_params': B_params,
            'pi': pi,
            'log_likelihood': log_likelihood
        }
        print(f"  - {emotion} HMM trained. Log-likelihood: {log_likelihood:.2f}")
        return emotion, params

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_train_single)(emo) for emo in emotion_classes
    )

    # Aggregate
    emotion_hmms = {emo: params for emo, params in results if params is not None}
    return emotion_hmms


def classify_emotion(observation_sequence, emotion_hmms, emission_probability_fn=None):
    """
    Classify an observation sequence by finding the emotion HMM with highest likelihood.
    
    Args:
        observation_sequence: Feature sequence to classify
        emotion_hmms: Dictionary mapping emotions to their trained HMM parameters
        emission_probability_fn: Custom emission probability function if needed
        
    Returns:
        predicted_emotion: The emotion class with highest likelihood
        log_likelihoods: Dictionary of log-likelihoods for each emotion
    """
    log_likelihoods = {}
    
    for emotion, hmm_params in emotion_hmms.items():
        # Extract HMM parameters
        A = hmm_params['A']
        B_params = hmm_params['B_params']
        pi = hmm_params['pi']
        
        # Create emission probability function based on the trained parameters
        if emission_probability_fn is None:
            # Define emission function based on B_params (Gaussian or GMM)
            means = B_params['means']
            covs = B_params['covariances']
            weights = B_params.get('weights', None)
            n_components = B_params.get('n_components', 1)
            
            # Create emission probability function for this emotion's HMM
            if n_components > 1 and weights is not None:
                # GMM emission
                def B(j, obs):
                    return hmm_model.gmm_emission_probability(weights[j], means[j], covs[j], obs)
            else:
                # Single Gaussian emission
                def B(j, obs):
                    return hmm_model.gaussian_emission_probability(means[j, 0], covs[j, 0], obs)
        else:
            # Use provided emission function
            B = emission_probability_fn
            
        # Calculate log-likelihood of the sequence given this emotion's HMM
        _, log_likelihood, _ = hmm_model.forward_algorithm(observation_sequence, A, B, pi)
        log_likelihoods[emotion] = log_likelihood
    
    # Find emotion with highest log-likelihood (closest to zero since log-likelihoods are negative)
    if log_likelihoods:
        # For negative log-likelihoods, the highest value (closest to zero) is the most probable
        predicted_emotion = max(log_likelihoods, key=lambda k: log_likelihoods[k])
    else:
        predicted_emotion = None
        
    return predicted_emotion, log_likelihoods


def cross_validate_hmm(observation_sequences, emotion_labels, emotion_classes,
                      n_states=5, n_components=1, n_folds=5, n_iter=100, n_jobs=-1,
                      return_paths=False, *, file_names=None, group_by_actor=False):
    """
    Perform cross-validation for the HMM emotion recognition system.
    
    Args:
        observation_sequences: List of observation sequences
        emotion_labels: Corresponding emotion labels
        emotion_classes: List of all emotion class names
        n_states: Number of hidden states for each HMM
        n_components: Number of Gaussian mixture components
        n_folds: Number of cross-validation folds
        n_iter: Maximum number of iterations for training
        n_jobs: Number of CPU jobs for parallel training
        return_paths: If True, also return the Viterbi-decoded state path for every
            utterance (aligned with the order of observation_sequences).
        file_names: List of file paths/names aligned with observation_sequences. Required
            if group_by_actor is True.
        group_by_actor: If True, perform speaker-independent CV by ensuring no actor
            appears in both train and test folds using GroupKFold. Actor IDs are
            extracted from RAVDESS filenames.
        
    Returns:
        Dictionary of cross-validated performance metrics
    """
    # Initialize performance metrics storage
    all_true_labels = []
    all_predicted_labels = []
    fold_metrics = []
    # Optional storage for per-utterance state paths
    if return_paths:
        all_viterbi_paths = []
    
    # Choose cross-validator
    if group_by_actor:
        if file_names is None:
            raise ValueError("file_names must be provided when group_by_actor=True")
        from sklearn.model_selection import GroupKFold

        def _actor_id(fname: str):
            # RAVDESS pattern: ...-<ActorID>.wav where ActorID is 2-digit
            actor_token = Path(fname).stem.split('-')[-1]
            return int(actor_token)

        groups = np.array([_actor_id(f) for f in file_names])
        cv_iter = GroupKFold(n_splits=n_folds).split(np.zeros(len(emotion_labels)), emotion_labels, groups)
    else:
        cv_iter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(
            np.zeros(len(emotion_labels)), emotion_labels)
    
    # Start cross-validation
    print(f"Starting {n_folds}-fold cross-validation...")
    
    for fold, (train_idx, test_idx) in enumerate(cv_iter):
        print(f"\nFold {fold+1}/{n_folds}:")
        
        # Split data into train and test sets
        X_train = observation_sequences[train_idx]
        y_train = emotion_labels[train_idx]
        X_test = observation_sequences[test_idx]
        y_test = emotion_labels[test_idx]
        
        # Organize training data by emotion
        training_data = {}
        for emotion in emotion_classes:
            emotion_indices = np.where(y_train == emotion)[0]
            training_data[emotion] = [X_train[i] for i in emotion_indices]
        
        # Train emotion-specific HMMs
        emotion_hmms = train_emotion_specific_hmms(
            training_data, 
            emotion_classes,
            n_states=n_states,
            n_components=n_components,
            n_iter=n_iter,
            n_jobs=n_jobs
        )
        
        # Classify test sequences
        y_pred = []
        for i, test_sequence in enumerate(X_test):
            predicted_emotion, _ = classify_emotion(test_sequence, emotion_hmms)
            y_pred.append(predicted_emotion)
            
            # Derive Viterbi state path if requested
            if return_paths:
                params = emotion_hmms.get(predicted_emotion)
                if params is not None:
                    A = params['A']
                    pi = params['pi']
                    B_params = params['B_params']
                    means = B_params['means']
                    covs = B_params['covariances']
                    weights = B_params['weights']
                    n_comp = B_params.get('n_components', 1)
                    
                    # Build emission probability function for this emotion model
                    if n_comp > 1:
                        def B(j, obs):
                            return hmm_model.gmm_emission_probability(weights[j], means[j], covs[j], obs)
                    else:
                        def B(j, obs):
                            return hmm_model.gaussian_emission_probability(means[j, 0], covs[j, 0], obs)
                    
                    path, _ = hmm_model.viterbi_algorithm(test_sequence, A, B, pi)
                    all_viterbi_paths.append(path)
                else:
                    all_viterbi_paths.append(None)
        
        # Store true and predicted labels
        all_true_labels.extend(y_test)
        all_predicted_labels.extend(y_pred)
        
        # Calculate metrics for this fold
        fold_metric = evaluate_hmm_performance(y_test, y_pred, emotion_classes)
        fold_metrics.append(fold_metric)
        print(f"Fold {fold+1} accuracy: {fold_metric['accuracy']:.4f}")
    
    # Calculate average metrics across folds
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
        'precision': np.mean([m['precision'] for m in fold_metrics]),
        'recall': np.mean([m['recall'] for m in fold_metrics]),
        'f1_score': np.mean([m['f1_score'] for m in fold_metrics])
    }
    
    # Overall metrics on combined predictions
    overall_metrics = evaluate_hmm_performance(all_true_labels, all_predicted_labels, emotion_classes)
    
    results_dict = {
        'fold_metrics': fold_metrics,
        'average_metrics': avg_metrics,
        'overall_metrics': overall_metrics,
        'all_true_labels': all_true_labels,
        'all_predicted_labels': all_predicted_labels
    }
    
    # Add Viterbi paths only if requested
    if return_paths:
        results_dict['viterbi_paths'] = all_viterbi_paths
    
    return results_dict


def plot_confusion_matrix(true_labels, predicted_labels, emotion_classes, normalized=True):
    """
    Create and plot a confusion matrix for emotion classification results.
    
    Args:
        true_labels: Ground truth emotion labels
        predicted_labels: Predicted emotion labels from the HMM
        emotion_classes: List of emotion class names
        normalized: Whether to normalize values by class support size
        
    Returns:
        fig: The matplotlib figure object (can be used to save the plot)
    """
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Normalize if requested
    if normalized:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_norm
        fmt = '.2f'
        title = 'Normalized Confusion Matrix for HMM Emotion Recognition'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix for HMM Emotion Recognition'
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=emotion_classes, yticklabels=emotion_classes,
                ax=ax)
    
    # Add labels and title
    ax.set_xlabel('Predicted Emotion')
    ax.set_ylabel('True Emotion')
    ax.set_title(title)
    
    # Adjust layout for better display
    plt.tight_layout()
    
    return fig


def plot_performance_comparison(metrics_dict, title='HMM Performance by Emotion'):
    """
    Create bar plots comparing performance metrics across emotions.
    
    Args:
        metrics_dict: Dictionary containing class_report from evaluate_hmm_performance
        title: Title for the plot
        
    Returns:
        fig: The matplotlib figure object
    """
    # Extract per-emotion metrics from classification report
    class_report = metrics_dict['class_report']
    
    # Get list of emotions (excluding avg/total)
    emotions = [label for label in class_report.keys() 
              if label not in ['accuracy', 'macro avg', 'weighted avg']]
    
    # Extract metrics for each emotion
    precision = [class_report[emotion]['precision'] for emotion in emotions]
    recall = [class_report[emotion]['recall'] for emotion in emotions]
    f1 = [class_report[emotion]['f1-score'] for emotion in emotions]
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set width of bars and positions
    barWidth = 0.25
    r1 = np.arange(len(emotions))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Create grouped bars
    ax.bar(r1, precision, width=barWidth, label='Precision', color='skyblue')
    ax.bar(r2, recall, width=barWidth, label='Recall', color='lightgreen')
    ax.bar(r3, f1, width=barWidth, label='F1 Score', color='salmon')
    
    # Add labels, title, and legend
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks([r + barWidth for r in range(len(emotions))])
    ax.set_xticklabels(emotions)
    ax.set_ylim(0, 1.0)  # Metrics are between 0 and 1
    ax.legend()
    
    # Add value annotations on top of bars
    for i, bars in enumerate([precision, recall, f1]):
        for j, val in enumerate(bars):
            ax.text(j + i*barWidth, val + 0.02, f'{val:.2f}', 
                   ha='center', va='bottom', rotation=0, fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_log_likelihood_evolution(log_likelihoods):
    """
    Plot the evolution of log-likelihood during HMM training.
    
    Args:
        log_likelihoods: List of log-likelihood values from training
        
    Returns:
        fig: The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(1, len(log_likelihoods) + 1)
    
    ax.plot(iterations, log_likelihoods, marker='o', linestyle='-', color='blue')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Log-Likelihood')
    ax.set_title('HMM Training Convergence: Log-Likelihood Evolution')
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig


def generate_results_table(cross_val_results, emotion_classes):
    """
    Generate a formatted table of results for inclusion in academic report.
    
    Args:
        cross_val_results: Results from cross_validate_hmm function
        emotion_classes: List of emotion class names
        
    Returns:
        DataFrame containing formatted results table
    """
    # Extract metrics
    overall_metrics = cross_val_results['overall_metrics']
    avg_metrics = cross_val_results['average_metrics']
    class_report = overall_metrics['class_report']
    
    # Create per-emotion metrics table
    emotion_data = []
    for emotion in emotion_classes:
        if emotion in class_report:
            emotion_metrics = class_report[emotion]
            emotion_data.append({
                'Emotion': emotion,
                'Precision': f"{emotion_metrics['precision']:.4f}",
                'Recall': f"{emotion_metrics['recall']:.4f}",
                'F1 Score': f"{emotion_metrics['f1-score']:.4f}",
                'Support': emotion_metrics['support']
            })
    
    # Create DataFrames
    emotion_df = pd.DataFrame(emotion_data)
    
    # Create summary metrics DataFrame
    summary_data = [
        {'Metric': 'Overall Accuracy', 'Value': f"{overall_metrics['accuracy']:.4f}"},
        {'Metric': 'Weighted Precision', 'Value': f"{overall_metrics['precision']:.4f}"},
        {'Metric': 'Weighted Recall', 'Value': f"{overall_metrics['recall']:.4f}"},
        {'Metric': 'Weighted F1 Score', 'Value': f"{overall_metrics['f1_score']:.4f}"},
        {'Metric': 'Cross-Val Avg Accuracy', 'Value': f"{avg_metrics['accuracy']:.4f}"},
        {'Metric': 'Cross-Val Avg F1', 'Value': f"{avg_metrics['f1_score']:.4f}"}
    ]
    summary_df = pd.DataFrame(summary_data)
    
    return {'emotion_metrics': emotion_df, 'summary_metrics': summary_df}


def parameter_sensitivity_analysis(observation_sequences, emotion_labels, emotion_classes, 
                                  n_states_range=[3, 5, 7, 9], 
                                  n_components_range=[1, 2, 3], 
                                  n_folds=3, n_iter=50):
    """
    Analyze how HMM performance varies with different parameter settings.
    
    Args:
        observation_sequences: List of observation sequences
        emotion_labels: Corresponding emotion labels
        emotion_classes: List of emotion class names
        n_states_range: Range of hidden state counts to test
        n_components_range: Range of GMM components to test
        n_folds: Number of cross-validation folds
        n_iter: Maximum number of iterations for training
        
    Returns:
        Dictionary of performance results for each parameter combination
    """
    results = {}
    
    print("Starting parameter sensitivity analysis...")
    
    # Test different parameter combinations
    for n_states in n_states_range:
        for n_components in n_components_range:
            print(f"\nTesting model with n_states={n_states}, n_components={n_components}")
            
            # Perform cross-validation with this parameter combination
            cv_results = cross_validate_hmm(
                observation_sequences, 
                emotion_labels,
                emotion_classes,
                n_states=n_states,
                n_components=n_components,
                n_folds=n_folds,
                n_iter=n_iter
            )
            
            # Store results
            key = f"states{n_states}_components{n_components}"
            results[key] = {
                'n_states': n_states,
                'n_components': n_components,
                'accuracy': cv_results['average_metrics']['accuracy'],
                'f1_score': cv_results['average_metrics']['f1_score']
            }
            
            print(f"Average accuracy: {cv_results['average_metrics']['accuracy']:.4f}, "
                  f"F1 score: {cv_results['average_metrics']['f1_score']:.4f}")
    
    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame({
        'N States': [results[key]['n_states'] for key in results],
        'N Components': [results[key]['n_components'] for key in results],
        'Accuracy': [results[key]['accuracy'] for key in results],
        'F1 Score': [results[key]['f1_score'] for key in results]
    })
    
    return {'detailed_results': results, 'summary_df': results_df}


def plot_parameter_sensitivity(sensitivity_results):
    """
    Plot the results of parameter sensitivity analysis.
    
    Args:
        sensitivity_results: Results from parameter_sensitivity_analysis
        
    Returns:
        Two matplotlib figures (states_fig, components_fig)
    """
    df = sensitivity_results['summary_df']
    
    # Create figure for states analysis
    states_fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Group by number of states and compute mean accuracy
    states_grouped = df.groupby('N States').mean()
    
    # Plot effect of number of states
    ax1.plot(states_grouped.index, states_grouped['Accuracy'], 
            marker='o', linestyle='-', color='blue', label='Accuracy')
    ax1.plot(states_grouped.index, states_grouped['F1 Score'], 
            marker='s', linestyle='-', color='red', label='F1 Score')
    
    # Add labels and legend
    ax1.set_xlabel('Number of Hidden States')
    ax1.set_ylabel('Score')
    ax1.set_title('Effect of Hidden States Count on HMM Performance')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Create figure for components analysis
    components_fig, ax2 = plt.subplots(figsize=(10, 6))
    
    # Group by number of components and compute mean accuracy
    components_grouped = df.groupby('N Components').mean()
    
    # Plot effect of number of Gaussian mixture components
    ax2.plot(components_grouped.index, components_grouped['Accuracy'], 
            marker='o', linestyle='-', color='blue', label='Accuracy')
    ax2.plot(components_grouped.index, components_grouped['F1 Score'], 
            marker='s', linestyle='-', color='red', label='F1 Score')
    
    # Add labels and legend
    ax2.set_xlabel('Number of Gaussian Mixture Components')
    ax2.set_ylabel('Score')
    ax2.set_title('Effect of GMM Components Count on HMM Performance')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    return states_fig, components_fig


def compute_perplexity(log_likelihood, sequence_length):
    """
    Compute perplexity from log-likelihood and sequence length.
    Perplexity is often used in language models but can be applied to HMMs as well.
    
    Args:
        log_likelihood: Log-likelihood of the sequence
        sequence_length: Length of the observation sequence
        
    Returns:
        Perplexity value
    """
    return np.exp(-log_likelihood / sequence_length)


def compare_models_with_likelihood(test_sequences, emotion_hmms):
    """
    Compare multiple HMM models using likelihood-based metrics.
    
    Args:
        test_sequences: List of observation sequences to evaluate
        emotion_hmms: Dictionary mapping emotions to trained HMM parameters
        
    Returns:
        DataFrame with log-likelihood and perplexity for each sequence and model
    """
    results = []
    
    for i, sequence in enumerate(test_sequences):
        sequence_length = len(sequence)
        for emotion, hmm_params in emotion_hmms.items():
            # Extract HMM parameters
            A = hmm_params['A']
            B_params = hmm_params['B_params']
            pi = hmm_params['pi']
            
            # Create emission probability function
            means = B_params['means']
            covs = B_params['covariances']
            weights = B_params.get('weights', None)
            n_components = B_params.get('n_components', 1)
            
            # Define appropriate emission function
            if n_components > 1 and weights is not None:
                # GMM emission
                def B(j, obs):
                    return hmm_model.gmm_emission_probability(weights[j], means[j], covs[j], obs)
            else:
                def B(j, obs):
                    return hmm_model.gaussian_emission_probability(means[j, 0], covs[j, 0], obs)
            
            # Compute log-likelihood using forward algorithm
            _, log_likelihood, _ = hmm_model.forward_algorithm(sequence, A, B, pi)
            
            # Compute perplexity
            perplexity = compute_perplexity(log_likelihood, sequence_length)
            
            # Store results
            results.append({
                'Sequence': i,
                'Emotion Model': emotion,
                'Log-Likelihood': log_likelihood,
                'Perplexity': perplexity
            })
    
    return pd.DataFrame(results)


def save_results_to_file(results_dict, filename='hmm_evaluation_results.pkl'):
    """
    Save evaluation results to a file for later analysis.
    
    Args:
        results_dict: Dictionary containing evaluation results
        filename: Output filename
        
    Returns:
        None
    """
    import pickle
    
    with open(filename, 'wb') as f:
        pickle.dump(results_dict, f)
    
    print(f"Results saved to {filename}")


def generate_academic_results_section(cv_results, emotion_classes, sensitivity_results=None, 
                                output_dir='results', save_figures=True):
    """
    Generate a comprehensive set of publication-ready figures and tables for academic reporting.
    This produces all necessary visualization elements for a complete results section.
    
    Args:
        cv_results: Results from cross_validate_hmm function
        emotion_classes: List of emotion class names
        sensitivity_results: Optional results from parameter_sensitivity_analysis
        output_dir: Directory to save output files
        save_figures: Whether to save figures to files
        
    Returns:
        Dictionary containing all generated tables and figure handles
    """
    import os
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    if save_figures and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize results container
    academic_results = {}
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Results tables - create formatted tables with LaTeX export option
    print("Generating results tables...")
    tables = generate_results_table(cv_results, emotion_classes)
    academic_results['tables'] = tables
    
    # 1.1 Save summary metrics table
    summary_table = tables['summary_metrics']
    latex_summary = summary_table.style.to_latex()
    academic_results['summary_table_latex'] = latex_summary
    
    if save_figures:
        summary_table.to_csv(f"{output_dir}/summary_metrics_{timestamp}.csv")
        with open(f"{output_dir}/summary_metrics_{timestamp}.tex", 'w') as f:
            f.write(latex_summary)
    
    # 1.2 Save per-emotion metrics table
    emotion_table = tables['emotion_metrics']
    latex_emotion = emotion_table.style.to_latex()
    academic_results['emotion_table_latex'] = latex_emotion
    
    if save_figures:
        emotion_table.to_csv(f"{output_dir}/emotion_metrics_{timestamp}.csv")
        with open(f"{output_dir}/emotion_metrics_{timestamp}.tex", 'w') as f:
            f.write(latex_emotion)
    
    # 2. Confusion Matrix - both normalized and non-normalized
    print("Generating confusion matrices...")
    true_labels = cv_results['all_true_labels']
    pred_labels = cv_results['all_predicted_labels']
    
    # 2.1 Normalized confusion matrix
    cm_norm_fig = plot_confusion_matrix(true_labels, pred_labels, emotion_classes, normalized=True)
    academic_results['confusion_matrix_normalized'] = cm_norm_fig
    
    if save_figures:
        cm_norm_fig.savefig(f"{output_dir}/confusion_matrix_norm_{timestamp}.pdf", bbox_inches='tight', dpi=300)
        cm_norm_fig.savefig(f"{output_dir}/confusion_matrix_norm_{timestamp}.png", bbox_inches='tight', dpi=300)
    
    # 2.2 Raw count confusion matrix
    cm_raw_fig = plot_confusion_matrix(true_labels, pred_labels, emotion_classes, normalized=False)
    academic_results['confusion_matrix_raw'] = cm_raw_fig
    
    if save_figures:
        cm_raw_fig.savefig(f"{output_dir}/confusion_matrix_raw_{timestamp}.pdf", bbox_inches='tight', dpi=300)
    
    # 3. Performance comparison across emotions
    print("Generating performance comparison charts...")
    perf_fig = plot_performance_comparison(cv_results['overall_metrics'])
    academic_results['performance_comparison'] = perf_fig
    
    if save_figures:
        perf_fig.savefig(f"{output_dir}/performance_by_emotion_{timestamp}.pdf", bbox_inches='tight', dpi=300)
        perf_fig.savefig(f"{output_dir}/performance_by_emotion_{timestamp}.png", bbox_inches='tight', dpi=300)
    
    # 4. Parameter sensitivity analysis if available
    if sensitivity_results is not None:
        print("Generating parameter sensitivity plots...")
        states_fig, components_fig = plot_parameter_sensitivity(sensitivity_results)
        academic_results['sensitivity_states'] = states_fig
        academic_results['sensitivity_components'] = components_fig
        
        if save_figures:
            states_fig.savefig(f"{output_dir}/sensitivity_states_{timestamp}.pdf", bbox_inches='tight', dpi=300)
            states_fig.savefig(f"{output_dir}/sensitivity_states_{timestamp}.png", bbox_inches='tight', dpi=300)
            components_fig.savefig(f"{output_dir}/sensitivity_components_{timestamp}.pdf", bbox_inches='tight', dpi=300)
            components_fig.savefig(f"{output_dir}/sensitivity_components_{timestamp}.png", bbox_inches='tight', dpi=300)
    
    # 5. Generate LaTeX code for including all figures in a report
    latex_code = generate_latex_figure_code(timestamp, sensitivity_results is not None)
    academic_results['latex_code'] = latex_code
    
    if save_figures:
        with open(f"{output_dir}/latex_figure_code_{timestamp}.tex", 'w') as f:
            f.write(latex_code)
    
    print(f"Academic results generation complete. Files saved to {output_dir}/")
    return academic_results


def generate_latex_figure_code(timestamp, include_sensitivity=True):
    """
    Generate LaTeX code for including the generated figures in an academic report.
    
    Args:
        timestamp: Timestamp string used in filenames
        include_sensitivity: Whether to include sensitivity analysis figures
        
    Returns:
        String containing LaTeX code
    """
    latex = []
    latex.append("% Include these code snippets in your LaTeX document")
    latex.append("% For the confusion matrix figure:")
    latex.append("\\begin{figure}[htbp]")
    latex.append("    \\centering")
    latex.append(f"    \\includegraphics[width=0.7\\textwidth]{{confusion_matrix_norm_{timestamp}.pdf}}")
    latex.append("    \\caption{Normalized confusion matrix for HMM emotion recognition. The model demonstrates strong diagonal dominance, indicating good classification performance across most emotion categories.}")
    latex.append("    \\label{fig:confusion_matrix}")
    latex.append("\\end{figure}")
    latex.append("")
    
    latex.append("% For the performance comparison figure:")
    latex.append("\\begin{figure}[htbp]")
    latex.append("    \\centering")
    latex.append(f"    \\includegraphics[width=0.8\\textwidth]{{performance_by_emotion_{timestamp}.pdf}}")
    latex.append("    \\caption{Performance metrics by emotion category. The figure shows precision, recall, and F1-score for each emotion class, highlighting the strengths and limitations of the HMM approach for different emotional expressions.}")
    latex.append("    \\label{fig:performance_comparison}")
    latex.append("\\end{figure}")
    latex.append("")
    
    if include_sensitivity:
        latex.append("% For the parameter sensitivity figures:")
        latex.append("\\begin{figure}[htbp]")
        latex.append("    \\centering")
        latex.append("    \\begin{subfigure}[b]{0.48\\textwidth}")
        latex.append(f"        \\includegraphics[width=\\textwidth]{{sensitivity_states_{timestamp}.pdf}}")
        latex.append("        \\caption{Effect of hidden states count}")
        latex.append("    \\end{subfigure}")
        latex.append("    \\hfill")
        latex.append("    \\begin{subfigure}[b]{0.48\\textwidth}")
        latex.append(f"        \\includegraphics[width=\\textwidth]{{sensitivity_components_{timestamp}.pdf}}")
        latex.append("        \\caption{Effect of GMM components count}")
        latex.append("    \\end{subfigure}")
        latex.append("    \\caption{Parameter sensitivity analysis showing how model performance varies with different hyperparameter settings. (a) Shows the impact of the number of hidden states on accuracy and F1-score, while (b) illustrates the effect of different numbers of Gaussian mixture components in emission probabilities.}")
        latex.append("    \\label{fig:parameter_sensitivity}")
        latex.append("\\end{figure}")
    
    return '\n'.join(latex)


if __name__ == "__main__":
    print("HMM Evaluation Module for Speech Emotion Recognition")
    print("=====================================================")
    
    # Example usage demonstrating the workflow
    print("\nExample workflow:")
    print("1. Load feature data from hmm_part1_load_data")
    print("2. Split data into training and testing sets")
    print("3. Train emotion-specific HMMs using cross_validate_hmm")
    print("4. Evaluate performance with evaluate_hmm_performance")
    print("5. Visualize results with plotting functions")
    print("6. Perform parameter sensitivity analysis")
    
    # Demonstrate with synthetic data if available
    try:
        # Try to import example data (small subset)
        from hmm_part1_load_data import load_example_data
        
        print("\nLoading example data...")
        observation_sequences, emotion_labels, emotion_classes = load_example_data()
        
        if len(observation_sequences) > 0:
            print(f"Loaded {len(observation_sequences)} sequences with {len(emotion_classes)} emotion classes")
            
            # Example cross-validation (with minimal parameters for demo)
            print("\nRunning quick demo cross-validation with minimal parameters...")
            cv_results = cross_validate_hmm(
                observation_sequences[:10],  # Use only first 10 for demo
                emotion_labels[:10],
                emotion_classes,
                n_states=3,
                n_components=1,
                n_folds=2,
                n_iter=5,  # Very minimal for demo
                n_jobs=2
            )
            
            # Display example results
            print(f"\nDemo cross-validation results:")
            print(f"Average accuracy: {cv_results['average_metrics']['accuracy']:.4f}")
            
            # Example visualization
            print("\nGenerating example confusion matrix...")
            cm_fig = plot_confusion_matrix(
                cv_results['all_true_labels'],
                cv_results['all_predicted_labels'],
                emotion_classes
            )
            
            # Save example figure
            cm_fig.savefig('example_confusion_matrix.png')
            print("Example confusion matrix saved as 'example_confusion_matrix.png'")
        
    except (ImportError, AttributeError) as e:
        print(f"\nCould not load example data: {str(e)}")
        print("Run this module after implementing the data loading functions.")
    
    print("\nModule ready for use in emotion recognition pipeline.")
    print("For full evaluation, import this module and call its functions")
    print("with your prepared feature data.")
    print("=====================================================")
