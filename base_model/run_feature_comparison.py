#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HMM Feature Type Comparison for Speech Emotion Recognition

This script compares the performance of different acoustic feature types (MFCC, spectral, prosodic, etc.)
for the task of speech emotion recognition using HMMs. It runs the full HMM pipeline on each feature type
and generates comparative visualizations and tables.

Related to the main HMM pipeline implementation.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import our pipeline
from run_hmm_pipeline import run_full_pipeline

# Configure matplotlib for high-quality figures
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'

def compare_feature_types(data_dir=None, 
                          feature_types=['mfcc', 'spectral', 'prosodic'],
                          n_states=5, n_components=1, n_folds=5, n_iter=100,
                          output_dir='results'):
    """
    Run the HMM pipeline on multiple feature types and compare performance.
    
    Args:
        data_dir: Directory containing feature files for all feature types
        feature_types: List of feature types to compare
        n_states: Number of hidden states for HMMs
        n_components: Number of Gaussian mixture components
        n_folds: Number of cross-validation folds
        n_iter: Maximum iterations for Baum-Welch algorithm
        output_dir: Directory to save comparison results
    
    Returns:
        Dictionary with comparative results
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create main output directory
    comparison_dir = os.path.join(output_dir, f'feature_comparison_{timestamp}')
    os.makedirs(comparison_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("HMM SPEECH EMOTION RECOGNITION: FEATURE TYPE COMPARISON")
    print("="*80)
    print(f"Comparing {len(feature_types)} feature types: {', '.join(feature_types)}")
    print(f"Using {n_states} states, {n_components} GMM components, {n_folds} folds")
    print("="*80 + "\n")
    
    # Run pipeline for each feature type
    results = {}
    performance_summary = []
    
    for feature_type in feature_types:
        print(f"\n{'='*30} PROCESSING FEATURE TYPE: {feature_type} {'='*30}\n")
        
        # Run pipeline for this feature type (without feature prefix in directory)
        feature_results = run_full_pipeline(
            data_dir=data_dir,
            feature_type=feature_type,
            n_states=n_states,
            n_components=n_components,
            n_folds=n_folds,
            n_iter=n_iter,
            run_sensitivity=False,
            output_dir=comparison_dir,
            feature_dir_prefix=True
        )
        
        if feature_results is None:
            print(f"Error processing {feature_type}, skipping...")
            continue
            
        # Store results
        results[feature_type] = feature_results
        
        # Extract key metrics for comparison
        metrics = feature_results['cv_results']['overall_metrics']
        performance_summary.append({
            'Feature Type': feature_type,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score']
        })
    
    # Create comparison dataframe
    if not performance_summary:
        print("No valid results to compare!")
        return None
        
    comparison_df = pd.DataFrame(performance_summary)
    
    # Generate comparative visualizations
    print("\n" + "="*50)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*50)
    
    try:
        # 1. Bar chart comparing key metrics across feature types
        performance_fig = plot_feature_performance_comparison(comparison_df)
        performance_fig.savefig(
            os.path.join(comparison_dir, f'feature_comparison_{timestamp}.pdf'),
            bbox_inches='tight', dpi=300
        )
        performance_fig.savefig(
            os.path.join(comparison_dir, f'feature_comparison_{timestamp}.png'),
            bbox_inches='tight', dpi=300
        )
        
        # 2. Confusion matrix grid (one per feature type)
        if len(feature_types) > 1:
            confusion_grid = plot_confusion_matrix_grid(results, feature_types)
            if confusion_grid:
                confusion_grid.savefig(
                    os.path.join(comparison_dir, f'confusion_matrix_comparison_{timestamp}.pdf'),
                    bbox_inches='tight', dpi=300
                )
                confusion_grid.savefig(
                    os.path.join(comparison_dir, f'confusion_matrix_comparison_{timestamp}.png'),
                    bbox_inches='tight', dpi=300
                )
        
        # 3. Generate radar chart for multi-dimensional visualization
        radar_fig = plot_feature_radar_chart(comparison_df)
        radar_fig.savefig(
            os.path.join(comparison_dir, f'feature_radar_chart_{timestamp}.pdf'),
            bbox_inches='tight', dpi=300
        )
        radar_fig.savefig(
            os.path.join(comparison_dir, f'feature_radar_chart_{timestamp}.png'),
            bbox_inches='tight', dpi=300
        )
        
        # 4. Per-emotion performance across feature types
        if len(feature_types) > 1 and all(ft in results for ft in feature_types):
            try:
                emotion_comp_fig = plot_per_emotion_feature_comparison(results, feature_types)
                emotion_comp_fig.savefig(
                    os.path.join(comparison_dir, f'emotion_feature_comparison_{timestamp}.pdf'),
                    bbox_inches='tight', dpi=300
                )
                emotion_comp_fig.savefig(
                    os.path.join(comparison_dir, f'emotion_feature_comparison_{timestamp}.png'),
                    bbox_inches='tight', dpi=300
                )
            except Exception as e:
                print(f"Error generating per-emotion comparison: {str(e)}")
        
        # 5. Generate LaTeX table
        latex_table = generate_feature_comparison_table(comparison_df)
        with open(os.path.join(comparison_dir, f'feature_comparison_table_{timestamp}.tex'), 'w') as f:
            f.write(latex_table)
        
        # Save comparison results
        comparison_results = {
            'timestamp': timestamp,
            'feature_types': feature_types,
            'comparison_df': comparison_df,
            'individual_results': results,
        }
        
        # Save to pickle
        with open(os.path.join(comparison_dir, f'feature_comparison_results_{timestamp}.pkl'), 'wb') as f:
            pickle.dump(comparison_results, f)
            
        # Print summary table
        print("\nFEATURE TYPE COMPARISON SUMMARY:")
        print(comparison_df.to_string(index=False))
        
        # Identify best feature type
        best_idx = comparison_df['F1 Score'].idxmax()
        best_feature = comparison_df.iloc[best_idx]['Feature Type']
        best_f1 = comparison_df.iloc[best_idx]['F1 Score']
        print(f"\nBest performing feature type: {best_feature} (F1 Score: {best_f1:.4f})")
        
    except Exception as e:
        print(f"Error generating comparison visualizations: {str(e)}")
    
    # Final timing
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("\n" + "="*80)
    print(f"Feature comparison completed in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print("="*80)
    
    return comparison_results


def plot_feature_performance_comparison(comparison_df):
    """
    Generate bar plot comparing performance metrics across feature types.
    """
    plt.figure(figsize=(12, 7))
    
    # Reshape data for seaborn
    plot_df = pd.melt(
        comparison_df, 
        id_vars=['Feature Type'],
        value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        var_name='Metric', value_name='Score'
    )
    
    # Create plot
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Feature Type', y='Score', hue='Metric', data=plot_df)
    
    # Customize
    plt.title('Performance Comparison Across Feature Types', fontsize=16)
    plt.xlabel('Feature Type', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Metric', fontsize=12)
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=10)
    
    plt.tight_layout()
    return plt.gcf()


def plot_confusion_matrix_grid(results, feature_types):
    """
    Generate grid of confusion matrices for each feature type.
    """
    try:
        n_features = len(feature_types)
        if n_features < 1:
            return None
            
        # Create figure with subplots
        fig, axes = plt.subplots(1, n_features, figsize=(6*n_features, 5))
        if n_features == 1:
            axes = [axes]  # Make it iterable
            
        # Plot confusion matrix for each feature type
        for i, feature_type in enumerate(feature_types):
            # Skip if feature type not in results
            if feature_type not in results:
                continue
                
            feature_results = results[feature_type]
            cv_results = feature_results['cv_results']
            
            # Import hmm_part3_evaluation for plotting function
            import hmm_part3_evaluation as evaluator
            
            # Get true and predicted labels
            true_labels = cv_results['all_true_labels']
            pred_labels = cv_results['all_predicted_labels']
            
            # Get confusion matrix
            from sklearn.metrics import confusion_matrix
            import numpy as np
            cm = confusion_matrix(true_labels, pred_labels)
            # Normalize
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot
            emotion_classes = cv_results.get('emotion_classes', range(cm.shape[0]))
            sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                        xticklabels=emotion_classes, 
                        yticklabels=emotion_classes,
                        ax=axes[i])
            axes[i].set_title(f'Feature Type: {feature_type}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error generating confusion matrix grid: {str(e)}")
        return None


def plot_feature_radar_chart(comparison_df):
    """
    Generate a radar chart comparing feature types across multiple metrics.
    This visualization is particularly effective for multivariate comparison.
    
    Args:
        comparison_df: DataFrame with feature types and performance metrics
        
    Returns:
        Matplotlib figure containing the radar chart
    """
    # Prepare data for radar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    feature_types = comparison_df['Feature Type'].tolist()
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of metrics
    N = len(metrics)
    
    # Angle for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Initialize plot
    ax.set_theta_offset(np.pi / 2)  # Start at top
    ax.set_theta_direction(-1)  # Go clockwise
    
    # Set labels
    plt.xticks(angles[:-1], metrics)
    
    # Draw each feature type
    colors = plt.cm.tab10(np.linspace(0, 1, len(feature_types)))
    for i, feature in enumerate(feature_types):
        row = comparison_df[comparison_df['Feature Type'] == feature]
        values = row[metrics].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', color=colors[i], label=feature)
        ax.fill(angles, values, color=colors[i], alpha=0.1)
    
    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Feature Type Performance Comparison', size=15, y=1.1)
    plt.tight_layout()
    
    return fig


def plot_per_emotion_feature_comparison(results, feature_types):
    """
    Generate a grouped bar chart showing how each feature type performs
    on individual emotions. This provides insight into which feature type
    is most effective for specific emotional expressions.
    
    Args:
        results: Dictionary containing results for each feature type
        feature_types: List of feature types to compare
        
    Returns:
        Matplotlib figure with the grouped bar chart
    """
    # Extract per-emotion metrics for each feature type
    emotion_data = []
    emotion_classes = None
    
    for feature_type in feature_types:
        if feature_type not in results:
            continue
            
        cv_results = results[feature_type]['cv_results']
        if 'per_emotion_metrics' not in cv_results:
            continue
            
        metrics = cv_results['per_emotion_metrics']
        if emotion_classes is None and 'emotion_classes' in cv_results:
            emotion_classes = cv_results['emotion_classes']
            
        for emotion, emotion_metrics in metrics.items():
            emotion_data.append({
                'Feature Type': feature_type,
                'Emotion': emotion,
                'F1 Score': emotion_metrics.get('f1_score', 0),
                'Precision': emotion_metrics.get('precision', 0),
                'Recall': emotion_metrics.get('recall', 0)
            })
    
    if not emotion_data:
        raise ValueError("No per-emotion metrics available in results")
        
    # Convert to DataFrame
    emotion_df = pd.DataFrame(emotion_data)
    
    # Create grouped bar chart
    plt.figure(figsize=(14, 8))
    
    # Create grouped bar chart for F1 scores
    emotion_pivot = emotion_df.pivot(index='Emotion', columns='Feature Type', values='F1 Score')
    
    ax = emotion_pivot.plot(kind='bar', rot=0, width=0.8)
    
    # Customize
    plt.title('Performance by Emotion Across Feature Types', fontsize=16)
    plt.xlabel('Emotion', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Feature Type')
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=8)
    
    plt.tight_layout()
    return plt.gcf()


def generate_feature_comparison_table(comparison_df):
    """
    Generate a LaTeX table for feature comparison.
    """
    styled_df = comparison_df.style.format({
        'Accuracy': '{:.4f}',
        'Precision': '{:.4f}',
        'Recall': '{:.4f}',
        'F1 Score': '{:.4f}'
    })
    
    latex_table = styled_df.to_latex(
        caption="Performance Comparison of Different Acoustic Feature Types",
        label="tab:feature_comparison",
        position="htbp",
        position_float="centering"
    )
    
    return latex_table


def main():
    """Command line interface for feature comparison"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Multiple Feature Types for HMM Speech Emotion Recognition')
    
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing feature data for multiple feature types')
    parser.add_argument('--feature_types', type=str, nargs='+',
                        default=['mfcc', 'spectral', 'prosodic'],
                        help='List of feature types to compare')
    parser.add_argument('--n_states', type=int, default=5,
                        help='Number of hidden states for HMMs')
    parser.add_argument('--n_components', type=int, default=1,
                        help='Number of Gaussian mixture components')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--n_iter', type=int, default=100,
                        help='Maximum iterations for Baum-Welch algorithm')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save comparison results')
    
    args = parser.parse_args()
    
    compare_feature_types(
        data_dir=args.data_dir,
        feature_types=args.feature_types,
        n_states=args.n_states,
        n_components=args.n_components,
        n_folds=args.n_folds,
        n_iter=args.n_iter,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
