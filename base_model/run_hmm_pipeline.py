#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HMM Speech Emotion Recognition Pipeline Runner

This script orchestrates the complete HMM-based speech emotion recognition pipeline:
1. Loads and preprocesses RAVDESS dataset features
2. Trains emotion-specific HMMs with cross-validation
3. Evaluates performance with comprehensive metrics
4. Performs parameter sensitivity analysis (optional)
5. Generates publication-quality results and visualizations
6. Saves trained models for later use

Based on Rabiner's tutorial and comprehensive HMM implementation guide.
"""

import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Import our custom modules
import debug_dataimport as data_loader
import hmm_part2_model_implementation as hmm_model
import hmm_part3_evaluation as evaluator

# Configure matplotlib for high-quality figures
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'

def run_full_pipeline(data_dir=None, feature_type='mfcc', n_states=5, 
                      n_components=1, n_folds=5, n_iter=100, 
                      run_sensitivity=False, output_dir='results',
                      feature_dir_prefix=True, *, group_by_actor=False):
    """
    Run the complete HMM speech emotion recognition pipeline.
    
    Args:
        data_dir: Directory containing RAVDESS dataset or features
        feature_type: Type of acoustic features ('mfcc', 'spectral', etc.)
        n_states: Number of hidden states for HMMs
        n_components: Number of Gaussian mixture components
        n_folds: Number of cross-validation folds
        n_iter: Maximum iterations for Baum-Welch algorithm
        run_sensitivity: Whether to run parameter sensitivity analysis
        output_dir: Directory to save results
        group_by_actor: Whether to group data by actor for speaker-independent CV
    
    Returns:
        Dictionary containing all results and trained models
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create organised output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir) / "hmm" / f"{ts}_{feature_type}_n{n_states}"
    metrics_dir = out_dir / "metrics"
    figs_dir = out_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir()
    
    # Configuration summary
    config = {
        'feature_type': feature_type,
        'n_states': n_states,
        'n_components': n_components,
        'n_folds': n_folds,
        'n_iter': n_iter,
        'timestamp': timestamp,
        'group_by_actor': group_by_actor
    }
    
    print("\n" + "="*70)
    print("HMM SPEECH EMOTION RECOGNITION PIPELINE")
    print("="*70)
    print(f"Starting pipeline with configuration:")
    print(f"  Feature type: {feature_type}")
    print(f"  HMM states: {n_states}")
    print(f"  GMM components: {n_components}")
    print(f"  CV folds: {n_folds}")
    print(f"  Max iterations: {n_iter}")
    print(f"  Group by actor: {group_by_actor}")
    print("="*70 + "\n")
    
    # Step 1: Load and preprocess data
    print("\n--- STEP 1: LOADING DATA ---")
    try:
        observation_sequences, emotion_labels, emotion_classes = data_loader.load_ravdess_features(
            data_dir, feature_type=feature_type
        )
        print(f"Successfully loaded {len(observation_sequences)} sequences")
        print(f"Emotion classes: {emotion_classes}")
        print(f"Feature dimensionality: {observation_sequences[0].shape[1]}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None
    
    # Step 2: Cross-validation and model training
    print("\n--- STEP 2: CROSS-VALIDATION AND MODEL TRAINING ---")
    try:
        file_names = None
        if group_by_actor:
            file_names = sorted(str(p) for p in Path(data_dir).rglob("*.wav"))
        cv_results = evaluator.cross_validate_hmm(
            observation_sequences, 
            emotion_labels,
            emotion_classes,
            n_states=n_states,
            n_components=n_components,
            n_folds=n_folds,
            n_iter=n_iter,
            group_by_actor=group_by_actor,
            file_names=file_names
        )
        
        # Print summary of cross-validation results
        print("\nCross-validation complete.")
        print(f"Overall accuracy: {cv_results['overall_metrics']['accuracy']:.4f}")
        print(f"Overall F1 score: {cv_results['overall_metrics']['f1_score']:.4f}")
    except Exception as e:
        print(f"Error in cross-validation: {str(e)}")
        return None
    
    # Step 3: Parameter sensitivity analysis (optional)
    sensitivity_results = None
    if run_sensitivity:
        print("\n--- STEP 3: PARAMETER SENSITIVITY ANALYSIS ---")
        try:
            # Use smaller parameter ranges and fewer folds for reasonable runtime
            n_states_range = [3, 5, 7] 
            n_components_range = [1, 2]
            
            sensitivity_results = evaluator.parameter_sensitivity_analysis(
                observation_sequences,
                emotion_labels,
                emotion_classes,
                n_states_range=n_states_range,
                n_components_range=n_components_range,
                n_folds=3,  # Fewer folds for efficiency
                n_iter=50   # Fewer iterations for efficiency
            )
            
            print("\nParameter sensitivity analysis complete.")
            print("Best performing configuration:")
            best_idx = sensitivity_results['summary_df']['Accuracy'].idxmax()
            best_config = sensitivity_results['summary_df'].iloc[best_idx]
            print(f"  States: {best_config['N States']}, Components: {best_config['N Components']}")
            print(f"  Accuracy: {best_config['Accuracy']:.4f}, F1: {best_config['F1 Score']:.4f}")
        except Exception as e:
            print(f"Error in sensitivity analysis: {str(e)}")
            sensitivity_results = None
    else:
        print("\n--- STEP 3: PARAMETER SENSITIVITY ANALYSIS (SKIPPED) ---")
    
    # Step 4: Generate final model with best parameters
    print("\n--- STEP 4: TRAINING FINAL MODELS ---")
    try:
        # If sensitivity analysis was run, use best parameters
        if sensitivity_results is not None:
            best_idx = sensitivity_results['summary_df']['Accuracy'].idxmax()
            best_config = sensitivity_results['summary_df'].iloc[best_idx]
            final_n_states = int(best_config['N States'])
            final_n_components = int(best_config['N Components'])
            print(f"Using best parameters from sensitivity analysis:")
            print(f"  States: {final_n_states}, Components: {final_n_components}")
        else:
            final_n_states = n_states
            final_n_components = n_components
        
        # Organize all data by emotion for final model training
        training_data = {}
        for emotion in emotion_classes:
            emotion_indices = np.where(np.array(emotion_labels) == emotion)[0]
            training_data[emotion] = [observation_sequences[i] for i in emotion_indices]
        
        # Train final models with all data
        final_models = evaluator.train_emotion_specific_hmms(
            training_data, 
            emotion_classes,
            n_states=final_n_states,
            n_components=final_n_components,
            n_iter=n_iter
        )
        
        print("\nFinal models trained successfully.")
    except Exception as e:
        print(f"Error training final models: {str(e)}")
        final_models = None
    
    # Step 5: Generate academic results
    print("\n--- STEP 5: GENERATING ACADEMIC RESULTS ---")
    try:
        academic_results = evaluator.generate_academic_results_section(
            cv_results, 
            emotion_classes, 
            sensitivity_results,
            output_dir=str(figs_dir),
            save_figures=True
        )
        print("\nAcademic results generated successfully.")
    except Exception as e:
        print(f"Error generating academic results: {str(e)}")
        academic_results = None
    
    # Step 6: Save all results and models
    print("\n--- STEP 6: SAVING RESULTS AND MODELS ---")
    try:
        # Compile all results
        all_results = {
            'config': config,
            'cv_results': cv_results,
            'sensitivity_results': sensitivity_results,
            'final_models': final_models,
            'elapsed_time': time.time() - start_time
        }
        
        # Save to pickle file
        results_file = metrics_dir / f'hmm_results_{timestamp}.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(all_results, f)
            
        # Save models separately (they can be large)
        if final_models is not None:
            models_file = metrics_dir / f'hmm_models_{timestamp}.pkl'
            with open(models_file, 'wb') as f:
                pickle.dump(final_models, f)
        
        print(f"\nResults saved to {results_file}")
        if final_models is not None:
            print(f"Models saved to {models_file}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")
    
    # Final timing report
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("\n" + "="*70)
    print(f"Pipeline completed in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print("="*70 + "\n")
    
    return all_results


def main():
    """Command line interface for the HMM pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run HMM Speech Emotion Recognition Pipeline')
    
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing RAVDESS dataset or precomputed features')
    parser.add_argument('--feature_type', type=str, default='mfcc',
                        help='Type of acoustic features (mfcc, spectral, etc.)')
    parser.add_argument('--n_states', type=int, default=5,
                        help='Number of hidden states for HMMs')
    parser.add_argument('--n_components', type=int, default=1,
                        help='Number of Gaussian mixture components')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--n_iter', type=int, default=100,
                        help='Maximum iterations for Baum-Welch algorithm')
    parser.add_argument('--sensitivity', action='store_true',
                        help='Run parameter sensitivity analysis')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--group_by_actor', action='store_true',
                        help='Use GroupKFold to ensure speaker-independent CV')
    
    args = parser.parse_args()
    
    run_full_pipeline(
        data_dir=args.data_dir,
        feature_type=args.feature_type,
        n_states=args.n_states,
        n_components=args.n_components,
        n_folds=args.n_folds,
        n_iter=args.n_iter,
        run_sensitivity=args.sensitivity,
        output_dir=args.output_dir,
        group_by_actor=args.group_by_actor
    )


if __name__ == "__main__":
    main()
