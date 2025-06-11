#!/usr/bin/env python
import os
import numpy as np
import debug_dataimport as di

# Test all feature types
print('=== TESTING FEATURE EXTRACTION ===')
data_path = '/Users/anyadecarlo/datasci224-final/ravdess-emotional-speech-audio'

feature_types = ['mfcc', 'spectral', 'prosodic', 'chroma', 'combined']

for ft in feature_types:
    print(f"\nExtracting {ft.upper()} features...")
    obs, labels, classes = di.load_ravdess_features(data_path, ft)
    print(f"  Sample count: {len(obs)}")
    print(f"  Feature dimensionality: {obs[0].shape[1]}")
    print(f"  Example sequence shape: {obs[0].shape}")
    
print('\nAll feature types extracted successfully')
