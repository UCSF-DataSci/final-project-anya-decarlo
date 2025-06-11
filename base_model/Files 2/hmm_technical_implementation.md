# Technical Implementation of Hidden Markov Models for Speech Emotion Recognition

This document provides a rigorous mathematical exposition of Hidden Markov Models (HMMs) as applied specifically to speech emotion recognition using the RAVDESS dataset. The implementation follows the theoretical framework established by Rabiner (1989) while incorporating considerations specific to acoustic feature modeling for emotional speech.

## 1. HMM Theoretical Foundation

### 1.1 Formal Definition

A Hidden Markov Model is formally characterized by the tuple λ = (A, B, π):

- **A** = {a<sub>ij</sub>} represents the state transition probability matrix, where:
  
  a<sub>ij</sub> = P(q<sub>t+1</sub> = S<sub>j</sub> | q<sub>t</sub> = S<sub>i</sub>), 1 ≤ i, j ≤ N
  
  subject to Σ<sub>j=1</sub><sup>N</sup> a<sub>ij</sub> = 1, ∀i

- **B** = {b<sub>j</sub>(k)} represents the emission probability distribution, where:

  b<sub>j</sub>(k) = P(o<sub>t</sub> = v<sub>k</sub> | q<sub>t</sub> = S<sub>j</sub>), 1 ≤ j ≤ N, 1 ≤ k ≤ M
  
  or for continuous observations:
  
  b<sub>j</sub>(o<sub>t</sub>) = N(o<sub>t</sub>; μ<sub>j</sub>, Σ<sub>j</sub>) = 
  (2π)<sup>-D/2</sup>|Σ<sub>j</sub>|<sup>-1/2</sup>exp{-1/2(o<sub>t</sub>-μ<sub>j</sub>)'Σ<sub>j</sub><sup>-1</sup>(o<sub>t</sub>-μ<sub>j</sub>)}
  
- **π** = {π<sub>i</sub>} represents the initial state distribution, where:

  π<sub>i</sub> = P(q<sub>1</sub> = S<sub>i</sub>), 1 ≤ i ≤ N

### 1.2 The Three Fundamental Problems

For speech emotion recognition, we must address the three fundamental problems of HMMs:

1. **Evaluation Problem**: Computing P(O|λ) - the probability of an observation sequence given the model
2. **Decoding Problem**: Finding the optimal state sequence that best explains the observations
3. **Learning Problem**: Adjusting model parameters λ = (A, B, π) to maximize P(O|λ)

## 2. Mathematical Implementation for Speech Emotion Recognition

### 2.1 Evaluation Problem: Forward Algorithm

For an emotional speech recording with T frames of acoustic features, we calculate the probability of the observation sequence O = {o<sub>1</sub>, o<sub>2</sub>, ..., o<sub>T</sub>} given a specific emotion model λ.

The forward variable α<sub>t</sub>(i) is defined as:

α<sub>t</sub>(i) = P(o<sub>1</sub>, o<sub>2</sub>, ..., o<sub>t</sub>, q<sub>t</sub> = S<sub>i</sub> | λ)

The forward algorithm computes this efficiently:

1. **Initialization**:
   
   α<sub>1</sub>(i) = π<sub>i</sub>b<sub>i</sub>(o<sub>1</sub>), 1 ≤ i ≤ N

2. **Recursion**:
   
   α<sub>t+1</sub>(j) = [Σ<sub>i=1</sub><sup>N</sup> α<sub>t</sub>(i)a<sub>ij</sub>]b<sub>j</sub>(o<sub>t+1</sub>), 1 ≤ t ≤ T-1, 1 ≤ j ≤ N

3. **Termination**:
   
   P(O|λ) = Σ<sub>i=1</sub><sup>N</sup> α<sub>T</sub>(i)

The computational complexity is O(N²T) compared to O(TN<sup>T</sup>) for direct calculation, making it feasible for real-time emotion recognition.

### 2.2 Decoding Problem: Viterbi Algorithm

When analyzing emotional speech, we often need to determine the most likely sequence of hidden states. The Viterbi algorithm finds the single best state sequence Q = {q<sub>1</sub>, q<sub>2</sub>, ..., q<sub>T</sub>} by defining:

δ<sub>t</sub>(i) = max<sub>q<sub>1</sub>,q<sub>2</sub>,...,q<sub>t-1</sub></sub> P(q<sub>1</sub>, q<sub>2</sub>, ..., q<sub>t-1</sub>, q<sub>t</sub> = S<sub>i</sub>, o<sub>1</sub>, o<sub>2</sub>, ..., o<sub>t</sub> | λ)

The algorithm proceeds as:

1. **Initialization**:
   
   δ<sub>1</sub>(i) = π<sub>i</sub>b<sub>i</sub>(o<sub>1</sub>), 1 ≤ i ≤ N
   
   ψ<sub>1</sub>(i) = 0

2. **Recursion**:
   
   δ<sub>t</sub>(j) = max<sub>1≤i≤N</sub> [δ<sub>t-1</sub>(i)a<sub>ij</sub>]b<sub>j</sub>(o<sub>t</sub>), 2 ≤ t ≤ T, 1 ≤ j ≤ N
   
   ψ<sub>t</sub>(j) = argmax<sub>1≤i≤N</sub> [δ<sub>t-1</sub>(i)a<sub>ij</sub>], 2 ≤ t ≤ T, 1 ≤ j ≤ N

3. **Termination**:
   
   P* = max<sub>1≤i≤N</sub> [δ<sub>T</sub>(i)]
   
   q<sub>T</sub>* = argmax<sub>1≤i≤N</sub> [δ<sub>T</sub>(i)]

4. **Path Backtracking**:
   
   q<sub>t</sub>* = ψ<sub>t+1</sub>(q<sub>t+1</sub>*), t = T-1, T-2, ..., 1

### 2.3 Learning Problem: Baum-Welch Algorithm

For each emotional category in the RAVDESS dataset, we train a separate HMM using the Baum-Welch algorithm, which maximizes P(O|λ) through iterative expectation-maximization:

1. **Define the auxiliary variables**:

   ξ<sub>t</sub>(i,j) = P(q<sub>t</sub> = S<sub>i</sub>, q<sub>t+1</sub> = S<sub>j</sub> | O, λ) = 
   (α<sub>t</sub>(i)a<sub>ij</sub>b<sub>j</sub>(o<sub>t+1</sub>)β<sub>t+1</sub>(j)) / P(O|λ)
   
   γ<sub>t</sub>(i) = P(q<sub>t</sub> = S<sub>i</sub> | O, λ) = Σ<sub>j=1</sub><sup>N</sup> ξ<sub>t</sub>(i,j)

2. **Update HMM parameters**:

   π̄<sub>i</sub> = expected frequency in state S<sub>i</sub> at t=1 = γ<sub>1</sub>(i)
   
   ā<sub>ij</sub> = (Σ<sub>t=1</sub><sup>T-1</sup> ξ<sub>t</sub>(i,j)) / (Σ<sub>t=1</sub><sup>T-1</sup> γ<sub>t</sub>(i))
   
   For continuous observations with Gaussian distributions:
   
   μ̄<sub>j</sub> = (Σ<sub>t=1</sub><sup>T</sup> γ<sub>t</sub>(j)o<sub>t</sub>) / (Σ<sub>t=1</sub><sup>T</sup> γ<sub>t</sub>(j))
   
   Σ̄<sub>j</sub> = (Σ<sub>t=1</sub><sup>T</sup> γ<sub>t</sub>(j)(o<sub>t</sub>-μ̄<sub>j</sub>)(o<sub>t</sub>-μ̄<sub>j</sub>)') / (Σ<sub>t=1</sub><sup>T</sup> γ<sub>t</sub>(j))

3. **Iterate** until convergence (when P(O|λ̄) ≥ P(O|λ))

## 3. Application to RAVDESS Dataset

### 3.1 Dataset Characteristics

The RAVDESS dataset contains 7356 recordings from 24 professional actors (12 male, 12 female) producing emotional expressions in speech and song, with:

- **Speech**: 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprise)
- **Song**: 6 emotions (neutral, calm, happy, sad, angry, fearful)
- **Intensity levels**: normal and strong (except neutral)
- **Modalities**: audio-video, video-only, audio-only

### 3.2 Feature Extraction for HMM

For each RAVDESS audio file (based on filename convention: Modality-Channel-Emotion-Intensity-Statement-Repetition-Actor):

1. **Extract frames** using 25ms windows with 10ms overlap
2. **Compute MFCC features** (13 coefficients + Δ + ΔΔ = 39 dimensions)
3. **Extract prosodic features**:
   - Fundamental frequency (F0)
   - Energy
   - Zero-crossing rate
   - Harmonics-to-noise ratio

The resulting observation sequence O = {o<sub>1</sub>, o<sub>2</sub>, ..., o<sub>T</sub>} represents the acoustic feature vectors for each time frame.

### 3.3 Emotion-Specific HMM Topology

For each emotion category e ∈ {neutral, calm, happy, sad, angry, fearful, disgust, surprise}:

1. **Model Structure**:
   - Left-to-right HMM topology (no backward transitions)
   - N = 5 states (optimal for short utterances)
   - Full covariance Gaussian mixture models (GMMs) for emission probabilities

2. **Training Strategy**:
   - Initialize A with left-to-right structure, with a<sub>ii</sub> = 0.7, a<sub>i,i+1</sub> = 0.3
   - Initialize π = [1, 0, 0, 0, 0]
   - Initialize B with k-means clustering of features
   - Train using Baum-Welch algorithm with convergence threshold of 10<sup>-4</sup>

3. **K-fold Cross-Validation**:
   - K=5 folds stratified by actor to ensure speaker independence
   - Ensures robust parameter estimation and prevents overfitting

### 3.4 Classification Strategy

For an unknown emotional utterance with observation sequence O:

1. Calculate P(O|λ<sub>e</sub>) for each emotion model e
2. Apply Bayesian decision rule with uniform priors:
   
   ê = argmax<sub>e</sub> P(O|λ<sub>e</sub>)

The likelihood P(O|λ<sub>e</sub>) is calculated using the Forward algorithm.

## 4. Evaluation Metrics

The performance of the HMM-based emotion classifier is evaluated using:

1. **Accuracy**: proportion of correctly classified emotions
2. **Unbiased Hit Rate (H<sub>u</sub>)**: accounts for response bias in emotion recognition
   
   H<sub>u</sub> = (Hits × Hits) / (Hits + False Alarms) × Number of categories
   
3. **Confusion Matrix Analysis**: examination of error patterns between emotions
4. **F1 Score**: harmonic mean of precision and recall for multi-class evaluation

These evaluation methods align with the validation methodology used in the original RAVDESS study, which achieved 80% mean accuracy across emotions with human raters.

## 5. Technical Optimizations

1. **Parameter Sensitivity Analysis**:
   - Number of states N ∈ {3, 5, 7, 9}
   - Covariance types ∈ {diagonal, full}
   - Feature dimensionality through PCA

2. **Baum-Welch Computation Efficiency**:
   - Scaling to prevent numerical underflow
   - Log-domain computation for improved numerical stability
   - Early stopping with validation set monitoring

3. **Acoustic Feature Selection**:
   - Information Gain Ratio for feature importance
   - Sequential Forward Selection to identify emotion-discriminative features
