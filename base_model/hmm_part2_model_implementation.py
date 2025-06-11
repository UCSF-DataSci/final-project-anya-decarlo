import numpy as np

def forward_algorithm(observations, A, B, pi):
    """
    Implement the forward algorithm (Problem 1 in Rabiner tutorial).
    Calculates P(O|λ), the probability of observation sequence given model parameters.
    
    Args:
        observations: Sequence of observations (T x D)
        A: State transition probability matrix (N x N)
        B: Emission probability function that takes (state_idx, observation) and returns probability
        pi: Initial state probability distribution (N)
        
    Returns:
        alpha: Forward probability matrix (T x N)
        log_likelihood: Log probability of observations given the model
        scale_factors: Scaling factors used for numerical stability
    """
    T = len(observations)  # Number of observations
    N = len(pi)            # Number of states
    
    # Initialize alpha matrix
    alpha = np.zeros((T, N))
    
    # Initialization step (t=0)
    for i in range(N):
        # For continuous observations, B is a function
        alpha[0, i] = pi[i] * B(i, observations[0])
    
    # Scaling to prevent numerical underflow (Rabiner convention: c_t = 1 / sum_alpha)
    scale_factors = np.zeros(T)
    sum_alpha = np.sum(alpha[0, :])
    if sum_alpha <= 0:
        sum_alpha = 1e-300  # numerical floor
    scale_factors[0] = 1.0 / sum_alpha  # c_0
    alpha[0, :] *= scale_factors[0]
    
    # Forward recursion
    for t in range(1, T):
        for j in range(N):
            # Calculate alpha[t,j] using the previous alpha values
            alpha[t, j] = 0
            for i in range(N):
                alpha[t, j] += alpha[t-1, i] * A[i, j]
            # Multiply by emission probability
            alpha[t, j] *= B(j, observations[t])
        
        # Scale alpha values for numerical stability (c_t = 1 / sum_alpha)
        sum_alpha = np.sum(alpha[t, :])
        if sum_alpha <= 0:
            sum_alpha = 1e-300
        scale_factors[t] = 1.0 / sum_alpha
        alpha[t, :] *= scale_factors[t]
    
    # Calculate log-likelihood using scaling factors (negative value)
    log_likelihood = -np.sum(np.log(scale_factors))
    
    return alpha, log_likelihood, scale_factors

def backward_algorithm(observations, A, B, scale_factors):
    """
    Implement the backward algorithm, needed for Baum-Welch parameter estimation.
    Calculates the backward probabilities used in Problem 3 (Eq. 24 in Rabiner).
    
    Args:
        observations: Sequence of observations (T x D)
        A: State transition probability matrix (N x N)
        B: Emission probability function that takes (state_idx, observation) and returns probability
        scale_factors: Scaling factors from the forward algorithm for numerical stability
        
    Returns:
        beta: Backward probability matrix (T x N)
    """
    T = len(observations)  # Number of observations
    N = A.shape[0]         # Number of states
    
    # Initialize beta matrix
    beta = np.zeros((T, N))
    
    # Initialization for the last time step (β_T(i) = c_{T-1})
    beta[T-1, :] = scale_factors[T-1]
    
    # Backward recursion (Eq. 24)
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t, i] = 0
            for j in range(N):
                beta[t, i] += A[i, j] * B(j, observations[t+1]) * beta[t+1, j]
        
        # Scale beta values using the corresponding scale factor (multiply)
        beta[t, :] *= scale_factors[t]
    
    return beta

def compute_xi_gamma(alpha, beta, A, B, observations):
    """
    Compute xi (joint probability of being in state i at time t and state j at time t+1) 
    and gamma (probability of being in state i at time t) for Baum-Welch algorithm.
    
    Args:
        alpha: Forward probability matrix (T x N)
        beta: Backward probability matrix (T x N)
        A: State transition probability matrix (N x N)
        B: Emission probability function
        observations: Sequence of observations (T x D)
        
    Returns:
        xi: Joint probability matrix for consecutive states (T-1 x N x N)
        gamma: State probability matrix (T x N)
    """
    T = len(observations)
    N = alpha.shape[1]
    
    # Initialize gamma and xi
    gamma = np.zeros((T, N))
    xi = np.zeros((T-1, N, N))
    
    # Compute gamma values for each time step
    for t in range(T):
        # gamma[t][i] = P(q_t = i | O, λ)
        for i in range(N):
            gamma[t, i] = alpha[t, i] * beta[t, i]
        
        # Normalize gamma to ensure it sums to 1
        gamma_sum = np.sum(gamma[t, :])
        if gamma_sum > 0:
            gamma[t, :] /= gamma_sum
    
    # Compute xi values for each time step (except the last one)
    for t in range(T-1):
        # For each possible state transition (i,j)
        for i in range(N):
            for j in range(N):
                # xi[t][i][j] = P(q_t = i, q_t+1 = j | O, λ)
                xi[t, i, j] = alpha[t, i] * A[i, j] * B(j, observations[t+1]) * beta[t+1, j]
        
        # Normalize xi to ensure it sums to 1
        xi_sum = np.sum(xi[t, :, :])
        if xi_sum > 0:
            xi[t, :, :] /= xi_sum
    
    return xi, gamma

def gaussian_emission_probability(mu, sigma, observation):
    """
    Calculate Gaussian emission probability for continuous observations.
    
    Args:
        mu: Mean vector for the Gaussian distribution (D)
        sigma: Covariance matrix (D x D)
        observation: Single observation vector (D)
        
    Returns:
        Probability density of the observation under the Gaussian
    """
    from scipy.stats import multivariate_normal
    # Check for numerical issues and regularize if needed
    try:
        prob = multivariate_normal.pdf(observation, mean=mu, cov=sigma)
        # Ensure non-zero probability to avoid underflow
        prob = max(prob, 1e-300)
        return prob
    except np.linalg.LinAlgError:
        # Add regularization if there's a numerical error
        regularized_sigma = sigma + np.eye(len(mu)) * 1e-4
        prob = multivariate_normal.pdf(observation, mean=mu, cov=regularized_sigma)
        # Ensure non-zero probability to avoid underflow
        prob = max(prob, 1e-300)
        return prob

def gmm_emission_probability(weights, means, covs, observation):
    """
    Calculate emission probability using a Gaussian Mixture Model.
    
    Args:
        weights: Mixture weights for each Gaussian component (M)
        means: List of mean vectors for each Gaussian component (M x D)
        covs: List of covariance matrices for each Gaussian component (M x D x D)
        observation: Single observation vector (D)
        
    Returns:
        Probability density of the observation under the GMM
    """
    M = len(weights)  # Number of mixture components
    prob = 0.0
    
    # Sum over all mixture components
    for m in range(M):
        # Weight times Gaussian probability
        prob += weights[m] * gaussian_emission_probability(means[m], covs[m], observation)
    
    # Ensure non-zero probability to avoid underflow
    prob = max(prob, 1e-300)
    return prob

def baum_welch_algorithm(observations, N, max_iter=100, tol=1e-6, n_components=1):
    """
    Implement the Baum-Welch algorithm (Problem 3 in Rabiner tutorial).
    Adjusts model parameters A, B, pi to maximize P(O|λ).
    
    Args:
        observations: Sequence of observations (T x D)
        N: Number of hidden states
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        n_components: Number of Gaussian mixture components for emission probabilities
        
    Returns:
        A: Optimized state transition probability matrix
        B_params: Parameters for emission probability function (means, covariances, weights)
        pi: Optimized initial state probability
        log_likelihood: Final log likelihood of the model
    """
    T, D = observations.shape  # Number of observations, Dimension of each observation
    M = n_components  # Number of mixture components
    
    # Initialize model parameters randomly
    A = np.random.rand(N, N)  # Transition probabilities
    A = A / np.sum(A, axis=1)[:, np.newaxis]  # Normalize rows to sum to 1
    
    pi = np.random.rand(N)  # Initial state probabilities
    pi = pi / np.sum(pi)  # Normalize to sum to 1
    
    # Initialize means, covariances, and weights for Gaussian mixture emissions
    # If n_components=1, this is equivalent to a simple Gaussian model
    means = np.zeros((N, M, D))
    covs = np.zeros((N, M, D, D))
    weights = np.ones((N, M)) / M  # Equal weights initially
    
    # Get min and max values for each dimension to help with initialization
    min_obs = np.min(observations, axis=0)
    max_obs = np.max(observations, axis=0)
    range_obs = max_obs - min_obs
    
    # Initialize means randomly within the range of observations
    for i in range(N):
        for m in range(M):
            # Spread means across the observation space
            means[i, m] = np.mean(observations, axis=0) + \
                         np.random.uniform(-0.5, 0.5, D) * range_obs * 0.5
            # Initialize with a reasonable covariance
            covs[i, m] = np.eye(D) * np.var(observations, axis=0).mean() * 0.5
    
    # Create emission probability wrapper function for Gaussian mixtures
    def create_emission_function(state_means, state_covs, state_weights):
        def emission_probability(j, obs):
            if n_components == 1:
                # Single Gaussian case
                return gaussian_emission_probability(state_means[j, 0], state_covs[j, 0], obs)
            else:
                # Mixture of Gaussians case
                return gmm_emission_probability(state_weights[j], state_means[j], state_covs[j], obs)
        return emission_probability
    
    # Initialize the emission probability function
    B = create_emission_function(means, covs, weights)
    
    # Iterative parameter estimation
    log_likelihood_history = []
    prev_log_likelihood = -np.inf
    
    for iteration in range(max_iter):
        # E-step: Calculate forward and backward variables
        alpha, log_likelihood, scale_factors = forward_algorithm(observations, A, B, pi)
        beta = backward_algorithm(observations, A, B, scale_factors)
        
        # Record log likelihood
        log_likelihood_history.append(log_likelihood)
        
        # Check for convergence
        if iteration > 0 and abs(log_likelihood - prev_log_likelihood) < tol:
            break
        
        prev_log_likelihood = log_likelihood
        
        # Calculate xi and gamma
        xi, gamma = compute_xi_gamma(alpha, beta, A, B, observations)
        
        # M-step: Re-estimate model parameters
        # Re-estimate pi (initial state probabilities)
        pi_new = gamma[0, :]
        
        # Re-estimate A (transition probabilities)
        A_new = np.zeros_like(A)
        for i in range(N):
            for j in range(N):
                numerator = np.sum(xi[:, i, j])
                denominator = np.sum(gamma[:-1, i])
                if denominator > 0:
                    A_new[i, j] = numerator / denominator
            
            # Ensure rows sum to 1 (handle numerical issues)
            if np.sum(A_new[i, :]) > 0:
                A_new[i, :] /= np.sum(A_new[i, :])
        
        # Re-estimate emission parameters for GMM (means, covariances, weights)
        means_new = np.zeros_like(means)
        covs_new = np.zeros_like(covs)
        weights_new = np.zeros_like(weights)
        
        # For GMM, we need to compute posteriors for each mixture component
        if n_components > 1:
            # Initialize mixture component posteriors
            # gamma_mix[t, j, m] = P(q_t = j, m_t = m | O, λ)
            gamma_mix = np.zeros((T, N, M))
            
            # Calculate mixture component posteriors for each time step
            for t in range(T):
                for j in range(N):
                    for m in range(M):
                        # Component likelihood * mixture weight
                        gamma_mix[t, j, m] = gamma[t, j] * weights[j, m] * \
                            gaussian_emission_probability(means[j, m], covs[j, m], observations[t])
                    
                    # Normalize by sum of all mixture components
                    mix_sum = np.sum(gamma_mix[t, j, :])
                    if mix_sum > 0:
                        gamma_mix[t, j, :] /= mix_sum
            
            # Re-estimate GMM parameters
            for j in range(N):
                gamma_sum = np.sum(gamma[:, j])
                
                if gamma_sum > 0:
                    # Update mixture weights
                    for m in range(M):
                        weights_new[j, m] = np.sum(gamma_mix[:, j, m]) / gamma_sum
                    
                    # Update means and covariances for each mixture component
                    for m in range(M):
                        # Calculate sum of posteriors for this component
                        gamma_mix_sum = np.sum(gamma_mix[:, j, m])
                        
                        if gamma_mix_sum > 0:
                            # Update mean - weighted sum of observations
                            means_new[j, m] = np.sum(gamma_mix[:, j, m].reshape(-1, 1) * observations, axis=0) / gamma_mix_sum
                            
                            # Update covariance
                            obs_diff = observations - means_new[j, m]
                            covs_new[j, m] = np.zeros((D, D))
                            for t in range(T):
                                covs_new[j, m] += gamma_mix[t, j, m] * np.outer(obs_diff[t], obs_diff[t])
                            covs_new[j, m] /= gamma_mix_sum
                            
                            # Calculate adaptive regularization based on data scale
                            var_scale = np.mean(np.diag(covs_new[j, m])) if np.any(np.diag(covs_new[j, m]) > 0) else 1.0
                            reg_value = max(1e-6, var_scale * 0.01)  # At least 1e-6, or 1% of variance
                            
                            # Add regularization to ensure positive definiteness
                            covs_new[j, m] += np.eye(D) * reg_value
                            
                            # Check for extremely small values on diagonal and correct them
                            min_var = 1e-6
                            for d in range(D):
                                if covs_new[j, m, d, d] < min_var:
                                    covs_new[j, m, d, d] = min_var  # Ensure minimum variance
                        else:
                            # If no data assigned to this component, keep previous values
                            means_new[j, m] = means[j, m]
                            covs_new[j, m] = covs[j, m]
                    
                    # Normalize weights to sum to 1
                    weights_new[j] = weights_new[j] / np.sum(weights_new[j])
                else:
                    # If no data assigned to this state, keep previous values
                    weights_new[j] = weights[j]
                    means_new[j] = means[j]
                    covs_new[j] = covs[j]
        else:
            # Single Gaussian case (original implementation)
            for j in range(N):
                gamma_sum = np.sum(gamma[:, j])
                
                # Update means
                if gamma_sum > 0:
                    # Weighted sum of observations according to gamma
                    means_new[j, 0] = np.sum(gamma[:, j].reshape(-1, 1) * observations, axis=0) / gamma_sum
                    
                    # Update covariances
                    obs_diff = observations - means_new[j, 0]
                    for t in range(T):
                        covs_new[j, 0] += gamma[t, j] * np.outer(obs_diff[t], obs_diff[t])
                    covs_new[j, 0] /= gamma_sum
                    
                    # Calculate adaptive regularization based on data scale
                    var_scale = np.mean(np.diag(covs_new[j, 0])) if np.any(np.diag(covs_new[j, 0]) > 0) else 1.0
                    reg_value = max(1e-6, var_scale * 0.01)  # At least 1e-6, or 1% of variance
                    
                    # Add regularization to ensure positive definiteness
                    covs_new[j, 0] += np.eye(D) * reg_value
                    
                    # Check for extremely small values on diagonal and correct them
                    min_var = 1e-6
                    for d in range(D):
                        if covs_new[j, 0, d, d] < min_var:
                            covs_new[j, 0, d, d] = min_var  # Ensure minimum variance
                else:
                    # If no data assigned to this state, keep previous values
                    means_new[j] = means[j]
                    covs_new[j] = covs[j]
                
            # Single component, so weights are just 1
            weights_new = np.ones((N, M))
        
        # Update parameters
        A = A_new
        pi = pi_new
        means = means_new
        covs = covs_new
        weights = weights_new
        
        # Update emission probability function with new parameters
        B = create_emission_function(means, covs, weights)
    
    # Return optimized model parameters
    B_params = {
        'means': means, 
        'covariances': covs, 
        'weights': weights, 
        'n_components': n_components
    }
    return A, B_params, pi, log_likelihood

def viterbi_algorithm(observations, A, B, pi):
    """
    Implement the Viterbi algorithm (Problem 2 in Rabiner tutorial).
    Finds the most likely sequence of hidden states given observations.
    
    Args:
        observations: Sequence of observations (T x D)
        A: State transition probability matrix (N x N)
        B: Emission probability function that takes (state_idx, observation) and returns probability
        pi: Initial state probability distribution (N)
        
    Returns:
        best_path: Most likely sequence of hidden states
        log_likelihood: Log probability of the best path
    """
    T = len(observations)  # Number of observations
    N = len(pi)            # Number of states
    
    # Initialize delta and psi matrices
    # delta holds the best score along a path ending in state i at time t
    # psi holds the state that led to the best path for state i at time t
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    
    # Initialization step (t=0) - Eq. 32a in Rabiner
    for i in range(N):
        delta[0, i] = pi[i] * B(i, observations[0])
        psi[0, i] = 0  # No previous state at t=0
    
    # Use separate scaling for Viterbi to preserve path selection
    viterbi_scale_factors = np.zeros(T)
    viterbi_scale_factors[0] = np.max(delta[0, :])  # Use max instead of sum for Viterbi
    if viterbi_scale_factors[0] > 0:
        delta[0, :] /= viterbi_scale_factors[0]
    else:
        viterbi_scale_factors[0] = 1.0
    
    # Recursion step - finding the best path - Eq. 32b in Rabiner
    for t in range(1, T):
        for j in range(N):
            # Find the previous state that maximizes delta[t,j]
            delta_temp = delta[t-1, :] * A[:, j]  # Vectorized computation
            
            # Find the max and argmax
            psi[t, j] = np.argmax(delta_temp)
            delta[t, j] = delta_temp[psi[t, j]] * B(j, observations[t])
        
        # Scale using max for Viterbi (different from forward algorithm)
        viterbi_scale_factors[t] = np.max(delta[t, :])
        if viterbi_scale_factors[t] > 0:
            delta[t, :] /= viterbi_scale_factors[t]
        else:
            viterbi_scale_factors[t] = 1.0
    
    # Termination: find the state with highest probability at final step - Eq. 32c
    best_path = np.zeros(T, dtype=int)
    best_path[T-1] = np.argmax(delta[T-1, :])  # q*_T = argmax_i [δ_T(i)]
    
    # Backtracking to find the best state sequence - Eq. 32d
    for t in range(T-2, -1, -1):
        best_path[t] = psi[t+1, best_path[t+1]]  # q*_t = ψ_{t+1}(q*_{t+1})
    
    # Calculate log-likelihood of the best path
    # Use negative sign like in forward algorithm for proper log-likelihood
    log_likelihood = -np.sum(np.log(viterbi_scale_factors))
    
    return best_path, log_likelihood
