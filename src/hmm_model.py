import numpy as np

class GaussianHMM:
    def __init__(self, n_states=3, n_features=None, tol=1e-4, max_iter=10):
        self.n_states = n_states
        self.n_features = n_features
        self.tol = tol
        self.max_iter = max_iter
        
        # Parameters
        self.start_prob = None
        self.trans_prob = None
        self.means = None
        self.covars = None
        
    def _init_params(self, X):
        """Initialize parameters if not already set."""
        if self.n_features is None:
            self.n_features = X.shape[1]
            
        # Random initialization
        self.start_prob = np.ones(self.n_states) / self.n_states
        
        # Transition matrix: slightly biased towards self-transition
        self.trans_prob = np.ones((self.n_states, self.n_states)) / (2 * self.n_states)
        np.fill_diagonal(self.trans_prob, 0.5 + 1/(2*self.n_states))
        # Normalize
        self.trans_prob /= self.trans_prob.sum(axis=1, keepdims=True)
        
        # Means: Randomly selected from data or random noise
        indices = np.random.choice(X.shape[0], self.n_states, replace=False)
        self.means = X[indices]

        # Diagonal covariances: Store only variance per dimension
        # Shape: (n_states, n_features) instead of (n_states, n_features, n_features)
        self.covars = np.ones((self.n_states, self.n_features))

    def _log_pdf(self, x, mean, var):
        """
        Compute log-probability of observation x given Gaussian with diagonal covariance.

        Args:
            x: Observations (T, n_features)
            mean: Mean vector (n_features,)
            var: Variance vector (n_features,) - diagonal elements only

        Returns:
            log_prob: Log-probability for each observation (T,)
        """
        n = self.n_features

        # Add small epsilon to variance for numerical stability
        var_reg = var + 1e-4

        diff = x - mean  # (T, n_features)

        # For diagonal covariance: -0.5 * sum((x-mu)^2 / sigma^2)
        mahalanobis = np.sum(diff**2 / var_reg, axis=1)

        # Log determinant: sum of log of diagonal elements
        logdet = np.sum(np.log(var_reg))

        log_prob = -0.5 * (n * np.log(2 * np.pi) + logdet + mahalanobis)

        return log_prob

    def _forward(self, X):
        """Compute forward probabilities alpha."""
        T = X.shape[0]
        log_alpha = np.zeros((T, self.n_states))
        
        # Precompute emission log probabilities
        log_emissions = np.zeros((T, self.n_states))
        for i in range(self.n_states):
            log_emissions[:, i] = self._log_pdf(X, self.means[i], self.covars[i])
            
        # Initialization
        log_alpha[0] = np.log(self.start_prob + 1e-10) + log_emissions[0]
        
        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                # Log-sum-exp trick
                # alpha[t, j] = sum_i(alpha[t-1, i] * A[i, j]) * B[j, x_t]
                # log_alpha[t, j] = logsumexp(log_alpha[t-1] + log_A[:, j]) + log_B[j, x_t]
                prev_log_alpha = log_alpha[t-1]
                log_trans = np.log(self.trans_prob[:, j] + 1e-10)
                
                # logsumexp
                m = np.max(prev_log_alpha + log_trans)
                log_sum = m + np.log(np.sum(np.exp(prev_log_alpha + log_trans - m)))
                
                log_alpha[t, j] = log_sum + log_emissions[t, j]
                
        return log_alpha

    def _backward(self, X):
        """Compute backward probabilities beta."""
        T = X.shape[0]
        log_beta = np.zeros((T, self.n_states))
        
        # Precompute emission log probabilities
        log_emissions = np.zeros((T, self.n_states))
        for i in range(self.n_states):
            log_emissions[:, i] = self._log_pdf(X, self.means[i], self.covars[i])
            
        # Initialization (at T-1, beta is 1, so log_beta is 0)
        log_beta[T-1] = 0.0
        
        # Recursion
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                # beta[t, i] = sum_j(A[i, j] * B[j, x_t+1] * beta[t+1, j])
                log_trans = np.log(self.trans_prob[i, :] + 1e-10)
                log_emit_next = log_emissions[t+1]
                log_beta_next = log_beta[t+1]
                
                term = log_trans + log_emit_next + log_beta_next
                
                m = np.max(term)
                log_beta[t, i] = m + np.log(np.sum(np.exp(term - m)))
                
        return log_beta

    def train(self, X_list):
        """Train the model using Baum-Welch algorithm."""
        # X_list can be a single array or list of arrays
        if isinstance(X_list, np.ndarray):
            X_list = [X_list]
            
        # Initialize if needed
        if self.means is None:
            # Use concatenated data for initialization
            X_concat = np.vstack(X_list)
            self._init_params(X_concat)
            
        for iter_idx in range(self.max_iter):
            total_log_likelihood = 0
            
            # Accumulators for sufficient statistics
            expected_start = np.zeros(self.n_states)
            expected_trans = np.zeros((self.n_states, self.n_states))
            expected_means_num = np.zeros((self.n_states, self.n_features))
            expected_means_den = np.zeros(self.n_states)
            # Diagonal covariance accumulator
            expected_covs_num = np.zeros((self.n_states, self.n_features))
            
            for X in X_list:
                T = X.shape[0]
                log_alpha = self._forward(X)
                log_beta = self._backward(X)
                
                # Compute posteriors (gamma)
                # gamma[t, i] = P(q_t = i | O, lambda)
                log_gamma = log_alpha + log_beta
                # Normalize
                log_evidence = np.max(log_alpha[T-1]) + np.log(np.sum(np.exp(log_alpha[T-1] - np.max(log_alpha[T-1]))))
                total_log_likelihood += log_evidence
                
                gamma = np.exp(log_gamma - log_evidence)
                
                # Compute xi (joint transition prob)
                # xi[t, i, j] = P(q_t=i, q_t+1=j | O, lambda)
                xi = np.zeros((T-1, self.n_states, self.n_states))
                
                log_emissions = np.zeros((T, self.n_states))
                for i in range(self.n_states):
                    log_emissions[:, i] = self._log_pdf(X, self.means[i], self.covars[i])
                
                for t in range(T-1):
                    for i in range(self.n_states):
                        for j in range(self.n_states):
                            # alpha[t, i] * A[i, j] * B[j, x_t+1] * beta[t+1, j]
                            log_val = log_alpha[t, i] + np.log(self.trans_prob[i, j] + 1e-10) + \
                                      log_emissions[t+1, j] + log_beta[t+1, j]
                            xi[t, i, j] = log_val
                    
                    # Normalize xi at t
                    m = np.max(xi[t])
                    xi[t] = np.exp(xi[t] - (log_evidence)) # Approximate normalization
                
                # Accumulate statistics
                expected_start += gamma[0]
                expected_trans += np.sum(xi, axis=0)
                
                for i in range(self.n_states):
                    expected_means_den[i] += np.sum(gamma[:, i])
                    expected_means_num[i] += np.sum(gamma[:, i][:, np.newaxis] * X, axis=0)
                    
                    diff = X - self.means[i]
                    # Weighted covariance for diagonal
                    weighted_sq_diff = gamma[:, i][:, np.newaxis] * (diff ** 2)
                    expected_covs_num[i] += np.sum(weighted_sq_diff, axis=0)
            
            # M-Step: Update parameters
            self.start_prob = expected_start / np.sum(expected_start)
            
            self.trans_prob = expected_trans / np.sum(expected_trans, axis=1, keepdims=True)
            
            for i in range(self.n_states):
                self.means[i] = expected_means_num[i] / (expected_means_den[i] + 1e-10)

                # Diagonal covariance update - store variance vector directly
                self.covars[i] = expected_covs_num[i] / (expected_means_den[i] + 1e-10) + 1e-4
                
            print(f"Iteration {iter_idx+1}/{self.max_iter}, Log-Likelihood: {total_log_likelihood:.4f}")

    def score(self, X):
        """Compute log-likelihood of X."""
        log_alpha = self._forward(X)
        T = X.shape[0]
        # log sum exp of last alpha
        m = np.max(log_alpha[T-1])
        log_prob = m + np.log(np.sum(np.exp(log_alpha[T-1] - m)))
        return log_prob

    def predict(self, X):
        """Find most likely state sequence using Viterbi algorithm."""
        T = X.shape[0]
        log_delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # Precompute emission log probabilities
        log_emissions = np.zeros((T, self.n_states))
        for i in range(self.n_states):
            log_emissions[:, i] = self._log_pdf(X, self.means[i], self.covars[i])
            
        # Initialization
        log_delta[0] = np.log(self.start_prob + 1e-10) + log_emissions[0]
        
        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                # max_i (delta[t-1, i] * A[i, j]) * B[j, x_t]
                vals = log_delta[t-1] + np.log(self.trans_prob[:, j] + 1e-10)
                psi[t, j] = np.argmax(vals)
                log_delta[t, j] = np.max(vals) + log_emissions[t, j]
                
        # Termination
        path = np.zeros(T, dtype=int)
        path[T-1] = np.argmax(log_delta[T-1])
        
        # Backtracking
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
            
        return path
