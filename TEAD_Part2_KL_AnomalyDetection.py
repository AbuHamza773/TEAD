
# === PART 2: Anomaly Detection using Gaussian + KL Divergence ===
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

# Define threshold for trust-based filtering
trust_threshold = 0.5

# Flag low-trust nodes for anomaly detection
suspicious_nodes = df[df['TrustScore'] < trust_threshold].copy()

# Select a feature for anomaly detection (can be extended)
feature = 'CFD'
normal_data = df[df['class'] == 0][feature]

# Estimate Gaussian distribution of normal behavior
mu, sigma = norm.fit(normal_data)

# Compute probability density of suspicious samples
suspicious_nodes['gaussian_prob'] = norm.pdf(suspicious_nodes[feature], mu, sigma)

# Calculate KL divergence (against normal distribution baseline)
def kl_divergence(p, q):
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    return np.sum(p * np.log(p / q))

# Compute KL divergence per node using their density vs normal
baseline_density = norm.pdf(normal_data, mu, sigma)
baseline_density /= np.sum(baseline_density)

kl_scores = []
for _, row in suspicious_nodes.iterrows():
    observed_density = norm.pdf([row[feature]], mu, sigma)
    observed_density /= np.sum(observed_density)
    kl = kl_divergence(observed_density, baseline_density)
    kl_scores.append(kl)

suspicious_nodes['KL_Divergence'] = kl_scores

# Set anomaly threshold based on percentile or fixed value
anomaly_threshold = np.percentile(kl_scores, 95)
suspicious_nodes['AnomalyFlag'] = suspicious_nodes['KL_Divergence'] > anomaly_threshold

# Display flagged anomalies
suspicious_nodes[['TrustScore', 'CFD', 'KL_Divergence', 'AnomalyFlag']].head(10)
