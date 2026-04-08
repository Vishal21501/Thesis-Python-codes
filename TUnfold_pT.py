import uproot
import awkward as ak
import vector
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# 1. CONFIGURATION

filename = "delphes_ppuulhe_high.root"
treename = "Delphes"

# Binning
pt_bins = np.linspace(0, 200, 16)
n_bins = len(pt_bins) - 1

# 2. TIKHONOV UNFOLDER CLASS

class TikhonovUnfolder:
    def __init__(self, response_matrix, measured_data):
        self.A = response_matrix
        self.y = measured_data
        
        # Variance (Data Errors), avoid div by zero
        # Assumption: Poisson statistics, Variance = Count (y)
        # We use max(y, 1) for weights to avoid inf, but will use y for error prop
        self.V_inv = np.diag(1.0 / np.maximum(measured_data, 1.0))
        
        # Regularization Matrix (Curvature/Laplacian)
        n_gen = response_matrix.shape[1]
        self.L = np.zeros((n_gen, n_gen))
        for i in range(1, n_gen - 1):
            self.L[i, i-1], self.L[i, i], self.L[i, i+1] = 1, -2, 1

        self.At_Vinv = np.dot(self.A.T, self.V_inv)
        self.LHS_base = np.dot(self.At_Vinv, self.A)
        self.RHS_base = np.dot(self.At_Vinv, self.y)
        self.M_inv = None # Store for error calc

    def unfold(self, tau):
        # Solve: (A^T V^-1 A + tau * L^T L) x = A^T V^-1 y
        # Let M = (A^T V^-1 A + tau * L^T L)
        M = self.LHS_base + tau * np.dot(self.L.T, self.L)
        
        try:
            # We explicitly compute inverse for error propagation later
            self.M_inv = np.linalg.inv(M)
            return np.dot(self.M_inv, self.RHS_base)
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix found during unfolding.")
            self.M_inv = np.zeros_like(M)
            return np.zeros_like(self.RHS_base)

    def get_errors(self):
        """
        Calculates the covariance matrix of the unfolded result.
        Propagates data errors: V_x = K * V_y * K^T
        Where K = M^-1 * A^T * V^-1
        """
        if self.M_inv is None:
            raise RuntimeError("Run unfold() before calculating errors.")

        # 1. Reconstruct Data Covariance Matrix (V_y)
        # Poisson assumption: V_ii = N_i
        V_y = np.diag(self.y)

        # 2. Construct Transformation Matrix K
        # K transforms measured data y into unfolded x
        # x = K * y
        K = np.dot(self.M_inv, self.At_Vinv)

        # 3. Propagate Covariance
        # V_x = K @ V_y @ K.T
        V_x = np.dot(K, np.dot(V_y, K.T))

        # Return the square root of diagonal elements (standard deviations)
        # We define errors as sqrt(Variance)
        return np.sqrt(np.diag(V_x))

# 3. DATA LOADING & MATCHING

print(f"Loading {filename}...")
# 
with uproot.open(f"{filename}:{treename}") as tree:
    gen_p4 = vector.zip({
        "pt": tree["Particle.PT"].array(),
        "eta": tree["Particle.Eta"].array(),
        "phi": tree["Particle.Phi"].array(),
        "pid": tree["Particle.PID"].array(),
        "status": tree["Particle.Status"].array()
    })
    reco_p4 = vector.zip({
        "pt": tree["Muon.PT"].array(),
        "eta": tree["Muon.Eta"].array(),
        "phi": tree["Muon.Phi"].array()
    })

# Filter Gen Muons
gen_muons = gen_p4[(abs(gen_p4.pid) == 13) & (gen_p4.status == 1)]

# Split Events: Odd=Train, Even=Test
indices = np.arange(len(gen_muons))
train_mask = (indices % 2 != 0)
test_mask  = ~train_mask

def get_matching_data(reco_sub, gen_sub):
    """Returns: matched_reco_pt, matched_gen_pt, all_reco_pt, all_gen_pt"""
    pairs = ak.cartesian({"reco": reco_sub, "gen": gen_sub}, nested=True)
    if len(pairs) == 0: return [], [], [], []

    dRs = pairs["reco"].deltaR(pairs["gen"])
    best_idx = ak.argmin(dRs, axis=2, keepdims=True)
    good_match = (dRs[best_idx] < 0.3)

    matched = pairs[best_idx][good_match]

    # Flatten arrays
    m_r = ak.flatten(matched["reco"].pt, axis=None).to_numpy()
    m_g = ak.flatten(matched["gen"].pt, axis=None).to_numpy()
    all_r = ak.flatten(reco_sub.pt, axis=None).to_numpy()
    all_g = ak.flatten(gen_sub.pt, axis=None).to_numpy()

    return m_r, m_g, all_r, all_g

print("Processing Training Data (for Matrix & Efficiency)...")
train_mr, train_mg, _, train_all_g = get_matching_data(reco_p4[train_mask], gen_muons[train_mask])

print("Processing Testing Data (Mock Experiment)...")
_, _, test_all_r, test_all_g = get_matching_data(reco_p4[test_mask], gen_muons[test_mask])

# 4. BUILD MATRICES & EFFICIENCY

# A. Response Matrix (Smearing)
response_hist, _, _ = np.histogram2d(train_mr, train_mg, bins=[pt_bins, pt_bins])

# Normalize columns
matched_gen_counts = response_hist.sum(axis=0)
matched_gen_counts[matched_gen_counts == 0] = 1.0
response_prob = response_hist / matched_gen_counts

# B. Efficiency Calculation
hist_gen_total, _ = np.histogram(train_all_g, bins=pt_bins)
is_in_acceptance = (train_mr >= pt_bins[0]) & (train_mr <= pt_bins[-1])
gen_pt_accepted = train_mg[is_in_acceptance]
hist_gen_accepted, _ = np.histogram(gen_pt_accepted, bins=pt_bins)

eff_map = np.divide(hist_gen_accepted, hist_gen_total, out=np.zeros_like(hist_gen_accepted, dtype=float), where=hist_gen_total!=0)

# C. Input Data
y_measured, _ = np.histogram(test_all_r, bins=pt_bins)
y_true_target, _ = np.histogram(test_all_g, bins=pt_bins)

# 5. UNFOLD + ERROR CALCULATION
solver = TikhonovUnfolder(response_prob, y_measured)
tau = 0.0000001

# 1. Unfold (Raw Result)
unfolded_raw = solver.unfold(tau)

# 2. Get Raw Errors (from unfolding matrix propagation)
errors_raw = solver.get_errors()

# 3. Apply Efficiency Correction to Values AND Errors
# Error scaling: sigma_corrected = sigma_raw / efficiency
mask_eff = eff_map > 0.01
unfolded_corrected = np.zeros_like(unfolded_raw)
errors_corrected = np.zeros_like(errors_raw)

unfolded_corrected[mask_eff] = unfolded_raw[mask_eff] / eff_map[mask_eff]
errors_corrected[mask_eff]   = errors_raw[mask_eff] / eff_map[mask_eff]

# 6. PRINT ERROR PER BIN

centers = (pt_bins[:-1] + pt_bins[1:]) / 2
width = pt_bins[1] - pt_bins[0]

print("\n" + "="*50)
print(f"{'Bin Center':^12} | {'Unfolded N':^12} | {'Error (+/-)':^12} | {'Rel.Err %':^10}")
print("-" * 50)
for c, val, err in zip(centers, unfolded_corrected, errors_corrected):
    rel_err = (err/val)*100 if val > 0 else 0.0
    print(f"{c:^12.1f} | {val:^12.1f} | {err:^12.1f} | {rel_err:^10.1f}")
print("="*50 + "\n")

# 7. VISUALIZATION

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Response Matrix 
im = ax[0].imshow(response_prob, origin='lower', cmap='viridis', norm=LogNorm(), extent=[0, 200, 0, 200], aspect='auto')
ax[0].set_title("Response Matrix P(Reco|Gen)")
ax[0].set_xlabel("Gen $p_T$ [GeV]")
ax[0].set_ylabel("Reco $p_T$ [GeV]")
fig.colorbar(im, ax=ax[0])


# Plot 2: Comparison with Error Bars 
ax[1].step(centers, y_measured, where='mid', label='Measured (Reco)', color='red', linestyle='--', linewidth=1.5)
ax[1].step(centers, y_true_target, where='mid', label='Truth (Gen)', color='grey', alpha=0.6, linewidth=3)

# Use errorbar for the unfolded data
ax[1].errorbar(centers, unfolded_corrected, 
               yerr=errors_corrected, 
               xerr=width/2, 
               fmt='o', 
               label=fr'Unfolded $\pm \sigma$', 
               color='blue', 
               capsize=3) # capsize adds the little bars at top/bottom

ax[1].set_title(f"Complete Unfolding (Tau={tau})")
ax[1].set_xlabel("Muon $p_T$ [GeV]")
ax[1].set_ylabel("Events")
ax[1].legend()
ax[1].set_yscale('log')
ax[1].grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("TUnfold_ppuu_WithErrors.png")
plt.show()
