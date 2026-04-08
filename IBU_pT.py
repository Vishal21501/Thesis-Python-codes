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

# IBU Parameters
N_ITERATIONS = 4
N_TOYS = 200  # Number of random experiments for error calculation

# 2. IBU UNFOLDER CLASS (With Error Calc)

class IBUUnfolder:
    def __init__(self, response_matrix, measured_data, efficiency):
        """
        response_matrix: Normalized P(Reco|Gen)
        measured_data: Vector of observed counts
        efficiency: Vector of epsilon per Gen bin
        """
        self.R = response_matrix
        self.data_nominal = measured_data
        # Protect against div by zero in efficiency
        self.eff = np.where(efficiency <= 0, 1.0, efficiency)

    def _run_single_unfold(self, data_input, iterations):
        """Internal worker function to run one unfolding pass."""
        # 1. Prior: Start with a flat guess
        n_gen = self.R.shape[1]
        unfolded = np.ones(n_gen) * np.sum(data_input) / n_gen
        
        # 2. Iteration Loop
        for i in range(iterations):
            # A. Fold: Predict Reco based on current Gen guess
            pred_meas = np.dot(self.R, unfolded)
            
            # Avoid division by zero
            pred_meas = np.where(pred_meas == 0, 1e-9, pred_meas)
            
            # B. Ratio: How wrong was our prediction?
            ratio = data_input / pred_meas
            
            # C. Update: Propagate correction back to Gen
            # (R.T * ratio) distributes the correction factor
            correction = np.dot(self.R.T, ratio)
            
            # D. Apply: New = Old * (Correction / Efficiency)
            unfolded = unfolded * (correction / self.eff)
            
        return unfolded

    def unfold(self, iterations):
        """Returns the nominal unfolded result."""
        return self._run_single_unfold(self.data_nominal, iterations)

    def get_errors_toy_mc(self, iterations, n_toys=100):
        """
        Estimates errors using Poisson bootstrapping (Toy MC).
        1. Generate n_toys variations of the input data (Poisson fluctuation).
        2. Unfold each toy.
        3. Calculate standard deviation of the results.
        """
        print(f"Running {n_toys} Toy MCs for error estimation...")
        toy_results = []
        
        for _ in range(n_toys):
            # Generate synthetic data: Poisson fluctuation of measured data
            toy_data = np.random.poisson(self.data_nominal)
            
            # Unfold this toy
            res = self._run_single_unfold(toy_data, iterations)
            toy_results.append(res)
            
        toy_results = np.array(toy_results)
        
        # Error is the standard deviation of the toys per bin
        errors = np.std(toy_results, axis=0)
        return errors

# 3. DATA LOADING & MATCHING

print(f"Loading {filename}...")
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

# Split Events
indices = np.arange(len(gen_muons))
train_mask = (indices % 2 != 0)
test_mask  = ~train_mask

def get_pt_data(reco, gen):
    pairs = ak.cartesian({"reco": reco, "gen": gen}, nested=True)
    if len(pairs) == 0: return [], [], []
    dRs = pairs["reco"].deltaR(pairs["gen"])
    best_idx = ak.argmin(dRs, axis=2, keepdims=True)
    good_match = (dRs[best_idx] < 0.3)
    matched = pairs[best_idx][good_match]
    
    m_r = ak.flatten(matched["reco"].pt, axis=None).to_numpy()
    m_g = ak.flatten(matched["gen"].pt, axis=None).to_numpy()
    all_g = ak.flatten(gen.pt, axis=None).to_numpy()
    return m_r, m_g, all_g

print("Processing Training Data...")
train_mr, train_mg, train_all_g = get_pt_data(reco_p4[train_mask], gen_muons[train_mask])

print("Processing Testing Data...")
# For test, we use the "Measured" Reco vector
test_reco = ak.flatten(reco_p4[test_mask].pt, axis=None).to_numpy()
test_gen  = ak.flatten(gen_muons[test_mask].pt, axis=None).to_numpy()

# 4. MATRICES & EFFICIENCY

# A. Response Matrix
response_hist, _, _ = np.histogram2d(train_mr, train_mg, bins=[pt_bins, pt_bins])
matched_counts = response_hist.sum(axis=0)
matched_counts[matched_counts == 0] = 1.0
response_prob = response_hist / matched_counts

# B. Efficiency
hist_gen_total, _ = np.histogram(train_all_g, bins=pt_bins)
is_in_acc = (train_mr >= pt_bins[0]) & (train_mr <= pt_bins[-1])
gen_accepted = train_mg[is_in_acc]
hist_gen_accepted, _ = np.histogram(gen_accepted, bins=pt_bins)

eff_map = np.divide(hist_gen_accepted, hist_gen_total, out=np.zeros_like(hist_gen_accepted, dtype=float), where=hist_gen_total!=0)

# C. Test Data (Histograms)
y_meas, _ = np.histogram(test_reco, bins=pt_bins)
y_true, _ = np.histogram(test_gen, bins=pt_bins)

# 5. EXECUTE IBU + ERROR ESTIMATION

solver = IBUUnfolder(response_prob, y_meas, eff_map)

# 1. Unfold Nominal
unfolded_ibu = solver.unfold(N_ITERATIONS)

# 2. Estimate Errors (Toy MC)
ibu_errors = solver.get_errors_toy_mc(N_ITERATIONS, N_TOYS)

# 6. PRINT ERROR PER BIN

centers = (pt_bins[:-1] + pt_bins[1:]) / 2
width = pt_bins[1] - pt_bins[0]

print("\n" + "="*55)
print(f" IBU RESULTS ({N_ITERATIONS} Iterations)")
print("="*55)
print(f"{'Bin Center':^12} | {'Unfolded N':^12} | {'Error (+/-)':^12} | {'Rel.Err %':^10}")
print("-" * 55)
for c, val, err in zip(centers, unfolded_ibu, ibu_errors):
    rel_err = (err/val)*100 if val > 0 else 0.0
    print(f"{c:^12.1f} | {val:^12.1f} | {err:^12.1f} | {rel_err:^10.1f}")
print("="*55 + "\n")

# 7. VISUALIZATION

# Create 1x2 grid
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Response Matrix
im = ax[0].imshow(response_prob, origin='lower', cmap='viridis', norm=LogNorm(), 
                  extent=[0, 200, 0, 200], aspect='auto')
ax[0].set_title("Response Matrix P(Reco|Gen)")
ax[0].set_xlabel("Gen $p_T$ [GeV]")
ax[0].set_ylabel("Reco $p_T$ [GeV]")
plt.colorbar(im, ax=ax[0])

# Plot 2: Result with Error Bars
ax[1].step(centers, y_meas, where='mid', label='Measured (Reco)', color='red', linestyle='--')
ax[1].step(centers, y_true, where='mid', label='Truth (Gen)', color='grey', alpha=0.6, linewidth=3)

# Errorbar plot
ax[1].errorbar(centers, unfolded_ibu, 
               yerr=ibu_errors, 
               xerr=width/2, 
               fmt='o', 
               label=f'IBU Unfolded', 
               color='blue', 
               capsize=3)

ax[1].set_title(f"Unfolding Results (N_iter={N_ITERATIONS})")
ax[1].set_xlabel("Muon $p_T$ [GeV]")
ax[1].set_ylabel("Events")
ax[1].legend()
ax[1].set_yscale('log')
ax[1].grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("IBU_Unfold_Errors.png")
plt.show()
