import uproot
import awkward as ak
import vector
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import minimize

# 1. CONFIGURATION

filename = "delphes_ppuulhe_high.root"
treename = "Delphes"

# Binning
pt_bins = np.linspace(0, 200, 16) 
n_bins = len(pt_bins) - 1

# MAXED Regularization Parameter (Beta)
# Controls trade-off between Fit (Chi2) and Entropy
BETA = 0.05

# Toy MC Parameters for Error Calculation
N_TOYS = 100  # Number of random experiments

# 2. MAXED UNFOLDER CLASS

class MaxEntUnfolder:
    def __init__(self, response_matrix, measured_data, efficiency):
        self.A = response_matrix
        self.y_nominal = measured_data
        self.eff = np.where(efficiency <= 1e-3, 1.0, efficiency) # Safety
        
        # The "Prior" (Default Model)
        # We assume a Flat Prior (Uniform)
        self.prior = np.ones(response_matrix.shape[1])

    def _loss_function(self, f_current, y_target, sigma2_target):
        """
        The Objective Function to Minimize:
        L = 0.5 * Chi2  -  Beta * Entropy
        """
        # 1. Fold: Predicted Data = Matrix * Unfolded * Efficiency
        y_pred = np.dot(self.A, f_current * self.eff)
        
        # 2. Chi-Squared Term (Fit Quality)
        chi2 = np.sum((y_target - y_pred)**2 / sigma2_target)
        
        # 3. Entropy Term (Regularization)
        # S = - Sum( p * ln(p / prior) )
        norm = np.sum(f_current) + 1e-9
        p = f_current / norm
        q = self.prior / np.sum(self.prior)
        
        entropy = -np.sum(p * np.log(p / q + 1e-12))
        
        return 0.5 * chi2 - BETA * entropy

    def _run_optimizer(self, data_input, verbose=False):
        """Helper to run the minimization for a specific data input."""
        n_gen = self.A.shape[1]
        
        # Variance for Chi2 (Neyman's)
        sigma2 = np.maximum(data_input, 1.0)
        
        # Initial Guess: Prior scaled to data counts
        x0 = np.ones(n_gen) * (np.sum(data_input) / n_gen)
        
        # Bounds: Strictly positive
        bounds = [(1e-6, None) for _ in range(n_gen)]
        
        options = {'disp': verbose, 'maxiter': 1000}
        
        result = minimize(
            self._loss_function, 
            x0, 
            args=(data_input, sigma2), # Pass data args to loss func
            method='L-BFGS-B', 
            bounds=bounds,
            options=options
        )
        return result.x

    def unfold(self):
        """Run nominal unfolding on the actual measured data."""
        print("Running Nominal MAXED Optimizer...")
        return self._run_optimizer(self.y_nominal, verbose=True)

    def get_errors_toy_mc(self, n_toys=50):
        """
        Estimates errors using Poisson bootstrapping.
        """
        print(f"Running {n_toys} Toy MCs for error estimation (this may take a moment)...")
        toy_results = []
        
        for i in range(n_toys):
            # Generate synthetic data: Poisson fluctuation
            toy_data = np.random.poisson(self.y_nominal)
            
            # Run optimizer (quietly)
            res = self._run_optimizer(toy_data, verbose=False)
            toy_results.append(res)
            
        toy_results = np.array(toy_results)
        
        # Error is standard deviation of the toys
        return np.std(toy_results, axis=0)

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

gen_muons = gen_p4[(abs(gen_p4.pid) == 13) & (gen_p4.status == 1)]
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

print("Processing Data...")
train_mr, train_mg, train_all_g = get_pt_data(reco_p4[train_mask], gen_muons[train_mask])
test_reco = ak.flatten(reco_p4[test_mask].pt, axis=None).to_numpy()
test_gen  = ak.flatten(gen_muons[test_mask].pt, axis=None).to_numpy()

# 4. MATRICES

# Response
response_hist, _, _ = np.histogram2d(train_mr, train_mg, bins=[pt_bins, pt_bins])
matched_counts = response_hist.sum(axis=0)
matched_counts[matched_counts == 0] = 1.0
response_prob = response_hist / matched_counts

# Efficiency
hist_gen_total, _ = np.histogram(train_all_g, bins=pt_bins)
is_in_acc = (train_mr >= pt_bins[0]) & (train_mr <= pt_bins[-1])
gen_accepted = train_mg[is_in_acc]
hist_gen_accepted, _ = np.histogram(gen_accepted, bins=pt_bins)
eff_map = np.divide(hist_gen_accepted, hist_gen_total, out=np.zeros_like(hist_gen_accepted, dtype=float), where=hist_gen_total!=0)

# Measured Data
y_meas, _ = np.histogram(test_reco, bins=pt_bins)
y_true, _ = np.histogram(test_gen, bins=pt_bins)

# 5. EXECUTE MAXED + ERRORS

solver = MaxEntUnfolder(response_prob, y_meas, eff_map)


# 1. Nominal Unfolding
unfolded_maxed = solver.unfold()

# 2. Error Estimation (Toy MC)
maxed_errors = solver.get_errors_toy_mc(n_toys=N_TOYS)

# 6. PRINT RESULTS TABLE

centers = (pt_bins[:-1] + pt_bins[1:]) / 2
width = pt_bins[1] - pt_bins[0]

print("\n" + "="*55)
print(f" MAXED RESULTS (Beta={BETA})")
print("="*55)
print(f"{'Bin Center':^12} | {'Unfolded N':^12} | {'Error (+/-)':^12} | {'Rel.Err %':^10}")
print("-" * 55)
for c, val, err in zip(centers, unfolded_maxed, maxed_errors):
    rel_err = (err/val)*100 if val > 0 else 0.0
    print(f"{c:^12.1f} | {val:^12.1f} | {err:^12.1f} | {rel_err:^10.1f}")
print("="*55 + "\n")

# 7. VISUALIZATION

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Response Matrix
im = ax[0].imshow(response_prob, origin='lower', cmap='viridis', norm=LogNorm(), 
                  extent=[0, 200, 0, 200], aspect='auto')
ax[0].set_title("Response Matrix")
ax[0].set_xlabel("Gen $p_T$ [GeV]")
ax[0].set_ylabel("Reco $p_T$ [GeV]")
plt.colorbar(im, ax=ax[0])

# Plot 2: Result with Errors
ax[1].step(centers, y_meas, where='mid', label='Measured (Reco)', color='red', linestyle='--')
ax[1].step(centers, y_true, where='mid', label='Truth (Gen)', color='grey', alpha=0.6, linewidth=3)

ax[1].errorbar(centers, unfolded_maxed, 
               yerr=maxed_errors, 
               xerr=width/2, 
               fmt='o', 
               label=f'MAXED Unfolded', 
               color='blue', 
               capsize=3)

ax[1].set_title(f"MAXED Result (Beta={BETA})")
ax[1].set_xlabel("Muon $p_T$ [GeV]")
ax[1].set_ylabel("Events")
ax[1].legend()
ax[1].set_yscale('log')
ax[1].grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("MAXED_Unfold_Errors.png")
plt.show()
