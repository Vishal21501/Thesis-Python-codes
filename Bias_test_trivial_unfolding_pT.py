import uproot
import awkward as ak
import vector
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import minimize

# 1. CONFIGURATION

FILENAME = "delphes_ppuulhe_high.root"
TREENAME = "Delphes"

# Binning
PT_BINS = np.linspace(0, 200, 16) 
CENTER = (PT_BINS[:-1] + PT_BINS[1:]) / 2
WIDTH  = PT_BINS[1] - PT_BINS[0]

# Solvers Settings
IBU_ITERS = 4          
TIK_TAU   = 0.0000001     
MAX_BETA  = 0.05     

# Signal Injection Settings
BUMP_MIN = 100         # Start of bump (GeV)
BUMP_MAX = 135         # End of bump (GeV)
AMPLIFICATION = 3.0    # Multiply events by 3x (Strong signal)

# 2. SOLVER CLASSES (The Math Engines)

# Method A: Tikhonov (Matrix Inversion)
class TikhonovUnfolder:
    def __init__(self, A, y):
        self.A, self.y = A, y
        self.V_inv = np.diag(1.0 / np.maximum(y, 1.0))
        n = A.shape[1]
        self.L = np.zeros((n, n))
        for i in range(1, n-1): self.L[i, i-1], self.L[i, i], self.L[i, i+1] = 1, -2, 1
        self.At_Vinv = np.dot(self.A.T, self.V_inv)

    def unfold(self, tau):
        M = np.dot(self.At_Vinv, self.A) + tau * np.dot(self.L.T, self.L)
        try: return np.linalg.solve(M, np.dot(self.At_Vinv, self.y))
        except: return np.zeros_like(self.y)

# Method B: IBU (Iterative Bayes) 
class IBUUnfolder:
    def __init__(self, A, y, eff):
        self.A, self.y, self.eff = A, y, np.where(eff <= 0, 1.0, eff)

    def unfold(self, iterations):
        unfolded = np.ones(self.A.shape[1]) * np.sum(self.y) / self.A.shape[1]
        for _ in range(iterations):
            pred = np.dot(self.A, unfolded)
            ratio = self.y / np.where(pred==0, 1e-9, pred)
            correction = np.dot(self.A.T, ratio)
            unfolded *= (correction / self.eff)
        return unfolded

# Method C: MAXED (Maximum Entropy)
class MaxEntUnfolder:
    def __init__(self, A, y, eff):
        self.A, self.y = A, y
        self.eff = np.where(eff <= 1e-3, 1.0, eff)
        self.sigma2 = np.maximum(y, 1.0)
        self.prior = np.ones(A.shape[1])

    def _loss(self, f):
        y_pred = np.dot(self.A, f * self.eff)
        chi2 = np.sum((self.y - y_pred)**2 / self.sigma2)
        norm = np.sum(f) + 1e-9
        p, q = f / norm, self.prior / np.sum(self.prior)
        entropy = -np.sum(p * np.log(p / q + 1e-12))
        return 0.5 * chi2 - MAX_BETA * entropy

    def unfold(self):
        x0 = np.ones(self.A.shape[1]) * (np.sum(self.y)/self.A.shape[1])
        res = minimize(self._loss, x0, method='L-BFGS-B', bounds=[(1e-6, None)]*len(x0))
        return res.x

# 3. DATA LOADING & PROCESSING

print(f"Loading {FILENAME}...")
with uproot.open(f"{FILENAME}:{TREENAME}") as tree:
    gen_p4 = vector.zip({"pt": tree["Particle.PT"].array(), "eta": tree["Particle.Eta"].array(), "phi": tree["Particle.Phi"].array(), "pid": tree["Particle.PID"].array(), "status": tree["Particle.Status"].array()})
    reco_p4 = vector.zip({"pt": tree["Muon.PT"].array(), "eta": tree["Muon.Eta"].array(), "phi": tree["Muon.Phi"].array()})

gen_muons = gen_p4[(abs(gen_p4.pid) == 13) & (gen_p4.status == 1)]
indices = np.arange(len(gen_muons))
train_mask, test_mask = (indices % 2 != 0), (indices % 2 == 0)

def get_data(reco, gen):
    pairs = ak.cartesian({"reco": reco, "gen": gen}, nested=True)
    if len(pairs)==0: return [], [], []
    dRs = pairs["reco"].deltaR(pairs["gen"])
    best_idx = ak.argmin(dRs, axis=2, keepdims=True)
    matched = pairs[best_idx][dRs[best_idx] < 0.3]
    return ak.flatten(matched["reco"].pt, axis=None).to_numpy(), ak.flatten(matched["gen"].pt, axis=None).to_numpy(), ak.flatten(gen.pt, axis=None).to_numpy()

# --- Build STANDARD MODEL Training Matrices ---
print("Building Standard Model Matrix...")
tr_mr, tr_mg, tr_all_g = get_data(reco_p4[train_mask], gen_muons[train_mask])

# Response Matrix
resp_h, _, _ = np.histogram2d(tr_mr, tr_mg, bins=[PT_BINS, PT_BINS])
resp_prob = resp_h / np.where(resp_h.sum(axis=0)==0, 1, resp_h.sum(axis=0))

# Efficiency
h_tot, _ = np.histogram(tr_all_g, bins=PT_BINS)
in_acc = (tr_mr >= PT_BINS[0]) & (tr_mr <= PT_BINS[-1])
h_acc, _ = np.histogram(tr_mg[in_acc], bins=PT_BINS)
eff = np.divide(h_acc, h_tot, out=np.zeros_like(h_acc, dtype=float), where=h_tot!=0)

# INJECT SIGNAL into Test Data 
print("Injecting Signal...")
test_reco = ak.flatten(reco_p4[test_mask].pt, axis=None).to_numpy()
test_gen  = ak.flatten(gen_muons[test_mask].pt, axis=None).to_numpy()

# Create copies of events in the bump region
reco_bump = test_reco[(test_reco > BUMP_MIN) & (test_reco < BUMP_MAX)]
gen_bump  = test_gen[(test_gen > BUMP_MIN) & (test_gen < BUMP_MAX)]

# Stack them to amplify signal
# We add (AMPLIFICATION - 1) copies
n_copies = int(AMPLIFICATION - 1)
reco_injected = np.concatenate([test_reco] + [reco_bump]*n_copies)
gen_injected  = np.concatenate([test_gen] + [gen_bump]*n_copies)

# Histogram the Injected Data
y_meas_inj, _ = np.histogram(reco_injected, bins=PT_BINS)
y_true_inj, _ = np.histogram(gen_injected, bins=PT_BINS)
y_true_std, _ = np.histogram(test_gen, bins=PT_BINS)

# 4. RUN ALL SOLVERS ON INJECTED DATA

print("Running Tikhonov on Injected Data...")
res_tik_raw = TikhonovUnfolder(resp_prob, y_meas_inj).unfold(TIK_TAU)
res_tik = np.divide(res_tik_raw, eff, out=np.zeros_like(res_tik_raw), where=eff > 0.001)

print("Running IBU on Injected Data...")
res_ibu = IBUUnfolder(resp_prob, y_meas_inj, eff).unfold(IBU_ITERS)

print("Running MAXED on Injected Data...")
res_maxed = MaxEntUnfolder(resp_prob, y_meas_inj, eff).unfold()

# 5. VISUALIZATION

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

# Upper Panel: Spectra 
# Reference: Standard Model (Grey)
ax1.fill_between(CENTER, y_true_std, step='mid', color='grey', alpha=0.15, label='Standard Model (Matrix)')

# Reference: True Signal (Black)
ax1.step(CENTER, y_true_inj, where='mid', color='black', linewidth=2, label='True Signal (Target)')

# Input: Measured (Red)
ax1.step(CENTER, y_meas_inj, where='mid', color='red', linestyle='--', label='Measured (Input)')

# Solvers
ax1.errorbar(CENTER, res_ibu, xerr=WIDTH/2, fmt='o', label='IBU', color='blue', markersize=5)
ax1.errorbar(CENTER, res_tik, xerr=WIDTH/2, fmt='s', label='Tikhonov', color='green', markersize=5, fillstyle='none')
ax1.errorbar(CENTER, res_maxed, xerr=WIDTH/2, fmt='^', label='MAXED', color='purple', markersize=5, fillstyle='none')

ax1.set_ylabel("Events")
ax1.set_yscale('log')
ax1.set_title(f"Bias Test: Recovering a {AMPLIFICATION}x Signal Bump [{BUMP_MIN}-{BUMP_MAX} GeV]")
ax1.axvspan(BUMP_MIN, BUMP_MAX, color='yellow', alpha=0.2, label='Signal Region')
ax1.legend()
ax1.grid(True, which='both', alpha=0.3)

# Lower Panel: Ratio to TRUE SIGNAL
# We compare Unfolded / True Injected Signal
# Ideally = 1.0. If < 1.0, the method is biased (smoothing out the bump).
r_ibu = np.divide(res_ibu, y_true_inj, out=np.zeros_like(res_ibu), where=y_true_inj>0)
r_tik = np.divide(res_tik, y_true_inj, out=np.zeros_like(res_tik), where=y_true_inj>0)
r_max = np.divide(res_maxed, y_true_inj, out=np.zeros_like(res_maxed), where=y_true_inj>0)

ax2.axhline(1.0, color='black', linestyle='-', linewidth=1)
ax2.plot(CENTER, r_ibu, 'o-', color='blue', markersize=4, label='IBU')
ax2.plot(CENTER, r_tik, 's-', color='green', markersize=4, fillstyle='none', label='Tikhonov')
ax2.plot(CENTER, r_max, '^-', color='purple', markersize=4, fillstyle='none', label='MAXED')

# Highlight Bump
ax2.axvspan(BUMP_MIN, BUMP_MAX, color='yellow', alpha=0.2, label='Signal Region')

ax2.set_ylabel("Unfolded / True Signal")
ax2.set_xlabel("Muon $p_T$ [GeV]")
ax2.set_ylim(0.5, 1.5)
ax2.grid(True, which='both', alpha=0.3)
ax2.legend(fontsize='small', ncol=3)

plt.tight_layout()
plt.subplots_adjust(hspace=0.05)
plt.savefig("Bias_Comparison_Unfold.png")
plt.show()
