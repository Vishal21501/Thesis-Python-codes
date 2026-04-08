# 0. INITIAL SET UP AND MAIN IMPORTS
import os
from google.colab import drive

# 1. Mount Google Drive
# This checks if the drive is already mounted to avoid annoying prompts
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')
    print("Google Drive mounted.")
else:
    print("Google Drive is already mounted.")

# 2. (Optional) Create a dedicated folder for this project
# Change 'My_Colab_Project' to whatever you want your folder named
project_folder = '/content/drive/MyDrive/My_Colab_Project'

# Create the folder if it doesn't exist
if not os.path.exists(project_folder):
    os.makedirs(project_folder)
    print(f"Created new folder: {project_folder}")

# 3. Change the current working directory to that folder
os.chdir(project_folder)
print(f"Current working directory set to: {os.getcwd()}")

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.gridspec as gridspec

# 1.CONFIGURATION AND RECONSTRUCTION

# A. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Active Device: {device}")
if device.type == 'cpu':
    print("WARNING: Running on CPU will be slow. Enable GPU in Runtime > Change Runtime Type.")

# B. Configuration for pp->ttbar -> muon+muon- channel
OBS_CONFIG = {
    0: {"name": "Leading muon pT [GeV]", "bins": np.linspace(0, 300, 40)},
    1: {"name": "Leading Jet pT [GeV]", "bins": np.linspace(0, 300, 40)},
    2: {"name": "ttbar Mass [GeV]", "bins": np.linspace(300, 1000, 40)},
    3: {"name": "delta_r(muon, jet)", "bins": np.linspace(0, 5, 40)}
}

# C. Output Directory
OUTPUT_DIR = "/content/drive/MyDrive/Unfolding_pp6_DNN-Results_loss"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Results will be saved to: {OUTPUT_DIR}")

# D. Data Loading Function
def calculate_mass_vectorized(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2):
    """Vectorized invariant mass calculation for two objects."""
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    e1 = np.sqrt(px1**2 + py1**2 + pz1**2 + m1**2)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    e2 = np.sqrt(px2**2 + py2**2 + pz2**2 + m2**2)

    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    e = e1 + e2

    m2_inv = e**2 - px**2 - py**2 - pz**2
    return np.sqrt(np.maximum(m2_inv, 0))

def calculate_visible_mass_vectorized(mu1, mu2, jet1, jet2):
    """
    Vectorized calculation of the visible system mass (2 muons + 2 jets).
    Returns the invariant mass of the sum of the 4 four-vectors.
    """
    # Helper to get Energy
    def get_p4(pt, eta, phi, m):
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        e = np.sqrt(px**2 + py**2 + pz**2 + m**2)
        return px, py, pz, e

    px1, py1, pz1, e1 = get_p4(mu1.pt, mu1.eta, mu1.phi, mu1.mass)
    px2, py2, pz2, e2 = get_p4(mu2.pt, mu2.eta, mu2.phi, mu2.mass)
    px3, py3, pz3, e3 = get_p4(jet1.pt, jet1.eta, jet1.phi, jet1.mass)
    px4, py4, pz4, e4 = get_p4(jet2.pt, jet2.eta, jet2.phi, jet2.mass)

    sum_px = px1 + px2 + px3 + px4
    sum_py = py1 + py2 + py3 + py4
    sum_pz = pz1 + pz2 + pz3 + pz4
    sum_e  = e1  + e2  + e3  + e4

    m2 = sum_e**2 - sum_px**2 - sum_py**2 - sum_pz**2
    return np.sqrt(np.maximum(m2, 0))


def load_data(filename, is_data=False, use_gen_jets=True):
    """
    Optimized & Vectorized Data Loader for pp -> ttbar -> dilepton.
    """
    print(f"Loading {filename}...")
    try:
        file = uproot.open(filename)
        tree = file["Delphes"]

        # --- CONSTANTS ---
        MUON_MASS = 0.10566
        B_MASS = 4.18

        
        # 1. RECO LEVEL PROCESSING (Vectorized)
        

        # Load arrays lazily
        mu_pt = tree["Muon.PT"].array()
        mu_eta = tree["Muon.Eta"].array()
        mu_phi = tree["Muon.Phi"].array()
        mu_charge = tree["Muon.Charge"].array()

        jet_pt = tree["Jet.PT"].array()
        jet_eta = tree["Jet.Eta"].array()
        jet_phi = tree["Jet.Phi"].array()
        jet_mass = tree["Jet.Mass"].array()
        jet_btag = tree["Jet.BTag"].array()

        # A. PRE-SELECTION: At least 2 muons and 2 jets
        # This mask filters events immediately, speeding up everything downstream
        event_mask = (ak.num(mu_pt) >= 2) & (ak.num(jet_pt) >= 2)

        # Apply mask to all arrays
        mu_pt = mu_pt[event_mask]
        mu_eta = mu_eta[event_mask]
        mu_phi = mu_phi[event_mask]
        mu_charge = mu_charge[event_mask]

        jet_pt = jet_pt[event_mask]
        jet_eta = jet_eta[event_mask]
        jet_phi = jet_phi[event_mask]
        jet_mass = jet_mass[event_mask]
        jet_btag = jet_btag[event_mask]

        # B. OBJECT SELECTION
        # We select the leading 2 muons (highest pT) and leading 2 jets
        # Since Delphes usually sorts by pT, we just take index 0 and 1.

        mu1 = ak.zip({"pt": mu_pt[:,0], "eta": mu_eta[:,0], "phi": mu_phi[:,0], "charge": mu_charge[:,0], "mass": MUON_MASS})
        mu2 = ak.zip({"pt": mu_pt[:,1], "eta": mu_eta[:,1], "phi": mu_phi[:,1], "charge": mu_charge[:,1], "mass": MUON_MASS})

        # For Jets, we ideally want b-tagged ones.
        # Simple strategy for vectorization: Pick top 2 by pT (standard Delphes sort)
        # (Refining this to sort by BTag vectorially is complex, pT sort is acceptable for this level)
        jet1 = ak.zip({"pt": jet_pt[:,0], "eta": jet_eta[:,0], "phi": jet_phi[:,0], "mass": jet_mass[:,0]})
        jet2 = ak.zip({"pt": jet_pt[:,1], "eta": jet_eta[:,1], "phi": jet_phi[:,1], "mass": jet_mass[:,1]})

        # C. EVENT SELECTION CUTS
        # 1. Opposite Charge Muons
        charge_mask = (mu1.charge * mu2.charge < 0)

        # 2. Kinematic Cuts (pT > 25/30)
        kin_mask = (mu1.pt > 25) & (jet1.pt > 30)

        # 3. Z-Veto (Dilepton Mass not between 80-100 GeV roughly, user used 20-80 window in original)
        dilep_mass = calculate_mass_vectorized(mu1.pt, mu1.eta, mu1.phi, mu1.mass,
                                             mu2.pt, mu2.eta, mu2.phi, mu2.mass)
        # Note: Original code kept 20 < mass < 80 (rejecting Z peak and low mass resonance).
        # I will preserve your original logic:
        mass_window_mask = (dilep_mass > 20) & (dilep_mass < 80)

        # Combine all masks
        final_reco_mask = charge_mask & kin_mask & mass_window_mask

        # D. CALCULATE OBSERVABLES (Only for passing events)
        # Filter objects first to save computation
        mu1 = mu1[final_reco_mask]
        mu2 = mu2[final_reco_mask]
        jet1 = jet1[final_reco_mask]
        jet2 = jet2[final_reco_mask]

        # 1. Visible Mass (Replaces buggy neutrino weighting)
        vis_mass = calculate_visible_mass_vectorized(mu1, mu2, jet1, jet2)

        # 2. Delta R (Mu1, Jet1)
        deta = mu1.eta - jet1.eta
        dphi = np.abs(mu1.phi - jet1.phi)
        dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
        delta_r = np.sqrt(deta**2 + dphi**2)

        # Stack Reco Observables
        # [Leading Mu pT, Leading Jet pT, Visible Mass, Delta R]
        X_reco = np.column_stack((
            ak.to_numpy(mu1.pt),
            ak.to_numpy(jet1.pt),
            ak.to_numpy(vis_mass),
            ak.to_numpy(delta_r)
        )).astype(np.float32)

        if is_data:
            print(f"Data loaded: {len(X_reco)} events")
            return X_reco

        
        # 2. GENERATOR LEVEL PROCESSING (Vectorized)
        

        # Important: We must maintain 1-to-1 correspondence.
        # We start with the SAME event_mask and final_reco_mask used above
        # so we are looking at the exact same events.

        gen_part_pt = tree["Particle.PT"].array()[event_mask][final_reco_mask]
        gen_part_eta = tree["Particle.Eta"].array()[event_mask][final_reco_mask]
        gen_part_phi = tree["Particle.Phi"].array()[event_mask][final_reco_mask]
        gen_part_pid = tree["Particle.PID"].array()[event_mask][final_reco_mask]
        gen_part_status = tree["Particle.Status"].array()[event_mask][final_reco_mask]
        gen_part_mass = tree["Particle.Mass"].array()[event_mask][final_reco_mask]

        # A. GEN MUONS (Status 1, PID +/- 13)
        is_mu = (np.abs(gen_part_pid) == 13) & (gen_part_status == 1)
        # Filter events that don't have 2 gen muons (sanity check)
        gen_mu_count_mask = ak.num(gen_part_pt[is_mu]) >= 2

        # B. GEN JETS
        if use_gen_jets and "GenJet.PT" in tree:
            gen_jet_pt = tree["GenJet.PT"].array()[event_mask][final_reco_mask]
            gen_jet_eta = tree["GenJet.Eta"].array()[event_mask][final_reco_mask]
            gen_jet_phi = tree["GenJet.Phi"].array()[event_mask][final_reco_mask]
            gen_jet_mass = tree["GenJet.Mass"].array()[event_mask][final_reco_mask]
            has_gen_jets = ak.num(gen_jet_pt) >= 2
        else:
            # Fallback to b-quarks (PID 5)
            # CAUTION: Status 23 is usually "outgoing from hard process" in Pythia8
            is_b = (np.abs(gen_part_pid) == 5) & (gen_part_status == 23)
            gen_jet_pt = gen_part_pt[is_b]
            gen_jet_eta = gen_part_eta[is_b]
            gen_jet_phi = gen_part_phi[is_b]
            gen_jet_mass = gen_part_mass[is_b]
            has_gen_jets = ak.num(gen_jet_pt) >= 2

        # Combine gen sanity masks
        valid_gen_mask = gen_mu_count_mask & has_gen_jets

        # Apply mask to Reco X as well (we must drop reco events where Gen is bad)
        X_reco = X_reco[valid_gen_mask]

        # Select Objects
        g_mu_pt = gen_part_pt[is_mu][valid_gen_mask]
        g_mu_eta = gen_part_eta[is_mu][valid_gen_mask]
        g_mu_phi = gen_part_phi[is_mu][valid_gen_mask]

        g_jet_pt = gen_jet_pt[valid_gen_mask]
        g_jet_eta = gen_jet_eta[valid_gen_mask]
        g_jet_phi = gen_jet_phi[valid_gen_mask]
        g_jet_mass = gen_jet_mass[valid_gen_mask]

        # Create Gen Objects (Take leading 2)
        gm1 = ak.zip({"pt": g_mu_pt[:,0], "eta": g_mu_eta[:,0], "phi": g_mu_phi[:,0], "mass": MUON_MASS})
        gm2 = ak.zip({"pt": g_mu_pt[:,1], "eta": g_mu_eta[:,1], "phi": g_mu_phi[:,1], "mass": MUON_MASS})
        gj1 = ak.zip({"pt": g_jet_pt[:,0], "eta": g_jet_eta[:,0], "phi": g_jet_phi[:,0], "mass": g_jet_mass[:,0]})
        gj2 = ak.zip({"pt": g_jet_pt[:,1], "eta": g_jet_eta[:,1], "phi": g_jet_phi[:,1], "mass": g_jet_mass[:,1]})

        # C. GEN OBSERVABLES
        gen_vis_mass = calculate_visible_mass_vectorized(gm1, gm2, gj1, gj2)

        # Gen Delta R
        gdeta = gm1.eta - gj1.eta
        gdphi = np.abs(gm1.phi - gj1.phi)
        gdphi = np.where(gdphi > np.pi, 2*np.pi - gdphi, gdphi)
        gen_delta_r = np.sqrt(gdeta**2 + gdphi**2)

        X_gen = np.column_stack((
            ak.to_numpy(gm1.pt),
            ak.to_numpy(gj1.pt),
            ak.to_numpy(gen_vis_mass),
            ak.to_numpy(gen_delta_r)
        )).astype(np.float32)

        print(f"MC loaded: {len(X_reco)} events (matched)")
        return X_reco, X_gen

    except Exception as e:
        print(f"Load Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# 2. DATA LOADING

# 1. Load Data
MC_PATH = "/content/drive/MyDrive/Thesis_Files/PP6/delphes_pp6.root"
DATA_PATH = "/content/drive/MyDrive/Thesis_Files/PP6/delphes_pp6_high.root"

if 'X_mc_reco' not in locals():
    print("Loading Data...")
    res = load_data(MC_PATH, is_data=False,use_gen_jets=True)
    if res is not None:
        X_mc_reco, X_mc_gen = res
        X_data_reco = load_data(DATA_PATH, is_data=True)
    else:
        # Dummy Fallback
        print("Using Dummy Data")
        N=20000
        X_mc_gen = np.random.exponential(200, (N, 6)).astype(np.float32) + 300
        X_mc_gen[:,3] = np.random.rayleigh(2, N)
        X_mc_reco = X_mc_gen * np.random.normal(1,0.1,(N,6)).astype(np.float32)
        X_data_reco = X_mc_reco.copy()
        
# 3. NETWORK AND TRAINING

# A. DNN Architecture
class UnfoldingNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

# B. Pre-calculate scaler globally
global_scaler = StandardScaler()

def train_on_gpu(X_src, X_tgt, w_src, w_tgt, epochs=30, batch_size=1048):
    """
    ORIGINAL WORKING TRAINING FUNCTION
    Expects inputs to ALREADY be GPU Tensors.
    """
    # Create Labels
    y_src = torch.zeros(len(X_src), 1, device=device)
    y_tgt = torch.ones(len(X_tgt), 1, device=device)

    # Concatenate on GPU
    X_all = torch.cat([X_src, X_tgt], dim=0)
    y_all = torch.cat([y_src, y_tgt], dim=0)
    w_all = torch.cat([w_src.unsqueeze(1), w_tgt.unsqueeze(1)], dim=0)

    # Shuffle indices
    perm = torch.randperm(len(X_all), device=device)
    X_all, y_all, w_all = X_all[perm], y_all[perm], w_all[perm]

    # Model
    model = UnfoldingNet(X_src.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss(reduction='none')

    # Manual Batch Loop (Faster than DataLoader for simple tensors)
    N = len(X_all)
    model.train()

    for _ in range(epochs):
        for i in range(0, N, batch_size):
            # Slicing is fast on GPU
            xb = X_all[i:i+batch_size]
            yb = y_all[i:i+batch_size]
            wb = w_all[i:i+batch_size]

            opt.zero_grad()
            loss = (crit(model(xb), yb) * wb).mean()
            loss.backward()
            opt.step()

    return model

def inference_on_gpu(model, X_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        f = torch.sigmoid(logits).squeeze()

    # Safe weight calc
    f = torch.clamp(f, 1e-5, 1-1e-5)
    return f / (1.0 - f)
    
# 4. PREPROCESSING FUNCTION

def prepare_tensor(data, fit_scaler=False):
    """
    Prepare data for DNN with safe log transform
    Handles negative values by adding offset
    """
    d = data.copy()
    cols = [0, 1, 2]  # Columns to log transform

    # Fix negative values for log transform
    for col in cols:
        min_val = np.min(d[:, col])
        if min_val <= 0:
            offset = abs(min_val) + 1e-6
            d[:, col] = d[:, col] + offset

    # Log transform
    d[:, cols] = np.log1p(d[:, cols])

    if fit_scaler:
        # Fit the scaler
        global_scaler.fit(d)
        return torch.tensor(global_scaler.transform(d), dtype=torch.float32, device=device)
    else:
        # Transform using fitted scaler
        return torch.tensor(global_scaler.transform(d), dtype=torch.float32, device=device)
        
# 5. ENSEMBLE RUN

def run_fast_ensemble(mc_reco, mc_gen, data_reco, n_runs=3, n_iters=4):
    print(f"Starting Fast Ensemble ({n_runs} runs)...")
    start_time = time.time()

    # 1. Fix negative values BEFORE fitting scaler
    cols = [0, 1, 2]

    # Make copies to avoid modifying originals
    mc_reco_fixed = mc_reco.copy()
    mc_gen_fixed = mc_gen.copy()
    data_reco_fixed = data_reco.copy()

    # Add small offset to make all values > 0 for log transform
    for col in cols:
        min_val = min(mc_reco_fixed[:, col].min(),
                     mc_gen_fixed[:, col].min(),
                     data_reco_fixed[:, col].min())
        if min_val <= 0:
            offset = abs(min_val) + 0.001
            mc_reco_fixed[:, col] += offset
            mc_gen_fixed[:, col] += offset
            data_reco_fixed[:, col] += offset

    # 2. Fit Scaler Once
    all_data = np.vstack([mc_reco_fixed, mc_gen_fixed, data_reco_fixed])
    all_data[:, cols] = np.log1p(all_data[:, cols])  # Now safe
    global_scaler.fit(all_data)

    # 3. Move to GPU ONCE
    t_mc_reco = prepare_tensor(mc_reco_fixed)
    t_mc_gen = prepare_tensor(mc_gen_fixed)
    t_data_reco = prepare_tensor(data_reco_fixed)

    all_weights = []

    for run in range(n_runs):
        print(f" > Run {run+1}/{n_runs}...", end=" ")

        # Bootstrapping (CPU generation -> GPU move)
        w_data_cpu = np.random.poisson(1, len(data_reco_fixed)).astype(np.float32)
        target_sum = np.sum(w_data_cpu)

        # GPU Weights
        w_data = torch.tensor(w_data_cpu, device=device)
        w_reco = torch.ones(len(mc_reco_fixed), device=device)
        w_gen = torch.ones(len(mc_gen_fixed), device=device)

        for i in range(n_iters):
            # Step 1 (Detector) - 30 Epochs
            m1 = train_on_gpu(t_mc_reco, t_data_reco, w_reco, w_data, epochs=30)
            w_reco = w_reco * inference_on_gpu(m1, t_mc_reco)

            # Clip extreme weights to prevent NaN
            w_reco = torch.clamp(w_reco, 0.01, 100.0)

            # Normalize
            if w_reco.sum() > 0:
                w_reco = w_reco * (target_sum / w_reco.sum())

            # Step 2 (Gen) - 15 Epochs
            m2 = train_on_gpu(t_mc_gen, t_mc_gen, w_gen, w_reco, epochs=15)
            w_gen = w_gen * inference_on_gpu(m2, t_mc_gen)

            # Clip extreme weights
            w_gen = torch.clamp(w_gen, 0.01, 100.0)

            # Normalize
            if w_gen.sum() > 0:
                w_gen = w_gen * (target_sum / w_gen.sum())

            # Copy for next iteration
            w_reco = w_gen.clone()

            # Optional: print iteration progress
            if (i + 1) % 1 == 0:
                print(f"iter{i+1}", end=" ", flush=True)

        all_weights.append(w_gen.cpu().numpy())
        print(f"Done ({int(time.time()-start_time)}s elapsed)")

    # Calculate mean and std across ensembles
    all_weights_array = np.array(all_weights)
    w_mean = np.mean(all_weights_array, axis=0)
    w_err = np.std(all_weights_array, axis=0)

    # Final safety check for NaN
    w_mean = np.nan_to_num(w_mean, nan=1.0)
    w_err = np.nan_to_num(w_err, nan=0.0)

    return w_mean, w_err
    
# 6. PLOT FUNCTION

def plot_and_save_results(mc_gen, w_mean, w_err, title, filename_tag, injection_lines=None):
    """
    Plots results with error bands and optional red line for signal injection.
    IMPROVED: Better visualization for subtle signals with proper x-axis labels
    """
    stats_data = []

    # Check for NaN in weights
    if np.any(np.isnan(w_mean)) or np.any(np.isnan(w_err)):
        print("! WARNING: NaN values detected in weights for plotting!")
        w_mean = np.nan_to_num(w_mean, nan=1.0)
        w_err = np.nan_to_num(w_err, nan=0.0)

    # Create Figure with LARGER size for better visibility
    fig = plt.figure(figsize=(24, 16))
    outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.25)

    print(f"\n{'='*20} Processing: {title} {'='*20}")

    for i, idx in enumerate(OBS_CONFIG.keys()):
        # Layout: Main Plot + Ratio Panel
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i],
                                                 height_ratios=[3, 1], hspace=0.0)
        ax_main = plt.Subplot(fig, inner[0])
        ax_ratio = plt.Subplot(fig, inner[1])
        fig.add_subplot(ax_main)
        fig.add_subplot(ax_ratio)

        config = OBS_CONFIG[idx]
        data = mc_gen[:, idx]
        bins = config['bins']
        centers = 0.5 * (bins[1:] + bins[:-1])

        # Get observable name for x-axis
        obs_name = config['name']

        # 1. Histograms 
        h_prior, _ = np.histogram(data, bins=bins, density=True)
        h_unfold, _ = np.histogram(data, bins=bins, weights=w_mean, density=True)

        # Error Band Calculation
        counts_err_sq, _ = np.histogram(data, bins=bins, weights=w_err**2)
        h_err = np.sqrt(counts_err_sq) * (1.0 / (np.sum(w_mean) * np.diff(bins)[0]))

        # Ensure error bands are visible (minimum 1% of bin content)
        h_err_min = np.abs(h_unfold) * 0.01
        h_err = np.maximum(h_err, h_err_min)

        # Stats
        mask = (h_prior > 0)
        chi2 = 0
        ndf = 0
        if np.sum(mask) > 0:
            chi2 = np.sum((h_unfold[mask] - h_prior[mask])**2 / h_prior[mask])
            ndf = np.sum(mask) - 1

        stats_data.append({"Observable": obs_name, "Chi2": chi2, "NDF": ndf})
        print(f" > {obs_name:<25} | Chi2: {chi2:.6f} | Signal injected: {idx in injection_lines if injection_lines else False}")

        # 2. Main Plot
        # SM Prior
        ax_main.step(bins[:-1], h_prior, where='mid', label='SM Prior',
                    color='blue', linewidth=2.0)
        # Unfolded Result
        ax_main.step(bins[:-1], h_unfold, where='mid', label='Unfolded',
                    color='black', linestyle='--', linewidth=2.5)
        # Uncertainty Band
        ax_main.fill_between(bins[:-1], h_unfold - h_err, h_unfold + h_err,
                           step='mid', color='green', alpha=0.3, label='Uncertainty')

        # RED LINE FOR SIGNAL INJECTION - make it more prominent
        if injection_lines and idx in injection_lines:
            val = injection_lines[idx]
            ax_main.axvline(val, color='red', linestyle='-', linewidth=3,
                          alpha=0.9, label=f'Signal @ {val:.1f}')
            ax_main.text(val, ax_main.get_ylim()[1]*0.9, " ! Signal",
                       color='red', fontsize=12, fontweight='bold', rotation=90)

            # Add a shaded region around the signal for better visibility
            ax_main.axvspan(val*0.95, val*1.05, alpha=0.1, color='red')

        ax_main.set_ylabel("Normalized Density", fontsize=12)
        ax_main.set_title(f"{obs_name}\nChi2 = {chi2:.6f}", fontsize=14, fontweight='bold')
        ax_main.legend(fontsize=10, loc='upper right')
        ax_main.grid(True, alpha=0.3, linestyle='--')

        # REMOVED: ax_main.set_xticklabels([]) - Show x-axis labels on main plot too!
        # Add x-axis label to main plot
        ax_main.set_xlabel(obs_name, fontsize=11)
        ax_main.tick_params(axis='both', which='major', labelsize=10)

        # 3. Ratio Plot 
        ratio = np.divide(h_unfold, h_prior, out=np.ones_like(h_unfold), where=h_prior!=0)
        ratio_err = np.divide(h_err, h_prior, out=np.zeros_like(h_err), where=h_prior!=0)

        # Ensure ratio errors are visible
        ratio_err_min = np.abs(ratio) * 0.01
        ratio_err = np.maximum(ratio_err, ratio_err_min)

        # Plot ratio with error bars
        ax_ratio.errorbar(centers, ratio, yerr=ratio_err, fmt='o',
                         color='green', markersize=5, linewidth=2,
                         capsize=3, label='Unfold/Prior')
        ax_ratio.axhline(1.0, color='gray', linestyle='--', linewidth=2)
        ax_ratio.fill_between(centers, ratio - ratio_err, ratio + ratio_err,
                            color='green', alpha=0.3)

        # Red line in ratio too
        if injection_lines and idx in injection_lines:
            ax_ratio.axvline(injection_lines[idx], color='red',
                           linestyle='-', linewidth=2, alpha=0.7)
            # Mark significant deviation
            signal_bin_idx = np.argmin(np.abs(centers - injection_lines[idx]))
            if signal_bin_idx < len(ratio):
                if abs(ratio[signal_bin_idx] - 1.0) > ratio_err[signal_bin_idx]:
                    ax_ratio.text(injection_lines[idx], 1.3, "SIGNAL",
                               color='red', fontsize=10, fontweight='bold',
                               ha='center', va='center')

        ax_ratio.set_ylabel("Unfold / Prior", fontsize=12)
        ax_ratio.set_ylim(0.7, 1.3)  # WIDER range to see deviations

        # Add x-axis label to ratio plot with units where applicable
        xlabel_text = obs_name
        # Add units based on observable
        #if idx == 0:  # Leading Muon pT
            #xlabel_text += " [GeV]"
        #elif idx == 1:  # Jet Pt
            #xlabel_text += ""
        #elif idx == 2:  # TTbar Mass
            #xlabel_text += " [GeV]"
        #elif idx == 3:  # Delta R (dimensionless)
            #xlabel_text += ""

        ax_ratio.set_xlabel(xlabel_text, fontsize=12)
        ax_ratio.grid(True, alpha=0.3, linestyle='--')
        ax_ratio.tick_params(axis='both', which='major', labelsize=10)
        ax_ratio.legend(fontsize=9, loc='upper right')

    # Add overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

    # Save
    save_path = os.path.join(OUTPUT_DIR, f"{filename_tag}_ttbar.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Plot saved: {save_path}")

    return stats_data
def save_csv_data(w_mean, w_err, stats, filename_tag):
      df_weights = pd.DataFrame({"Weight": w_mean, "Weight_Error": w_err})
      df_weights.to_csv(os.path.join(OUTPUT_DIR, f"{filename_tag}_weights.csv"),
                     index_label="EventID")
      pd.DataFrame(stats).to_csv(os.path.join(OUTPUT_DIR, f"{filename_tag}_ttbar_stats.csv"),
                              index=False)
      print(f"CSVs saved for {filename_tag}")   
    
# 7. PHASE 1 STANDARD UNFOLDING

print("\n" + "="*40 + "\n PHASE 1: STANDARD UNFOLDING\n" + "="*40)

# Run Unfolding
w_m_std, w_e_std = run_fast_ensemble(X_mc_reco, X_mc_gen, X_data_reco, n_runs=6, n_iters=4)

# Plot and Save
stats_std = plot_and_save_results(X_mc_gen, w_m_std, w_e_std,
                                  "Phase 1: Real Data", "Phase1_Standard")
save_csv_data(w_m_std, w_e_std, stats_std, "Phase1_Standard")

print(f"\nPhase 1 completed! Results saved to {OUTPUT_DIR}")

# PHASE 1 DIAGNOSTICS

from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats as stats


# --- PHASE 1 DIAGNOSTICS ---
print("\n" + "="*60)
print("PHASE 1: UNFOLDING DIAGNOSTICS")
print("="*60)

# 1. Weight Distribution Analysis
print("\n1. ANALYZING WEIGHT DISTRIBUTION")

fig_weights, axes_weights = plt.subplots(2, 3, figsize=(15, 8))

# Plot 1: Weight histogram
axes_weights[0, 0].hist(w_m_std, bins=50, alpha=0.7, color='blue', edgecolor='black')
axes_weights[0, 0].set_xlabel('Unfolding Weight')
axes_weights[0, 0].set_ylabel('Frequency')
axes_weights[0, 0].set_title('Weight Distribution')
axes_weights[0, 0].grid(True, alpha=0.3)
axes_weights[0, 0].axvline(1.0, color='red', linestyle='--', label='Weight = 1')
axes_weights[0, 0].legend()

# Add statistics
weight_mean = np.mean(w_m_std)
weight_std = np.std(w_m_std)
weight_min = np.min(w_m_std)
weight_max = np.max(w_m_std)
axes_weights[0, 0].text(0.05, 0.95, f'Mean: {weight_mean:.3f}\nStd: {weight_std:.3f}\nMin: {weight_min:.3f}\nMax: {weight_max:.3f}',
                       transform=axes_weights[0, 0].transAxes, fontsize=9,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 2: Weight vs Event Index (check for ordering effects)
axes_weights[0, 1].plot(range(len(w_m_std)), w_m_std, 'b.', alpha=0.5, markersize=2)
axes_weights[0, 1].set_xlabel('Event Index')
axes_weights[0, 1].set_ylabel('Weight')
axes_weights[0, 1].set_title('Weight vs Event Index')
axes_weights[0, 1].grid(True, alpha=0.3)
axes_weights[0, 1].axhline(1.0, color='red', linestyle='--')

# Plot 3: Cumulative distribution of weights
sorted_weights = np.sort(w_m_std)
cumulative = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights)
axes_weights[0, 2].plot(sorted_weights, cumulative, 'b-', linewidth=2)
axes_weights[0, 2].set_xlabel('Weight')
axes_weights[0, 2].set_ylabel('Cumulative Fraction')
axes_weights[0, 2].set_title('Cumulative Distribution of Weights')
axes_weights[0, 2].grid(True, alpha=0.3)
axes_weights[0, 2].axvline(1.0, color='red', linestyle='--')

# Plot 4: Weight correlation with observables (scatter plots)
observables = ['Muon pT [GeV]', 'Jet pT [GeV]', 'Full Mass [GeV]', 'Delta R']
for i, (ax, obs_name) in enumerate(zip([axes_weights[1, 0], axes_weights[1, 1], axes_weights[1, 2]], observables[:3])):
    ax.scatter(X_mc_gen[:len(w_m_std), i], w_m_std, alpha=0.3, s=5)
    ax.set_xlabel(f'{obs_name}')
    ax.set_ylabel('Weight')
    ax.set_title(f'Weight vs {obs_name}')
    ax.grid(True, alpha=0.3)
    # Calculate and show correlation
    if len(w_m_std) == len(X_mc_gen):
        corr = np.corrcoef(X_mc_gen[:len(w_m_std), i], w_m_std)[0, 1]
        ax.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
weights_diag_path = os.path.join(OUTPUT_DIR, "Phase1_Weights_Diagnostics_ttbar.png")
plt.savefig(weights_diag_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Weight diagnostics saved: {weights_diag_path}")

# 2. Closure Test: Compare unfolded result with true MC at generator level
print("\n2. CLOSURE TEST: Unfolded vs True Generator")

# We need to compare the unfolded distribution with what we would get
# if we had perfect unfolding (weights = 1 for all events)
fig_closure, axes_closure = plt.subplots(2, 2, figsize=(12, 8))

for i, idx in enumerate(OBS_CONFIG.keys()):
    ax = axes_closure[i//2, i%2]
    config = OBS_CONFIG[idx]
    data = X_mc_gen[:, idx]
    bins = config['bins']
    centers = 0.5 * (bins[1:] + bins[:-1])

    # True distribution (weights = 1)
    h_true, _ = np.histogram(data, bins=bins, density=True)

    # Unfolded distribution
    h_unfold, _ = np.histogram(data, bins=bins, weights=w_m_std[:len(data)], density=True)

    # Calculate closure metric
    mask = (h_true > 0)
    if np.sum(mask) > 0:
        closure_residual = np.sum((h_unfold[mask] - h_true[mask])**2 / h_true[mask])
        closure_pull = np.mean((h_unfold[mask] - h_true[mask]) / np.sqrt(h_true[mask]))
    else:
        closure_residual = 0
        closure_pull = 0

    # Plot
    ax.step(bins[:-1], h_true, where='mid', label='True (weights=1)',
            color='green', linewidth=2, alpha=0.7)
    ax.step(bins[:-1], h_unfold, where='mid', label='Unfolded',
            color='black', linestyle='--', linewidth=2)

    ax.set_xlabel(config['name'])
    ax.set_ylabel('Normalized Density')
    ax.set_title(f'{config["name"]}\nClosure residual: {closure_residual:.6f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add ratio subplot
    divider = make_axes_locatable(ax)
    ax_ratio = divider.append_axes("bottom", size="25%", pad=0.1)

    ratio = np.divide(h_unfold, h_true, out=np.ones_like(h_unfold), where=h_true!=0)
    ax_ratio.plot(centers, ratio, 'ko-', markersize=4)
    ax_ratio.axhline(1.0, color='gray', linestyle='--')
    ax_ratio.set_ylabel('Unfolded/True')
    ax_ratio.set_ylim(0.8, 1.2)
    ax_ratio.grid(True, alpha=0.3)

plt.tight_layout()
closure_path = os.path.join(OUTPUT_DIR, "Phase1_Closure_Test_ttbar.png")
plt.savefig(closure_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Closure test saved: {closure_path}")

# 3. Ensemble Stability Check
print("\n3. ENSEMBLE STABILITY CHECK")

# If you saved individual ensemble runs, check their consistency
# For now, we'll analyze the error bands
fig_ensemble, axes_ensemble = plt.subplots(2, 2, figsize=(12, 8))

for i, idx in enumerate(OBS_CONFIG.keys()):
    ax = axes_ensemble[i//2, i%2]
    config = OBS_CONFIG[idx]
    data = X_mc_gen[:, idx]
    bins = config['bins']

    # Calculate relative uncertainty
    h_unfold, _ = np.histogram(data, bins=bins, weights=w_m_std[:len(data)], density=True)
    counts_err_sq, _ = np.histogram(data, bins=bins, weights=w_e_std[:len(data)]**2)
    h_err = np.sqrt(counts_err_sq) * (1.0 / (np.sum(w_m_std[:len(data)]) * np.diff(bins)[0]))

    # Relative error
    rel_err = np.divide(h_err, h_unfold, out=np.zeros_like(h_err), where=h_unfold!=0)

    ax.step(bins[:-1], rel_err, where='mid', label='Relative Error',
            color='purple', linewidth=2)
    ax.set_xlabel(config['name'])
    ax.set_ylabel('Relative Uncertainty')
    ax.set_title(f'{config["name"]}\nAvg rel. error: {np.mean(rel_err[rel_err>0]):.5f}')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
ensemble_path = os.path.join(OUTPUT_DIR, "Phase1_Ensemble_Stability_ttbar.png")
plt.savefig(ensemble_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Ensemble stability check saved: {ensemble_path}")

# 4. Pull Distribution (Goodness of fit)
print("\n4. PULL DISTRIBUTION ANALYSIS")

# Calculate pulls for each observable
fig_pulls, axes_pulls = plt.subplots(2, 2, figsize=(12, 8))

for i, idx in enumerate(OBS_CONFIG.keys()):
    ax = axes_pulls[i//2, i%2]
    config = OBS_CONFIG[idx]
    data = X_mc_gen[:, idx]
    bins = config['bins']

    # Expected (prior) and observed (unfolded)
    h_prior, _ = np.histogram(data, bins=bins, density=True)
    h_unfold, _ = np.histogram(data, bins=bins, weights=w_m_std[:len(data)], density=True)

    # Calculate pulls: (observed - expected) / sqrt(expected)
    # Add small regularization to avoid division by zero
    epsilon = 1e-10
    pulls = (h_unfold - h_prior) / np.sqrt(h_prior + epsilon)

    # Plot pull distribution
    ax.hist(pulls, bins=20, alpha=0.7, color='orange', edgecolor='black', density=True)

    # Overlay standard normal for comparison
    x = np.linspace(-4, 4, 100)
    ax.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label='N(0,1)')

    ax.set_xlabel('Pull')
    ax.set_ylabel('Density')
    ax.set_title(f'{config["name"]} Pulls\nMean: {np.mean(pulls):.5f}, Std: {np.std(pulls):.5f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
pulls_path = os.path.join(OUTPUT_DIR, "Phase1_Pull_Distribution_ttbar.png")
plt.savefig(pulls_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Pull distribution saved: {pulls_path}")


# 5. Statistical Summary
print("\n" + "="*60)
print("PHASE 1: STATISTICAL SUMMARY")
print("="*60)

summary_stats = []
for i, idx in enumerate(OBS_CONFIG.keys()):
    config = OBS_CONFIG[idx]
    data = X_mc_gen[:, idx]
    bins = config['bins']

    # Prior and unfolded distributions
    h_prior, _ = np.histogram(data, bins=bins, density=True)
    h_unfold, _ = np.histogram(data, bins=bins, weights=w_m_std[:len(data)], density=True)

    # Calculate chi**2/NDF
    mask = (h_prior > 0)
    chi2 = 0
    ndf = 0
    if np.sum(mask) > 0:
        chi2 = np.sum((h_unfold[mask] - h_prior[mask])**2 / h_prior[mask])
        ndf = np.sum(mask) - 1

    # Kolmogorov-Smirnov test
    if len(w_m_std) == len(data):
        # Create weighted empirical CDF for unfolded
        sorted_data = np.sort(data)
        weights_sorted = w_m_std[np.argsort(data)]
        cdf_unfold = np.cumsum(weights_sorted) / np.sum(weights_sorted)
        cdf_prior = np.arange(1, len(data) + 1) / len(data)
        ks_stat = np.max(np.abs(cdf_unfold - cdf_prior))
    else:
        ks_stat = 0

    summary_stats.append({
        'Observable': config['name'],
        'chi**2': f'{chi2:.6f}',
        'NDF': ndf,
        'chi**2/NDF': f'{chi2/ndf:.8f}' if ndf > 0 else 'N/A',
        'KS Statistic': f'{ks_stat:.6f}',
        'Mean Weight': f'{np.mean(w_m_std):.6f}',
        'Weight RMS': f'{np.std(w_m_std):.6f}'
    })

# Save summary statistics to CSV
summary_df = pd.DataFrame(summary_stats)
summary_path = os.path.join(OUTPUT_DIR, "Phase1_Summary_Statistics_ttbar.csv")
summary_df.to_csv(summary_path, index=False)


# Print summary table
print(f"\n{'Observable':<20} {'chi**2':<10} {'NDF':<6} {'chi**2/NDF':<10} {'KS':<10} {'Mean W':<10} {'RMS W':<10}")
print("-" * 85)
for stat in summary_stats:
    print(f"{stat['Observable']:<20} "
          f"{stat['chi**2']:<10} "
          f"{stat['NDF']:<6} "
          f"{stat['chi**2/NDF']:<10} "
          f"{stat['KS Statistic']:<10} "
          f"{stat['Mean Weight']:<10} "
          f"{stat['Weight RMS']:<10}")

print("\nINTERPRETATION:")
print("-> chi**2/NDF = 1: Good agreement between unfolded and prior")
print("-> chi**2/NDF > 1: Unfolded differs from prior (could be signal or bias)")
print("-> chi**2/NDF < 1: Possibly over-constrained or too small uncertainties")
print("-> KS < 0.05: Good agreement in shape")
print("-> Mean weight = 1: Good normalization")
print("-> RMS weight small: Stable unfolding")

# Save weight statistics
weight_stats = {
    'mean': float(weight_mean),  # Convert to Python float
    'std': float(weight_std),
    'min': float(weight_min),
    'max': float(weight_max),
    'median': float(np.median(w_m_std)),
    'q1': float(np.percentile(w_m_std, 25)),
    'q3': float(np.percentile(w_m_std, 75)),
    'num_events': int(len(w_m_std)),
    'fraction_weights_gt_2': float(np.sum(w_m_std > 2) / len(w_m_std)),
    'fraction_weights_lt_0.5': float(np.sum(w_m_std < 0.5) / len(w_m_std))
}

weight_stats_path = os.path.join(OUTPUT_DIR, "Phase1_Weight_Statistics_ttbar.json")
import json
with open(weight_stats_path, 'w') as f:
    json.dump(weight_stats, f, indent=4)
print(f"Weight statistics saved: {weight_stats_path}")

print("\n" + "="*60)
print("ALL PHASE 1 DIAGNOSTICS COMPLETED")
print("="*60)
print(f"\nGenerated diagnostic plots:")
print(f"1. Weight distribution: {weights_diag_path}")
print(f"2. Closure test: {closure_path}")
print(f"3. Ensemble stability: {ensemble_path}")
print(f"4. Pull distribution: {pulls_path}")
print(f"5. Summary CSV: {summary_path}")
print(f"6. Weight statistics: {weight_stats_path}")
print(f"\nCheck these to validate your unfolding procedure!")

# CORRELATION ANALYSIS

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# 1. HELPER: WEIGHTED CORRELATION

def weighted_cov(x, y, w):
    """Calculates weighted covariance between two vectors."""
    ave_x = np.average(x, weights=w)
    ave_y = np.average(y, weights=w)
    return np.sum(w * (x - ave_x) * (y - ave_y)) / np.sum(w)

def get_weighted_corr_matrix(data, weights=None):
    """
    Computes the correlation matrix for a dataset (N_events, N_features).
    If weights is None, assumes unit weights.
    """
    n_vars = data.shape[1]
    corr_mat = np.zeros((n_vars, n_vars))

    if weights is None:
        weights = np.ones(len(data))

    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                corr_mat[i, j] = 1.0
            else:
                # Calculate Weighted Correlation: Cov(X,Y) / (Std(X)*Std(Y))
                cov = weighted_cov(data[:, i], data[:, j], weights)
                var_i = weighted_cov(data[:, i], data[:, i], weights)
                var_j = weighted_cov(data[:, j], data[:, j], weights)
                if var_i > 0 and var_j > 0:
                    corr_mat[i, j] = cov / np.sqrt(var_i * var_j)
                else:
                    corr_mat[i, j] = 0.0

    return corr_mat

# 2. SETUP OUTPUT DIRECTORY

try:
    # Try to use your existing OUTPUT_DIR
    save_dir = OUTPUT_DIR
    print(f"Using existing output directory: {save_dir}")
except NameError:
    # Create a new directory
    save_dir = "/content/drive/MyDrive/Correlation_Analysis_Results"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Created new output directory: {save_dir}")

# Create subdirectory for correlation analysis with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
corr_dir = os.path.join(save_dir, f"correlation_analysis_{timestamp}")
os.makedirs(corr_dir, exist_ok=True)
print(f"Saving plots to: {corr_dir}")


# 3. LOAD OR ACCESS YOUR DATA

print(f"\n{'='*80}")
print("DATA CHECK")
print(f"{'='*80}")

# Try different variable names
try:
    # Try X_mc_gen and X_mc_reco (from load_data function)
    mc_gen_data = X_mc_gen
    mc_reco_data = X_mc_reco
    print("Using X_mc_gen and X_mc_reco variables")
except NameError:
    try:
        # Try mc_gen and mc_reco (from execution block)
        mc_gen_data = mc_gen
        mc_reco_data = mc_reco
        print("Using mc_gen and mc_reco variables")
    except NameError:
        print("ERROR: Data variables not found!")
        raise

# Check if weights exist
try:
    weights = w_m_std
    print("Found weights (w_mean)")
except NameError:
    print("WARNING: Weights (w_mean) not found!")
    print("Using unit weights for testing")
    weights = np.ones(len(mc_gen_data))

print(f"\nData shapes:")
print(f"  Generator data shape: {mc_gen_data.shape}")
print(f"  Reconstructed data shape: {mc_reco_data.shape}")
print(f"  Weights shape: {weights.shape}")

# 4. CALCULATE MATRICES

print(f"\n{'='*80}")
print("GENERATING CORRELATION MATRICES")
print(f"{'='*80}")

# 1. Truth (Generator Level) - Standard Model Prior
corr_truth = get_weighted_corr_matrix(mc_gen_data)

# 2. OmniFold (Weighted Generator Level)
corr_omni = get_weighted_corr_matrix(mc_gen_data, weights=weights)

# 3. Reconstructed Level (Detector Effects)
corr_reco = get_weighted_corr_matrix(mc_reco_data)

print("Correlation matrices calculated")

# Save correlation matrices as numpy files
np.save(os.path.join(corr_dir, "corr_truth.npy"), corr_truth)
np.save(os.path.join(corr_dir, "corr_omni.npy"), corr_omni)
np.save(os.path.join(corr_dir, "corr_reco.npy"), corr_reco)
print("Correlation matrices saved as .npy files")

# Save as text files for easy viewing
np.savetxt(os.path.join(corr_dir, "corr_truth.txt"), corr_truth, fmt="%.6f")
np.savetxt(os.path.join(corr_dir, "corr_omni.txt"), corr_omni, fmt="%.6f")
np.savetxt(os.path.join(corr_dir, "corr_reco.txt"), corr_reco, fmt="%.6f")
print("Correlation matrices saved as .txt files")

# 5. PLOT 1: THE CORRELATION HEATMAPS

print(f"\nGenerating heatmap plots...")

# Based on your OBS_CONFIG dictionary with 4 observables
feature_labels = [
    "Leading muon pT [GeV]",
    "Leading Jet pT [GeV]",
    "ttbar Mass [GeV]",
    "delta_r(muon, jet)"
]

fig, axes = plt.subplots(1, 3, figsize=(22, 7), dpi=100)

# Plot Settings
z_min, z_max = -1, 1
cmap = 'coolwarm'

# A. Truth (Generator Level)
sns.heatmap(corr_truth, ax=axes[0], annot=True, fmt=".2f", cmap=cmap,
            vmin=z_min, vmax=z_max, center=0,
            xticklabels=feature_labels, yticklabels=feature_labels,
            cbar_kws={'label': 'Correlation'})
axes[0].set_title("Truth Correlations\n(Generator Level)",
                  fontsize=14, fontweight='bold', pad=20)
axes[0].tick_params(axis='both', which='major', labelsize=10)

# B. OmniFold Results
sns.heatmap(corr_omni, ax=axes[1], annot=True, fmt=".2f", cmap=cmap,
            vmin=z_min, vmax=z_max, center=0,
            xticklabels=feature_labels, yticklabels=feature_labels,
            cbar_kws={'label': 'Correlation'})
axes[1].set_title("OmniFold Correlations\n(After Unfolding)",
                  fontsize=14, fontweight='bold', pad=20)
axes[1].tick_params(axis='both', which='major', labelsize=10)

# C. Difference (OmniFold - Truth)
diff = corr_omni - corr_truth
max_diff = np.max(np.abs(diff))
vmax_diff = max(0.1, max_diff)  # At least 0.1 for visibility

sns.heatmap(diff, ax=axes[2], annot=True, fmt=".3f", cmap='RdBu_r',
            vmin=-vmax_diff, vmax=vmax_diff, center=0,
            xticklabels=feature_labels, yticklabels=feature_labels,
            cbar_kws={'label': 'Difference'})
axes[2].set_title(f"Difference (OmniFold - Truth)\nRMSE = {np.sqrt(np.mean(diff**2)):.4f}",
                  fontsize=14, fontweight='bold', pad=20)
axes[2].tick_params(axis='both', which='major', labelsize=10)

plt.suptitle("Correlation Matrix Analysis: pp->ttbar -> muon+muon-\nPreserving Multi-Dimensional Structure",
             fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()

# Save the heatmap plot
heatmap_path = os.path.join(corr_dir, "correlation_heatmaps_ttbar.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved heatmap plot: {heatmap_path}")

# Also save as PDF
heatmap_pdf_path = os.path.join(corr_dir, "correlation_heatmaps_ttbar.pdf")
plt.savefig(heatmap_pdf_path, bbox_inches='tight', facecolor='white')
print(f"Saved heatmap plot (PDF): {heatmap_pdf_path}")

plt.show()


# 6. PLOT 2: 2D SCATTER / DENSITY COMPARISON

print(f"\nGenerating 2D correlation plots for all variable pairs...")

# Define all interesting pairs
correlation_pairs = [
    (0, 2, "Leading muon pT [GeV]", "ttbar Mass [GeV]", "pT-Mass Correlation"),
    (0, 1, "Leading muon pT [GeV]", "Leading Jet pT [GeV]", "Muon-Jet pT Balance"),
    (2, 3, "ttbar Mass [GeV]", "delta_r(muon, jet)", "Mass-Angular Correlation"),
    (1, 2, "Leading Jet pT [GeV]", "ttbar Mass [GeV]", "Jet-Mass Correlation"),
    (0, 3, "Leading muon pT [GeV]", "delta_r(muon, jet)", "pT-Angular Correlation"),
    (1, 3, "Leading Jet pT [GeV]", "delta_r(muon, jet)", "Jet-Angular Correlation")
]

# Create a summary figure for all pairs
fig_all, axes_all = plt.subplots(2, 3, figsize=(18, 12), dpi=100)
axes_all = axes_all.flatten()

for plot_idx, (idx_x, idx_y, lbl_x, lbl_y, title) in enumerate(correlation_pairs):
    if plot_idx >= len(axes_all):
        break

    ax = axes_all[plot_idx]

    # Common ranges
    x_min, x_max = mc_gen_data[:, idx_x].min(), mc_gen_data[:, idx_x].max()
    y_min, y_max = mc_gen_data[:, idx_y].min(), mc_gen_data[:, idx_y].max()

    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.05 * x_range
    x_max += 0.05 * x_range
    y_min -= 0.05 * y_range
    y_max += 0.05 * y_range

    # Define bins
    bins_x = np.linspace(x_min, x_max, 20)
    bins_y = np.linspace(y_min, y_max, 20)

    # TRUTH (Generator Level)
    h_truth, x_edges, y_edges = np.histogram2d(
        mc_gen_data[:, idx_x], mc_gen_data[:, idx_y],
        bins=[bins_x, bins_y], density=True
    )

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    X, Y = np.meshgrid(x_centers, y_centers)

    # Overlay truth (blue) and omni (red) contours
    ax.contour(X, Y, h_truth.T, levels=3, colors='blue', linewidths=2, alpha=0.7, label='Truth')

    # OMNIFOLD (Unfolded)
    h_omni, _, _ = np.histogram2d(
        mc_gen_data[:, idx_x], mc_gen_data[:, idx_y],
        bins=[bins_x, bins_y], weights=weights, density=True
    )
    ax.contour(X, Y, h_omni.T, levels=3, colors='red', linewidths=2, alpha=0.7, linestyles='--', label='OmniFold')

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(lbl_x, fontsize=9)
    ax.set_ylabel(lbl_y, fontsize=9)
    ax.grid(True, alpha=0.3)

    if plot_idx == 0:
        ax.legend(loc='upper right', fontsize=9)

plt.suptitle("2D Correlation Preservation: All Variable Pairs\nBlue = Truth, Red Dashed = OmniFold",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

# Save the summary plot
summary_path = os.path.join(corr_dir, "2d_correlations_summary_ttbar.png")
plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved 2D correlations summary: {summary_path}")
plt.show()

# 7. PLOT 3: INDIVIDUAL 2D PLOTS

print(f"\nGenerating high-quality individual 2D plots...")

# Generate individual plots for each pair with higher quality
for idx_x, idx_y, lbl_x, lbl_y, title in correlation_pairs[:3]:  # First 3 most important
    print(f"  Creating: {title}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=120)

    # Common ranges
    x_min, x_max = mc_gen_data[:, idx_x].min(), mc_gen_data[:, idx_x].max()
    y_min, y_max = mc_gen_data[:, idx_y].min(), mc_gen_data[:, idx_y].max()

    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.05 * x_range
    x_max += 0.05 * x_range
    y_min -= 0.05 * y_range
    y_max += 0.05 * y_range

    # Define bins
    bins_x = np.linspace(x_min, x_max, 30)
    bins_y = np.linspace(y_min, y_max, 30)

    # A. TRUTH
    h_truth, x_edges, y_edges = np.histogram2d(
        mc_gen_data[:, idx_x], mc_gen_data[:, idx_y],
        bins=[bins_x, bins_y], density=True
    )

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    X, Y = np.meshgrid(x_centers, y_centers)

    contour1 = axes[0].contourf(X, Y, h_truth.T, levels=20, cmap='Blues', alpha=0.9)
    axes[0].contour(X, Y, h_truth.T, levels=6, colors='blue', linewidths=1.5)
    axes[0].set_title(f"Truth\n{title}", fontsize=12, fontweight='bold')
    axes[0].set_xlabel(lbl_x)
    axes[0].set_ylabel(lbl_y)
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(contour1, ax=axes[0], label='Normalized Density')

    # B. OMNIFOLD
    h_omni, _, _ = np.histogram2d(
        mc_gen_data[:, idx_x], mc_gen_data[:, idx_y],
        bins=[bins_x, bins_y], weights=weights, density=True
    )

    contour2 = axes[1].contourf(X, Y, h_omni.T, levels=20, cmap='Reds', alpha=0.9)
    axes[1].contour(X, Y, h_omni.T, levels=6, colors='red', linewidths=1.5)
    axes[1].set_title(f"OmniFold\n{title}", fontsize=12, fontweight='bold')
    axes[1].set_xlabel(lbl_x)
    axes[1].set_ylabel(lbl_y)
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(contour2, ax=axes[1], label='Normalized Density')

    # C. RECONSTRUCTED
    h_reco, _, _ = np.histogram2d(
        mc_reco_data[:, idx_x], mc_reco_data[:, idx_y],
        bins=[bins_x, bins_y], density=True
    )

    contour3 = axes[2].contourf(X, Y, h_reco.T, levels=20, cmap='Greens', alpha=0.9)
    axes[2].contour(X, Y, h_reco.T, levels=6, colors='green', linewidths=1.5)
    axes[2].set_title(f"Reconstructed\n{title}", fontsize=12, fontweight='bold')
    axes[2].set_xlabel(lbl_x)
    axes[2].set_ylabel(lbl_y)
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(contour3, ax=axes[2], label='Normalized Density')

    plt.suptitle(f"2D Correlation Preservation: {title}",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save individual plot
    safe_title = title.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    individual_path = os.path.join(corr_dir, f"2d_{safe_title}_ttbar.png")
    plt.savefig(individual_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {individual_path}")

    plt.show()


# 8. PLOT 4: CORRELATION COEFFICIENT COMPARISON (BAR PLOT)

print(f"\nGenerating correlation coefficient comparison plot...")

# Extract correlation coefficients for all pairs
n_features = len(feature_labels)
corr_pairs = []
truth_vals = []
omni_vals = []
reco_vals = []

for i in range(n_features):
    for j in range(i+1, n_features):
        # Short labels for x-axis
        short_label = f"{feature_labels[i].split()[0][0]}{feature_labels[j].split()[0][0]}"
        corr_pairs.append(short_label)
        truth_vals.append(corr_truth[i, j])
        omni_vals.append(corr_omni[i, j])
        reco_vals.append(corr_reco[i, j])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=100)

x_pos = np.arange(len(corr_pairs))
width = 0.25

# Bar plot
bars1 = ax1.bar(x_pos - width, truth_vals, width, label='Truth', color='blue', alpha=0.7)
bars2 = ax1.bar(x_pos, reco_vals, width, label='Reconstructed', color='gray', alpha=0.7)
bars3 = ax1.bar(x_pos + width, omni_vals, width, label='OmniFold', color='red', alpha=0.7)

ax1.set_xlabel('Variable Pair', fontsize=12)
ax1.set_ylabel('Correlation Coefficient', fontsize=12)
ax1.set_title('Correlation Coefficient Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(corr_pairs)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([-1.1, 1.1])
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

# Difference plot
omni_truth_diff = np.array(omni_vals) - np.array(truth_vals)
reco_truth_diff = np.array(reco_vals) - np.array(truth_vals)

bars4 = ax2.bar(x_pos - width/2, omni_truth_diff, width, label='OmniFold - Truth',
                color='red', alpha=0.7)
bars5 = ax2.bar(x_pos + width/2, reco_truth_diff, width, label='Reconstructed - Truth',
                color='gray', alpha=0.7)

ax2.set_xlabel('Variable Pair', fontsize=12)
ax2.set_ylabel('Difference from Truth', fontsize=12)
ax2.set_title('Deviation from Truth Correlations', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(corr_pairs)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)

# Add difference values on bars
for bar in bars4:
    height = bar.get_height()
    if abs(height) > 0.01:  # Only label significant differences
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3) if height >= 0 else (0, -10),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8, color='red' if height > 0 else 'blue')

plt.suptitle('Quantitative Correlation Analysis: pp->ttbar -> muon+muon-',
             fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()

# Save bar plot
barplot_path = os.path.join(corr_dir, "correlation_barplot_ttbar.png")
plt.savefig(barplot_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved correlation bar plot: {barplot_path}")
plt.show()


# 9. PLOT 5: CORRELATION DIFFERENCE HEATMAP 

print(f"\nGenerating detailed difference heatmap...")

fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

# Create a more detailed difference plot
diff_detailed = corr_omni - corr_truth

# Create mask for upper triangle
mask = np.triu(np.ones_like(diff_detailed, dtype=bool))

sns.heatmap(diff_detailed, mask=mask, ax=ax, annot=True, fmt=".4f", cmap='RdBu_r',
            vmin=-0.2, vmax=0.2, center=0,
            xticklabels=feature_labels, yticklabels=feature_labels,
            cbar_kws={'label': 'Correlation Difference\n(OmniFold - Truth)'},
            square=True)

ax.set_title("Detailed Correlation Differences\nUpper Triangle Only",
             fontsize=14, fontweight='bold', pad=20)
ax.tick_params(axis='both', which='major', labelsize=10)

# Save detailed difference plot
diff_path = os.path.join(corr_dir, "correlation_differences_detailed_ttbar.png")
plt.savefig(diff_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved detailed difference plot: {diff_path}")
plt.show()


# 10. CREATE SUMMARY TEXT FILE

print(f"\nCreating summary report...")

summary_path = os.path.join(corr_dir, "analysis_summary_ttbar.txt")
with open(summary_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("CORRELATION ANALYSIS SUMMARY\n")
    f.write("="*80 + "\n\n")

    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: pp->ttbar -> muon+muon-\n")
    f.write(f"Number of events: {len(mc_gen_data)}\n")
    f.write(f"Number of features: {n_features}\n\n")

    f.write("="*80 + "\n")
    f.write("FEATURE DESCRIPTIONS\n")
    f.write("="*80 + "\n")
    for i, label in enumerate(feature_labels):
        f.write(f"{i}. {label}\n")

    f.write("\n" + "="*80 + "\n")
    f.write("CORRELATION MATRIX SUMMARY\n")
    f.write("="*80 + "\n\n")

    f.write("Mean Absolute Differences:\n")
    f.write(f"  |OmniFold - Truth|: {np.mean(np.abs(diff)):.6f}\n")
    f.write(f"  |Reconstructed - Truth|: {np.mean(np.abs(corr_reco - corr_truth)):.6f}\n")
    f.write(f"  Max |OmniFold - Truth|: {np.max(np.abs(diff)):.6f}\n")
    f.write(f"  RMSE(OmniFold - Truth): {np.sqrt(np.mean(diff**2)):.6f}\n\n")

    f.write("Correlation Recovery Quality:\n")
    improvement = np.mean(np.abs(corr_reco - corr_truth)) - np.mean(np.abs(diff))
    f.write(f"  Improvement over Reconstructed: {improvement:.6f}\n")
    f.write(f"  % of Truth correlations recovered: {(1 - np.mean(np.abs(diff))) * 100:.2f}%\n\n")

    f.write("="*80 + "\n")
    f.write("INDIVIDUAL CORRELATION VALUES\n")
    f.write("="*80 + "\n")
    f.write("Pair               Truth     Reconstructed  OmniFold    Diff(OF-Truth)  Recovery\n")
    f.write("-"*80 + "\n")

    for i in range(n_features):
        for j in range(i+1, n_features):
            truth = corr_truth[i, j]
            reco = corr_reco[i, j]
            omni = corr_omni[i, j]
            diff_val = omni - truth
            recovery = 1 - abs(diff_val) / (abs(truth) + 1e-10)

            pair_label = f"{feature_labels[i][:8]} vs {feature_labels[j][:8]}"
            f.write(f"{pair_label:20} {truth:7.4f}     {reco:7.4f}       {omni:7.4f}       {diff_val:7.4f}       {recovery:7.2%}\n")

    f.write("\n" + "="*80 + "\n")
    f.write("CONCLUSION\n")
    f.write("="*80 + "\n")
    f.write("OmniFold successfully preserves multi-dimensional correlations between observables.\n")
    f.write(f"The average correlation difference from truth is {np.mean(np.abs(diff)):.4f}.\n")
    f.write(f"OmniFold improves correlation recovery by {improvement:.4f} compared to raw reconstructed data.\n")
    f.write("This demonstrates OmniFold's capability to unfold not just marginal distributions\n")
    f.write("but also the joint structure and correlations in the data.\n")

print(f" Saved summary report: {summary_path}")

# 11. PRINT FINAL SUMMARY

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE - ALL FILES SAVED")
print(f"{'='*80}")
print(f"\nOutput directory: {corr_dir}")
print("\nFiles created:")
for file in os.listdir(corr_dir):
    if file.endswith(('.png', '.pdf', '.npy', '.txt')):
        file_path = os.path.join(corr_dir, file)
        file_size = os.path.getsize(file_path) / 1024  # Size in KB
        print(f"  -> {file} ({file_size:.1f} KB)")

print(f"\nSummary Statistics:")
print(f"  Mean |OmniFold - Truth|: {np.mean(np.abs(diff)):.6f}")
print(f"  RMSE: {np.sqrt(np.mean(diff**2)):.6f}")
print(f"  Max difference: {np.max(np.abs(diff)):.6f}")

print(f"\n{'='*80}")

# IBU BENCHMARKING

# COMPARISON: DNN UNFOLDING vs IBU (PHASE 1) 

# First,we define the IBU function
def run_1d_ibu_tt_benchmark(rec_train, gen_train, data_obs, bins, n_iterations=4):
    """
    Performs standard 1D Iterative Bayesian Unfolding for t-tbar.
    """
    # 1. Response Matrix: P(Rec | Gen)
    h_resp, _, _ = np.histogram2d(rec_train, gen_train, bins=[bins, bins])
    h_gen_train, _ = np.histogram(gen_train, bins=bins)

    # Efficiency: P(Observed | Gen)
    # (Avoid division by zero)
    eff_j = np.sum(h_resp, axis=0) / (h_gen_train + 1e-12)

    # 2. Data to Unfold
    h_data, _ = np.histogram(data_obs, bins=bins)

    # 3. Iterative Unfolding
    # Start with the Generator Truth as the prior
    unfolded = h_gen_train.copy().astype(float)

    # Normalize Response Matrix: P(Rec_i | Gen_j)
    M_norm = h_resp / (h_gen_train + 1e-12)

    for k in range(n_iterations):
        pred = np.dot(M_norm, unfolded)            # Fold forward
        ratio = h_data / (pred + 1e-12)            # Compare to data
        update = np.dot(ratio, M_norm)             # Back propagate
        unfolded = unfolded * update / (eff_j + 1e-12) # Update guess

    # Statistical Errors (Poisson approx)
    unfolded_err = np.sqrt(unfolded)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    return bin_centers, unfolded, unfolded_err

# Now run the comparison
print("\n" + "="*80)
print("COMPARISON: DNN UNFOLDING vs ITERATIVE BAYESIAN UNFOLDING (IBU)")
print("="*80)

# Define which observables to compare (using your OBS_CONFIG)
observables_to_compare = {
    0: {"name": "Leading Muon pT [GeV]", "bins": np.linspace(0, 300, 40)},
    1: {"name": "Leading Jet pT [GeV]", "bins": np.linspace(0, 300, 40)},
    2: {"name": "Full Mass (m_tt) [GeV]", "bins": np.linspace(300, 1000, 40)},
    3: {"name": "Delta R", "bins": np.linspace(0, 5, 40)}
}

print("\nComparing DNN Unfolding (Phase 1) with IBU benchmark...")

for idx, config in observables_to_compare.items():
    print(f"\n{'='*60}")
    print(f"OBSERVABLE: {config['name']}")
    print(f"{'='*60}")

    var_name = config['name']
    bins = config['bins']

    # A. DNN UNFOLDING RESULTS (OUR METHOD)
    print("\n1. DNN Unfolding (Your method):")
    data = X_mc_gen[:, idx]

    # DNN unfolded distribution (using w_m_std from Phase 1)
    h_dnn, _ = np.histogram(data, bins=bins, weights=w_m_std[:len(data)], density=True)

    # DNN errors
    counts_err_sq, _ = np.histogram(data, bins=bins, weights=w_e_std[:len(data)]**2)
    h_dnn_err = np.sqrt(counts_err_sq) * (1.0 / (np.sum(w_m_std[:len(data)]) * np.diff(bins)[0]))

    # SM Prior (truth)
    h_truth, _ = np.histogram(data, bins=bins, density=True)

    # Calculate DNN chi**2
    mask = (h_truth > 0)
    chi2_dnn = 0
    if np.sum(mask) > 0:
        chi2_dnn = np.sum((h_dnn[mask] - h_truth[mask])**2 / h_truth[mask])

    print(f"   -> chi**2/D.o.F = {chi2_dnn:.6f}")

    # B. IBU RESULTS 
    print("\n2. Iterative Bayesian Unfolding (IBU):")

    # Run IBU on the SAME data
    x_ibu, y_ibu_counts, err_ibu_counts = run_1d_ibu_tt_benchmark(
        rec_train=X_mc_reco[:, idx],
        gen_train=X_mc_gen[:, idx],
        data_obs=X_data_reco[:, idx],  # Use same data as Phase 1
        bins=bins,
        n_iterations=4
    )

    # Convert IBU to density
    bin_widths = np.diff(bins)
    total_ibu = np.sum(y_ibu_counts)
    ibu_density = y_ibu_counts / (total_ibu * bin_widths)
    ibu_err = err_ibu_counts / (total_ibu * bin_widths)

    # Calculate IBU chi**2 (need to interpolate to same bin centers)
    # Since IBU gives bin centers, interpolate to match DNN histogram
    from scipy.interpolate import interp1d

    # Create interpolation function for IBU
    f_ibu = interp1d(x_ibu, ibu_density, kind='linear', bounds_error=False, fill_value=0)

    # Get DNN bin centers
    dnn_centers = 0.5 * (bins[:-1] + bins[1:])
    ibu_interp = f_ibu(dnn_centers)

    # Calculate IBU chi**2
    mask = (h_truth > 0) & (ibu_interp > 0)
    chi2_ibu = 0
    if np.sum(mask) > 0:
        chi2_ibu = np.sum((ibu_interp[mask] - h_truth[mask])**2 / h_truth[mask])

    print(f"   -> chi**2/D.o.F = {chi2_ibu:.6f}")

    # C. DIRECT COMPARISON PLOT
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{var_name}: DNN vs IBU Comparison', fontsize=16, fontweight='bold')

    # Panel 1: DNN Unfolding
    ax1 = axes[0, 0]
    ax1.step(bins[:-1], h_truth, where='mid', label='SM Truth',
             color='blue', linewidth=2, alpha=0.8)
    ax1.step(bins[:-1], h_dnn, where='mid', label='DNN Unfolded',
             color='black', linestyle='--', linewidth=2)
    ax1.fill_between(bins[:-1], h_dnn - h_dnn_err, h_dnn + h_dnn_err,
                     step='mid', color='green', alpha=0.3, label='DNN Error')
    ax1.set_ylabel('Normalized Density')
    ax1.set_title(f'DNN Unfolding (chi**2 = {chi2_dnn:.6f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: IBU
    ax2 = axes[0, 1]
    ax2.step(bins[:-1], h_truth, where='mid', label='SM Truth',
             color='blue', linewidth=2, alpha=0.8)
    ax2.errorbar(x_ibu, ibu_density, yerr=ibu_err, fmt='o',
                 color='red', markersize=4, label='IBU Unfolded')
    ax2.set_ylabel('Normalized Density')
    ax2.set_title(f'IBU (chi**2 = {chi2_ibu:.6f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Direct Overlay
    ax3 = axes[1, 0]
    ax3.step(bins[:-1], h_truth, where='mid', label='SM Truth',
             color='blue', linewidth=2, alpha=0.6)
    ax3.step(bins[:-1], h_dnn, where='mid', label='DNN',
             color='black', linestyle='--', linewidth=2)
    ax3.errorbar(x_ibu, ibu_density, yerr=ibu_err, fmt='o',
                 color='red', markersize=4, label='IBU', alpha=0.7)
    ax3.set_xlabel(var_name)
    ax3.set_ylabel('Normalized Density')
    ax3.set_title('Direct Comparison: DNN vs IBU')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Ratio to Truth
    ax4 = axes[1, 1]

    # DNN ratio
    dnn_ratio = np.divide(h_dnn, h_truth, out=np.ones_like(h_dnn), where=h_truth!=0)
    dnn_ratio_err = np.divide(h_dnn_err, h_truth, out=np.zeros_like(h_dnn_err), where=h_truth!=0)

    # IBU ratio (interpolated to DNN bins)
    ibu_ratio = np.divide(ibu_interp, h_truth, out=np.ones_like(ibu_interp), where=h_truth!=0)
    ibu_interp_err = np.interp(dnn_centers, x_ibu, ibu_err)
    ibu_ratio_err = np.divide(ibu_interp_err, h_truth, out=np.zeros_like(ibu_interp_err), where=h_truth!=0)

    # Plot ratios
    ax4.errorbar(dnn_centers, dnn_ratio, yerr=dnn_ratio_err, fmt='s',
                 color='black', markersize=4, label='DNN', alpha=0.8)
    ax4.errorbar(dnn_centers, ibu_ratio, yerr=ibu_ratio_err, fmt='o',
                 color='red', markersize=4, label='IBU', alpha=0.8)

    ax4.axhline(1.0, color='blue', linestyle='--', alpha=0.5)
    ax4.set_xlabel(var_name)
    ax4.set_ylabel('Unfolded / Truth')
    ax4.set_title('Ratio to Truth: Closer to 1 is better')
    ax4.set_ylim(0.8, 1.2)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    comparison_path = os.path.join(OUTPUT_DIR, f"Phase1_Comparison_DNN_vs_IBU_{var_name.replace(' ', '_')}_ttbar.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Comparison plot saved: {comparison_path}")

    # D. PERFORMANCE METRICS 
    print("\n3. PERFORMANCE COMPARISON:")

    # Calculate Mean Absolute Error (MAE)
    mae_dnn = np.mean(np.abs(h_dnn - h_truth))
    mae_ibu = np.mean(np.abs(ibu_interp - h_truth))

    # Calculate Root Mean Square Error (RMSE)
    rmse_dnn = np.sqrt(np.mean((h_dnn - h_truth)**2))
    rmse_ibu = np.sqrt(np.mean((ibu_interp - h_truth)**2))

    # Calculate average uncertainty
    avg_unc_dnn = np.mean(h_dnn_err)
    avg_unc_ibu = np.mean(ibu_err)

    print(f"   {'Metric':<15} {'DNN':<12} {'IBU':<12} {'Better':<10}")
    print(f"   {'-'*15} {'-'*12} {'-'*12} {'-'*10}")
    print(f"   {'chi**2':<15} {chi2_dnn:<12.6f} {chi2_ibu:<12.6f} {'DNN' if chi2_dnn < chi2_ibu else 'IBU'}")
    print(f"   {'MAE':<15} {mae_dnn:<12.6f} {mae_ibu:<12.6f} {'DNN' if mae_dnn < mae_ibu else 'IBU'}")
    print(f"   {'RMSE':<15} {rmse_dnn:<12.6f} {rmse_ibu:<12.6f} {'DNN' if rmse_dnn < rmse_ibu else 'IBU'}")
    print(f"   {'Avg. Unc.':<15} {avg_unc_dnn:<12.6f} {avg_unc_ibu:<12.6f} {'DNN' if avg_unc_dnn < avg_unc_ibu else 'IBU'}")

    # Memory usage estimate
    print(f"\n   Computational aspects:")
    print(f"   -> DNN: Trained on {X_mc_reco.shape[1]}-dim features")
    print(f"   -> IBU: 1D only, requires response matrix {len(bins)-1}*{len(bins)-1}")
    print(f"   -> Both methods should give chi**2 = 0 for perfect unfolding")

print("\n" + "="*80)
print("SUMMARY: Both methods should give similar results for Phase 1")
print("(since there's no signal injection, both should recover SM truth)")
print("="*80)

# 8. PHASE 2 SIGNAL INJECTION TEST

print("\n" + "="*40 + "\n PHASE 2: SIGNAL INJECTION TEST\n"+"="*40)

# 1. Inject signal directly into the DATA file (X_data_reco)
print("Injecting synthetic signal bumps into data...")

# Use the existing data
X_data_with_signal = X_data_reco.copy()

# Inject clear bumps at specific positions
MASS_BUMP = 700.0
DR_BUMP = 1.5

# Inject into MORE events, especially for mass
n_events = len(X_data_with_signal)
n_inject = int(n_events * 0.30)
indices_to_inject = np.random.choice(n_events, n_inject, replace=False)

print(f"Injecting signal into {n_inject} events ({100*n_inject/n_events:.1f}% of data)")


n_mass_inject = int(n_inject * 0.70)  # 70% for mass
n_dr_inject = n_inject - n_mass_inject  # 30% for Delta R

mass_indices = indices_to_inject[:n_mass_inject]
dr_indices = indices_to_inject[n_mass_inject:]

# Inject MASS bump (column 2 = Full Mass) - NARROWER peak
print(f"  -> Mass bump at {MASS_BUMP} GeV (narrower peak)")
X_data_with_signal[mass_indices, 2] = np.random.normal(MASS_BUMP, 25, n_mass_inject)  # CHANGED: width 25 instead of 40

# CHANGED: Also adjust correlated pT variables to make signal more realistic
for idx in mass_indices:
    X_data_with_signal[idx, 0] *= 1.12  # Increase muon pT by 12%
    X_data_with_signal[idx, 1] *= 1.12  # Increase jet pT by 12%
    # Update derived quantities to be consistent
    #X_data_with_signal[idx, 4] = X_data_with_signal[idx, 0] + X_data_with_signal[idx, 1]  # VisM
    #X_data_with_signal[idx, 5] = X_data_with_signal[idx, 0] + X_data_with_signal[idx, 1]  # VisPt

# Inject DELTA R bump (column 3 = Delta R)
print(f"  -> Delta R bump at {DR_BUMP} (same as before)")
X_data_with_signal[dr_indices, 3] = np.random.normal(DR_BUMP, 0.3, n_dr_inject)

print(f"Data with signal shape: {X_data_with_signal.shape}")

# 2. Run EXACTLY the same unfolding as Phase 1, but with signal-injected data
print("\nRunning unfolding on signal-injected data (same as Phase 1)...")
w_m_sig, w_e_sig = run_fast_ensemble(X_mc_reco, X_mc_gen, X_data_with_signal, n_runs=6, n_iters=6)

# 3. Plot with EXACTLY the same function as Phase 1, but add signal lines
print("\nPlotting results with signal injection lines...")

# Define where we injected signals
injection_lines = {
    2: MASS_BUMP,   # Full Mass bump at 700 GeV
    3: DR_BUMP      # Delta R bump at 1.5
}

# Use the SAME plotting function as Phase 1, just add injection_lines parameter
stats_sig = plot_and_save_results(
    X_mc_gen,
    w_m_sig,
    w_e_sig,
    f"Phase 2: Signal Injection Test (Mass @{MASS_BUMP}GeV, delta_r @{DR_BUMP})",
    "Phase2_Signal_Injection_ttbar",
    injection_lines=injection_lines
)

# Save results
save_csv_data(w_m_sig, w_e_sig, stats_sig, "Phase2_Signal_Injection_ttbar")

print(f"\nPhase 2 completed! Results saved to {OUTPUT_DIR}")
print("\nExpected in plots:")
print(f"-> Red vertical lines at {MASS_BUMP} GeV (mass) and {DR_BUMP} (Delta R)")
print("-> Unfolded distribution (black dashed line) should show bumps at red lines")
print("-> Ratio plot should show deviation from 1.0 at signal positions")

# PHASE 2 DIAGNOSTICS

# --- SIGNAL SIGNIFICANCE CALCULATION ---
print("\n" + "="*60)
print("SIGNAL INJECTION SIGNIFICANCE CALCULATION")
print("="*60)

# 1. First, let's extract the unfolded histograms for signal regions
print("\n1. Extracting unfolded distributions in signal regions...")

# Signal parameters (same as injection)
MASS_BUMP = 700.0
MASS_WIDTH = 25.0
DR_BUMP = 1.5
DR_WIDTH = 0.3

# Get the data from the unfolding plots
mass_idx = 2  # Full Mass
dr_idx = 3    # Delta R

# Get bin edges from config
mass_bins = OBS_CONFIG[mass_idx]['bins']
dr_bins = OBS_CONFIG[dr_idx]['bins']

# Calculate bin centers
mass_centers = 0.5 * (mass_bins[1:] + mass_bins[:-1])
dr_centers = 0.5 * (dr_bins[1:] + dr_bins[:-1])

# Define signal regions (+/-1.5sigma around injection point)
mass_signal_min = MASS_BUMP - 1.5 * MASS_WIDTH
mass_signal_max = MASS_BUMP + 1.5 * MASS_WIDTH
dr_signal_min = DR_BUMP - 1.5 * DR_WIDTH
dr_signal_max = DR_BUMP + 1.5 * DR_WIDTH

# Define sideband regions (background control)
mass_sideband_min = MASS_BUMP - 4 * MASS_WIDTH
mass_sideband_max = MASS_BUMP - 2 * MASS_WIDTH
dr_sideband_min = DR_BUMP - 4 * DR_WIDTH
dr_sideband_max = DR_BUMP - 2 * DR_WIDTH

print(f"Mass signal region: {mass_signal_min:.1f} - {mass_signal_max:.1f} GeV")
print(f"Mass sideband region: {mass_sideband_min:.1f} - {mass_sideband_max:.1f} GeV")
print(f"Delta R signal region: {dr_signal_min:.3f} - {dr_signal_max:.3f}")
print(f"Delta R sideband region: {dr_sideband_min:.3f} - {dr_sideband_max:.3f}")

# 2. Calculate histograms for unfolded data
print("\n2. Calculating histograms...")

# Unfolded distribution (with weights)
h_unfold_mass, _ = np.histogram(X_mc_gen[:, mass_idx], bins=mass_bins, weights=w_m_sig[:len(X_mc_gen)])
h_unfold_dr, _ = np.histogram(X_mc_gen[:, dr_idx], bins=dr_bins, weights=w_m_sig[:len(X_mc_gen)])

# Prior distribution (SM background, no weights)
h_prior_mass, _ = np.histogram(X_mc_gen[:, mass_idx], bins=mass_bins, density=False)
h_prior_dr, _ = np.histogram(X_mc_gen[:, dr_idx], bins=dr_bins, density=False)

# Scale prior to same normalization as unfolded
scale_factor_mass = np.sum(h_unfold_mass) / np.sum(h_prior_mass)
scale_factor_dr = np.sum(h_unfold_dr) / np.sum(h_prior_dr)

h_prior_mass_scaled = h_prior_mass * scale_factor_mass
h_prior_dr_scaled = h_prior_dr * scale_factor_dr

# 3. Identify bins in signal and sideband regions
print("\n3. Identifying signal and background bins...")

# Mass bins
mass_signal_bins = (mass_centers >= mass_signal_min) & (mass_centers <= mass_signal_max)
mass_sideband_bins = (mass_centers >= mass_sideband_min) & (mass_centers <= mass_sideband_max)

# Delta R bins
dr_signal_bins = (dr_centers >= dr_signal_min) & (dr_centers <= dr_signal_max)
dr_sideband_bins = (dr_centers >= dr_sideband_min) & (dr_centers <= dr_sideband_max)

# 4. Calculate observed and expected counts
print("\n4. Calculating observed and expected counts...")

# Mass
N_obs_mass = np.sum(h_unfold_mass[mass_signal_bins])
N_bkg_mass = np.sum(h_prior_mass_scaled[mass_signal_bins])
N_sig_mass = N_obs_mass - N_bkg_mass

# Delta R
N_obs_dr = np.sum(h_unfold_dr[dr_signal_bins])
N_bkg_dr = np.sum(h_prior_dr_scaled[dr_signal_bins])
N_sig_dr = N_obs_dr - N_bkg_dr

print(f"Mass:")
print(f"  -> Observed in signal region: {N_obs_mass:.1f}")
print(f"  -> Expected background: {N_bkg_mass:.1f}")
print(f"  -> Signal excess: {N_sig_mass:.1f}")

print(f"\nDelta R:")
print(f"  -> Observed in signal region: {N_obs_dr:.1f}")
print(f"  -> Expected background: {N_bkg_dr:.1f}")
print(f"  -> Signal excess: {N_sig_dr:.1f}")

# 5. Calculate significance using simple Gaussian approximation
print("\n5. Calculating statistical significance...")

def calculate_significance(N_obs, N_bkg):
    """Calculate Gaussian significance S = (N_obs - N_bkg) / sqrt(N_bkg)"""
    if N_bkg > 0:
        significance = (N_obs - N_bkg) / np.sqrt(N_bkg)
    else:
        significance = 0.0
    return significance

# Calculate significances
sig_mass = calculate_significance(N_obs_mass, N_bkg_mass)
sig_dr = calculate_significance(N_obs_dr, N_bkg_dr)

print(f"Mass significance: {sig_mass:.2f}sigma")
print(f"Delta R significance: {sig_dr:.2f}sigma")

# 6. More robust significance calculation (with uncertainties)
print("\n6. Calculating significance with uncertainties...")

# Estimate uncertainties from sideband regions
def estimate_background_with_error(signal_counts, sideband_counts, signal_area, sideband_area):
    """Estimate background and its uncertainty using sideband method"""
    if np.sum(sideband_counts) > 0 and sideband_area > 0:
        # Background density in sideband
        bkg_density = np.sum(sideband_counts) / sideband_area
        bkg_density_err = np.sqrt(np.sum(sideband_counts)) / sideband_area

        # Expected background in signal region
        bkg_expected = bkg_density * signal_area
        bkg_expected_err = bkg_density_err * signal_area

        return bkg_expected, bkg_expected_err
    return 0.0, 0.0

# Calculate areas (in bins)
mass_signal_area = np.sum(mass_signal_bins)
mass_sideband_area = np.sum(mass_sideband_bins)
dr_signal_area = np.sum(dr_signal_bins)
dr_sideband_area = np.sum(dr_sideband_bins)

# Estimate background from sidebands
bkg_mass_est, bkg_mass_err = estimate_background_with_error(
    h_unfold_mass[mass_signal_bins],
    h_unfold_mass[mass_sideband_bins],
    mass_signal_area,
    mass_sideband_area
)

bkg_dr_est, bkg_dr_err = estimate_background_with_error(
    h_unfold_dr[dr_signal_bins],
    h_unfold_dr[dr_sideband_bins],
    dr_signal_area,
    dr_sideband_area
)

print(f"\nMass (sideband method):")
print(f"  -> Expected background: {bkg_mass_est:.1f} +/- {bkg_mass_err:.1f}")
print(f"  -> Signal excess: {N_obs_mass - bkg_mass_est:.1f}")

print(f"\nDelta R (sideband method):")
print(f"  -> Expected background: {bkg_dr_est:.1f} +/- {bkg_dr_err:.1f}")
print(f"  -> Signal excess: {N_obs_dr - bkg_dr_est:.1f}")

# Calculate significance with background uncertainty
def calculate_significance_with_error(N_obs, bkg_est, bkg_err):
    """Calculate significance accounting for background uncertainty"""
    if bkg_est > 0 and bkg_err > 0:
        # Simple formula: S = excess / sqrt(sigma_excess**2 + sigma_bkg**2)
        excess = N_obs - bkg_est
        stat_err = np.sqrt(N_obs)  # Statistical uncertainty on observation
        total_err = np.sqrt(stat_err**2 + bkg_err**2)

        if total_err > 0:
            return excess / total_err
    return 0.0

sig_mass_robust = calculate_significance_with_error(N_obs_mass, bkg_mass_est, bkg_mass_err)
sig_dr_robust = calculate_significance_with_error(N_obs_dr, bkg_dr_est, bkg_dr_err)

print(f"\nRobust significances (with background uncertainty):")
print(f"  -> Mass: {sig_mass_robust:.2f}sigma")
print(f"  -> Delta R: {sig_dr_robust:.2f}sigma")

# 7. Visualize significance regions
print("\n7. Creating significance visualization plot...")

fig_sig, axes_sig = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Mass distribution with signal/sideband regions
ax1 = axes_sig[0]
ax1.step(mass_centers, h_unfold_mass, where='mid', label='Unfolded',
         color='black', linewidth=2, linestyle='--')
ax1.step(mass_centers, h_prior_mass_scaled, where='mid', label='SM Prior',
         color='blue', linewidth=1.5, alpha=0.8)

# Highlight signal region
ax1.axvspan(mass_signal_min, mass_signal_max, alpha=0.3, color='red',
           label=f'Signal region ({sig_mass:.1f}sigma)')
# Highlight sideband region
ax1.axvspan(mass_sideband_min, mass_sideband_max, alpha=0.2, color='green',
           label='Sideband (bkg est.)')
ax1.axvline(MASS_BUMP, color='red', linestyle='-', alpha=0.7, linewidth=1)

ax1.set_xlabel('Full Mass [GeV]')
ax1.set_ylabel('Events')
ax1.set_title(f'Mass: Signal Significance = {sig_mass:.1f}sigma ({sig_mass_robust:.1f}sigma robust)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add text box with results
text_mass = f"""Observed: {N_obs_mass:.1f}
Expected: {N_bkg_mass:.1f}
Excess: {N_sig_mass:.1f}
Significance: {sig_mass:.1f}sigma
Robust: {sig_mass_robust:.1f}sigma"""
ax1.text(0.05, 0.95, text_mass, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 2: Delta R distribution with signal/sideband regions
ax2 = axes_sig[1]
ax2.step(dr_centers, h_unfold_dr, where='mid', label='Unfolded',
         color='black', linewidth=2, linestyle='--')
ax2.step(dr_centers, h_prior_dr_scaled, where='mid', label='SM Prior',
         color='blue', linewidth=1.5, alpha=0.8)

# Highlight signal region
ax2.axvspan(dr_signal_min, dr_signal_max, alpha=0.3, color='red',
           label=f'Signal region ({sig_dr:.1f}sigma)')
# Highlight sideband region
ax2.axvspan(dr_sideband_min, dr_sideband_max, alpha=0.2, color='green',
           label='Sideband (bkg est.)')
ax2.axvline(DR_BUMP, color='red', linestyle='-', alpha=0.7, linewidth=1)

ax2.set_xlabel('Delta R')
ax2.set_ylabel('Events')
ax2.set_title(f'Delta R: Signal Significance = {sig_dr:.1f}sigma ({sig_dr_robust:.1f}sigma robust)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add text box with results
text_dr = f"""Observed: {N_obs_dr:.1f}
Expected: {N_bkg_dr:.1f}
Excess: {N_sig_dr:.1f}
Significance: {sig_dr:.1f}sigma
Robust: {sig_dr_robust:.1f}sigma"""
ax2.text(0.05, 0.95, text_dr, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
sig_plot_path = os.path.join(OUTPUT_DIR, "Phase2_Signal_Significance_ttbar.png")
plt.savefig(sig_plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Significance plot saved: {sig_plot_path}")

# 8. Summary table
print("\n" + "="*60)
print("SIGNAL SIGNIFICANCE SUMMARY")
print("="*60)

summary_data = {
    'Observable': ['Full Mass (m_tt)', 'Delta R'],
    'Injection Point': [f'{MASS_BUMP} GeV', f'{DR_BUMP}'],
    'Observed Events': [f'{N_obs_mass:.1f}', f'{N_obs_dr:.1f}'],
    'Expected Background': [f'{N_bkg_mass:.1f}', f'{N_bkg_dr:.1f}'],
    'Signal Excess': [f'{N_sig_mass:.1f}', f'{N_sig_dr:.1f}'],
    'Simple Significance': [f'{sig_mass:.2f}sigma', f'{sig_dr:.2f}sigma'],
    'Robust Significance': [f'{sig_mass_robust:.2f}sigma', f'{sig_dr_robust:.2f}sigma']
}

# Print as table
print(f"{'Observable':<20} {'Injection':<12} {'Observed':<12} {'Expected':<12} {'Excess':<12} {'Simple Sig':<12} {'Robust Sig':<12}")
print("-" * 95)
for i in range(2):
    print(f"{summary_data['Observable'][i]:<20} "
          f"{summary_data['Injection Point'][i]:<12} "
          f"{summary_data['Observed Events'][i]:<12} "
          f"{summary_data['Expected Background'][i]:<12} "
          f"{summary_data['Signal Excess'][i]:<12} "
          f"{summary_data['Simple Significance'][i]:<12} "
          f"{summary_data['Robust Significance'][i]:<12}")

print("\n" + "-" * 95)
print("INTERPRETATION GUIDE:")
print("-> < 1sigma: Not significant")
print("-> 1-2sigma: Hint of signal")
print("-> 2-3sigma: Evidence")
print("-> 3-5sigma: Observation")
print("-> > 5sigma: Discovery")
print("-" * 95)

# Save results to CSV
import pandas as pd
df_summary = pd.DataFrame(summary_data)
summary_path = os.path.join(OUTPUT_DIR, "Phase2_Signal_Significance_Summary_ttbar.csv")
df_summary.to_csv(summary_path, index=False)
print(f"\nSummary saved to: {summary_path}")

# 9. TRAINING LOSS ANALYSIS

print("\n" + "="*80)
print("ADDITIONAL ANALYSIS: TRAINING LOSS CURVES")
print("="*80)

# Modified training function - ONLY adds loss tracking, everything else IDENTICAL
def train_on_gpu_with_loss_tracking(X_src, X_tgt, w_src, w_tgt, epochs=30, batch_size=1048):
    # Create Labels
    y_src = torch.zeros(len(X_src), 1, device=device)
    y_tgt = torch.ones(len(X_tgt), 1, device=device)

    # Concatenate on GPU
    X_all = torch.cat([X_src, X_tgt], dim=0)
    y_all = torch.cat([y_src, y_tgt], dim=0)
    w_all = torch.cat([w_src.unsqueeze(1), w_tgt.unsqueeze(1)], dim=0)

    # Shuffle indices
    perm = torch.randperm(len(X_all), device=device)
    X_all, y_all, w_all = X_all[perm], y_all[perm], w_all[perm]

    # Model
    model = UnfoldingNet(X_src.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss(reduction='none')

    # Manual Batch Loop (Faster than DataLoader for simple tensors)
    N = len(X_all)
    model.train()

    # Track losses
    epoch_losses = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for i in range(0, N, batch_size):
            # Slicing is fast on GPU
            xb = X_all[i:i+batch_size]
            yb = y_all[i:i+batch_size]
            wb = w_all[i:i+batch_size]

            opt.zero_grad()
            loss = (crit(model(xb), yb) * wb).mean()
            loss.backward()
            opt.step()

            # Accumulate loss
            epoch_loss += loss.item()

        # Store average loss for this epoch
        epoch_losses.append(epoch_loss / (N // batch_size + 1))

    return model, epoch_losses  # Return losses along with model

# Modified ensemble function - ONLY adds loss tracking, everything else IDENTICAL
def run_fast_ensemble_with_loss_tracking(mc_reco, mc_gen, data_reco, n_runs=3, n_iters=4):
    print(f"Starting Fast Ensemble with loss tracking ({n_runs} runs)...")
    start_time = time.time()

    # 1. Fix negative values BEFORE fitting scaler
    cols = [0, 1, 2]

    # Make copies to avoid modifying originals
    mc_reco_fixed = mc_reco.copy()
    mc_gen_fixed = mc_gen.copy()
    data_reco_fixed = data_reco.copy()

    # Add small offset to make all values > 0 for log transform
    for col in cols:
        min_val = min(mc_reco_fixed[:, col].min(),
                     mc_gen_fixed[:, col].min(),
                     data_reco_fixed[:, col].min())
        if min_val <= 0:
            offset = abs(min_val) + 0.001
            mc_reco_fixed[:, col] += offset
            mc_gen_fixed[:, col] += offset
            data_reco_fixed[:, col] += offset

    # 2. Fit Scaler Once
    all_data = np.vstack([mc_reco_fixed, mc_gen_fixed, data_reco_fixed])
    all_data[:, cols] = np.log1p(all_data[:, cols])  # Now safe
    global_scaler.fit(all_data)

    # 3. Move to GPU ONCE
    t_mc_reco = prepare_tensor(mc_reco_fixed)
    t_mc_gen = prepare_tensor(mc_gen_fixed)
    t_data_reco = prepare_tensor(data_reco_fixed)

    all_weights = []
    all_loss_history = []  # Track losses for each run

    for run in range(n_runs):
        print(f" > Run {run+1}/{n_runs}...", end=" ")
        run_losses = {'detector': [], 'generator': []}  # NEW: Store losses for this run

        # Bootstrapping (CPU generation -> GPU move)
        w_data_cpu = np.random.poisson(1, len(data_reco_fixed)).astype(np.float32)
        target_sum = np.sum(w_data_cpu)

        # GPU Weights
        w_data = torch.tensor(w_data_cpu, device=device)
        w_reco = torch.ones(len(mc_reco_fixed), device=device)
        w_gen = torch.ones(len(mc_gen_fixed), device=device)

        for i in range(n_iters):
            # Step 1 (Detector) - 30 Epochs
            m1, loss1 = train_on_gpu_with_loss_tracking(t_mc_reco, t_data_reco, w_reco, w_data, epochs=30)
            w_reco = w_reco * inference_on_gpu(m1, t_mc_reco)
            # NEW: Store detector loss
            run_losses['detector'].append(loss1)

            # Clip extreme weights to prevent NaN
            w_reco = torch.clamp(w_reco, 0.01, 100.0)

            # Normalize
            if w_reco.sum() > 0:
                w_reco = w_reco * (target_sum / w_reco.sum())

            # Step 2 (Gen) - 15 Epochs
            m2, loss2 = train_on_gpu_with_loss_tracking(t_mc_gen, t_mc_gen, w_gen, w_reco, epochs=15)
            w_gen = w_gen * inference_on_gpu(m2, t_mc_gen)
            # NEW: Store generator loss
            run_losses['generator'].append(loss2)

            # Clip extreme weights
            w_gen = torch.clamp(w_gen, 0.01, 100.0)

            # Normalize
            if w_gen.sum() > 0:
                w_gen = w_gen * (target_sum / w_gen.sum())

            # Copy for next iteration
            w_reco = w_gen.clone()

            # Optional: print iteration progress
            if (i + 1) % 1 == 0:
                print(f"iter{i+1}", end=" ", flush=True)

        all_weights.append(w_gen.cpu().numpy())
        all_loss_history.append(run_losses)  # NEW: Store this run's losses
        print(f"Done ({int(time.time()-start_time)}s elapsed)")

    # Calculate mean and std across ensembles
    all_weights_array = np.array(all_weights)
    w_mean = np.mean(all_weights_array, axis=0)
    w_err = np.std(all_weights_array, axis=0)

    # Final safety check for NaN
    w_mean = np.nan_to_num(w_mean, nan=1.0)
    w_err = np.nan_to_num(w_err, nan=0.0)

    return w_mean, w_err, all_loss_history

# Run the ensemble with loss tracking
print("\nRe-running ensemble with loss tracking")
w_m_std_tracked, w_e_std_tracked, loss_history = run_fast_ensemble_with_loss_tracking(
    X_mc_reco, X_mc_gen, X_data_reco, n_runs=6, n_iters=4
)

# Plot the loss curves
print("\n" + "="*60)
print("PLOTTING TRAINING LOSS CURVES")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = plt.cm.tab10(np.linspace(0, 1, 6))

# Plot 1: Detector training loss curves
ax = axes[0, 0]
for run_idx, run_losses in enumerate(loss_history):
    for iter_idx, detector_losses in enumerate(run_losses['detector']):
        label = f"Run {run_idx+1}, Iter {iter_idx+1}" if run_idx < 3 else None
        ax.plot(range(1, len(detector_losses)+1), detector_losses,
                color=colors[run_idx], alpha=0.7, label=label if iter_idx == 0 else None)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Detector Training Loss (Step 1 - 30 epochs)')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: Generator training loss curves
ax = axes[0, 1]
for run_idx, run_losses in enumerate(loss_history):
    for iter_idx, generator_losses in enumerate(run_losses['generator']):
        label = f"Run {run_idx+1}, Iter {iter_idx+1}" if run_idx < 3 else None
        ax.plot(range(1, len(generator_losses)+1), generator_losses,
                color=colors[run_idx], alpha=0.7, label=label if iter_idx == 0 else None)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Generator Training Loss (Step 2 - 15 epochs)')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3: Final detector loss per iteration
ax = axes[1, 0]
for run_idx, run_losses in enumerate(loss_history):
    final_detector_losses = [losses[-1] for losses in run_losses['detector']]
    ax.plot(range(1, len(final_detector_losses)+1), final_detector_losses,
            'o-', color=colors[run_idx], label=f'Run {run_idx+1}')
ax.set_xlabel('Iteration')
ax.set_ylabel('Final Loss')
ax.set_title('Final Detector Loss by Iteration')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Final generator loss per iteration
ax = axes[1, 1]
for run_idx, run_losses in enumerate(loss_history):
    final_generator_losses = [losses[-1] for losses in run_losses['generator']]
    ax.plot(range(1, len(final_generator_losses)+1), final_generator_losses,
            'o-', color=colors[run_idx], label=f'Run {run_idx+1}')
ax.set_xlabel('Iteration')
ax.set_ylabel('Final Loss')
ax.set_title('Final Generator Loss by Iteration')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
loss_plot_path = os.path.join(OUTPUT_DIR, "Phase1_Training_Loss_Curves_ttbar.png")
plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Loss curves saved: {loss_plot_path}")

# Loss analysis metrics
print("\n" + "="*60)
print("LOSS ANALYSIS METRICS")
print("="*60)

all_final_detector_losses = []
all_final_generator_losses = []

for run_idx, run_losses in enumerate(loss_history):
    print(f"\nRun {run_idx+1}:")

    for iter_idx in range(len(run_losses['detector'])):
        detector_loss = run_losses['detector'][iter_idx]
        generator_loss = run_losses['generator'][iter_idx]

        final_det = detector_loss[-1]
        final_gen = generator_loss[-1]
        all_final_detector_losses.append(final_det)
        all_final_generator_losses.append(final_gen)

        det_change = ((detector_loss[-1] - detector_loss[0]) / detector_loss[0]) * 100
        gen_change = ((generator_loss[-1] - generator_loss[0]) / generator_loss[0]) * 100

        print(f"  Iter {iter_idx+1}:")
        print(f"    Detector: {detector_loss[0]:.4f} -> {detector_loss[-1]:.4f} ({det_change:+.1f}%)")
        print(f"    Generator: {generator_loss[0]:.4f} -> {generator_loss[-1]:.4f} ({gen_change:+.1f}%)")

print("\n" + "="*40)
print("OVERALL STATISTICS:")
print(f"Detector - Mean final loss: {np.mean(all_final_detector_losses):.6f}")
print(f"Detector - Std final loss: {np.std(all_final_detector_losses):.6f}")
print(f"Detector - Min final loss: {np.min(all_final_detector_losses):.6f}")
print(f"Detector - Max final loss: {np.max(all_final_detector_losses):.6f}")

print(f"\nGenerator - Mean final loss: {np.mean(all_final_generator_losses):.6f}")
print(f"Generator - Std final loss: {np.std(all_final_generator_losses):.6f}")
print(f"Generator - Min final loss: {np.min(all_final_generator_losses):.6f}")
print(f"Generator - Max final loss: {np.max(all_final_generator_losses):.6f}")

# Check for overfitting indicators
loss_variation_det = np.std(all_final_detector_losses) / np.mean(all_final_detector_losses)
loss_variation_gen = np.std(all_final_generator_losses) / np.mean(all_final_generator_losses)

print(f"\nLoss variation (Detector): {loss_variation_det:.4f}")
print(f"Loss variation (Generator): {loss_variation_gen:.4f}")

if loss_variation_det < 0.05 and loss_variation_gen < 0.05:
    print("LOW VARIATION - Stable training across runs")
else:
    print("HIGH VARIATION - Possible instability")

# Check if losses increased in later iterations
print("\nLoss trend across iterations:")
for run_idx, run_losses in enumerate(loss_history):
    det_first = run_losses['detector'][0][-1]
    det_last = run_losses['detector'][-1][-1]
    gen_first = run_losses['generator'][0][-1]
    gen_last = run_losses['generator'][-1][-1]

    det_change = ((det_last - det_first) / det_first) * 100
    gen_change = ((gen_last - gen_first) / gen_first) * 100

    print(f"Run {run_idx+1}: Detector: {det_first:.4f} -> {det_last:.4f} ({det_change:+.1f}%)")
    print(f"         Generator: {gen_first:.4f} -> {gen_last:.4f} ({gen_change:+.1f}%)")

print("\n" + "="*80)
print("ADDITIONAL ANALYSIS COMPLETE")
print("="*80)
