import pylhe
import pyhepmc
import uproot
import awkward as ak
import numpy as np
import vector
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Define PDG muon mass for uproot analysis
MUON_MASS = 0.105658 # GeV

#  DATA PROCESSING FUNCTIONS

def process_lhe_file(file_path):
    """Reads an LHE file and returns kinematic data as NumPy arrays."""
    print(f"Processing LHE file: {file_path}")
    pt_data, eta_data, mass_data = [], [], []
    try:
        events = pylhe.read_lhe(file_path)
        for event in events:
            muons = []
            for p in event.particles:
                if p.status == 1 and abs(p.id) == 13:
                    muons.append(vector.obj(px=p.px, py=p.py, pz=p.pz, E=p.e))
            
            if len(muons) == 2:
                pt_data.extend([mu.pt for mu in muons])
                eta_data.extend([mu.eta for mu in muons])
                dimuon = muons[0] + muons[1]
                mass_data.append(dimuon.mass)
    except Exception as e:
        print(f"Error processing LHE file: {e}")
    return np.array(pt_data), np.array(eta_data), np.array(mass_data)

def process_hepmc_file(file_path):
    """Reads a HEPMC file and returns kinematic data as NumPy arrays."""
    print(f"Processing HEPMC file: {file_path}")
    pt_data, eta_data, mass_data = [], [], []
    try:
        with pyhepmc.open(file_path) as f:
            for event in f:
                muons = []
                for p in event.particles:
                    if p.status == 1 and abs(p.pid) == 13:
                        mom = p.momentum
                        muons.append(vector.obj(px=mom.px, py=mom.py, pz=mom.pz, E=mom.e))
                
                if len(muons) == 2:
                    pt_data.extend([mu.pt for mu in muons])
                    eta_data.extend([mu.eta for mu in muons])
                    dimuon = muons[0] + muons[1]
                    mass_data.append(dimuon.mass)
    except Exception as e:
        print(f"Error processing HEPMC file: {e}")
    return np.array(pt_data), np.array(eta_data), np.array(mass_data)

def process_root_file_uproot(file_path):
    """Reads a ROOT file using uproot and returns kinematic data as NumPy arrays."""
    print(f"Processing ROOT file with uproot: {file_path}")
    pt_data, eta_data, mass_data = [], [], []
    try:
        with uproot.open(file_path) as f:
            tree = f["Delphes"]
            muon_branches = ["Muon/Muon.PT", "Muon/Muon.Eta", "Muon/Muon.Phi"]
            
            for arrays in tree.iterate(muon_branches, library="ak", step_size=1000):
                event_mask = (ak.num(arrays["Muon/Muon.PT"]) >= 2)
                valid_events = arrays[event_mask]
                if len(valid_events) == 0:
                    continue
                
                pts = valid_events["Muon/Muon.PT"]
                etas = valid_events["Muon/Muon.Eta"]
                phis = valid_events["Muon/Muon.Phi"]
                masses = ak.full_like(pts, MUON_MASS)
                
                muons_p4 = vector.zip({ "pt": pts, "eta": etas, "phi": phis, "mass": masses })
                muon1 = muons_p4[:, 0]
                muon2 = muons_p4[:, 1]
                
                pt_for_hist = ak.concatenate([muon1.pt, muon2.pt], axis=0)
                eta_for_hist = ak.concatenate([muon1.eta, muon2.eta], axis=0)
                pt_data.append(ak.to_numpy(pt_for_hist))
                eta_data.append(ak.to_numpy(eta_for_hist))
                
                dimuon = muon1 + muon2
                mass_data.append(ak.to_numpy(dimuon.mass))
    except Exception as e:
        print(f"An error occurred during ROOT file processing: {e}")
    
    pt_np = np.concatenate(pt_data) if pt_data else np.array([])
    eta_np = np.concatenate(eta_data) if eta_data else np.array([])
    mass_np = np.concatenate(mass_data) if mass_data else np.array([])
    
    print(f"Found {len(mass_np)} events in ROOT file with >= 2 muons.")
    return pt_np, eta_np, mass_np

#  PLOTTING FUNCTION

def plot_single_dataset(pt, eta, mass, label, color, filename):
    """Creates a 1x3 figure for a single dataset (LHE, HepMC, or ROOT)."""
    
    # Check if data exists before plotting
    if pt is None or len(pt) == 0:
        print(f"Skipping {label} plots: No data found.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(8, 14))
    fig.suptitle(f"Kinematic Distributions: {label} Data", fontsize=18, y=1.05)

    # --- Plot 1: Transverse Momentum ---
    axes[0].hist(pt, bins=100, range=(0, 200), histtype='step', 
                 linewidth=2, density=True, color=color)
    axes[0].set_title(f"Transverse Momentum ($p_T$)", fontsize=14)
    axes[0].set_xlabel("$p_T$ [GeV]", fontsize=12)
    axes[0].set_ylabel("Normalized Entries", fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # --- Plot 2: Pseudorapidity ---
    axes[1].hist(eta, bins=100, range=(-5, 5), histtype='step', 
                 linewidth=2, density=True, color=color)
    axes[1].set_title(f"Pseudorapidity ($\eta$)", fontsize=14)
    axes[1].set_xlabel("$\eta$", fontsize=12)
    axes[1].set_ylabel("Normalized Entries", fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # --- Plot 3: Invariant Mass ---
    axes[2].hist(mass, bins=100, range=(50, 250), histtype='step', 
                 linewidth=2, density=True, color=color)
    axes[2].set_title(f"Invariant Mass ($m_{{\mu^+\mu^-}}$)", fontsize=14)
    axes[2].set_xlabel("$m_{\mu^+\mu^-}$ [GeV]", fontsize=12)
    axes[2].set_ylabel("Normalized Entries", fontsize=12)
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close() # Close figure to free memory
    print(f"Saved {filename}")

#  MAIN EXECUTION

if __name__ == "__main__":
    # --- !!! UPDATE THESE FILE PATHS !!! ---
    lhe_file = "!"
    hepmc_file = "!"
    root_file = "!"
    
    # --- Process all data sources ---
    pt_lhe, eta_lhe, mass_lhe = process_lhe_file(lhe_file)
    pt_hepmc, eta_hepmc, mass_hepmc = process_hepmc_file(hepmc_file)
    pt_root, eta_root, mass_root = process_root_file_uproot(root_file)

    # --- Create individual plots for each file type ---
    plot_single_dataset(pt_lhe, eta_lhe, mass_lhe, label="LHE", color="red", filename="plots_lhe.png")
    plot_single_dataset(pt_hepmc, eta_hepmc, mass_hepmc, label="HepMC", color="blue", filename="plots_hepmc.png")
    plot_single_dataset(pt_root, eta_root, mass_root, label="ROOT (Delphes)", color="green", filename="plots_root.png")
