import pylhe
import pyhepmc
import uproot
import awkward as ak
import numpy as np
import vector
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive saving
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
                # Find final-state muons (status 1)
                if p.status == 1 and abs(p.id) == 13:
                    muons.append(vector.obj(px=p.px, py=p.py, pz=p.pz, E=p.e))
            
            # Require exactly 2 muons
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
                    # Find stable, final-state muons (status 1)
                    if p.status == 1 and abs(p.pid) == 13:
                        mom = p.momentum
                        muons.append(vector.obj(px=mom.px, py=mom.py, pz=mom.pz, E=mom.e))
                
                # Require exactly 2 muons
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
            
            # Iterate in chunks for memory efficiency
            for arrays in tree.iterate(muon_branches, library="ak", step_size=1000):
                # Select events with at least 2 muons
                event_mask = (ak.num(arrays["Muon/Muon.PT"]) >= 2)
                valid_events = arrays[event_mask]
                if len(valid_events) == 0:
                    continue
                
                # Get the kinematics of all muons
                pts = valid_events["Muon/Muon.PT"]
                etas = valid_events["Muon/Muon.Eta"]
                phis = valid_events["Muon/Muon.Phi"]
                masses = ak.full_like(pts, MUON_MASS)
                
                # Create four-vectors for all muons in valid events
                muons_p4 = vector.zip({ "pt": pts, "eta": etas, "phi": phis, "mass": masses })
                
                # Select the leading two muons for the analysis
                muon1 = muons_p4[:, 0]
                muon2 = muons_p4[:, 1]
                
                # Store kinematics for histograms
                pt_for_hist = ak.concatenate([muon1.pt, muon2.pt], axis=0)
                eta_for_hist = ak.concatenate([muon1.eta, muon2.eta], axis=0)
                pt_data.append(ak.to_numpy(pt_for_hist))
                eta_data.append(ak.to_numpy(eta_for_hist))
                
                # Calculate and store invariant mass
                dimuon = muon1 + muon2
                mass_data.append(ak.to_numpy(dimuon.mass))
    except Exception as e:
        print(f"An error occurred during ROOT file processing: {e}")
    
    # Concatenate chunks into final NumPy arrays
    pt_np = np.concatenate(pt_data) if pt_data else np.array([])
    eta_np = np.concatenate(eta_data) if eta_data else np.array([])
    mass_np = np.concatenate(mass_data) if mass_data else np.array([])
    
    print(f"Found {len(mass_np)} events in ROOT file with >= 2 muons.")
    return pt_np, eta_np, mass_np


#  PLOTTING FUNCTION

def plot_all_on_one_canvas(pt_datasets, eta_datasets, mass_datasets):
    """Creates a single canvas with three subplots for comparison."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    colors = {'LHE': 'red', 'HEPMC': 'blue', 'ROOT': 'green'}
    
    # Plot 1: Transverse Momentum
    ax1 = axes[0]
    for label, data in pt_datasets.items():
        if data.size > 0:
            ax1.hist(data, bins=100, range=(0, 200), histtype='step', 
                     linewidth=2, density=True, label=label, color=colors[label])
    ax1.set_title("Transverse Momentum Comparison", fontsize=16)
    ax1.set_xlabel("$p_T$ [GeV]", fontsize=12)
    ax1.set_ylabel("Normalized Entries", fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Pseudorapidity
    ax2 = axes[1]
    for label, data in eta_datasets.items():
        if data.size > 0:
            ax2.hist(data, bins=100, range=(-5, 5), histtype='step', 
                     linewidth=2, density=True, label=label, color=colors[label])
    ax2.set_title("Pseudorapidity Comparison", fontsize=16)
    ax2.set_xlabel("$\eta$", fontsize=12)
    ax2.set_ylabel("Normalized Entries", fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Invariant Mass 
    ax3 = axes[2]
    for label, data in mass_datasets.items():
        if data.size > 0:
            ax3.hist(data, bins=100, range=(50, 250), histtype='step', 
                     linewidth=2, density=True, label=label, color=colors[label])
    ax3.set_title("Invariant Mass Comparison ($m_{\mu^+\mu^-}$)", fontsize=16)
    ax3.set_xlabel("$m_{\mu^+\mu^-}$ [GeV]", fontsize=12)
    ax3.set_ylabel("Normalized Entries", fontsize=12)
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("comparison_plots.png")
    print("\nPlots saved as comparison_plots.png")


#  MAIN EXECUTION (Example paths)

if __name__ == "__main__":
    # File paths used for the analysis
    lhe_file = "path/to/unweighted_events.lhe.gz"
    hepmc_file = "path/to/pythia_events.hepmc"
    root_file = "path/to/delphes_events.root"
    
    # Process all data sources
    pt_lhe, eta_lhe, mass_lhe = process_lhe_file(lhe_file)
    pt_hepmc, eta_hepmc, mass_hepmc = process_hepmc_file(hepmc_file)
    pt_root, eta_root, mass_root = process_root_file_uproot(root_file)

    # Organize data for plotting 
    pt_data_all = {"LHE": pt_lhe, "HEPMC": pt_hepmc, "ROOT": pt_root}
    eta_data_all = {"LHE": eta_lhe, "HEPMC": eta_hepmc, "ROOT": eta_root}
    mass_data_all = {"LHE": mass_lhe, "HEPMC": mass_hepmc, "ROOT": mass_root}

    # Create the combined plot
    plot_all_on_one_canvas(pt_data_all, eta_data_all, mass_data_all)
