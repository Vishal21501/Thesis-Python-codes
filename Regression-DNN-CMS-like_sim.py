import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

DATA_DIR = '/content/drive/MyDrive/CMS-Simple-Sim/no_smearing/'

# Config
FEATURES    = ['x', 'y', 'z', 't_ns', 'hit', 'layer']
TARGET_COLS = ['px', 'py', 'pz', 'pT', 'E', 'charge']
colors      = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800', '#9C27B0', '#607D8B']
units       = {'px':'GeV/c', 'py':'GeV/c', 'pz':'GeV/c',
               'pT':'GeV/c', 'E':'GeV', 'charge':'e'}

LAYERS      = list(range(1, 11))
BATCH_SIZE  = 512
EPOCHS      = 500
LR          = 5e-4
PATIENCE    = 50            # early stopping patience

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Load data
print("\nLoading data...")
hits = pd.read_csv(DATA_DIR + 'output_hits_no-loopers_pT_max-5GeV.csv')
mom  = pd.read_csv(DATA_DIR + 'input_momentum_no-loopers_pT_max-5GeV.csv')

# Pivot: all hits per event flattened by hit_n
# Each event has variable hits - flatten by hit_n, zero-pad to MAX_HITS
MAX_HITS     = 100
hit_features = [f'hit{h}_{feat}' for h in range(1, MAX_HITS + 1) for feat in FEATURES]

print("  Pivoting hits to flat feature matrix...")
hits['hit_n_capped'] = hits['hit_n'].clip(upper=MAX_HITS)

hits_flat = hits.pivot_table(
    index='event_id',
    columns='hit_n_capped',
    values=FEATURES,
    aggfunc='first'
)
hits_flat.columns = [f'hit{int(h)}_{feat}' for feat, h in hits_flat.columns]
hits_flat = hits_flat.reindex(columns=hit_features, fill_value=0.0)

df = hits_flat.merge(
    mom.set_index('event_id')[TARGET_COLS],
    left_index=True, right_index=True
)
df = df.fillna(0.0)
feature_cols = hit_features
print(f"  Events: {len(df)}   Features per event: {len(feature_cols)}")

X = df[feature_cols].values.astype(np.float32)
y = df[TARGET_COLS].values.astype(np.float32)

# Scale
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_sc = scaler_X.fit_transform(X)
y_sc = scaler_y.fit_transform(y)

# Train / test split
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X_sc, y_sc, np.arange(len(df)), test_size=0.2, random_state=42)
print(f"  Train: {len(X_tr)}  |  Test: {len(X_te)}")

# Convert to tensors
X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)
X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)
y_te_t = torch.tensor(y_te, dtype=torch.float32).to(device)

train_loader = DataLoader(
    TensorDataset(X_tr_t, y_tr_t),
    batch_size=BATCH_SIZE, shuffle=True
)

# DNN model
class TrackDNN(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 512),  nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.1),
            nn.Linear(512, 256),  nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 128),  nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64),   nn.ReLU(),
            nn.Linear(64,  n_out)
        )
    def forward(self, x):
        return self.net(x)

n_features = X_tr.shape[1]
n_targets  = y_tr.shape[1]
model      = TrackDNN(n_features, n_targets).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n  Model parameters: {total_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5)
criterion = nn.MSELoss()

# Training loop
print(f"\nTraining on {device} for up to {EPOCHS} epochs...")
print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'LR':>10}")

train_losses = []
val_losses   = []
best_val     = float('inf')
patience_ctr = 0
best_weights = None

for epoch in range(1, EPOCHS + 1):
    # train
    model.train()
    batch_losses = []
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_loss = np.mean(batch_losses)

    # validate
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_te_t), y_te_t).item()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    # print every 10 epochs
    if epoch % 10 == 0 or epoch == 1:
        lr_now = optimizer.param_groups[0]['lr']
        print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  {lr_now:>10.2e}")

    # early stopping
    if val_loss < best_val - 1e-7:
        best_val     = val_loss
        patience_ctr = 0
        best_weights = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}  (best val loss={best_val:.6f})")
            break

# restore best weights
model.load_state_dict(best_weights)
print(f"\n  Training complete. Best val loss: {best_val:.6f}")

# Predict & inverse-scale
model.eval()
with torch.no_grad():
    y_pred_sc = model(X_te_t).cpu().numpy()

y_pred = scaler_y.inverse_transform(y_pred_sc)
y_true = scaler_y.inverse_transform(y_te)

# round charge to nearest integer
i_charge              = TARGET_COLS.index('charge')
y_pred_charge_rounded = np.round(y_pred[:, i_charge]).astype(int)
y_true_charge         = np.round(y_true[:, i_charge]).astype(int)

# R2 scores
print("\nR2 scores on test set:")
r2 = {}
for i, t in enumerate(TARGET_COLS):
    r2[t] = r2_score(y_true[:, i], y_pred[:, i])
    print(f"  {t:<8}: {r2[t]:.5f}")

charge_acc = accuracy_score(y_true_charge, y_pred_charge_rounded) * 100
print(f"\n  Charge classification accuracy: {charge_acc:.2f}%")

# Save predictions CSV
event_ids = df.index.values[idx_te]
df_out    = pd.DataFrame({'event_id': event_ids})
for i, t in enumerate(TARGET_COLS):
    if t == 'charge':
        df_out['charge_pred'] = y_pred_charge_rounded
        df_out['charge_true'] = y_true_charge
    else:
        df_out[f'{t}_pred'] = np.round(y_pred[:, i], 6)
        df_out[f'{t}_true'] = np.round(y_true[:, i], 6)

pred_path = DATA_DIR + 'dnn_predictions_pT_max-5GeV.csv'
df_out.to_csv(pred_path, index=False)
print(f"\nSaved -> {pred_path}  ({len(df_out)} test events)")

# Generate organized plots
print("\nGenerating performance plots...")

# Canvas 1: R**2 scatter plots (px, py, pz, pT, E, charge)
fig1 = plt.figure(figsize=(18, 12))
fig1.suptitle('R**2 Scatter Plots - True vs Predicted pT_max-5Gev', fontsize=14, fontweight='bold')
gs1 = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.3, wspace=0.3)

for idx, t in enumerate(['px', 'py', 'pz', 'pT', 'E', 'charge']):
    i = TARGET_COLS.index(t)
    ax = fig1.add_subplot(gs1[idx // 3, idx % 3])

    if t == 'charge':
        # For charge, use jitter to see overlapping points
        jitter = 0.05
        y_true_jitter = y_true[:, i] + np.random.normal(0, jitter, len(y_true[:, i]))
        y_pred_jitter = y_pred[:, i] + np.random.normal(0, jitter, len(y_pred[:, i]))
        ax.scatter(y_true_jitter, y_pred_jitter, alpha=0.15, s=5, color=colors[i])
    else:
        vmin = min(y_true[:, i].min(), y_pred[:, i].min())
        vmax = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.12, s=5, color=colors[i])
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1.2)

    if t == 'charge':
        ax.plot([-1.5, 1.5], [-1.5, 1.5], 'k--', lw=1.2)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_title(f'{t}   R**2={r2[t]:.4f}\nacc={charge_acc:.1f}%')
    else:
        ax.set_title(f'{t}   R**2={r2[t]:.4f}')

    ax.set_xlabel(f'True {t} ({units[t]})')
    ax.set_ylabel(f'Pred {t} ({units[t]})')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(DATA_DIR + 'canvas1_R2_scatter_plots_pT_max-5GeV.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved -> canvas1_R2_scatter_plots_pT_max-5GeV.png")

# Canvas 2: Residual plots (px, py, pz, pT, E, charge)
fig2 = plt.figure(figsize=(18, 12))
fig2.suptitle('Residual Distributions pT_max-5Gev', fontsize=14, fontweight='bold')
gs2 = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.3, wspace=0.3)

for idx, t in enumerate(['px', 'py', 'pz', 'pT', 'E', 'charge']):
    i = TARGET_COLS.index(t)
    ax = fig2.add_subplot(gs2[idx // 3, idx % 3])

    if t == 'charge':
        # For charge, show classification errors
        correct = (y_pred_charge_rounded == y_true_charge)
        wrong = ~correct
        ax.hist(y_true_charge[correct], bins=[-1.5, -0.5, 0.5, 1.5], alpha=0.5,
                color='green', label='Correct', rwidth=0.8)
        ax.hist(y_true_charge[wrong], bins=[-1.5, -0.5, 0.5, 1.5], alpha=0.5,
                color='red', label='Wrong', rwidth=0.8)
        ax.set_xticks([-1, 0, 1])
        ax.set_xlabel(f'True {t} (e)')
        ax.set_ylabel('Count')
        ax.set_title(f'{t} classification\naccuracy={charge_acc:.1f}%')
        ax.legend()
    else:
        res = y_pred[:, i] - y_true[:, i]
        ax.hist(res, bins=70, color=colors[i], alpha=0.75, edgecolor='none')
        ax.axvline(0, color='black', lw=1.3, linestyle='--')
        ax.set_xlabel(f'D{t} ({units[t]})')
        ax.set_ylabel('Count')
        ax.set_title(f'{t} residuals\nmu={res.mean():.3f}  sigma={res.std():.3f}')

    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(DATA_DIR + 'canvas2_residual_plots_pT_max-5GeV.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved -> canvas2_residual_plots_pT_max-5GeV.png")

# Canvas 3: Training curve and relative errors
fig3 = plt.figure(figsize=(14, 10))
fig3.suptitle('Training Progress & Relative Errors pT_max-5Gev', fontsize=14, fontweight='bold')
gs3 = gridspec.GridSpec(2, 2, figure=fig3, hspace=0.3, wspace=0.3)

# Training curve
ax1 = fig3.add_subplot(gs3[0, 0])
ax1.plot(train_losses, color='steelblue', lw=1.8, label='Train loss')
ax1.plot(val_losses, color='tomato', lw=1.8, label='Val loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss')
ax1.set_title('Training curve')
ax1.legend()
ax1.grid(True, alpha=0.3)

# pT relative error
i_pT = TARGET_COLS.index('pT')
i_E = TARGET_COLS.index('E')
ax2 = fig3.add_subplot(gs3[0, 1])
rel_pT = np.abs(y_pred[:, i_pT] - y_true[:, i_pT]) / (np.abs(y_true[:, i_pT]) + 1e-9) * 100
ax2.hist(rel_pT, bins=70, color=colors[i_pT], alpha=0.75, edgecolor='none')
ax2.set_xlabel('|DpT|/|pT| (%)')
ax2.set_ylabel('Count')
ax2.set_title(f'pT relative error\nmedian={np.median(rel_pT):.2f}%')
ax2.grid(True, alpha=0.3)

# E relative error
ax3 = fig3.add_subplot(gs3[1, 0])
rel_E = np.abs(y_pred[:, i_E] - y_true[:, i_E]) / (np.abs(y_true[:, i_E]) + 1e-9) * 100
ax3.hist(rel_E, bins=70, color=colors[i_E], alpha=0.75, edgecolor='none')
ax3.set_xlabel('|DE|/|E| (%)')
ax3.set_ylabel('Count')
ax3.set_title(f'E relative error\nmedian={np.median(rel_E):.2f}%')
ax3.grid(True, alpha=0.3)

# Summary text
ax4 = fig3.add_subplot(gs3[1, 1])
ax4.axis('off')
summary = (
    f"Summary\n"
    f"Events (train) : {len(X_tr):>7,}\n"
    f"Events (test)  : {len(X_te):>7,}\n"
    f"Epochs trained : {len(train_losses):>7,}\n"
    f"{'-'*28}\n"
    f"R2 px     : {r2['px']:>8.4f}\n"
    f"R2 py     : {r2['py']:>8.4f}\n"
    f"R2 pz     : {r2['pz']:>8.4f}\n"
    f"R2 pT     : {r2['pT']:>8.4f}\n"
    f"R2 E      : {r2['E']:>8.4f}\n"
    f"R2 charge : {r2['charge']:>8.4f}\n"
    f"Charge acc: {charge_acc:>7.1f}%\n"
    f"pT rel err: {np.median(rel_pT):>7.2f}%\n"
    f"E  rel err: {np.median(rel_E):>7.2f}%\n"
)
ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))

plt.tight_layout()
plt.savefig(DATA_DIR + 'canvas3_training_and_errors_pT_max-5GeV.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved -> canvas3_training_and_errors_pT_max-5GeV.png")

# Save model checkpoint
model_path = DATA_DIR + 'dnn_model_pT_max-5GeV.pt'
torch.save({
    'model_state_dict'    : model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_X_mean'       : scaler_X.mean_,
    'scaler_X_scale'      : scaler_X.scale_,
    'scaler_y_mean'       : scaler_y.mean_,
    'scaler_y_scale'      : scaler_y.scale_,
    'feature_cols'        : feature_cols,
    'target_cols'         : TARGET_COLS,
    'best_val_loss'       : best_val,
    'epochs_trained'      : len(train_losses),
}, model_path)
print(f"Saved -> {model_path}")
