import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Constants
C        = 299792458.0
E_CONV   = 1.602e-10
E_CHARGE = 1.602e-19
B_FIELD  = 3.8
LAYERS   = np.array([0.044, 0.073, 0.102, 0.25, 0.35, 0.50, 0.70, 0.90, 1.10, 1.29])
Z_MAX    = 2.80

PARTICLE_TABLE = [
    ('pion+'     , 0.13957,  +1),
    ('pion-'     , 0.13957,  -1),
    ('kaon+'     , 0.49368,  +1),
    ('kaon-'     , 0.49368,  -1),
    ('proton'    , 0.93827,  +1),
    ('antiproton', 0.93827,  -1),
    ('electron'  , 0.000511, -1),
    ('positron'  , 0.000511, +1),
    ('muon'      , 0.10566,  -1),
    ('antimuon'  , 0.10566,  +1),
]
NAMES   = [p[0] for p in PARTICLE_TABLE]
MASSES  = {p[0]: p[1] for p in PARTICLE_TABLE}
CHARGES = {p[0]: p[2] for p in PARTICLE_TABLE}

# Kinematic Sampler
def sample_pT(n, pT_min=0.001, pT_max=5.0, pT0=0.5):
    """Truncated exponential - naturally peaks at low pT, falls steeply."""
    u = np.random.uniform(0, 1, n)
    a = np.exp(-pT_min / pT0)
    b = np.exp(-pT_max / pT0)
    return np.clip(-pT0 * np.log(a - u * (a - b)), pT_min, pT_max)

# compute_hits
def compute_hits(px, py, pz, mass_gev, charge, x0=0.0, y0=0.0, z0=0.0, N_rev=5):
    pT      = np.sqrt(px**2 + py**2)
    energy  = np.sqrt(px**2 + py**2 + pz**2 + mass_gev**2)
    gamma_m = energy * E_CONV / C**2
    r       = (pT * E_CONV / C) / (abs(charge) * E_CHARGE * B_FIELD)
    omega   = (charge * E_CHARGE * B_FIELD) / gamma_m
    vz      = (pz * E_CONV / C) / gamma_m
    phi0    = np.arctan2(py, px)
    xc      = x0 - r * np.sin(phi0)
    yc      = y0 + r * np.cos(phi0)

    d_centre = np.sqrt(xc**2 + yc**2)
    r_min    = abs(d_centre - r)
    r_max    = d_centre + r
    T_period = 2 * np.pi / abs(omega)

    layer_hits = {}
    for i, R in enumerate(LAYERS):
        if R < r_min or R > r_max:
            continue

        a      = (R**2 - r**2 + d_centre**2) / (2 * d_centre)
        h      = np.sqrt(np.clip(R**2 - a**2, 0, None))
        ux, uy = xc / d_centre, yc / d_centre

        candidates = [
            (a * ux + h * (-uy),  a * uy + h * (ux)),
            (a * ux - h * (-uy),  a * uy - h * (ux)),
        ]

        hits_on_layer = []
        for ix, iy in candidates:
            phi_hit = np.arctan2(ix - xc, -(iy - yc))
            dphi    = (phi_hit - phi0) / omega
            if dphi < 0:
                dphi += T_period
            for n in range(N_rev):
                t_hit = dphi + n * T_period
                x_hit = xc + r * np.sin(phi0 + omega * t_hit)
                y_hit = yc - r * np.cos(phi0 + omega * t_hit)
                z_hit = z0 + vz * t_hit
                if not np.isclose(np.sqrt(x_hit**2 + y_hit**2), R, rtol=1e-3):
                    continue
                if abs(z_hit) > Z_MAX:
                    break
                hits_on_layer.append((x_hit, y_hit, z_hit, t_hit))

        if hits_on_layer:
            hits_on_layer.sort(key=lambda h: h[3])
            layer_hits[i + 1] = hits_on_layer

    helix_params = dict(r=r, omega=omega, vz=vz, phi0=phi0, xc=xc, yc=yc,
                        x0=x0, y0=y0, z0=z0)
    return energy, layer_hits, helix_params

# wrapper: give pT instead of px, py, pz
def compute_hits_pT(pT, mass_gev, charge, eta=0.0, x0=0.0, y0=0.0, z0=0.0, N_rev=5):
    phi = np.random.uniform(0, 2 * np.pi)
    px  = pT * np.cos(phi)
    py  = pT * np.sin(phi)
    pz  = pT * np.sinh(eta)
    return compute_hits(px, py, pz, mass_gev, charge, x0, y0, z0, N_rev)

# classify track type from geometry
def classify_track(layer_hits, helix_params):
    r        = helix_params['r']
    xc       = helix_params['xc']
    yc       = helix_params['yc']
    d_centre = np.sqrt(xc**2 + yc**2)
    R_outer  = LAYERS[-1]
    n_layers = len(layer_hits)

    if n_layers == 0:
        return 'pure_looper'                     # orbit never reaches any layer
    elif (d_centre + r) < R_outer:
        return 'partial_looper'                  # orbit contained inside detector
    else:
        return 'full_track'                      # orbit escapes outermost layer

# plot_event
def plot_event(event_id, df_hits, df_mom):
    """Plot xy and rz projections for a given event_id."""

    row       = df_mom[df_mom['event_id'] == event_id].iloc[0]
    hits      = df_hits[df_hits['event_id'] == event_id].copy()
    track_type = row['track_type']

    # colour by track type
    color_map  = {
        'pure_looper'    : '#ff6b6b',
        'partial_looper' : '#ffd93d',
        'full_track'     : '#6bcfff',
    }
    color = color_map.get(track_type, 'tomato')

    # recompute helix arc for drawing
    px, py, pz = row['px'], row['py'], row['pz']
    charge     = row['charge']
    energy     = row['E']
    gamma_m    = energy * E_CONV / C**2
    r          = (row['pT'] * E_CONV / C) / (abs(charge) * E_CHARGE * B_FIELD)
    omega      = (charge * E_CHARGE * B_FIELD) / gamma_m
    vz         = (pz * E_CONV / C) / gamma_m
    phi0       = np.arctan2(py, px)
    xc         = -r * np.sin(phi0)
    yc         =  r * np.cos(phi0)
    T_period   = 2 * np.pi / abs(omega)
    t_arr      = np.linspace(0, 5 * T_period, 2000)
    x_arc      = xc + r * np.sin(phi0 + omega * t_arr)
    y_arc      = yc - r * np.cos(phi0 + omega * t_arr)
    z_arc      = vz * t_arr
    r_arc      = np.sqrt(x_arc**2 + y_arc**2)

    fig, (ax_xy, ax_rz) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        f'Event {event_id}   [{track_type.upper().replace("_", " ")}]   '
        f'pT={row["pT"]:.3f} GeV/c   E={row["E"]:.3f} GeV   '
        f'charge={int(row["charge"]):+d}   r_L={r*100:.2f} cm   '
        f'hits={len(hits)}',
        fontsize=9, fontweight='bold'
    )

    for ax in (ax_xy, ax_rz):
        ax.tick_params(colors='#666')
        ax.xaxis.label.set_color('#444')
        ax.yaxis.label.set_color('#444')
        for sp in ax.spines.values():
            sp.set_edgecolor('#ccc')

    # x-y plot
    ax_xy.set_aspect('equal')
    ax_xy.set_xlim(-1.45, 1.45)
    ax_xy.set_ylim(-1.45, 1.45)
    ax_xy.set_xlabel('x [m]')
    ax_xy.set_ylabel('y [m]')
    ax_xy.set_title('x-y view')
    for j, R in enumerate(LAYERS):
        ax_xy.add_patch(plt.Circle((0, 0), R, fill=False,
            color='steelblue' if j < 3 else 'slategray',
            lw=1.0 if j < 3 else 0.5,
            ls='-'  if j < 3 else '--'))
        ax_xy.text(0, R + 0.02, f'L{j+1}', ha='center',
                   color='steelblue' if j < 3 else 'slategray', fontsize=6)
    ax_xy.plot(0, 0, 'k+', ms=8)
    ax_xy.plot(x_arc, y_arc, color=color, lw=1.0)
    ax_xy.scatter(hits['x'], hits['y'],
                  color=color, s=25, zorder=5, edgecolors='black', lw=0.4)
    i_arr = len(t_arr) // 8
    ax_xy.annotate('', xy=(x_arc[i_arr+1], y_arc[i_arr+1]),
                   xytext=(x_arc[i_arr], y_arc[i_arr]),
                   arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # r-z plot
    ax_rz.set_xlim(-300, 300)
    ax_rz.set_ylim(0, 1.45)
    ax_rz.set_xlabel('z [cm]')
    ax_rz.set_ylabel('r [m]')
    ax_rz.set_title('r-z view')
    for R in LAYERS:
        ax_rz.axhline(R, color='slategray', lw=0.5, ls='--')
    ax_rz.plot(z_arc * 100, r_arc, color=color, lw=1.0)
    r_hits = np.sqrt(hits['x']**2 + hits['y']**2)
    ax_rz.scatter(hits['z'] * 100, r_hits,
                  color=color, s=25, zorder=5, edgecolors='black', lw=0.4)

    plt.tight_layout()
    plt.savefig(f'event_{event_id}_pT0-0.5GeV.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"  [{track_type}]  event {event_id}  "
          f"pT={row['pT']:.3f} GeV/c  r_L={r*100:.2f} cm  "
          f"hits={len(hits)}")


# Sample kinematics
N_TOTAL = 10000
np.random.seed(42)

pT_arr   = sample_pT(N_TOTAL)
phi_arr  = np.random.uniform(0, 2 * np.pi, N_TOTAL)
eta_arr  = np.random.uniform(-1.5, 1.5, N_TOTAL)
name_arr = np.random.choice(NAMES, N_TOTAL)

# 15% displaced vertex, 85% prompt
is_displaced = np.random.random(N_TOTAL) < 0.15
d_xy  = np.random.uniform(0.005, 0.05, N_TOTAL)
d_phi = np.random.uniform(0, 2 * np.pi, N_TOTAL)
x0_arr = np.where(is_displaced, d_xy * np.cos(d_phi), 0.0)
y0_arr = np.where(is_displaced, d_xy * np.sin(d_phi), 0.0)
z0_arr = np.where(is_displaced, np.random.uniform(-0.02, 0.02, N_TOTAL), 0.0)

# Run simulation
momentum_rows = []
hit_rows      = []

for event_id in range(N_TOTAL):
    name   = name_arr[event_id]
    mass   = MASSES[name]
    charge = CHARGES[name]
    pT     = pT_arr[event_id]
    phi    = phi_arr[event_id]
    eta    = eta_arr[event_id]
    x0, y0, z0 = x0_arr[event_id], y0_arr[event_id], z0_arr[event_id]

    px = pT * np.cos(phi)
    py = pT * np.sin(phi)
    pz = pT * np.sinh(eta)

    energy, layer_hits, helix_params = compute_hits(
        px, py, pz, mass, charge, x0, y0, z0, N_rev=5
    )

    track_type = classify_track(layer_hits, helix_params)

    # input_momentum rows
    momentum_rows.append({
        'event_id'   : event_id,
        'particle'   : name,
        'charge'     : charge,
        'px'         : round(px,     6),
        'py'         : round(py,     6),
        'pz'         : round(pz,     6),
        'pT'         : round(pT,     6),
        'E'          : round(energy, 6),
        'track_type' : track_type,
        'x0' : round(x0, 6),
        'y0' : round(y0, 6),
        'z0' : round(z0, 6),
    })

    # output_hits rows
    hit_n = 1
    for layer in sorted(layer_hits.keys()):
        for (x, y, z, t) in layer_hits[layer]:
            hit_rows.append({
                'event_id' : event_id,
                'layer'    : layer,
                'hit_n'    : hit_n,
                'x'        : round(x,       5),
                'y'        : round(y,       5),
                'z'        : round(z,       5),
                't_ns'     : round(t * 1e9, 5),
                'hit'      : 1,
            })
            hit_n += 1
            
# Save CSVs
df_mom  = pd.DataFrame(momentum_rows)
df_hits = pd.DataFrame(hit_rows)

df_mom.to_csv( 'input_momentum_pT0-0.5GeV.csv', index=False)
df_hits.to_csv('output_hits_pT0-0.5GeV.csv',    index=False)

# Summary + first 5 rows
print(f"  Simulation complete")
print(f"  Events : {len(df_mom):,}")
print(f"  Hits   : {len(df_hits):,}   "
      f"(avg {len(df_hits)/len(df_mom):.1f} hits/event)")

print("input_momentum_pT0-0.5GeV.csv - first 5 rows:")
print(df_mom.head().to_string(index=False))
print()
print("output_hits_pT0-0.5GeV.csv - first 5 rows:")
print(df_hits.head().to_string(index=False))
displaced_events = df_mom[np.sqrt(df_mom['x0']**2 + df_mom['y0']**2 + df_mom['z0']**2) > 0.001]['event_id'].values
print('displaced events - ', displaced_events)

# Plot a desired event  (change event_id to any value 0-9999)
plot_event(event_id=136, df_hits=df_hits, df_mom=df_mom)
