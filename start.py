# ================================================================
# The code was supported with AI DeepSeek-R1
# Version: 2024.06
# kosma.szutkowski@gmail.com 
# Date: 24 April 2025
# Usage: put all your PDB files from MD simulation in the directory.
# ================================================================
from __future__ import annotations
import os, re, math, logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from tqdm.notebook import tqdm    
from MDAnalysis import Universe
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull, QhullError
import pyvista as pv                    

try:
    import alphashape                   
    USE_ALPHASHAPE = True
except ImportError:
    alphashape = None
    USE_ALPHASHAPE = False               # fallback: PyVista/ConvexHull

# —— konfiguracja usera ——————————————————————————————
INPUT_DIR   = Path("PDB1")
PATTERN     = "pda_swelling*.pdb"
SELECTION   = "element C N O H"  
N_PROC      = max(1, (os.cpu_count() or 4)//2)
K_NN        = 6
ALPHA_F     = 1.8
OUTPUT_CSV  = "alpha_area_volume.csv"
LOG_FILE    = "alpha.log"

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(processName)s :: %(message)s")

# ================================================================
# Aux functions
# ================================================================
_DIG = re.compile(r"\d+")

def extract_time_ps(stem: str) -> float:
    m = _DIG.search(stem); return int(m.group()) if m else math.nan
#   m = _DIG.search(stem); return int(m.group())/1000 if m else math.nan
def ensure_xyz(arr: np.ndarray) -> np.ndarray:
    """Matrix (N, 3)."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        if arr.size % 3:
            raise ValueError("Number of coordinates not multiplies of 3")
        arr = arr.reshape(-1, 3)
    return arr

def alpha_stats(coords: np.ndarray) -> tuple[float, float]:
    """
    Surface Å², Volume Å³ alpha‑shape or PyVista+Convex methods.
    """
    coords = ensure_xyz(coords)
    n = coords.shape[0]
    if n < 4:
        raise ValueError("≥4 points required for 3D volume")

    # — heurystyczny α —
    k = min(K_NN, n-1)                   
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    mean_nn = dists[:, 1:].mean()        
    alpha = ALPHA_F * mean_nn

    # — alfa‑shape or PyVista —
    try:
        if USE_ALPHASHAPE:
            shell = pv.wrap(alphashape.alphashape(coords, alpha))
        else:
            shell = pv.PolyData(coords).delaunay_3d(alpha=alpha).extract_geometry()
        return shell.area, shell.volume
    except Exception:
        # alternative approach
        hull = ConvexHull(coords)
        return hull.area, hull.volume

def analyse_one(pdb: Path) -> Dict | None:
    try:
        u  = Universe(str(pdb))
        at = u.select_atoms(SELECTION)
        if len(at) < 4:
            raise ValueError("Too few atoms in the selection")

        area, vol = alpha_stats(at.positions)

        return dict(file=pdb.name,
                    time_ps=extract_time_ps(pdb.stem),
                    surface_A2=round(area, 2),
                    volume_A3 =round(vol,  2),
                    n_atoms=len(at))
    except Exception as e:
        logging.error(f"{pdb.name}: {e}")
        return None

def natural_key(p: Path):
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r"(\d+)", p.stem)]

# ================================================================
# Files stats
# ================================================================
pdb_files: List[Path] = sorted(Path(INPUT_DIR).glob(PATTERN), key=natural_key)
if not pdb_files:
    raise SystemExit(f"❌ No files {PATTERN} w {INPUT_DIR.resolve()}")

print(f"▪ Files: {len(pdb_files)} | proc={N_PROC} | "
      f"backend={'alphashape' if USE_ALPHASHAPE else 'pyvista+convex'}")

# ================================================================
# Parallel processing. Tested on apple silicon M4
# ================================================================
results = Parallel(n_jobs=N_PROC, backend="loky")(
    delayed(analyse_one)(p) for p in tqdm(pdb_files, desc="Alpha 3‑D")
)
results = [r for r in results if r]

if not results:
    raise RuntimeError("All snapshots were not accepted – check alpha.log")

# ================================================================
# Data 
# ================================================================
df = (pd.DataFrame.from_records(results)
        .dropna(subset=["time_ps"])
        .sort_values("time_ps"))
df.to_csv(OUTPUT_CSV, index=False)
print(f"✓ Results written to {OUTPUT_CSV}")
df.head()           


# ===============  Plotting =========================================

CSV_FILE = "alpha_area_volume.csv"   

FIGSIZE_SURF = (4, 4)                   
FIGSIZE_VOL  = (4, 4)                     

X_SCALE        = 'linear'
Y_SCALE_SURF   = 'linear'
Y_SCALE_VOL    = 'linear'

X_LIM          = (0, 5000)        

Y_LIM_SURF     = None           
Y_LIM_VOL      = (8000,13000)          

# ====================================================================

df = pd.read_csv(CSV_FILE)

# Surface plot

plt.figure(figsize=FIGSIZE_SURF)
plt.plot(df["time_ps"], df["surface_A2"])
plt.xlabel("Time [ps]")
plt.ylabel("Surface [Å²]")
plt.title("PDA surface vs time")
plt.xscale(X_SCALE); plt.yscale(Y_SCALE_SURF)
if X_LIM:      plt.xlim(*X_LIM)
if Y_LIM_SURF: plt.ylim(*Y_LIM_SURF)
plt.tight_layout(); plt.show()

# Volume plot
plt.figure(figsize=FIGSIZE_VOL)
plt.plot(df["time_ps"], df["volume_A3"])
plt.xlabel("Time [ps]")
plt.ylabel("Volume [Å³]")
plt.title("PDA volume vs time")
plt.xscale(X_SCALE); plt.yscale(Y_SCALE_VOL)
if X_LIM:     plt.xlim(*X_LIM)
if Y_LIM_VOL: plt.ylim(*Y_LIM_VOL)
plt.tight_layout(); plt.show()
