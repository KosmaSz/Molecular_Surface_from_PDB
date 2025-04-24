# Molecular_Surface_from_PDB
# 3D Surface and Volume Analysis of Molecular Structures from PDB Files

## Project Description  
This Python script automates the calculation of surface area and enclosed volume for molecular structures stored in PDB format. Its core functionality relies on the α-shape algorithm, with adaptive parameter selection, and provides robust fallbacks to ensure reliable geometry extraction.

## Key Features  
- **α-Shape Surface and Volume Calculation**  
  - Heuristic determination of the α parameter from the mean distance to the kᵗʰ nearest neighbor (k=6, scaling factor α_F=1.8).  
  - Primary backend: alphashape (wrapped by PyVista).  
  - Secondary backend: Delaunay 3D triangulation via PyVista.  
  - Tertiary fallback: Convex Hull algorithm (scipy.spatial.ConvexHull) for minimal overestimation when mesh generation fails.  
- **Batch Processing**  
  - Sequential (single-core) or optional parallel execution via joblib for high-throughput analysis of large PDB sets.  
- **Results Export**  
  - Aggregates simulation time, surface area (Å²), volume (Å³), and atom count into a CSV file for downstream analysis.

## Use Cases  
- Quantitative comparison of molecular morphologies across simulation snapshots.  
- Monitoring conformational changes by tracking surface and volume over time.  
- Generating datasets of geometric descriptors for machine-learning or statistical modeling.  

## Quick Start  
1. Clone the repository  
   ```bash
   git clone https://github.com/KosmaSz/Molecular_Surface_from_PDB.git  
   cd your-repository  
   ```  
2. Install dependencies  
   ```bash
   pip install -r requirements.txt  
   ```  
3. Prepare PDB input  
   - Place all `*.pdb` files into the `PDB/` folder.  
4. Run the analysis  
   ```bash
   python start.py or paste it to the jupyter notebook 
   ```  

---

**Customization:** Adjust input directory, filename pattern, atom selection, and α-shape parameters in the script header to suit your dataset.
