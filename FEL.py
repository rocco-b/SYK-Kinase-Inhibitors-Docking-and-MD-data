import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ==========================================
# PUBLICATION QUALITY PLOT SETTINGS
# ==========================================
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.linewidth': 1.5,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']
})

# ==========================================
# SYSTEM PARAMETERS
# ==========================================
# Input files
ligand_files = {
    "vemurafenib": "1836_minimized_pca.tab",
    "palbociclib": "1863_minimized_pca.tab",
    "imatinib": "2061_minimized_pca.tab"
}

T = 300.0  # Simulation temperature in Kelvin
R = 0.0019872041  # Ideal gas constant in kcal/(mol·K)
KT = R * T
G_cut = 6.0  # Maximum Free Energy cutoff (kcal/mol)
colormap = 'RdYlBu_r'

# ==========================================
# FEL CALCULATION & PLOTTING (Independent Axes)
# ==========================================
print("Calculating Free Energy Landscapes...")

for ligand_name, file_path in ligand_files.items():
    print(f"Processing {ligand_name}...")
    
    # 1. READ AND CLEAN DATA
    df = pd.read_csv(file_path, sep=r'\s+', comment='@')
    if '#' in df.columns[0]:  
        df.columns = df.columns.str.replace('#', '').str.strip()
        
    time_col = df.columns[0]
    df = df[~df[time_col].astype(str).str.contains('_')]
    
    df['PCA1'] = pd.to_numeric(df['PCA1'])
    df['PCA2'] = pd.to_numeric(df['PCA2'])
    
    x = df['PCA1'].values
    y = df['PCA2'].values
    
    pad_x = (x.max() - x.min()) * 0.45
    pad_y = (y.max() - y.min()) * 0.45
    
    xmin, xmax = x.min() - pad_x, x.max() + pad_x
    ymin, ymax = y.min() - pad_y, y.max() + pad_y
    
    grid_points = 100j
    X, Y = np.mgrid[xmin:xmax:grid_points, ymin:ymax:grid_points]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    
    # 3. KERNEL DENSITY ESTIMATION & BOLTZMANN INVERSION
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    
    Z_safeguard = np.where(Z > 0, Z, Z[Z > 0].min())
    G = -KT * np.log(Z_safeguard)
    
    # Normalize data
    G = G - G.min()
    G[G > G_cut] = G_cut

    # ------------------------------------------
    # 2D CONTOUR PLOT
    # ------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    
    contour_fill = ax.contourf(X, Y, G, levels=40, cmap=colormap, vmin=0, vmax=G_cut)
    contour_lines = ax.contour(X, Y, G, levels=[1.0, 2.0, 3.0, 4.0, 5.0], 
                               colors='black', linewidths=0.8, alpha=0.7)
    ax.clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=10)
    
    cbar = plt.colorbar(contour_fill, ax=ax)
    cbar.set_label('Free Energy $\Delta G$ (kcal/mol)', rotation=270, labelpad=20)
    
    #ax.set_title(f'{ligand_name}', pad=15, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
        
    plt.savefig(f'FEL_2D_{ligand_name}.png', dpi=600, bbox_inches='tight')
    plt.close()

    # ------------------------------------------
    # 3D SURFACE PLOT
    # ------------------------------------------
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, G, cmap=colormap, edgecolor='none', 
                           rstride=1, cstride=1, alpha=0.9, vmin=0, vmax=G_cut)
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.1)
    cbar.set_label('$\Delta G$ (kcal/mol)')
    
    #ax.set_title(f'{ligand_name}', pad=10, fontweight='bold')
    ax.set_xlabel('PC1', labelpad=10)
    ax.set_ylabel('PC2', labelpad=10)
    ax.set_zlabel('$\Delta G$ (kcal/mol)', labelpad=10)
    ax.set_zlim(0, G_cut)
        
    ax.view_init(elev=35, azim=45)
    
    plt.savefig(f'FEL_3D_{ligand_name}.png', dpi=600, bbox_inches='tight')
    plt.close()

print("\nProcess completed!")