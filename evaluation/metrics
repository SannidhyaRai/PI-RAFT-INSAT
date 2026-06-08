import numpy as np
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def calculate_amv_metrics(u_pred, v_pred, u_true, v_true, mask=None):
    """
    Calculates standard meteorological validation metrics for AMVs.
    Tensors can be shape (B, H, W) or flattened vectors.
    """
    if mask is not None:
        u_pred = u_pred[mask]
        v_pred = v_pred[mask]
        u_true = u_true[mask]
        v_true = v_true[mask]
        
    # 1. Vector Difference 
    u_diff = u_pred - u_true
    v_diff = v_pred - v_true
    vector_difference = torch.sqrt(u_diff**2 + v_diff**2)
    
    # 2. RMSVD (Root Mean Square Vector Difference)
    rmsvd = torch.sqrt(torch.mean(vector_difference**2))
    
    # 3. MVD (Mean Vector Difference)
    mvd = torch.mean(vector_difference)
    
    # 4. Speed Bias (Predicted Speed - True Speed)
    speed_pred = torch.sqrt(u_pred**2 + v_pred**2)
    speed_true = torch.sqrt(u_true**2 + v_true**2)
    bias = torch.mean(speed_pred - speed_true)
    
    return {
        "RMSVD": rmsvd.item(),
        "MVD": mvd.item(),
        "Speed_Bias": bias.item()
    }

def plot_insat3ds_diagnostic(satellite_img, u_pred, v_pred, save_path="insat_diagnostic.png"):
    """
    Plots PI-RAFT wind vectors overlaid on the INSAT-3DS Geostationary Full Disk (82E).
    """
    fig = plt.figure(figsize=(10, 10))
    
    # Set up the Geostationary projection exactly at INSAT-3DS orbital slot
    insat_proj = ccrs.Geostationary(central_longitude=82.0, satellite_height=35786000)
    
    ax = plt.axes(projection=insat_proj)
    ax.set_global()
    
    # Add geographical outlines for context over South Asia/Indian Ocean
    ax.add_feature(cfeature.COASTLINE, linestyle='-', edgecolor='black', linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray', linewidth=0.8)
    
    # Display the background channel (e.g., Thermal IR Brightness Temperature)
    # Extent bounds must match your crop or the full disk coordinates
    im = ax.imshow(satellite_img, cmap='gray', transform=insat_proj, origin='upper')
    
    # Grid of coordinates for vector arrows (subsample for visual cleanliness)
    h, w = u_pred.shape
    x, y = np.meshgrid(np.linspace(-5500000, 5500000, w), np.linspace(5500000, -5500000, h))
    
    skip = max(1, h // 32)  # Show roughly a 32x32 grid of arrows to avoid clutter
    
    # Overlay the predicted vectors as arrows
    ax.quiver(x[::skip, ::skip], y[::skip, ::skip], 
              u_pred[::skip, ::skip], v_pred[::skip, ::skip], 
              color='cyan', scale=500, transform=insat_proj, headwidth=3)
    
    plt.title("PI-RAFT Predicted AMV Fields over INSAT-3DS (82°E)", fontsize=14)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

# --- QUICK VERIFICATION UNIT TEST ---
if __name__ == "__main__":
    print("Testing Evaluation Suite with Mock INSAT-3DS Data...")
    # Simulate a 256x256 patch of a TIR channel
    mock_img = np.sin(np.linspace(0, 10, 256))[:, None] * np.cos(np.linspace(0, 10, 256))[None, :]
    
    # Mock physical wind speed grids (m/s)
    mock_u_true = torch.ones((256, 256)) * 15.0
    mock_v_true = torch.ones((256, 256)) * -5.0
    
    # Add random errors to simulate model predictions
    mock_u_pred = mock_u_true + torch.randn((256, 256)) * 2.0
    mock_v_pred = mock_v_true + torch.randn((256, 256)) * 2.0
    
    metrics = calculate_amv_metrics(mock_u_pred, mock_v_pred, mock_u_true, mock_v_true)
    print(f"Calculated Verification Metrics: {metrics}")
    
    # Test if plotting runs without coordinate crashes
    plot_insat3ds_diagnostic(mock_img, mock_u_pred.numpy(), mock_v_pred.numpy())
    print("Success: Diagnostics saved to insat_diagnostic.png")
