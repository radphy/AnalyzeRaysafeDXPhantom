import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from skimage import exposure
from skimage.filters import median
from skimage.morphology import disk

def wedge_segmentation(I, radius, r_outer_scale=74/80, r_inner_scale=56/80):
    """
    Segments the Pb-step phantom into 16 wedge segments + central region.
    Performs the polar unwrap and angular profile peak detection.
    """
    # 1. Pre-processing
    I_med = median(I, disk(1)) # 3x3 median filter
    I_adj = exposure.equalize_adapthist(I_med, nbins=256, clip_limit=0.01)
    
    h, w = I.shape
    ctr = np.array([w / 2.0, h / 2.0])
    r_outer = round(radius * r_outer_scale)
    r_inner = round(radius * r_inner_scale)
    
    # 2. Build masks & Polar Unwrap
    y, x = np.ogrid[:h, :w]
    dist_sq = (x - ctr[0])**2 + (y - ctr[1])**2
    mask_ring = (dist_sq <= r_outer**2) & (dist_sq >= r_inner**2)
    
    N = 4 # Sampling multiplier
    delta = 1.0 / N
    theta_samples = np.arange(0, 360 + delta, delta)
    rho_samples = np.arange(r_inner, r_outer + 1)
    
    Theta, Rho = np.meshgrid(np.deg2rad(theta_samples), rho_samples)
    Xi = ctr[0] + Rho * np.cos(Theta)
    Yi = ctr[1] + Rho * np.sin(Theta)
    
    # Map coordinates (replaces sub2ind logic)
    I_pol = ndimage.map_coordinates(I_adj, [Yi, Xi], order=1, mode='constant', cval=0.0)
    
    # Apply ring mask constraint to polar
    ring_rows = np.any(I_pol > 0, axis=1)
    i_prof = np.median(I_pol[ring_rows, :], axis=0)
    
    # 3. Angular profile & edge detection
    dprof = np.diff(i_prof)
    
    # Find strongest edges (wedge boundaries)
    peaks, _ = find_peaks(np.abs(dprof), distance=13*N, prominence=0.04)
    peaks = np.sort(peaks)
    angles = peaks / N
    
    # Interpolate missing expected angles (similar to the MATLAB logic)
    Nexp = 18
    if len(angles) > 1:
        total_span = angles[-1] - angles[0]
        spacing = total_span / (Nexp - 1)
        tol = spacing / 2.0
        
        expected = angles[0] + spacing * np.arange(Nexp)
        # Check presence
        present = np.abs(angles[:, None] - expected[None, :]) < tol
        is_present = np.max(present, axis=0)
        missing = expected[~is_present]
        
        angles_complete = np.sort(np.concatenate([angles, missing]))
        locs = np.round(angles_complete * N).astype(int)
    else:
        locs = peaks # Fallback
        
    # 4. Reconstruct Labels in Polar, Map to Cartesian
    Nth = len(i_prof)
    L_pol = np.zeros_like(I_pol, dtype=int)
    
    for k in range(1, 17): # 16 wedges
        if k < len(locs):
            c1 = locs[k-1] + 1
            c2 = locs[k] - 1
            if c2 <= Nth:
                L_pol[ring_rows, c1:c2] = k
            else:
                L_pol[ring_rows, c1:Nth] = k
                L_pol[ring_rows, 0:(c2 % Nth)] = k
                
    # Inverse map back to cartesian coordinates
    dx = x - ctr[0]
    dy = y - ctr[1]
    rho_img = np.sqrt(dx**2 + dy**2)
    theta_img = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    
    i_rho = np.round(rho_img - r_inner).astype(int)
    i_theta = np.round(np.rad2deg(theta_img) / delta).astype(int)
    
    # Clamp and wrap
    i_theta[i_theta >= Nth] = i_theta[i_theta >= Nth] - Nth
    i_theta[i_theta < 0] = i_theta[i_theta < 0] + Nth
    
    L_cart = np.zeros((h, w), dtype=int)
    valid = mask_ring & (i_rho >= 0) & (i_rho < len(rho_samples))
    
    # Extract valid coordinates
    v_y, v_x = np.nonzero(valid)
    valid_rho = i_rho[v_y, v_x]
    valid_theta = i_theta[v_y, v_x]
    
    L_cart[v_y, v_x] = L_pol[valid_rho, valid_theta]
    
    return L_cart

def calculate_cnr(sig_vals, bg_vals):
    """
    Standard CNR Calculation: (Mean_Sig - Mean_Bg) / Std_Bg
    """
    mean_sig = np.mean(sig_vals)
    mean_bg = np.mean(bg_vals)
    std_bg = np.std(bg_vals)
    
    if std_bg == 0:
        return 0.0 # Prevent division by zero
        
    return (mean_sig - mean_bg) / std_bg