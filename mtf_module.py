import numpy as np
from scipy.optimize import curve_fit, fsolve
from scipy.interpolate import PchipInterpolator
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def get_bar_direction(mask):
    """
    Equivalent to getBarDirection. Uses PCA to find the principal axis of the mask.
    """
    y, x = np.nonzero(mask)
    if len(x) == 0:
        return np.array([1, 0])
    
    coords = np.column_stack((x, y))
    pca = PCA(n_components=2)
    pca.fit(coords)
    return pca.components_[0]

def fit_two_gauss_unsharp_mask_mtf0(f, C):
    """
    Fits sampled MTF using a two-Gaussian detector model multiplied by a 
    one-band unsharp-mask sharpening stage.
    """
    f = np.asarray(f, dtype=float)
    C = np.asarray(C, dtype=float)
    
    # Enforce MTF(0) = 1
    MTF = C / C[0]
    fmax = np.max(f)
    
    # Mathematical Model
    def model(x, w, s1, s2, beta, ss):
        # Two-Gaussian detector MTF
        two_gauss = w * np.exp(-((np.pi / s1 * x)**2) / 4) + (1 - w) * np.exp(-((np.pi / s2 * x)**2) / 4)
        # One-band unsharp mask
        um = 1 + beta * (1 - np.exp(-((np.pi / ss * x)**2) / 4))
        return two_gauss * um

    # Initial Guesses (t0) and Bounds (lb, ub) matched exactly to MATLAB
    p0 = [0.5, 0.8 * fmax, 0.4 * fmax, 0.5, 2.5]
    bounds = (
        [0.0, 0.2 * fmax, 0.05 * fmax, 0.3, 0.5], # Lower bounds
        [1.0, np.inf, np.inf, 1.0, 10.0]          # Upper bounds
    )
    
    try:
        # curve_fit utilizes Levenberg-Marquardt or TRF (Trust Region Reflective for bounded problems)
        theta_hat, _ = curve_fit(model, f, MTF, p0=p0, bounds=bounds)
    except Exception as e:
        print(f"Curve fitting failed: {e}")
        theta_hat = p0 # Fallback
        
    # Calculate Goodness of Fit (R^2)
    MTF_fit = model(f, *theta_hat)
    RSS = np.sum((MTF - MTF_fit)**2)
    TSS = np.sum((MTF - np.mean(MTF))**2)
    R2 = 1 - (RSS / TSS) if TSS != 0 else 0
    
    # Resolution Metrics (MTF50, MTF10)
    mf = lambda x: model(x, *theta_hat)
    
    # Find theoretical f50 and f10 using fsolve
    f50 = fsolve(lambda x: mf(x) - 0.5, np.median(f))[0]
    f10 = fsolve(lambda x: mf(x) - 0.1, np.median(f))[0]
    
    # Effective Empirical (PCHIP interpolation)
    sort_idx = np.argsort(MTF)[::-1]
    MTF_s, f_s = MTF[sort_idx], f[sort_idx]
    
    # Ensure strictly increasing for interpolation
    _, unique_idx = np.unique(MTF_s, return_index=True)
    MTF_uniq, f_uniq = MTF_s[unique_idx], f_s[unique_idx]
    
    if len(MTF_uniq) > 1:
        pchip = PchipInterpolator(MTF_uniq[::-1], f_uniq[::-1])
        f50e = float(pchip(0.5)) if 0.5 <= np.max(MTF_uniq) else np.nan
        f10e = float(pchip(0.1)) if 0.1 <= np.max(MTF_uniq) else np.nan
    else:
        f50e, f10e = np.nan, np.nan
        
    core_FWHM = 2.3548 / theta_hat[1]
    halo_FWHM = 2.3548 / theta_hat[2]
    
    results = {
        'theta_hat': theta_hat,
        'R2': R2,
        'f50': f50, 'f10': f10,
        'f50e': f50e, 'f10e': f10e,
        'core_FWHM': core_FWHM,
        'halo_FWHM': halo_FWHM
    }
    
    return results, model