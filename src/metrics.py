import os
import glob
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.transform import radon
from scipy.ndimage import convolve, distance_transform_edt
from skimage.morphology import medial_axis, reconstruction
from scipy.stats import expon
from skimage.measure import moments, moments_central, moments_normalized, moments_hu
from tqdm.auto import tqdm


def compute_area_fraction(arr: np.ndarray) -> float:
    """
    Compute the area fraction of the foreground phase using Otsu thresholding.
    """
    thr = threshold_otsu(arr)
    return float((arr > thr).mean())


def compute_morans_i(arr: np.ndarray) -> float:
    """
    Compute Moran's I for an image array using a 4-neighbor adjacency.
    """
    x = arr.flatten()
    xm, n = x.mean(), x.size
    # 4-neighbor kernel
    kernel = np.array([[0,1,0],[1,0,1],[0,1,0]])
    conv = convolve(arr, kernel, mode='reflect').flatten()
    W = 4 * n
    num = ((conv - 4 * xm) * (x - xm)).sum()
    denom = ((x - xm)**2).sum()
    return float((n / W) * (num / denom))


def compute_band_metrics(arr: np.ndarray, angle_step: int = 1):
    """
    Compute banding index and principal orientation via Radon transform histogram.
    """
    # Compute Radon sinogram
    theta = np.arange(0, 181, angle_step)
    sinogram = radon(arr, theta=theta, circle=False)
    # Threshold and mirror
    sig = np.where(sinogram >= 0.7 * sinogram.max(), sinogram, 0)
    sig += np.fliplr(sig)
    # Build half-histogram
    half = sig[:, :sig.shape[1]//2 + 1]
    hist = half.sum(axis=0)
    k = hist.argmax()
    # Neighbors for mode interpolation
    nxt = hist[k+1] if k+1 < hist.size else hist[k-1]
    prv = hist[k-1] if k-1 >= 0 else hist[k+1]
    upper = k*angle_step + 0.5*angle_step
    lower = k*angle_step - 0.5*angle_step
    theta_mode = (upper*nxt + lower*prv) / (nxt + prv)
    theta_max = 90.0 - theta_mode
    # Band index via normalized std
    x = np.arange(hist.size)
    x_norm = (x - x.min()) / (x.max() - x.min())
    mean_n = (x_norm * hist).sum() / hist.sum()
    std_n = np.sqrt((hist * (x_norm - mean_n)**2).sum() / (hist.sum() - 1))
    band_idx = max(1.0 - (std_n / 0.2887), 0.0)
    return float(band_idx), float(theta_max)


def compute_hu_features(arr: np.ndarray) -> np.ndarray:
    """
    Compute log-scaled Hu moments from binary mask.
    """
    thr = threshold_otsu(arr)
    mask = (arr > thr).astype(np.uint8)
    m = moments(mask, order=3)
    mu = moments_central(mask,
                         center=(m[1,0]/m[0,0], m[0,1]/m[0,0]),
                         order=3)
    nu = moments_normalized(mu)
    hu = moments_hu(nu)
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)


def compute_thickness_lambda(arr: np.ndarray) -> float:
    """
    Fit exponential to local thickness distribution and return rate (lambda).
    """
    thr = threshold_otsu(arr)
    mask = (arr > thr).astype(np.uint8)
    dist = distance_transform_edt(mask)
    skel, d2 = medial_axis(mask, return_distance=True)
    seed = d2 * skel
    local_rad = reconstruction(seed, dist, method='dilation')
    thickness = 2 * local_rad * mask
    vals = thickness[mask.astype(bool)]
    _, scale_hat = expon.fit(vals, floc=2)
    return float(1.0 / scale_hat)


def extract_metrics_from_paths(image_pattern: str, max_files: int = None) -> pd.DataFrame:
    """
    Process all images matching the glob, compute metrics, and return a DataFrame.
    """
    paths = sorted(glob.glob(image_pattern))
    if max_files:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(f"No images found for pattern: {image_pattern}")

    records = []
    for p in tqdm(paths, desc='Processing images'):
        img = imread(p, as_gray=True).astype(float) / 255.0
        af = compute_area_fraction(img)
        mi = compute_morans_i(img)
        bi, ori = compute_band_metrics(img)
        hu = compute_hu_features(img)
        lam = compute_thickness_lambda(img)
        rec = [os.path.basename(p), af, mi, bi, ori, *hu.tolist(), lam]
        records.append(rec)

    cols = ['filename', 'AF', 'MI', 'BI', 'ORI'] + [f'Hu{i+1}' for i in range(7)] + ['lambda']
    return pd.DataFrame(records, columns=cols)