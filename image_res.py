import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ==========================================
# PART 1: SCALING (MIN-MAX NORMALIZATION)
# ==========================================
def scale_image(A: np.ndarray) -> np.ndarray:
  """Scales an input image matrix A to the range [0, 1]."""
  # Convert input matrix to float64 to prevent integer division truncation
  A_float = A.astype(np.float64)

  # Compute global minimum and maximum intensity values
  a_min = np.min(A_float)
  a_max = np.max(A_float)

  # Avoid division by zero for constant intensity images
  if a_max == a_min:
    return np.zeros_like(A_float)

  # Normalize intensities linearly to [0, 1]
  return (A_float - a_min) / (a_max - a_min)


# ==========================================
# PART 2: INTERPOLATION ROUTINES
# ==========================================
def bilinear_interpolation(
    img: np.ndarray, target_shape: tuple
) -> np.ndarray:
  """Resizes a 2D image to target_shape using bilinear interpolation."""
  h_in, w_in = img.shape
  h_out, w_out = target_shape

  # Map output grid coordinates back to input image coordinates
  r = np.linspace(0, h_in - 1, h_out)
  c = np.linspace(0, w_in - 1, w_out)
  r_grid, c_grid = np.meshgrid(r, c, indexing='ij')

  # Identify surrounding integer pixel coordinates
  r0 = np.floor(r_grid).astype(int)
  c0 = np.floor(c_grid).astype(int)
  r1 = np.clip(r0 + 1, 0, h_in - 1)
  c1 = np.clip(c0 + 1, 0, w_in - 1)

  # Calculate fractional weights
  dr = r_grid - r0
  dc = c_grid - c0

  # Compute weighted sum of the 4 neighboring pixels
  I00 = img[r0, c0]
  I01 = img[r0, c1]
  I10 = img[r1, c0]
  I11 = img[r1, c1]

  out = (
      (1 - dr) * (1 - dc) * I00
      + (1 - dr) * dc * I01
      + dr * (1 - dc) * I10
      + dr * dc * I11
  )
  return out


def cubic_kernel(t: np.ndarray, a: float = -0.5) -> np.ndarray:
  """Evaluates the Keys bicubic convolution kernel for distance t."""
  abs_t = np.abs(t)
  weights = np.zeros_like(abs_t)

  # Inner region: |t| <= 1
  mask1 = abs_t <= 1
  weights[mask1] = (a + 2) * (abs_t[mask1] ** 3) - (a + 3) * (
      abs_t[mask1] ** 2
  ) + 1

  # Outer region: 1 < |t| < 2
  mask2 = (abs_t > 1) & (abs_t < 2)
  weights[mask2] = (
      a * (abs_t[mask2] ** 3)
      - 5 * a * (abs_t[mask2] ** 2)
      + 8 * a * abs_t[mask2]
      - 4 * a
  )
  return weights


def bicubic_interpolation(
    img: np.ndarray, target_shape: tuple
) -> np.ndarray:
  """Resizes a 2D image to target_shape using separable bicubic interpolation."""
  h_in, w_in = img.shape
  h_out, w_out = target_shape

  # Step 1: Interpolate along rows (width) first, then columns (height)
  def interpolate_1d_along_axis0(
      arr: np.ndarray, out_len: int
  ) -> np.ndarray:
    in_len = arr.shape[0]
    coords = np.linspace(0, in_len - 1, out_len)
    x0 = np.floor(coords).astype(int)

    result = np.zeros((out_len, arr.shape[1]), dtype=np.float64)
    # Sum over the 4 neighboring points in 1D
    for m in [-1, 0, 1, 2]:
      idx = np.clip(x0 + m, 0, in_len - 1)
      w = cubic_kernel(coords - (x0 + m))[:, np.newaxis]
      result += arr[idx, :] * w
    return result

  # Separable execution: vertical pass followed by horizontal pass
  temp = interpolate_1d_along_axis0(img, h_out)
  out = interpolate_1d_along_axis0(temp.T, w_out).T

  # Clip intensity values to valid dynamic range of input
  return np.clip(out, np.min(img), np.max(img))


# ==========================================
# PART 3: PERFORMANCE METRICS & EVALUATION
# ==========================================
def compute_metrics(orig: np.ndarray, recon: np.ndarray) -> tuple:
  """Computes Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR)."""
  mse = np.mean((orig - recon) ** 2)
  if mse == 0:
    return 0.0, float('inf')
  # Using dynamic range L = 1.0 for normalized [0, 1] images
  psnr = 10 * np.log10(1.0 / mse)
  return mse, psnr


if __name__ == '__main__':
  files = [
      'Chronometer.tif',
      'Chronometerlow2.tif',
      'Chronometerlow34.tif',
      'Chronometerlow5.tif',
  ]
  images = {}

  print('--- 2(a) IMAGE RESOLUTIONS ---')
  for f in files:
    if os.path.exists(f):
      im = np.array(Image.open(f))
      # If multi-channel, take the first channel
      if im.ndim > 2:
        im = im[:, :, 0]
      images[f] = scale_image(im)
      print(f'{f:22s}: {im.shape[0]} x {im.shape[1]} pixels')

  orig_img = images['Chronometer.tif']
  target_res = orig_img.shape

  low_files = [
      'Chronometerlow2.tif',
      'Chronometerlow34.tif',
      'Chronometerlow5.tif',
  ]
  results = {}

  print('\n--- 2(c) PERFORMANCE METRICS (ON [0, 1] SCALED IMAGES) ---')
  print(f"{'Image File':22s} | {'Method':10s} | {'MSE':12s} | {'PSNR (dB)':10s}")
  print('-' * 62)

  for lf in low_files:
    low_img = images[lf]

    # Apply Bilinear and Bicubic interpolation
    rec_lin = bilinear_interpolation(low_img, target_res)
    rec_cub = bicubic_interpolation(low_img, target_res)

    mse_lin, psnr_lin = compute_metrics(orig_img, rec_lin)
    mse_cub, psnr_cub = compute_metrics(orig_img, rec_cub)

    results[lf] = (rec_lin, rec_cub, mse_lin, psnr_lin, mse_cub, psnr_cub)

    print(f"{lf:22s} | {'Bilinear':10s} | {mse_lin:.6f}     | {psnr_lin:.2f}")
    print(f"{lf:22s} | {'Bicubic':10s} | {mse_cub:.6f}     | {psnr_cub:.2f}")

  # Generate and save comparison figure for LaTeX report
  fig, axes = plt.subplots(3, 3, figsize=(12, 12))
  # Crop region around "REAUMUR" and dial numbers to highlight edge sharpness
  h, w = target_res
  r_sl = slice(int(h * 0.55), int(h * 0.75))
  c_sl = slice(int(w * 0.40), int(w * 0.65))

  for idx, lf in enumerate(low_files):
    rec_lin, rec_cub, _, psnr_lin, _, psnr_cub = results[lf]

    axes[idx, 0].imshow(orig_img[r_sl, c_sl], cmap='gray', vmin=0, vmax=1)
    axes[idx, 0].set_title('Original (Cropped ROI)')
    axes[idx, 0].axis('off')

    axes[idx, 1].imshow(rec_lin[r_sl, c_sl], cmap='gray', vmin=0, vmax=1)
    axes[idx, 1].set_title(f'{lf}\nBilinear (PSNR: {psnr_lin:.2f} dB)')
    axes[idx, 1].axis('off')

    axes[idx, 2].imshow(rec_cub[r_sl, c_sl], cmap='gray', vmin=0, vmax=1)
    axes[idx, 2].set_title(f'{lf}\nBicubic (PSNR: {psnr_cub:.2f} dB)')
    axes[idx, 2].axis('off')

  plt.tight_layout()
  plt.savefig('interpolation_comparison.png', dpi=300)
  print("\nSaved figure 'interpolation_comparison.png' for LaTeX report.")