import cv2
import numpy as np
import sys
import os
import csv
from natsort import natsorted
import torch
import lpips

def compute_psnr(img1, img2):
    img1_f = img1.astype(np.float64)
    img2_f = img2.astype(np.float64)
    mse = np.mean((img1_f - img2_f) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def _compute_ssim_single_channel(img1, img2):
    img1_f = img1.astype(np.float64)
    img2_f = img2.astype(np.float64)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    kernel = (11, 11)
    sigma = 1.5

    mu1 = cv2.GaussianBlur(img1_f, kernel, sigma)
    mu2 = cv2.GaussianBlur(img2_f, kernel, sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1_f * img1_f, kernel, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2_f * img2_f, kernel, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1_f * img2_f, kernel, sigma) - mu1_mu2

    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(np.mean(ssim_map))


def compute_ssim(img1, img2):
    if img1.ndim == 2:
        return _compute_ssim_single_channel(img1, img2)
    channel_scores = [
        _compute_ssim_single_channel(img1[..., c], img2[..., c])
        for c in range(img1.shape[2])
    ]
    return float(np.mean(channel_scores))


def make_lpips_model():
    if torch is None or lpips is None:
        return None
    try:
        model = lpips.LPIPS(net="vgg")
        model.eval()
        return model
    except Exception as exc:
        print(f"Warning: LPIPS unavailable ({exc})")
        return None


def compute_lpips(img1, img2, lpips_model):
    if lpips_model is None:
        return None

    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img1_t = torch.from_numpy(img1_rgb).float().permute(2, 0, 1).unsqueeze(0)
    img2_t = torch.from_numpy(img2_rgb).float().permute(2, 0, 1).unsqueeze(0)
    img1_t = img1_t / 127.5 - 1.0
    img2_t = img2_t / 127.5 - 1.0

    with torch.no_grad():
        score = lpips_model(img1_t, img2_t)
    return float(score.item())


def compare_image_directories(dir1_path, dir2_path, gt_dir_path, output_dir):
    """
    Compare images from two directories pixel-wise and save differences.
    
    Args:
        dir1_path: Path to first directory
        dir2_path: Path to second directory
        gt_dir_path: Path to ground-truth image directory
        output_dir: Path to save the difference images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sorted image files
    img_files1 = natsorted([f for f in os.listdir(dir1_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    img_files2 = natsorted([f for f in os.listdir(dir2_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    gt_files = natsorted([f for f in os.listdir(gt_dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    if len(img_files1) != len(img_files2):
        print(f"Error: Directories have different number of images ({len(img_files1)} vs {len(img_files2)})")
        return
    if len(img_files1) != len(gt_files):
        print(f"Error: Ground-truth directory has different number of images ({len(gt_files)} vs {len(img_files1)})")
        return

    lpips_model = make_lpips_model()
    if lpips_model is None:
        print("Warning: LPIPS metric disabled because torch/lpips is not available.")

    metrics_rows = []
    
    # Compare each pair of images
    for img_file1, img_file2, gt_file in zip(img_files1, img_files2, gt_files):
        img1_path = os.path.join(dir1_path, img_file1)
        img2_path = os.path.join(dir2_path, img_file2)
        gt_path = os.path.join(gt_dir_path, gt_file)
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        img_gt = cv2.imread(gt_path)
        
        if img1 is None or img2 is None or img_gt is None:
            print(f"Error: Could not read {img_file1}, {img_file2}, or {gt_file}")
            continue

        if img1.shape != img2.shape or img1.shape != img_gt.shape:
            print(
                f"Error: Shape mismatch for {img_file1}, {img_file2}, {gt_file}: "
                f"{img1.shape} vs {img2.shape} vs {img_gt.shape}"
            )
            continue
        
        # Calculate pixel-wise difference
        diff = cv2.absdiff(img1, img2)
        
        # Save the difference image
        output_path = os.path.join(output_dir, f"diff_{img_file1}")
        cv2.imwrite(output_path, diff)
        
        mean_diff = np.mean(diff)
        psnr_1 = compute_psnr(img1, img_gt)
        ssim_1 = compute_ssim(img1, img_gt)
        lpips_1 = compute_lpips(img1, img_gt, lpips_model)
        psnr_2 = compute_psnr(img2, img_gt)
        ssim_2 = compute_ssim(img2, img_gt)
        lpips_2 = compute_lpips(img2, img_gt, lpips_model)

        metrics_rows.append(
            {
                "image_1": img_file1,
                "image_2": img_file2,
                "image_gt": gt_file,
                "mean_pixel_diff": mean_diff,
                "psnr_dir1_vs_gt": psnr_1,
                "ssim_dir1_vs_gt": ssim_1,
                "lpips_dir1_vs_gt": lpips_1,
                "psnr_dir2_vs_gt": psnr_2,
                "ssim_dir2_vs_gt": ssim_2,
                "lpips_dir2_vs_gt": lpips_2,
            }
        )

        lpips_1_str = "N/A" if lpips_1 is None else f"{lpips_1:.6f}"
        lpips_2_str = "N/A" if lpips_2 is None else f"{lpips_2:.6f}"
        psnr_1_str = "inf" if np.isinf(psnr_1) else f"{psnr_1:.4f}"
        psnr_2_str = "inf" if np.isinf(psnr_2) else f"{psnr_2:.4f}"
        print(
            f"{img_file1} vs {img_file2}: mean_diff={mean_diff:.2f} | "
            f"dir1_vs_gt(psnr={psnr_1_str}, ssim={ssim_1:.6f}, lpips={lpips_1_str}) | "
            f"dir2_vs_gt(psnr={psnr_2_str}, ssim={ssim_2:.6f}, lpips={lpips_2_str})"
        )

    if len(metrics_rows) == 0:
        print("No valid image pairs were compared.")
        return

    metrics_csv = os.path.join(output_dir, "metrics.csv")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_1",
                "image_2",
                "image_gt",
                "mean_pixel_diff",
                "psnr_dir1_vs_gt",
                "ssim_dir1_vs_gt",
                "lpips_dir1_vs_gt",
                "psnr_dir2_vs_gt",
                "ssim_dir2_vs_gt",
                "lpips_dir2_vs_gt",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    mean_diff_avg = float(np.mean([row["mean_pixel_diff"] for row in metrics_rows]))
    psnr_1_values = [row["psnr_dir1_vs_gt"] for row in metrics_rows if np.isfinite(row["psnr_dir1_vs_gt"])]
    ssim_1_avg = float(np.mean([row["ssim_dir1_vs_gt"] for row in metrics_rows]))
    lpips_1_values = [row["lpips_dir1_vs_gt"] for row in metrics_rows if row["lpips_dir1_vs_gt"] is not None]
    psnr_2_values = [row["psnr_dir2_vs_gt"] for row in metrics_rows if np.isfinite(row["psnr_dir2_vs_gt"])]
    ssim_2_avg = float(np.mean([row["ssim_dir2_vs_gt"] for row in metrics_rows]))
    lpips_2_values = [row["lpips_dir2_vs_gt"] for row in metrics_rows if row["lpips_dir2_vs_gt"] is not None]

    print("\nSummary")
    print(f"Compared {len(metrics_rows)} image pairs")
    print(f"Mean pixel difference: {mean_diff_avg:.4f}")
    if len(psnr_1_values) > 0:
        print(f"Mean PSNR dir1 vs gt: {float(np.mean(psnr_1_values)):.6f}")
    else:
        print("Mean PSNR dir1 vs gt: inf")
    print(f"Mean SSIM dir1 vs gt: {ssim_1_avg:.6f}")
    if len(lpips_1_values) > 0:
        print(f"Mean LPIPS dir1 vs gt: {float(np.mean(lpips_1_values)):.6f}")
    else:
        print("Mean LPIPS dir1 vs gt: N/A")
    if len(psnr_2_values) > 0:
        print(f"Mean PSNR dir2 vs gt: {float(np.mean(psnr_2_values)):.6f}")
    else:
        print("Mean PSNR dir2 vs gt: inf")
    print(f"Mean SSIM dir2 vs gt: {ssim_2_avg:.6f}")
    if len(lpips_2_values) > 0:
        print(f"Mean LPIPS dir2 vs gt: {float(np.mean(lpips_2_values)):.6f}")
    else:
        print("Mean LPIPS dir2 vs gt: N/A")
    print(f"Saved per-image metrics to: {metrics_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python compare_two_imgs.py <dir1_path> <dir2_path> <gt_dir_path> <output_dir>")
        sys.exit(1)
    
    dir1_path = sys.argv[1]
    dir2_path = sys.argv[2]
    gt_dir_path = sys.argv[3]
    output_dir = sys.argv[4]
    
    compare_image_directories(dir1_path, dir2_path, gt_dir_path, output_dir)
