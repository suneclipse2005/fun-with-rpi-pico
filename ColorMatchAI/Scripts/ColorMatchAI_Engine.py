import sys
import os
import json
import argparse
import numpy as np
import cv2

# Ensure we have the required packages.
# For ONNX model support, you would also import onnxruntime here.
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

def create_lut_3d(size, mapping_func):
    """
    Creates a 3D LUT of given size using a mapping function.
    """
    lut = np.zeros((size, size, size, 3), dtype=np.float32)

    for b in range(size):
        for g in range(size):
            for r in range(size):
                # Normalize coordinates to 0.0 - 1.0
                r_norm = r / (size - 1.0)
                g_norm = g / (size - 1.0)
                b_norm = b / (size - 1.0)

                # Apply mapping
                r_mapped, g_mapped, b_mapped = mapping_func(r_norm, g_norm, b_norm)

                lut[b, g, r, 0] = r_mapped
                lut[b, g, r, 1] = g_mapped
                lut[b, g, r, 2] = b_mapped

    return lut

def write_cube_lut(lut, filename):
    """
    Writes a 3D numpy array to a .cube file format.
    """
    size = lut.shape[0]
    with open(filename, 'w') as f:
        f.write("TITLE \"ColorMatchAI_Generated\"\n")
        f.write(f"LUT_3D_SIZE {size}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n")

        # .cube format iterates R first, then G, then B.
        for b in range(size):
            for g in range(size):
                for r in range(size):
                    val = lut[b, g, r]
                    f.write(f"{val[0]:.6f} {val[1]:.6f} {val[2]:.6f}\n")

def match_histograms(target_img, reference_img):
    """
    Matches the histogram of the target image to the reference image in LAB color space.
    Images should be floating point 0.0 - 1.0.
    """
    # Convert to 8-bit for OpenCV LAB conversion
    tgt_8u = (np.clip(target_img, 0, 1) * 255).astype(np.uint8)
    ref_8u = (np.clip(reference_img, 0, 1) * 255).astype(np.uint8)

    tgt_lab = cv2.cvtColor(tgt_8u, cv2.COLOR_RGB2LAB)
    ref_lab = cv2.cvtColor(ref_8u, cv2.COLOR_RGB2LAB)

    matched_lab = np.zeros_like(tgt_lab)

    # Match each channel (L, A, B)
    for i in range(3):
        # Calculate histograms and CDFs
        tgt_hist, _ = np.histogram(tgt_lab[:, :, i].ravel(), bins=256, range=[0, 256])
        ref_hist, _ = np.histogram(ref_lab[:, :, i].ravel(), bins=256, range=[0, 256])

        tgt_cdf = tgt_hist.cumsum()
        tgt_cdf = tgt_cdf / tgt_cdf[-1]

        ref_cdf = ref_hist.cumsum()
        ref_cdf = ref_cdf / ref_cdf[-1]

        # Create mapping lookup table
        lookup_table = np.zeros(256)
        lookup_val = 0
        for tgt_val in range(256):
            while lookup_val < 255 and ref_cdf[lookup_val] < tgt_cdf[tgt_val]:
                lookup_val += 1
            lookup_table[tgt_val] = lookup_val

        # Apply mapping
        matched_lab[:, :, i] = cv2.LUT(tgt_lab[:, :, i], lookup_table.astype(np.uint8))

    matched_rgb_8u = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
    matched_rgb = matched_rgb_8u.astype(np.float32) / 255.0
    return matched_rgb


def calculate_stats(img):
    """Calculate basic image stats."""
    mean_val = np.mean(img, axis=(0, 1))
    std_val = np.std(img, axis=(0, 1))

    # Calculate perceived luminance
    lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    contrast = np.std(lum)

    return {
        "mean_r": float(mean_val[0]),
        "mean_g": float(mean_val[1]),
        "mean_b": float(mean_val[2]),
        "std_r": float(std_val[0]),
        "std_g": float(std_val[1]),
        "std_b": float(std_val[2]),
        "contrast": float(contrast)
    }


def build_color_match_lut(ref_img, tgt_img, lut_size=33):
    """
    Builds a 3D LUT that maps target colors to reference colors.
    """
    print("Extracting statistical data...")
    # For a robust, fast approximation in the LUT, we calculate global mean/std mapping
    # This is a linear matching (Color Transfer between Images, Reinhard et al.)

    # Convert to LAB for Reinhard transfer
    tgt_8u = (np.clip(tgt_img, 0, 1) * 255).astype(np.uint8)
    ref_8u = (np.clip(ref_img, 0, 1) * 255).astype(np.uint8)

    tgt_lab = cv2.cvtColor(tgt_8u, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_8u, cv2.COLOR_RGB2LAB).astype(np.float32)

    tgt_mean = np.mean(tgt_lab, axis=(0, 1))
    tgt_std = np.std(tgt_lab, axis=(0, 1))

    ref_mean = np.mean(ref_lab, axis=(0, 1))
    ref_std = np.std(ref_lab, axis=(0, 1))

    # Avoid division by zero
    tgt_std[tgt_std == 0] = 1.0

    def mapping_func(r, g, b):
        # Input RGB
        rgb_pixel = np.array([[[r, g, b]]], dtype=np.float32)
        # Convert to LAB
        rgb_8u = (rgb_pixel * 255).astype(np.uint8)
        lab_pixel = cv2.cvtColor(rgb_8u, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Apply Reinhard Transfer
        mapped_lab = (lab_pixel - tgt_mean) * (ref_std / tgt_std) + ref_mean
        mapped_lab = np.clip(mapped_lab, 0, 255).astype(np.uint8)

        # Convert back to RGB
        mapped_rgb = cv2.cvtColor(mapped_lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        return mapped_rgb[0, 0, 0], mapped_rgb[0, 0, 1], mapped_rgb[0, 0, 2]

    print(f"Generating {lut_size}x{lut_size}x{lut_size} 3D LUT...")
    lut = create_lut_3d(lut_size, mapping_func)
    return lut

def main():
    parser = argparse.ArgumentParser(description="ColorMatchAI Engine")
    parser.add_argument("--ref", required=True, help="Path to reference image (PNG/EXR)")
    parser.add_argument("--tgt", required=True, help="Path to target image (PNG/EXR)")
    parser.add_argument("--out_lut", required=True, help="Path to save the generated .cube LUT")
    parser.add_argument("--out_json", required=True, help="Path to save the analysis JSON")
    parser.add_argument("--mode", default="Fast", choices=["Fast", "AI_Precision"], help="Match mode")

    args = parser.parse_args()

    print(f"ColorMatchAI Engine - Mode: {args.mode}")
    print(f"Reference: {args.ref}")
    print(f"Target: {args.tgt}")

    # Load images (Assume RGB)
    # Note: For EXR, you would use OpenEXR or imageio. Using cv2.imread for general support.
    # cv2 loads as BGR by default.
    ref_bgr = cv2.imread(args.ref, cv2.IMREAD_UNCHANGED)
    tgt_bgr = cv2.imread(args.tgt, cv2.IMREAD_UNCHANGED)

    if ref_bgr is None or tgt_bgr is None:
        print("Error: Could not read input images.")
        sys.exit(1)

    # Convert to RGB and normalize to 0.0 - 1.0
    # Handle both 8-bit (uint8) and 16-bit (uint16) or floats
    if ref_bgr.dtype == np.uint8:
        ref_img = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    elif ref_bgr.dtype == np.uint16:
        ref_img = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
    else:
        ref_img = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    if tgt_bgr.dtype == np.uint8:
        tgt_img = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    elif tgt_bgr.dtype == np.uint16:
        tgt_img = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
    else:
        tgt_img = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Calculate statistics
    ref_stats = calculate_stats(ref_img)
    tgt_stats = calculate_stats(tgt_img)

    analysis_data = {
        "reference": ref_stats,
        "target": tgt_stats,
        "mode": args.mode,
        "status": "success",
        "has_onnx": HAS_ONNX
    }

    if args.mode == "AI_Precision" and HAS_ONNX:
        print("AI Precision Mode requested. Would load ONNX model here.")
        # ort_session = ort.InferenceSession("path_to_model.onnx")
        # apply neural color transfer...

    # Generate LUT
    lut_size = 33
    lut = build_color_match_lut(ref_img, tgt_img, lut_size)

    # Save LUT
    print(f"Saving LUT to {args.out_lut}...")
    write_cube_lut(lut, args.out_lut)

    # Save JSON Profile
    print(f"Saving JSON profile to {args.out_json}...")
    with open(args.out_json, 'w') as f:
        json.dump(analysis_data, f, indent=4)

    print("Analysis complete.")

if __name__ == "__main__":
    main()
