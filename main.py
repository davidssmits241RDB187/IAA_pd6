import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    is_gray = len(img.shape) == 2 or img.shape[2] == 1

    arr = img.astype(np.float32) / 255.0

    return arr, is_gray


def remove_noise(image):

    blurred = cv2.GaussianBlur(image, (5, 5), 3)
    return blurred.astype(np.float32)


def gamma_correction(image, gamma_value=0.8):

    C = np.power(image, gamma_value)
    return np.clip(C, 0.0, 1.0).astype(np.float32)


def histogram_linear_transformation(image):
    g_min = np.min(image)
    g_max = np.max(image)

    if g_max <= g_min:
        return image.copy()

    k = 1.0 / (g_max - g_min)

    C = k * (image - g_min)

    return np.clip(C, 0.0, 1.0).astype(np.float32)


def mean_brightness(img, is_gray):
    if is_gray:
        return float(np.mean(img))
    else:
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 2]))


def activate_appropriate_contrast_processor(img, is_gray):
    mean_v = mean_brightness(img, is_gray)

    if mean_v > 200:
        processed = gamma_correction(img)
        proc_title = "Processed: Gamma"
    else:
        processed = histogram_linear_transformation(img)
        proc_title = "Processed: Histogram linear"

    return processed, proc_title


def save_image_comparison(original, processed, out_path, is_gray, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].set_title("Original")
    axes[1].set_title("Processed")

    if is_gray:
        axes[0].imshow(original, cmap="gray")
        axes[1].imshow(processed, cmap="gray")
    else:
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

    for ax in axes:
        ax.axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_and_save_histograms(original, processed, out_path, is_gray):
    n = 50
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if is_gray:
        orig_hist, bins = np.histogram(original.flatten(), bins=n, range=(0, 1))
        proc_hist, _ = np.histogram(processed.flatten(), bins=n, range=(0, 1))

        axes[0].plot(bins[:-1], orig_hist)
        axes[1].plot(bins[:-1], proc_hist)
    else:
        colors = ["r", "g", "b"]
        for c, col in enumerate(colors):
            orig_hist, bins = np.histogram(
                original[:, :, c].flatten(), bins=n, range=(0, 1)
            )
            proc_hist, _ = np.histogram(
                processed[:, :, c].flatten(), bins=n, range=(0, 1)
            )

            axes[0].plot(bins[:-1], orig_hist, color=col)
            axes[1].plot(bins[:-1], proc_hist, color=col)

    axes[0].set_title("Original histogram")
    axes[1].set_title("Processed histogram")

    for ax in axes:
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_single_image(path, idx):
    img, is_gray = load_image(path)

    denoised = remove_noise(img)
    processed, proc_title = activate_appropriate_contrast_processor(denoised, is_gray)

    base = os.path.splitext(os.path.basename(path))[0]
    img_out = os.path.join(OUTPUT_DIR, f"{idx:02d}_{base}_comparison.png")
    hist_out = os.path.join(OUTPUT_DIR, f"{idx:02d}_{base}_histogram.png")

    save_image_comparison(img, processed, img_out, is_gray, proc_title)
    plot_and_save_histograms(img, processed, hist_out, is_gray)

    print(f"Saved: {img_out}")
    print(f"Saved: {hist_out}")
    print(f"{base}: {proc_title}")


def main():
    files = [
        "image.jpg",
        "image2.jpg",
        "image3.jpg",
        "image4.jpeg",
        "image5.png",
    ]

    for i, f in enumerate(files, start=1):
        if os.path.exists(f):
            process_single_image(f, i)


if __name__ == "__main__":
    main()
