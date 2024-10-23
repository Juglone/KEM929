import cv2
import numpy as np
import sys

def compute_image_statistics(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")

    # Convert the image to different color spaces
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # Compute mean RGB
    mean_rgb = np.mean(image_rgb, axis=(0, 1))

    # Compute mean HSV
    mean_hsv = np.mean(image_hsv, axis=(0, 1))

    # Compute mean HSL
    mean_hls = np.mean(image_hls, axis=(0, 1))

    # Compute mean brightness (mean of the grayscale image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray_image)

    return {
        "mean_brightness": mean_brightness,
        "mean_rgb": mean_rgb,
        "mean_hsv": mean_hsv,
        "mean_hls": mean_hls
    }

# Example usage
image_path = sys.argv[1]
results = compute_image_statistics(image_path)

print("Mean Brightness:", results["mean_brightness"])
print("Mean RGB:", results["mean_rgb"])
print("Mean HSV:", results["mean_hsv"])
print("Mean HSL:", results["mean_hls"])

