import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import sys
import os
from multiprocessing import Pool

DEBUG = False
plt.rcParams['font.size'] = 20
MARGIN = 0.15
plt.rcParams['figure.subplot.left'] = MARGIN
plt.rcParams['figure.subplot.bottom'] = MARGIN
plt.rcParams['figure.subplot.top'] = 1 - MARGIN * 2 / 3
plt.rcParams['figure.subplot.right'] = 1 - MARGIN / 4

def process_video(video_path):
    global DEBUG
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open the video: {video_path}")
        exit()

    cv_metric_values = []

    MASK_RETAIN = 40
    retained_masks = None
    next_mask_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if retained_masks is None:
            retained_masks = np.ones((MASK_RETAIN,100,100), dtype=np.uint8) * 255

        if not ret:
            break

        h, w, _ = frame.shape

        square_size = min(h, w) // 2

        x_start = w // 2 - square_size // 2
        y_start = h // 2 - square_size // 2
        x_end = x_start + square_size
        y_end = y_start + square_size

        square_frame = frame[y_start:y_end, x_start:x_end]

        square_frame = cv2.resize(square_frame, (100, 100), interpolation=cv2.INTER_AREA)

        grayscale_highcontrast = cv2.cvtColor(square_frame,cv2.COLOR_BGR2GRAY)
        grayscale_highcontrast = cv2.GaussianBlur(grayscale_highcontrast, (5,5), 2)
        grayscale_highcontrast = cv2.equalizeHist(grayscale_highcontrast)

        cannyThres = 200
        circles = cv2.HoughCircles(
            grayscale_highcontrast,
            cv2.HOUGH_GRADIENT,
            dp=1,           # Inverse ratio of the accumulator resolution to the image resolution
            minDist=3,       # Minimum distance between the centers of the detected circles
            param1=cannyThres,        # Higher threshold for the Canny edge detector
            param2=13,        # Accumulator threshold for circle detection
            minRadius=1,      # Minimum circle radius
            maxRadius=10      # Maximum circle radius
        )

        # Create a mask with white background
        mask = np.ones_like(grayscale_highcontrast, dtype=np.uint8) * 255

        if circles is not None:
            # Round the circle parameters and convert to integer
            circles = np.uint16(np.around(circles))
            for x_center, y_center, radius in circles[0, :]:
                # Draw filled black circles on the mask to exclude bubble regions
                MARGIN = 2
                cv2.circle(mask, (x_center, y_center), radius+MARGIN, 0, -1)

        assert mask.shape == retained_masks.shape[1:]
        retained_masks[next_mask_index] = mask
        next_mask_index = (next_mask_index + 1) % MASK_RETAIN
        mask = np.percentile(retained_masks, 20, axis=0).astype(np.uint8)

        # Compute the mean grayscale value excluding the bubbles

        if DEBUG:
            cv2.imshow("Edges", cv2.Canny(grayscale_highcontrast, cannyThres/2, cannyThres))
            cv2.imshow("grayscale_highcontrast", grayscale_highcontrast)
            cv2.imshow("mask", mask)
            cv2.waitKey(25)

        val = cv2.cvtColor(square_frame, cv2.COLOR_RGB2HSV)[:, :, 1].astype("float32")
        val = cv2.mean(val, mask=mask)[0] / 255
        cv_metric_values.append(val)

    # Release the video capture object
    cap.release()

    return video_path, cv_metric_values

def plot_values(video_path, cv_metric_values):
    # Convert list to numpy array
    cv_metric_values = np.array(cv_metric_values)
    CLAMP = 0.001
    #cv_metric_values = np.cumsum(np.clip(np.diff(cv_metric_values), a_min=-CLAMP, a_max=CLAMP))
    print(f"{len(cv_metric_values) = }")
    hours = np.arange(len(cv_metric_values)) * 200 / (30 * 3600)  # Adjusted time calculation if needed

    # Perform linear regression using scipy
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(hours, cv_metric_values)

    # Extract relevant parts of file name
    parts = video_path.split('_')
    name = ""
    name += {
        "pH1.5": "1.5",
        "pH3": "3",
        "pH7": "7",
    }[parts[0]]
    name += {
        "IS-0": "O",
        "IS-S": "S",
        "IS-I": "I",
    }[parts[1]]
    name += {
        "pepsin": "p",
        "nopepsin": "",
        "nogel": " blank"
    }[parts[2]]
    date = parts[3]
    version = parts[4].split(".")[0]


    # Plot grayscale values over time with linear regression line
    plt.figure(figsize=(10, 6))
    plt.plot(hours, cv_metric_values, label='Data', linestyle='-')
    plt.plot(hours, slope * hours + intercept, 'r', label=f'Slope={slope:.4f}')
    plt.xlabel('Hours')
    plt.ylabel('Value')
    title_text = f"{name}, {date}{version}"
    plt.title(title_text)
    plt.legend()

    # Save plot to folder
    output_folder = "bubble_graphs_discontinous"
    #output_folder = "bubble_graphs"
    output_filename = "bubble_" + os.path.splitext(os.path.basename(video_path))[0] + ".png"
    output_path = os.path.join(output_folder, output_filename)

    os.makedirs(output_folder, exist_ok=True)

    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

    plt.close()

    return slope

def main():
    global DEBUG

    if len(sys.argv) < 3:
        print("Usage: python script.py [rerun-debug|rerun|cache] video_path1 video_path2 ...")
        sys.exit(1)

    mode = sys.argv[1]
    video_paths = sys.argv[2:]  # Paths are passed as command-line arguments

    DEBUG = (mode == "rerun-debug")

    cache = {}
    if mode == "rerun" or mode == "rerun-debug":
        with Pool(processes=16) as pool:
            cache = dict(pool.map(process_video, video_paths))
        with open("cache.json", "w") as f:
            json.dump(cache, f)
    elif mode == "cache":
        if not os.path.exists("cache.json"):
            print("Cache file not found. Please run with 'rerun' mode first.")
            sys.exit(1)
        with open("cache.json", "r") as f:
            cache = json.load(f)
    else:
        print("First argument must be 'rerun' or 'cache'")
        sys.exit(1)

    slopes = {video_path: plot_values(video_path, cache[video_path])
              for video_path in video_paths}

    for video_path, slope in slopes.items():
        print(f"Video: {video_path}, Slope: {slope}")

if __name__ == "__main__":
    main()
