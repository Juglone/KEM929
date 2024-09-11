import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import sys
import os
from multiprocessing import Pool


def process_video(video_path):
    # Open the video with cv2.VideoCapture
    cap = cv2.VideoCapture(video_path)

    # Control that the video was opened successfully
    if not cap.isOpened():
        print("Could not open the video")
        exit()

    grayscale_values = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        h, w, _ = frame.shape

        square_size = min(h, w)

        x_start = w // 2 - square_size // 2
        y_start = h // 2 - square_size // 2
        x_end = x_start + square_size
        y_end = y_start + square_size

        square_frame = frame[y_start:y_end, x_start:x_end]

        square_frame = cv2.resize(square_frame, (100, 100), interpolation=cv2.INTER_AREA)

        grayscale_values.append(square_frame.astype("float32").mean() / 255)


    # Release the video capture object and close any OpenCV windows
    cap.release()

    return grayscale_values


def plot_values(video_path, grayscale_values):
    # Convert list to numpy array
    grayscale_values = np.array(grayscale_values)
    hours = np.arange(len(grayscale_values))*200/(30*3600)

    # Perform linear regression using scipy
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(hours, grayscale_values)

    # Extract relevant parts of file name
    parts = video_path.split('_')
    pH_value = parts[0]
    IS_value = parts[1].replace('-', ' ')
    if parts[2] == "without":
        condition = f"{parts[2]} {parts[3]}"
        date = parts[4]
    else:
        condition = f"{parts[2]} {parts[3]} {parts[4]}"
        date = parts[5]

    # Plot grayscale values over time with linear regression line
    plt.plot(hours, grayscale_values, label='Data')
    plt.plot(hours, slope * hours + intercept, 'r', label=f'Linear Fit (slope={slope:.4f})')
    plt.xlabel('Hours')
    plt.ylabel('Average Grayscale Value')
    title_text = (
        f"Grayscale Value Over Time for Gel in a Medium with {pH_value} and {IS_value}, \n"
        f"{condition}, {date}"
    )
    plt.title(title_text)
    plt.legend()

    # Save plot to folder
    output_folder = "grayscale_graphs"
    output_filename = "grayscaleplot_" + video_path.split('.')[0] + ".png"
    output_path = os.path.join(output_folder, output_filename)

    os.makedirs(output_folder, exist_ok=True)

    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

    plt.close()

    return slope


def main():
    video_paths = sys.argv[2:]  # Assuming the paths are passed as command-line arguments

    if sys.argv[1] == "rerun":
        with Pool(16) as pool:
            cache = pool.map(process_video, video_paths)
            with open("cache.json", "w") as f:
                f.truncate()
                f.write(json.dumps(cache))
    elif sys.argv[1] == "cache":
        with open("cache.json", "r") as f:
            cache = json.loads(f.read())
    else:
        assert(sys.argv[1] == "rerun" or sys.argv[1] == "cache")

    slopes = {video_path : plot_values(video_path, gs_values)
              for video_path, gs_values in zip(video_paths, cache)}

    for slope in slopes:
        print(slope)

if __name__ == "__main__":
    main()

