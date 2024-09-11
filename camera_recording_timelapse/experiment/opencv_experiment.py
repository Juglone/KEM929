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

        cv2.imshow('Processed Video', square_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return grayscale_values


def plot_values(video_path, grayscale_values):
    grayscale_values = np.array(grayscale_values)
    hours = np.arange(len(grayscale_values))*200/(30*3600)

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(hours, grayscale_values)

    parts = video_path.split('_')
    pH_value = parts[0]
    IS_value = parts[1].replace('-', ' ')
    if parts[2] == "without":
        condition = f"{parts[2]} {parts[3]}"
        date = parts[4]
    else:
        condition = f"{parts[2]} {parts[3]} {parts[4]}"
        date = parts[5]

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

    output_folder = "grayscale_graphs"
    output_filename = "grayscaleplot_" + video_path.split('.')[0] + ".png"
    output_path = os.path.join(output_folder, output_filename)

    os.makedirs(output_folder, exist_ok=True)

    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

    plt.close()

    return slope


def main():
    video_path = sys.argv[1]
    process_video(video_path)


main()
