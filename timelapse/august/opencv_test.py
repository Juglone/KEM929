import cv2
import matplotlib.pyplot as plt
import sys
import os

# Open the video with cv2.VideoCapture
video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)

# Extract relevant parts of file name
parts = video_path.split('_')
pH_value = parts[0]
IS_value = parts[1].replace('-', ' ')
if parts[2] == "without":
    condition = f"{parts[2]} {parts[3]}"
    date = parts[4].split('.')[0]
else:
    condition = f"{parts[2]} {parts[3]} {parts[4]}"
    date = parts[5].split('.')[0]

# Control that the video was opened successfully
if not cap.isOpened():
    print("Kunde inte öppna videon.")
    exit()

def plot():
    grayscale_values = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        image = cv2.resize(frame, (100, 100), interpolation=cv2.INTER_AREA)
        image_16bit = image.astype("float32")
        grayscale_values.append(image_16bit.mean()/255)
        #use mean?? Want to weigh everything equally (mean) or not?

        # Convert frame to grayscale
        #gray_frame = cv2.cvtColor(image_16bit, cv2.COLOR_BGR2GRAY)

        # Calculate the average grayscale value
        #grayscale_values.append(gray_frame[0,0])

    cap.release()

    print("cap released")

    # Plot grayscale values over time
    plt.plot(grayscale_values)
    plt.xlabel('Frame Number')
    plt.ylabel('Average Grayscale Value')
    plt.title(f"Grayscale Value Over Time for gel in a medium with {pH_value} and {IS_value}, {condition}, {date}")

    # Save plot to folder
    output_folder = "grayscale_graphs"
    output_filename = "grayscaleplot_" + video_path.split('.')[0] + ".png"
    output_path = os.path.join(output_folder, output_filename)

    os.makedirs(output_folder, exist_ok=True)

    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")


plot()

# Free the videosource and close all open windows
cap.release()
cv2.destroyAllWindows()
