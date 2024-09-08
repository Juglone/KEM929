import cv2
import matplotlib.pyplot as plt

# Open the video with cv2.VideoCapture
video_path = "/home/naph/files/uni/kem929/KEM929/timelapse/august/pH7_IS-0_with_straw_6mm_2024_08_12.mkv.timelapse.mp4"
cap = cv2.VideoCapture(video_path)

# Control that the video was opened successfully
if not cap.isOpened():
    print("Kunde inte öppna videon.")
    exit()

# Downscaling-faktorn
scale_factor = 0.5  # Decreasing the size of the video to 50% of the original

def downscale_video():
    # Read frame from the video
    ret, frame = cap.read()

    # If the reading was not successful (end of video reached), break the loop
    if not ret:
        return 1

    # Downscale bilden
    width = int(frame.shape[1] * scale_factor)
    height = int(frame.shape[0] * scale_factor)
    downscaled_frame1 = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    downscaled_frame2 = cv2.resize(cv2.resize(frame, (1, 1), interpolation=cv2.INTER_AREA), (40,40))

    # Show the downscaled frame
    cv2.imshow('Downscaled Video 1', downscaled_frame1)
    cv2.imshow('Downscaled Video 2', downscaled_frame2)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return 1

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
    plt.title('Grayscale Value Over Time')
    plt.show()

plot()

# while True:
    #plot() # donẗ generate image forever
    #downscale_video()



# Free the videosource and close all open windows
cap.release()
cv2.destroyAllWindows()
