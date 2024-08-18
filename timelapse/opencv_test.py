import cv2
import matplotlib.pyplot as plt

# Öppna videon med hjälp av cv2.VideoCapture
video_path = "/home/naph/files/uni/kem929/KEM929/timelapse/july/pH3_IS-I_with_straw_2024-07-24.timelapse.mp4"  # Byt ut med sökvägen till din video
cap = cv2.VideoCapture(video_path)

# Kontrollera om videon öppnades framgångsrikt
if not cap.isOpened():
    print("Kunde inte öppna videon.")
    exit()

# Ange downscaling-faktorn
scale_factor = 0.5  # Minskar storleken till 50% av originalet

def downscale_video():
    # Läs en bildruta från videon
    ret, frame = cap.read()

    # Om läsningen inte lyckades (slutet av videon nått), bryt loopen
    if not ret:
        return 1

    # Downscalea bilden
    width = int(frame.shape[1] * scale_factor)
    height = int(frame.shape[0] * scale_factor)
    downscaled_frame1 = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    downscaled_frame2 = cv2.resize(cv2.resize(frame, (1, 1), interpolation=cv2.INTER_AREA), (40,40))

    # Visa upp den downscalade bilden
    cv2.imshow('Downscaled Video 1', downscaled_frame1)
    cv2.imshow('Downscaled Video 2', downscaled_frame2)

    # Avbryt om användaren trycker på 'q'
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
        grayscale_values.append(image_16bit.mean()/255) #use mean?? viktar alla lika mycket
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



# Frigör videokällan och stäng alla öppna fönster
cap.release()
cv2.destroyAllWindows()
