import cv2
import numpy as np

# Load the Haar cascades (use correct XMLs)
face_cascade = cv2.CascadeClassifier(r"D:\Data Science & AI\Deep_Learning_Algorithms\MediaPipe\haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(r"D:\Data Science & AI\Deep_Learning_Algorithms\MediaPipe\haarcascade_eye.xml")

# Load the image
image = cv2.imread(r"D:\Pictures\Saved Pictures\Dastagiri.jpg")
if image is None:
    print("Error: Image not found or cannot be loaded!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

if len(faces) == 0:
    print("No faces found!")
else:
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)

        # Define ROI in grayscale and color images for eye detection
        roi_gray  = gray[y : y + h, x : x + w]
        roi_color = image[y : y + h, x : x + w]

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Show result
    cv2.imshow('Face and Eye detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
