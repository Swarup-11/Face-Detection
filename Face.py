import cv2
import numpy as np

# Load the Haar Cascade classifier for frontal face detection
face_cascade = cv2.CascadeClassifier("C:/Users/kiran/OneDrive/Desktop/open-cv/myenv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

# Load the reference image for matching
reference_image = cv2.imread("C:/Users/kiran/OneDrive/Desktop/open-cv/llb.jpg")
# Convert the reference image to grayscale
reference_image_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

# Create an ORB (Oriented FAST and Rotated BRIEF) detector
orb = cv2.ORB_create()

# Extract keypoints and descriptors from the reference image
keypoints_ref, descriptors_ref = orb.detectAndCompute(reference_image_gray, None)
# Ensure descriptors are of type CV_8U
descriptors_ref = descriptors_ref.astype(np.uint8)

# Open the webcam
video_cap = cv2.VideoCapture(0)

# Define a threshold for the minimum number of matches required to consider it a match
threshold = 20  # Adjust this value as needed

while True:
    ret, video_data = video_cap.read()
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the video frame
    faces = face_cascade.detectMultiScale(col, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    match_found = False  # Flag to track if a match is found
    
    for (x, y, w, h) in faces:
        # Crop the detected face from the video frame
        face_roi = col[y:y + h, x:x + w]

        # Extract keypoints and descriptors from the detected face
        keypoints_face, descriptors_face = orb.detectAndCompute(face_roi, None)
        # Ensure descriptors are of type CV_8U
        descriptors_face = descriptors_face.astype(np.uint8)

        # Create a BFMatcher (Brute-Force Matcher)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match the descriptors of the reference image and the detected face
        matches = bf.match(descriptors_ref, descriptors_face)

        # Sort the matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # If there is a match (enough matches), set the flag to True
        if len(matches) > threshold:
            match_found = True
            # Draw a "Match Found" text on the video frame
            cv2.putText(video_data, "Match Found", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # If no match is found for the current face, display "No Match Found."
            cv2.putText(video_data, "No Match Found", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw a rectangle around the detected face
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("video_live", video_data)

    if cv2.waitKey(100) == ord("a"):
        break

video_cap.release()
cv2.destroyAllWindows()