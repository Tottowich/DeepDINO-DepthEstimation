import cv2
import os

def capture_and_save_image(n=10,folder_path="captured_images"):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Capture a single frame
    def capture_frame():
        ret, frame = cap.read()

        # Save the image if the capture is successful
        if ret:

            image_path = os.path.join(folder_path, "captured_image.jpg")
            if os.path.exists(image_path):
                i = len(os.listdir(folder_path))
                image_path = os.path.join(folder_path, f"captured_image_{i}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Image saved at {image_path}")
        else:
            print("Error: Could not capture an image.")
    for i in range(n):
        capture_frame()
    # Release the webcam
    cap.release()

# Run the function to capture and save the image
capture_and_save_image(n=10)
