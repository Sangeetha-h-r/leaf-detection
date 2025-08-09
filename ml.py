from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import time

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Try different camera backends and indices
def initialize_camera():
    """Initialize camera with fallback options"""
    # Try DirectShow backend first (more reliable on Windows)
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not camera.isOpened():
        print("DirectShow backend failed, trying MSMF...")
        camera = cv2.VideoCapture(0, cv2.CAP_MSMF)
    
    if not camera.isOpened():
        print("MSMF backend failed, trying default...")
        camera = cv2.VideoCapture(0)
    
    # Try different camera indices
    for cam_index in range(3):
        if camera.isOpened():
            break
        print(f"Trying camera index {cam_index}...")
        camera = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    
    if not camera.isOpened():
        raise RuntimeError("Could not open any camera. Please check camera connection and permissions.")
    
    # Set camera properties for better performance
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    # Warm up camera
    for _ in range(5):
        camera.read()
    
    return camera

try:
    print("Initializing camera...")
    camera = initialize_camera()
    print("Camera initialized successfully!")
    
    # Counter for failed frame attempts
    failed_attempts = 0
    max_failed_attempts = 10
    
    while True:
        # Grab the webcamera's image.
        ret, image = camera.read()
        
        # Check if frame was successfully captured
        if not ret:
            failed_attempts += 1
            print(f"Failed to grab frame (attempt {failed_attempts}/{max_failed_attempts})")
            
            if failed_attempts >= max_failed_attempts:
                print("Too many failed attempts. Please check camera connection.")
                break
            
            time.sleep(0.1)
            continue
        
        # Reset failed attempts counter on successful frame
        failed_attempts = 0
        
        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Show the image in a window
        cv2.imshow("Webcam Image", image)

        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predicts the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

        # Listen to the keyboard for presses.
        keyboard_input = cv2.waitKey(1)

        # 27 is the ASCII for the esc key on your keyboard.
        if keyboard_input == 27:
            break

except Exception as e:
    print(f"Error: {e}")
    print("Please check if your camera is connected and not being used by another application.")

finally:
    if 'camera' in locals():
        camera.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")
