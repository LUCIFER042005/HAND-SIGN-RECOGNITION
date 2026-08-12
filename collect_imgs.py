import os
import time
import cv2

# Directory to save dataset
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Ask user which letter to capture
letter = input("Which letter do you want to capture? ").strip()

# Ask user how many images to capture
while True:
    try:
        dataset_size = int(input("How many images do you want to capture? "))
        if dataset_size <= 0:
            print("Please enter a positive number.")
            continue
        break
    except ValueError:
        print("Please enter a valid number.")

# Directory for this letter
class_dir = os.path.join(DATA_DIR, letter)
if not os.path.exists(class_dir):
    os.makedirs(class_dir)

print(f'\nCollecting data for letter: {letter}, total images: {dataset_size}')

# Use webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Wait for user to type 's' to start capturing
while True:
    user_input = input("\nPosition your hand sign in front of the camera, then type 's' and press Enter to start capturing: ").strip().lower()
    if user_input == 's':
        break
    elif user_input == 'q':
        print('Stopping capture and closing camera...')
        cap.release()
        exit()
    else:
        print("Invalid input. Type 's' to start or 'q' to quit.")

print("\nCapturing started...")
counter = 0

while counter < dataset_size:
    ret, frame = cap.read()
    if not ret:
        continue

    # Save frame directly to disk (Headless)
    filename = os.path.join(class_dir, '{}.jpg'.format(counter))
    cv2.imwrite(filename, frame)
    counter += 1

    # Show live terminal progress
    print(f"Capturing... {counter}/{dataset_size}", end="\r")

    # Slight delay between frame captures
    time.sleep(0.05)

print(f'\n\nFinished capturing {dataset_size} images for letter {letter}.')

# Release resources
cap.release()