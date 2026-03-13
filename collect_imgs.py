import os
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

print(f'Collecting data for letter: {letter}, total images: {dataset_size}')

# Use webcam (change index if needed)
cap = cv2.VideoCapture(0)

capturing = False
counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Show instructions or capturing status
    if not capturing:
        cv2.putText(frame, 'Press "S" to start capturing', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, f'Capturing... {counter}/{dataset_size}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(25) & 0xFF

    if key == ord('s'):
        capturing = True  # Start capturing
    elif key == ord('q'):
        print('Stopping capture and closing camera...')
        break  # Exit the loop and close camera

    # Capture images if capturing is True
    if capturing and counter < dataset_size:
        filename = os.path.join(class_dir, '{}.jpg'.format(counter))
        cv2.imwrite(filename, frame)
        counter += 1

    # Stop capturing automatically if reached dataset_size
    if counter >= dataset_size:
        print(f'Finished capturing {dataset_size} images for letter {letter}.')
        break

# Release resources
cap.release()
cv2.destroyAllWindows()