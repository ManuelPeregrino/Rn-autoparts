import cv2
import os
from tkinter import Tk, filedialog

def select_folder():
    root = Tk()
    root.withdraw()  # Hide the main window
    folder_selected = filedialog.askdirectory(title="Select folder to save images")
    return folder_selected

def capture_images(folder_path, total_images=110, skip_initial=10):
    folder_name = os.path.basename(folder_path.rstrip("/\\"))
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Starting image capture...\nSaving images to: {folder_path}")
    img_count = 0
    saved_count = 0

    while saved_count < total_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        if img_count >= skip_initial:
            filename = f"{folder_name}_{saved_count+1:03d}.jpg"
            filepath = os.path.join(folder_path, filename)
            cv2.imwrite(filepath, frame)
            print(f"Saved: {filepath}")
            saved_count += 1

        img_count += 1
        cv2.waitKey(50)  # Slight delay between captures

    cap.release()
    print("Image capture complete.")

if __name__ == "__main__":
    folder = select_folder()
    if folder:
        capture_images(folder)
    else:
        print("No folder selected.")
