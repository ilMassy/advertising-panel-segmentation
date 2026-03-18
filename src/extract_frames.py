import cv2
import os
import sys


def extract_frames(video_full_path, target_folder):
    """
    Extracts frames from a video and saves them to a specified sub-directory.
    The frames are named following the pattern: {video_prefix}_{index}.jpeg
    """

    # 1. DEFINE ABSOLUTE PROJECT PATHS
    BASE_DIR = "/home/massimiliano/advertising-panel-segmentation"
    # Dynamic output directory based on user input (e.g., train, val, test)
    OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "images", target_folder)

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # 2. VALIDATE VIDEO PATH
    if not os.path.exists(video_full_path):
        print(f"Error: Video file not found at {video_full_path}")
        return

    # 3. INITIALIZE VIDEO CAPTURE
    cap = cv2.VideoCapture(video_full_path)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # --- FPS ADAPTIVE LOGIC ---
    # Retrieve the video's frames per second (FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Warning: Could not detect FPS, falling back to default interval (60 frames).")
        interval = 60
    else:
        # Define the time gap (in seconds) between extracted frames
        # Change this value to increase or decrease the number of saved frames
        seconds_between_frames = 2
        interval = int(fps * seconds_between_frames)

    # Extract the filename without extension to use as a prefix (e.g., match8)
    video_basename = os.path.basename(video_full_path)
    prefix = os.path.splitext(video_basename)[0]

    print(f"Processing video: {video_basename}")
    print(f"Target folder: {target_folder}")
    print(f"Detected FPS: {fps:.2f}")
    print(f"Sampling interval: {interval} frames (approx. every {interval / fps:.1f}s)")

    count = 0
    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Save frame if current index is a multiple of the calculated interval
        if count % interval == 0:
            filename = f"{prefix}_{saved_count:04d}.jpeg"
            save_path = os.path.join(OUTPUT_DIR, filename)

            # Save as high-quality JPEG (95)
            cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1

        count += 1

    cap.release()
    print(f"\nExtraction complete!")
    print(f"Total saved frames: {saved_count}")
    print(f"Destination: {OUTPUT_DIR}")


if __name__ == "__main__":
    # Ensure both the video path and the target sub-folder are provided
    if len(sys.argv) < 3:
        print("Usage: python3 extract_frames.py <full_path_to_video.mp4> <target_folder>")
        print("Example: python3 extract_frames.py data/raw/match8.mp4 val")
    else:
        extract_frames(sys.argv[1], sys.argv[2])