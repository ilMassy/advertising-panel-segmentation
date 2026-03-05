import cv2
import os
import argparse


def extract_frames(video_name, interval):
    # Define paths
    video_path = os.path.join("data/raw", video_name)
    output_dir = "data/processed/train/images"

    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: The file {video_path} was not found in data/raw/")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Check if the video is valid
    if fps == 0:
        print("Error: Invalid video file or unable to read FPS.")
        return

    # Calculate how many frames to skip (hop)
    hop = int(fps * interval)
    count = 0
    saved_count = 0

    # Get the base name (e.g., 'highlights' from 'highlights.mp4')
    base_name = os.path.splitext(video_name)[0]

    print(f"Extraction started: {video_name} (1 frame every {interval}s)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Save the frame based on the interval
        if count % hop == 0:
            # Simple prefix: V_ + videoname + counter
            filename = f"V_{base_name}_{saved_count:03d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved_count += 1

        count += 1

    # Release resources
    cap.release()
    print(f"Success! Saved {saved_count} images with 'V_' prefix in: {output_dir}")


if __name__ == "__main__":
    # Command line argument configuration
    parser = argparse.ArgumentParser(description="Extract frames from a video for YOLOv8-seg training.")
    parser.add_argument("video", help="Name of the video file inside data/raw/")
    parser.add_argument("-i", "--interval", type=int, default=2, help="Seconds between frames (default: 2)")

    args = parser.parse_args()

    extract_frames(args.video, args.interval)