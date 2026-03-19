import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def check_random_mask(split='train'):
    base_path = f"data/processed/{split}"
    img_dir = os.path.join(base_path, 'images')
    mask_dir = os.path.join(base_path, 'masks')

    # Pick a random image
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpeg', '.jpg'))]
    random_img = random.choice(img_files)
    mask_file = os.path.splitext(random_img)[0] + '.png'

    # Load image and mask
    image = cv2.imread(os.path.join(img_dir, random_img))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Error: Mask not found for {random_img}")
        return

    # Create an overlay (Green color for the panel)
    overlay = image.copy()
    overlay[mask > 0] = [0, 255, 0] # Highlight panel in green

    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Original: {random_img}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.addWeighted(image, 0.6, overlay, 0.4, 0))
    plt.title("Overlay (Mask Check)")
    plt.axis('off')

    # Save preview
    plt.savefig('mask_check_preview.png')
    print(f"Preview saved as 'mask_check_preview.png'!")
    plt.show()

if __name__ == "__main__":
    check_random_mask()
