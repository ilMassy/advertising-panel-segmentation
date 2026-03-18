import os
import numpy as np
import cv2
from pycocotools.coco import COCO


def process_dataset_folder(set_name):
    # Set up paths for the specific split (train, val, or test)
    base_path = f"data/processed/{set_name}"
    json_path = os.path.join(base_path, 'metadata.json')
    img_dir = os.path.join(base_path, 'images')
    mask_dir = os.path.join(base_path, 'masks')

    # Create the masks directory if it doesn't exist
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    # Check if the annotation file exists
    if not os.path.exists(json_path):
        print(f"Skipping {set_name}: metadata.json not found.")
        return

    # Initialize COCO API for instance annotations
    coco = COCO(json_path)
    image_ids = coco.getImgIds()

    # 1. Synchronization: Remove image files not present in the JSON annotations
    annotated_filenames = {coco.loadImgs(i)[0]['file_name'] for i in image_ids}

    for filename in os.listdir(img_dir):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            if filename not in annotated_filenames:
                file_path = os.path.join(img_dir, filename)
                os.remove(file_path)
                print(f"[{set_name}] Deleted unannotated file: {filename}")

    # 2. Mask Generation: Convert polygons to PNG masks
    print(f"[{set_name}] Generating {len(image_ids)} masks...")
    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)

        # Create a blank black mask (height x width)
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        # Fill the mask with polygons (value 1 for the target class)
        for ann in coco.loadAnns(ann_ids):
            binary_mask = coco.annToMask(ann)
            mask[binary_mask > 0] = 1

            # Save as PNG with the same filename as the image
        mask_filename = os.path.splitext(img_info['file_name'])[0] + '.png'
        cv2.imwrite(os.path.join(mask_dir, mask_filename), mask)


# Run the process for all dataset splits
if __name__ == "__main__":
    for split in ['train', 'val', 'test']:
        process_dataset_folder(split)
    print("\nDataset preparation complete!")