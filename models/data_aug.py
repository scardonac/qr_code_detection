import albumentations as A
import cv2
import os
import random

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.3),
    A.Rotate(limit=10, p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def augment_and_save_images(image_folder: str,
                            label_folder: str,
                            output_folder: str, 
                            augment_fraction: float = 0.2
                            ) -> None:
    """
    Performs augmentation on a fraction of images and their labels in YOLO format
    and saves the results.

    Args:
        image_folder (str): Folder containing the original images.
        label_folder (str): Folder containing the YOLO labels corresponding to
        the images.
        output_folder (str): Output folder where the augmented images and labels will be
        saved.
        augment_fraction (float, optional): Fraction of the images that will be augmented.
        Defaults to 0.2 (20%).

    Returns:
        None: The function returns nothing, but saves the augmented images and labels
        in the output directory.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, image_file.replace('.jpg', '.txt'))

        image = cv2.imread(image_path)
        with open(label_path, 'r') as f:
            labels = f.readlines()

        bboxes = []
        for label in labels:
            parts = label.strip().split()
            class_id = parts[0]
            bbox = list(map(float, parts[1:]))
            bboxes.append(bbox)

        if random.random() < augment_fraction:
            augmented = transform(image=image, bboxes=bboxes, class_labels=[class_id for _ in bboxes])
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']

            output_image_path = os.path.join(output_folder, 'images', f"aug_{image_file}")
            cv2.imwrite(output_image_path, aug_image)

            output_label_path = os.path.join(output_folder, 'labels', f"aug_{image_file.replace('.jpg', '.txt')}")
            with open(output_label_path, 'w') as out_f:
                for bbox in aug_bboxes:
                    out_f.write(f"{class_id} {' '.join(map(str, bbox))}\n")

# augment_and_save_images(
#     image_folder='C:/Users/Usuario/Downloads/qr_detection_dataset/train/images',
#     label_folder='C:/Users/Usuario/Downloads/qr_detection_dataset/train/labels',
#     output_folder='C:/Users/Usuario/Downloads/qr_detection_dataset/train'
# )
