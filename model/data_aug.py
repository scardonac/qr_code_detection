import albumentations as A
import cv2
import os
import random

# Definir transformaciones con Albumentations
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
    Realiza augmentación sobre una fracción de imágenes y sus etiquetas en formato YOLO
    y guarda los resultados.

    Args:
        image_folder (str): Carpeta que contiene las imágenes originales.
        label_folder (str): Carpeta que contiene las etiquetas YOLO correspondientes a
        las imágenes.
        output_folder (str): Carpeta de salida donde se guardarán las imágenes y etiquetas
        augmentadas.
        augment_fraction (float, optional): Fracción de las imágenes que serán augmentadas.
        Por defecto a 0.2 (20%).
        
    Returns:
        None: La función no devuelve nada, pero guarda las imágenes y etiquetas augmentadas
        en el directorio de salida.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_file in os.listdir(image_folder):
        # Cargar imagen y etiqueta
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, image_file.replace('.jpg', '.txt'))

        image = cv2.imread(image_path)
        with open(label_path, 'r') as f:
            labels = f.readlines()

        # Leer bounding boxes en formato YOLO
        bboxes = []
        for label in labels:
            parts = label.strip().split()
            class_id = parts[0]
            bbox = list(map(float, parts[1:]))
            bboxes.append(bbox)

        # Aplicar augmentations al 30% de las imágenes
        if random.random() < augment_fraction:
            augmented = transform(image=image, bboxes=bboxes, class_labels=[class_id for _ in bboxes])
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']

            # Guardar imagen augmentada
            output_image_path = os.path.join(output_folder, 'images', f"aug_{image_file}")
            cv2.imwrite(output_image_path, aug_image)

            # Guardar etiquetas augmentadas
            output_label_path = os.path.join(output_folder, 'labels', f"aug_{image_file.replace('.jpg', '.txt')}")
            with open(output_label_path, 'w') as out_f:
                for bbox in aug_bboxes:
                    out_f.write(f"{class_id} {' '.join(map(str, bbox))}\n")

# Aplicar augmentations y guardar las nuevas imágenes en una carpeta de salida
augment_and_save_images(
    image_folder='C:/Users/Usuario/Downloads/qr_detection_dataset/train/images',
    label_folder='C:/Users/Usuario/Downloads/qr_detection_dataset/train/labels',
    output_folder='C:/Users/Usuario/Downloads/qr_detection_dataset/train'
)
