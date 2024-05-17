import albumentations as A
import cv2
import os

# Define the transformation
transform = A.Compose([
    A.HorizontalFlip(always_apply=True),
    A.CLAHE(p=0.5, tile_grid_size=(8, 8)),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def read_labels(label_path):
    with open(label_path, 'r') as f:
        labels = f.readlines()
    bboxes = []
    class_labels = []
    for label in labels:
        parts = label.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        bboxes.append([x_center, y_center, width, height])
        class_labels.append(class_id)
    return bboxes, class_labels

def save_labels(label_path, bboxes, class_labels):
    with open(label_path, 'w') as f:
        for bbox, class_id in zip(bboxes, class_labels):
            x_center = bbox[0]
            y_center = bbox[1]
            width = bbox[2]
            height = bbox[3]
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def process_image_and_labels(image_path, label_path, output_image_path, output_label_path):
    # Read image
    image = cv2.imread(image_path)

    # Read labels
    bboxes, class_labels = read_labels(label_path)

    # Apply transformation
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    transformed_class_labels = transformed['class_labels']

    # Save transformed image
    cv2.imwrite(output_image_path, transformed_image)

    # Save transformed labels
    save_labels(output_label_path, transformed_bboxes, transformed_class_labels)

def main(images_dir, labels_dir):
    for image_file in os.listdir(images_dir):
        if image_file.endswith(('jpg', 'jpeg', 'png')):
            image_path = os.path.join(images_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)

            output_image_path = os.path.join(images_dir, os.path.splitext(image_file)[0] + '_augmented.jpg')
            output_label_path = os.path.join(labels_dir, os.path.splitext(image_file)[0] + '_augmented.txt')

            process_image_and_labels(image_path, label_path, output_image_path, output_label_path)

if __name__ == "__main__":
    images_dir = "/home/leducdat/deverlopment/Albumentations/images"
    labels_dir = "/home/leducdat/deverlopment/Albumentations/labels"

    main(images_dir, labels_dir)
