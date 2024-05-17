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

def draw_bboxes(image, bboxes, class_labels, class_names=None):
    height, width, _ = image.shape
    for bbox, class_id in zip(bboxes, class_labels):
        x_center, y_center, w, h = bbox
        x1 = int((x_center - w / 2) * width)
        y1 = int((y_center - h / 2) * height)
        x2 = int((x_center + w / 2) * width)
        y2 = int((y_center + h / 2) * height)
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = str(class_id) if class_names is None else class_names[class_id]
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image

def process_image_and_labels(image_path, label_path, output_image_path, output_label_path, class_names=None):
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

    # Draw bounding boxes on the transformed image
    image_with_bboxes = draw_bboxes(transformed_image.copy(), transformed_bboxes, transformed_class_labels, class_names)
    cv2.imwrite(output_image_path.replace('.jpg', '_with_bboxes.jpg'), image_with_bboxes)

def main(images_dir, labels_dir, output_images_dir, output_labels_dir, class_names=None):
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)
    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)

    for image_file in os.listdir(images_dir):
        if image_file.endswith(('jpg', 'jpeg', 'png')):
            image_path = os.path.join(images_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)

            output_image_path = os.path.join(output_images_dir, os.path.splitext(image_file)[0] + '_augmented.jpg')
            output_label_path = os.path.join(output_labels_dir, os.path.splitext(image_file)[0] + '_augmented.txt')

            process_image_and_labels(image_path, label_path, output_image_path, output_label_path, class_names)

if __name__ == "__main__":
    images_dir = "images"
    labels_dir = "labels"
    output_images_dir = "output_images"
    output_labels_dir = "output_labels"
    
    # If you have a list of class names, you can declare it here or pass it the value none
    class_names = [
        'label1',
        'label2',
        ...
    ]

    main(images_dir, labels_dir, output_images_dir, output_labels_dir, class_names)
