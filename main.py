import albumentations as A
import cv2
import os

# Define the transformation
transform = A.Compose([  # Add transformation algorithms here.
    A.HorizontalFlip(p=0.5),  # Apply horizontal flip with a probability of 50%
    A.RandomBrightnessContrast(
        brightness_limit=(0.0, 0.2),  # Set the limit for brightness adjustment
        contrast_limit=(0.0, 0.2),    # Set the limit for contrast adjustment
        p=0.5  # Apply this transformation with a probability of 50%
    ),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))  # Define bounding box parameters

def read_labels(label_path):
    with open(label_path, 'r') as f:
        labels = f.readlines()  # Read all lines from the label file
    bboxes = []
    class_labels = []
    for label in labels:
        parts = label.strip().split()  # Split the line into parts
        class_id = int(parts[0])  # First part is the class ID
        x_center = float(parts[1])  # Second part is the x-center of the bounding box
        y_center = float(parts[2])  # Third part is the y-center of the bounding box
        width = float(parts[3])  # Fourth part is the width of the bounding box
        height = float(parts[4])  # Fifth part is the height of the bounding box
        # Validate bounding box values
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
            raise ValueError(f"Invalid bounding box values in file {label_path}: {label.strip()}")
        bboxes.append([x_center, y_center, width, height])  # Add bounding box to the list
        class_labels.append(class_id)  # Add class ID to the list
    return bboxes, class_labels  # Return bounding boxes and class labels

def save_labels(label_path, bboxes, class_labels):
    with open(label_path, 'w') as f:
        for bbox, class_id in zip(bboxes, class_labels):
            x_center = round(bbox[0], 6)  # Round x-center to 6 decimal places
            y_center = round(bbox[1], 6)  # Round y-center to 6 decimal places
            width = round(bbox[2], 6)  # Round width to 6 decimal places
            height = round(bbox[3], 6)  # Round height to 6 decimal places
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")  # Write the label to the file

def process_image_and_labels(image_path, label_path, output_image_path, output_label_path):
    try:
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
    except ValueError as e:
        print(e)

def main(images_dir, labels_dir):
    for image_file in os.listdir(images_dir):
        if image_file.endswith(('jpg', 'jpeg', 'png')):  # Check if the file is an image
            image_path = os.path.join(images_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'  # Corresponding label file
            label_path = os.path.join(labels_dir, label_file)

            output_image_path = os.path.join(images_dir, os.path.splitext(image_file)[0] + '_augmented.jpg')
            output_label_path = os.path.join(labels_dir, os.path.splitext(image_file)[0] + '_augmented.txt')

            process_image_and_labels(image_path, label_path, output_image_path, output_label_path)  # Process and save augmented images and labels
    
if __name__ == "__main__":
    images_dir = "./images"
    labels_dir = "./labels"

    main(images_dir, labels_dir)  # Run the main function with specified directories
