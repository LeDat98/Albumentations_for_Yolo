import albumentations as A
import cv2

# Declare an augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(always_apply=True),
    A.CLAHE(p=0.5,tile_grid_size=(8, 8)),
    A.RandomBrightnessContrast(p=0.2),
])

# Read an image with OpenCV and convert it to the RGB colorspace
image = cv2.imread("image2124.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Augment an image
transformed = transform(image=image)
transformed_image = transformed["image"]

#show the original and augmented images
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.axis('off')
plt.imshow(image)
plt.show()

plt.figure(figsize=(8, 8))
plt.axis('off')
plt.imshow(transformed_image)
plt.show()

