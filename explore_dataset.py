from pathlib import Path

import cv2

image_dir = Path("datasets/coco128/images/train2017")

images = list(image_dir.glob("*.jpg"))

print("Total Images:", len(images))

image = cv2.imread("datasets/coco128/images/train2017/000000000009.jpg")

cv2.imshow("Image", image)

cv2.waitKey(0)
