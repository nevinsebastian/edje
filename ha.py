import cv2
import numpy as np
import os

# Ensure output folder exists
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest

img = cv2.imread('input/6.jpeg')
img_original = img.copy()
original_width = img_original.shape[1]  # Store the original width

# Image modification
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 20, 30, 30)
edged = cv2.Canny(gray, 10, 20)

# Contour detection
contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

biggest = biggest_contour(contours)
cv2.drawContours(img, [biggest], -1, (0, 255, 0), 3)

# Pixel values in the original image
points = biggest.reshape(4, 2)
input_points = np.zeros((4, 2), dtype="float32")

points_sum = points.sum(axis=1)
input_points[0] = points[np.argmin(points_sum)]
input_points[3] = points[np.argmax(points_sum)]

points_diff = np.diff(points, axis=1)
input_points[1] = points[np.argmin(points_diff)]
input_points[2] = points[np.argmax(points_diff)]

# Compute widths and heights
top_width = np.linalg.norm(input_points[1] - input_points[0])
bottom_width = np.linalg.norm(input_points[2] - input_points[3])
right_height = np.linalg.norm(input_points[1] - input_points[2])
left_height = np.linalg.norm(input_points[0] - input_points[3])

# Set output width and calculate height based on the contour's aspect ratio
max_width = original_width
aspect_ratio = max(top_width, bottom_width) / original_width
max_height = int((right_height + left_height) / 2 / aspect_ratio)  # Maintain aspect ratio

# Desired points values in the output image
converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])

# Perspective transformation
matrix = cv2.getPerspectiveTransform(input_points, converted_points)
img_output = cv2.warpPerspective(img_original, matrix, (0, 0))  # No fixed width, let OpenCV determine

# Save the output image in the output folder
output_path = os.path.join(output_folder, 'document.jpg')
cv2.imwrite(output_path, img_output)

# Display the images
gray = np.stack((gray,) * 3, axis=-1)
edged = np.stack((edged,) * 3, axis=-1)

img_hor = np.hstack((img_original, gray, edged, img))
cv2.imshow("Contour detection", img_hor)
cv2.imshow("Warped perspective", img_output)

cv2.waitKey(0)
cv2.destroyAllWindows()