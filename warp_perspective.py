import cv2
import numpy as np
import os

# Ensure output folder exists
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

def biggest_contour(contours, aspect_ratio_threshold=1.58, min_area=5000):
    biggest = np.array([])
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            if len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                
                # Check if contour is within an acceptable aspect ratio range
                if (0.7 * aspect_ratio_threshold) < aspect_ratio < (1.3 * aspect_ratio_threshold) and area > max_area:
                    biggest = approx
                    max_area = area
    return biggest

# Load image
img = cv2.imread('input/5.jpg')
img_original = img.copy()
original_width = img_original.shape[1]

# Image modification
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Gaussian Blur for smoothing
edged = cv2.Canny(gray, 5, 15)  # Adjusted Canny edge detection thresholds

# Display intermediate images for debugging
cv2.imshow("Grayscale", gray)
cv2.imshow("Edged", edged)

# Contour detection with external hierarchy
contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on hierarchy to get external ones only
filtered_contours = [contour for i, contour in enumerate(contours) if hierarchy[0][i][3] == -1]

# Find the biggest contour
biggest = biggest_contour(filtered_contours, aspect_ratio_threshold=1.58, min_area=5000)
if biggest.size != 0:
    cv2.drawContours(img, [biggest], -1, (0, 255, 0), 3)  # Draw the biggest contour
else:
    print("No suitable contour found.")

# Perspective transformation
if biggest.size != 0:
    points = biggest.reshape(4, 2)
    input_points = np.zeros((4, 2), dtype="float32")

    # Determine the new ordered points
    points_sum = points.sum(axis=1)
    input_points[0] = points[np.argmin(points_sum)]  # Top-left point
    input_points[3] = points[np.argmax(points_sum)]  # Bottom-right point

    points_diff = np.diff(points, axis=1)
    input_points[1] = points[np.argmin(points_diff)]  # Top-right point
    input_points[2] = points[np.argmax(points_diff)]  # Bottom-left point

    # Calculate dimensions for output
    top_width = np.linalg.norm(input_points[1] - input_points[0])
    bottom_width = np.linalg.norm(input_points[2] - input_points[3])
    right_height = np.linalg.norm(input_points[1] - input_points[2])
    left_height = np.linalg.norm(input_points[0] - input_points[3])

    max_width = original_width
    aspect_ratio = max(top_width, bottom_width) / original_width
    max_height = int((right_height + left_height) / 2 / aspect_ratio)

    converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])
    matrix = cv2.getPerspectiveTransform(input_points, converted_points)
    img_output = cv2.warpPerspective(img_original, matrix, (max_width, max_height))

    # Save the output image in the output folder
    output_path = os.path.join(output_folder, 'document.jpg')
    cv2.imwrite(output_path, img_output)

    # Display images
    gray = np.stack((gray,) * 3, axis=-1)
    edged = np.stack((edged,) * 3, axis=-1)
    img_hor = np.hstack((img_original, gray, edged, img))
    cv2.imshow("Contour detection", img_hor)
    cv2.imshow("Warped perspective", img_output)
else:
    print("No appropriate document detected.")

cv2.waitKey(0)
cv2.destroyAllWindows()
