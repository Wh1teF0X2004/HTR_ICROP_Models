## To select coordinates of the crop fields
import cv2

# Load the image
image = cv2.imread('Form_Template.jpg')

# Resize the image for display
scale_percent = 50  # Adjust this percentage as needed
width = int(image.shape[1] * scale_percent / 150)
height = int(image.shape[0] * scale_percent / 150)
dim = (width, height)

# Resize image
resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Select ROI on the resized image
roi = cv2.selectROI("Select ROI", resized_image)

# Print the coordinates of the selected ROI on the resized image
print("Selected ROI on resized image:", roi)

# Close the window
cv2.destroyAllWindows()

# Calculate the scaling factor
scale_x = image.shape[1] / resized_image.shape[1]
scale_y = image.shape[0] / resized_image.shape[0]

# Map the coordinates back to the original image size
x, y, w, h = roi
original_roi = (int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y))

# Print the mapped coordinates
print("Mapped ROI on original image:", original_roi)
