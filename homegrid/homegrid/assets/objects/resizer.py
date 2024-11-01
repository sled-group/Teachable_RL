from PIL import Image
import numpy as np

# Load the original image
original_image = Image.open("tomato.png")

# Convert the image to a numpy array
original_array = np.array(original_image)

# Create a new 32x32 array filled with zeros (black pixels)
new_array = np.zeros((32, 32, 4), dtype=np.uint8)

# Define the start point for the original image in the new array
start_point = 8  # since (32 - 16) / 2 = 8

# Copy the original image into the center of the new array
new_array[start_point : start_point + 16, start_point : start_point + 16] = (
    original_array
)

# Convert the new array back to an image
new_image = Image.fromarray(new_array)

# Save or display the new image
new_image.save("tomato_revised.png")
new_image.show()
