import cv2
import numpy as np
from fourteen_seg import Segment

# Create a black image
height = 300
width = 1800
image = np.zeros((height, width, 3), dtype=np.uint8)

# Define the text and its properties
text = "HELLO WORLD"

speed = 15 # pixels
seg = Segment(offset=20, spacing=20)
seg_height = height
seg_width = min(width, 200)
# Initialize the position for scrolling
x = image.shape[1]
y = image.shape[0]

COLORS = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))

# # Scroll the text
while True:
    # Create a black background
    image.fill(0)

    start = x
    abc = seg.apply(start, 0, seg_width, seg_height, text)
    # abc = seg.apply_B(x, 0, seg_width, seg_height)
    for c in abc:
        cv2.line(image,(c[0],c[1]),(c[2],c[3]), COLORS, 25)

    # Decrement the x-coordinate for scrolling effect
    x -= speed
    
    # Reset the text position
    if x < -len(text) * (seg_width + seg.offset):
        x = image.shape[1]
        COLORS = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))

    # Display the image
    cv2.imshow('Hello World', image)

    # Delay and check for exit
    if cv2.waitKey(30) & 0xFF == 27:
        break

# # Release the window
cv2.destroyAllWindows()