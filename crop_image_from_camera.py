#####################################################
####    DO NOT RUN IN PYCHARM  #####
#### Use only virtual environment with installed SURF and SIFT   ###

#######################################################




"""
import cv2
import numpy as np

image = cv2.imread('images/input.jpg')
height, width = image.shape[:2]

# Let's get the starting pixel coordiantes (top  left of cropping rectangle)
start_row, start_col = int(height * .25), int(width * .25)

# Let's get the ending pixel coordinates (bottom right)
end_row, end_col = int(height * .75), int(width * .75)

# Simply use indexing to crop out the rectangle we desire
cropped = image[start_row:end_row , start_col:end_col]

cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.imshow("Cropped Image", cropped)
cv2.waitKey(0)
cv2.imwrite('piramide.jpg', cropped)
cv2.imwrite('piramide.png', cropped)
cv2.destroyAllWindows()
"""

import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("Object1")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    # cv2.imshow("Object 1", frame)

    # Get height and width of webcam frame
    height, width = frame.shape[:2]

    # Define ROI Box Dimensions
    top_left_x = int(width / 3)
    top_left_y = int((height / 2) + (height / 4))
    bottom_right_x = int((width / 3) * 2)
    bottom_right_y = int((height / 2) - (height / 4))

    # print(top_left_x)
    # print(top_left_y)
    # print(bottom_right_x)
    # print(bottom_right_y)

    # Draw rectangular window for our region of interest   
    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 255, 3)

    # print(frame)
    # Crop window of observation we defined above
    cropped = frame[bottom_right_y:top_left_y, top_left_x:bottom_right_x]
    # print(cropped)
    cv2.imshow('want recognize this product? ESC for yes, space fore next photo', cropped)  # comment later
    cv2.imshow('Set object in blue square press SPACE if ok then press escape', frame)

    # Flip frame orientation horizontally we do that because it is easier for us to put he object in front of the camera
    frame = cv2.flip(frame, 1)  # comment this later
    # cv2.imshow('flipped frame', frame)#comment this later

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape will closing a window and jump to next stage")
        break
    elif k % 256 == 32:
        # SPACE pressed
        # img_name = "opencv_frame_{}.png".format(img_counter)
        img_name = "maro_image/Object1.png".format(img_counter)
        # cv2.imwrite(img_name, frame)
        cv2.imwrite(img_name, cropped)
        print("{} written!".format(img_name))
        img_counter += 1
        cv2.imshow('Press escape If you happy with this photo', cropped)

cam.release()

cv2.destroyAllWindows()
