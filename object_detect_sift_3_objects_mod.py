###### It is Working test surf an sift;-)))))) #####
# ubuntu 20 and python 3.8 opencv 3.4.4

#####################################################
####    DO NOT RUN IN PYCHARM  #####
#### Use only virtual environment with installed SURF and SIFT   ###

#######################################################

import cv2
import numpy as np


def sift_detector(new_image, image_template):
    # Function that compares input image to template
    # It then returns the number of SIFT matches between them

    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    image2 = image_template

    # Create SIFT detector object
    # sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()

    # Obtain the keypoints and descriptors using SIFT
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)

    # Define parameters for our Flann Matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
    search_params = dict(checks=100)

    # Create the Flann Matcher object
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Obtain matches using K-Nearest Neighbor Method
    # the result 'matches' is the number of similar matches found in both images
    matchesA = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Store good matches using Lowe's ratio test
    good_matches = []
    for m, n in matchesA:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return len(good_matches)


def sift_detector2(new_image, image_template2):
    # Function that compares input image to template
    # It then returns the number of SIFT matches between them

    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    image2 = image_template2

    # Create SIFT detector object
    # sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()

    # Obtain the keypoints and descriptors using SIFT
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)

    # Define parameters for our Flann Matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
    search_params = dict(checks=100)

    # Create the Flann Matcher object
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Obtain matches using K-Nearest Neighbor Method
    # the result 'matches' is the number of similar matches found in both images
    matchesB = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Store good matches using Lowe's ratio test
    good_matches = []
    for m, n in matchesB:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return len(good_matches)


def sift_detector3(new_image, image_template3):
    # Function that compares input image to template
    # It then returns the number of SIFT matches between them

    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    image2 = image_template3

    # Create SIFT detector object
    # sift = cv2.SIFT()#this command is for older python
    sift3 = cv2.xfeatures2d.SIFT_create()
    # Unhide this program and hide SIFT if you want to try SURF
    # surf = cv2.xfeatures2d.SURF_create()
    # Obtain the keypoints and descriptors using SIFT
    keypoints_1, descriptors_1 = sift3.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift3.detectAndCompute(image2, None)

    # Unhide this program and hide SIFT if you want to try SURF
    # surf = cv2.xfeatures2d.SURF_create()
    # (kps, descs) = surf.detectAndCompute(gray, None)
    # print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
    # show kp points
    # image1 = cv2.drawKeypoints(image1, keypoints_1, None)
    # image2 = cv2.drawKeypoints(image2, keypoints_2, None)
    # cv2.imshow('Image1', image1)
    # cv2.imshow('Image2', image2)
    # cv2.waitKey()

    # Define parameters for our Flann Matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
    search_params = dict(checks=100)

    # Create the Flann Matcher object
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    ### create brute force maching
    bf = cv2.BFMatcher()

    # Obtain matches using K-Nearest Neighbor Method
    # the result 'matches' is the number of similar matches found in both images
    matchesC = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    # Store good matches using Lowe's ratio test

    good_matches = []
    for m, n in matchesC:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return len(good_matches)


cap = cv2.VideoCapture(0)
# This is for testing only
# Load our image template, this is our reference image
# image_template = cv2.imread('images/box_in_scene.png', 0)
# image_template = cv2.imread('maro_image/marek_headphone.jpg', 0) 
# image_template = cv2.imread('maro_image/notatnik.jpg', 0) # zamienia na czarnobiale
# image_template = cv2.imread('maro_image/pasta_zegarek.jpg', 0) # zamienia na czarnobiale
image_template = cv2.imread('maro_image/Object1.png', 0)  # tylko pasta do zebow
image_template2 = cv2.imread('maro_image/object2.jpg', 0)  # g-shock
image_template3 = cv2.imread('maro_image/object3.jpg', 0)  # g-shock

# This is for testing only
# cv2.imshow('Image template', image_template)
# cv2.waitKey()
img_scaled = cv2.resize(image_template, (250, 250),
                        interpolation=cv2.INTER_AREA)  # i change size of my image because is too big
img_scaled2 = cv2.resize(image_template2, (250, 250),
                         interpolation=cv2.INTER_AREA)  # i change size of my image because is too big
img_scaled3 = cv2.resize(image_template3, (250, 250), interpolation=cv2.INTER_AREA)
# This is for testing only
# cv2.imshow('Scaling pasta', img_scaled)
# cv2.imshow('Scaling zegarek', img_scaled2)
# cv2.waitKey()
image_template = img_scaled
image_template2 = img_scaled2
image_template3 = img_scaled3
cv2.imshow('Object number 1 to recognize', image_template)
# This is for testing only if you press any button on the keyboard this will be continuing with the program
# cv2.waitKey()
cv2.imshow('Object number 2 to recognize', image_template2)
# This is for testing only
# cv2.waitKey()
cv2.imshow('Object number 3 to recognize', image_template3)
# This is for testing only
# cv2.waitKey()


while True:

    # Get webcam images
    ret, frame = cap.read()

    # Get height and width of webcam frame
    height, width = frame.shape[:2]

    # Define ROI Box Dimensions
    top_left_x = int(width / 3)
    top_left_y = int((height / 2) + (height / 4))
    bottom_right_x = int((width / 3) * 2)
    bottom_right_y = int((height / 2) - (height / 4))

    # This is for testing only
    # print(top_left_x)
    # print(top_left_y)
    # print(bottom_right_x)
    # print(bottom_right_y)

    # Draw rectangular window for our region of interest   
    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 255, 3)

    # print(frame)
    # Crop window of observation we defined above
    cropped = frame[bottom_right_y:top_left_y, top_left_x:bottom_right_x]
    # This is for testing only
    # print(cropped)
    # cv2.imshow('Image cropped', cropped)#comment later
    # cv2.imshow('frame before flipping', frame)

    # ########## Jak zrobie zdjecie zdjecia na komputerze telefonem i wyswietle i potem pokaze przed kamera to dziala
    # ale jak chce sie pokaza na ekranie to tez rozpoznaje ;) ale mniej dokladnie
    #Flip frame orientation horizontally we do that because
    # it is easer for us to put he object in fron of the camera
    frame = cv2.flip(frame, 1)  # comment this later
    cv2.imshow('flipped frame', frame)  # comment this later

    '''
    img_counter = 0
    if cv2.waitKey(1)%256==32:
     img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        print("are you happy with this photo")
    '''

    # Get number of SIFT matches
    #Provide photos
    matches = sift_detector(cropped, image_template)
    matches2 = sift_detector2(cropped, image_template2)
    matches3 = sift_detector3(cropped, image_template3)

    # Display status string showing the current no. of matches
    cv2.putText(frame, str(matches), (100, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 1)
    cv2.putText(frame, str(matches2), (250, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 1)
    cv2.putText(frame, str(matches3), (500, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 1)

    # Our threshold to indicate object deteciton
    # We use 10 since the SIFT detector returns little false positves
    threshold = 10

    # If matches exceed our threshold then object has been detected
    if matches > threshold:
        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 3)
        cv2.putText(frame, 'First object ', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    if matches2 > threshold:
        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 3)
        cv2.putText(frame, 'Oral-B', (50, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    if matches3 > threshold - 2:
        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 3)
        cv2.putText(frame, 'G-Shock', (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    # print(matches3)
    cv2.imshow('Object Detector using SIFT', frame)
    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()
