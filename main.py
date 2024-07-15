import cv2
import pickle
import cvzone
import numpy as np

# video feed
cap = cv2.VideoCapture('carPark.mp4')

with open('CarParkPos', 'rb') as f:
    pos_list = pickle.load(f)

width = 107
height = 48

def draw_gradient_rectangle(img, start_point, end_point, start_color, end_color):
    x1, y1 = start_point
    x2, y2 = end_point
    for i in range(y1, y2):
        alpha = (i - y1) / (y2 - y1)
        color = tuple([int(start_color[j] * (1 - alpha) + end_color[j] * alpha) for j in range(3)])
        cv2.line(img, (x1, i), (x2, i), color, 1)

def check_parking_space(img_pro, img):
    space_counter = 0

    # overlay the rectangles that were drawn on the image
    for pos in pos_list:
        x, y = pos

        # crop the individual rectangles to get images of each parking spot
        img_crop = img_pro[y:y + height, x:x + width]

        # gives count number of pixels
        count = cv2.countNonZero(img_crop)

        if count < 860:
            color = (0, 255, 0)
            thickness = 5
            space_counter += 1
        else:
            color = (0, 0, 255)
            thickness = 2

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)

    # Draw a rounded rectangle with a gradient background for the counter
    start_point = (50, 10) 
    end_point = (400, 80)   
    draw_gradient_rectangle(img, start_point, end_point, (0, 200, 0), (0, 100, 0))
    cv2.rectangle(img, start_point, end_point, (50, 50, 50), 2, cv2.LINE_AA)
    
    # Put the text with a shadow effect
    text = f'Free: {space_counter}/{len(pos_list)}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5  
    text_thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
    text_x = start_point[0] + (end_point[0] - start_point[0] - text_size[0]) // 2
    text_y = start_point[1] + (end_point[1] - start_point[1] + text_size[1]) // 2
    cv2.putText(img, text, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

# get the frames from video
while True:

    # check if the current position is the last frame and restart video if true
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()

    # convert image to gray and blurred, and then convert to inverse binary
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
    img_threshold = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 25, 16)

    # get rid of stray dots to make image cleaner
    img_median = cv2.medianBlur(img_threshold, 5)

    kernel = np.ones((3, 3), np.uint8)
    img_dilate = cv2.dilate(img_median, kernel, iterations=1)

    # calls function to crop the images using the edited photo
    check_parking_space(img_dilate, img)

    cv2.imshow("Image", img)
    cv2.waitKey(10)
