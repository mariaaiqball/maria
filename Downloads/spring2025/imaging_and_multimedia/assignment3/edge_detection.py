import cv2 
import os 
import numpy as np

# Display image
def display_img(img):
    cv2.imshow('Frame View', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
GRAY IMAGE: IRELAND
'''

current_directory = os.path.dirname(__file__) 
image_path = os.path.join(current_directory, "HW2images", "ireland-03gray.tif")
img = cv2.imread(image_path)

# Edge Detection
blur = cv2.GaussianBlur(img, (5,5), 0)
canny = cv2.Canny(blur, threshold1=180, threshold2=200)
display_img(canny)

highlighted = img.copy()
highlighted[canny != 0] = 255 
display_img(highlighted)
cv2.imwrite("save_output_images/edge_detection_ireland.tif", highlighted)



'''
COLOR IMAGE: AMSTERDAM
'''
image_path = os.path.join(current_directory, "HW2images", "Amsterdam.JPG")
img = cv2.imread(image_path)

# Edge Detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
canny = cv2.Canny(blur, threshold1=180, threshold2=200)
display_img(canny)

highlighted = img.copy()
highlighted[canny != 0] = 255  
display_img(highlighted)
cv2.imwrite("save_output_images/edge_detection_amsterdam_white.JPG", highlighted)


'''
COLOR IMAGE: AMSTERDAM
'''
image_path = os.path.join(current_directory, "HW2images", "Amsterdam.JPG")
img = cv2.imread(image_path)

# Edge Detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
canny = cv2.Canny(blur, threshold1=180, threshold2=200)
display_img(canny)

highlighted = img.copy()
highlighted[canny != 0] = 0  
display_img(highlighted)
cv2.imwrite("save_output_images/edge_detection_amsterdam_dark.JPG", highlighted)