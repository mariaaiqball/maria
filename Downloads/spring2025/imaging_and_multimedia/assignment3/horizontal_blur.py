import cv2
import numpy as np

def blur(path, kernel_size=7):
    img = cv2.imread(path)
    kernel = np.ones((1, kernel_size), np.float32) / kernel_size
    blur = cv2.filter2D(img, -1, kernel)
    return blur

def main():
    # image "100.tif"
    image1_path = "HW2images/100.tif"   
    blur_100 = blur(image1_path, kernel_size=7)
    cv2.imshow("horizontal blur: 100.tif", blur_100)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("save_output_images/horizontal_blur_100.tif", blur_100)
    
    # selfie image
    selfie_path = "HW2images/selfie.png"  
    blur_selfie = blur(selfie_path, kernel_size=7)
    cv2.imshow("horizontal blur: selfie", blur_selfie)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("save_output_images/horizontal_blur_selfie.jpg", blur_selfie)

main()
print('Saved all motion blur images into save_output_images')
