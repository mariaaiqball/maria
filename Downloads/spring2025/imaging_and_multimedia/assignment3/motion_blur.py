import cv2
import numpy as np

def blur(a, paths):
    #initialize B = I
    # recursively use formula B = (alpha)(B) + (1-alpha)(I) where I = 0,...T
    B = cv2.imread(paths[0], cv2.IMREAD_UNCHANGED)
    B = B.astype(np.float32)
    for path in paths[1:]:
        I = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        I = I.astype(np.float32)
        B = a * B + (1 - a) * I
    B = np.clip(B, 0, 255).astype(np.uint8)
    return B

def main():
    paths = [f"HW2images/{i}.tif" for i in range(100, 111)]
    alphas = [0.1, 0.2, 0.5, 0.8] #alpha values  
    for a in alphas:
        result = blur(a, paths)
        cv2.imshow(f"motion blur: a = {a}", result)
        cv2.waitKey(0)
        cv2.destroyWindow(f"motion blur: a = {a}")
        cv2.imwrite(f'save_output_images/f"motion_blur_a_{a}.tif"', result)
    cv2.destroyAllWindows()


main()
print('Saved all motion blur images into save_output_images')
