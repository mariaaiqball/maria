import cv2
import numpy as np
import matplotlib.pyplot as plt


def stretch(difference):
    minimum = np.min(difference)
    maximum = np.max(difference)
    if minimum - maximum == 0:
        return difference.astype(np.uint8)
    difference_stretched = ((difference - minimum) / (maximum - minimum)) * 255.0
    return difference_stretched.astype(np.uint8)
    

def binary_mask(difference, p):
    threshold = np.percentile(difference, 100 - p)
    bm = difference > threshold
    return bm, threshold

def apply_mask(I2, mask):
    image_mask = np.zeros_like(I2)
    image_mask[mask] = I2[mask]
    return image_mask
.
def object_detect(I1_path, I2_path, p_values, soccer=True):
    # Read in images
    I1 = cv2.imread(I1_path, cv2.IMREAD_GRAYSCALE)
    I2 = cv2.imread(I2_path, cv2.IMREAD_GRAYSCALE)

    # Part 1
    difference = np.abs(I1.astype(np.float32) - I2.astype(np.float32))    

    # Part 2
    plt.figure()
    plt.hist(difference.ravel(), bins=256, range=(0, 255), color='black')
    plt.title('Histogram of Absolute Difference between Image 1 and 2')
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.show()
    
    # Part 3 --> better visualization
    diff_stretched = stretch(difference)
    plt.figure()
    plt.imshow(diff_stretched, cmap='gray')
    plt.title("Contrast Stretched Difference Image")
    plt.axis('off')
    plt.show()
    
    # Part 4
    masks = {}
    plt.figure(figsize=(12, 8))
    for i, p in enumerate(p_values):
        mask, threshold_value = binary_mask(difference, p)
        masks[p] = mask
        plt.subplot(2, 2, i+1)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Mask = {p}% and threshold = {threshold_value:.2f}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    best_p = 1
    best_mask = masks[best_p]
    
    # Part 5
    im = apply_mask(I2, best_mask)
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.title(f"Moving Pixels using p = {best_p}%")
    plt.axis('off')
    plt.show()


# Soccer images
soccer_image1 = "HW2images/soccer1.bmp"
soccer_image2 = "HW2images/soccer2.bmp"
p_values = [2, 1, 0.1, 0.01]
object_detect(soccer_image1, soccer_image2, p_values, soccer=True)

# Parking lot images 
parking_image1 = "HW2images/1015.jpg"
parking_image2 = "HW2images/1020.jpg"
object_detect(parking_image1, parking_image2, p_values, soccer=False)

