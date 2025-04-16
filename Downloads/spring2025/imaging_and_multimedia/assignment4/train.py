# import all libraries 
import numpy as np
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage.morphology import closing, opening, disk
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import warnings
import os
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
warnings.filterwarnings("ignore")

'''
Trains the data using train_data(). Image path is hardcoded, but if needed can be changed. 
'''


'''READING IMAGES AND BINARIZATION'''
# reading an image file 
def read_img(img): 
    return io.imread(img)

# visualizing an image/matrix
def display(img, title):
    io.imshow(img)
    plt.title(title)
    io.show()

#image histogram 
def display_histogram(img):
    hist = exposure.histogram(img) 
    plt.bar(hist[1], hist[0])
    plt.title('Histogram')
    plt.show()

# Binarization by Thresholding
def binarization(img, th):
    return (img < th).astype(np.double)

'''EXTRACTING CHARACTERS AND THEIR FEATURES'''

# outputs the final image with object detection complete 
# displaying component bounding boxes
def bounding_boxes(img, threshold):
    img_binary = binarization(img, threshold)
    #image morphology
    # mg_binary = closing(img_binary, disk(dialation))
    # get components and level each
    img_label = label(img_binary, background=0)
    regions = regionprops(img_label)
    io.imshow(img_binary)    
    ax = plt.gca()
    Features = [] # Storing Features 
    count = 1
    for props in regions:
        minr, minc, maxr, maxc = props.bbox 
        if (maxr - minr) < 15 or (maxc - minc) < 15 or (maxr - minr) > 200 or (maxc - minc) > 200:
            continue
        # computing hu moments and removing small components 
        roi = img_binary[minr:maxr, minc:maxc]
        m = moments(roi)
        cc = m[0, 1] / m[0, 0]
        cr = m[1, 0] / m[0, 0]
        mu = moments_central(roi, center=(cr, cc))
        nu = moments_normalized(mu)
        hu = moments_hu(nu)
        # Storing features
        Features.append(hu)  
        ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
    
    print(f'Number of Connected Components: {len(Features)}')
    ax.set_title("Bounding Boxes")
    io.show()
    return Features, img_binary, img_label, regions

def normalize_features_list(Features):
    normalized = []
    for f in Features:
        mean = np.mean(f, axis=0)
        std = np.std(f, axis=0)
        normalized_f = (f - mean) / std
        normalized.append(normalized_f)
    return np.array(normalized, dtype=object), mean, std

def train_data(th=205, path='images'): 
    Features = []
    Labels = []
    i = 0
    
    for image in os.listdir(path):
        if image.endswith('.bmp'):
            print(f'Processing: {image}')
            img = read_img(f'images/{image}')
            
            # Get letter
            letter = image.split(".")[0]
            if len(letter) == 1:
                # Extract Hu features
                feat, _, _, _= bounding_boxes(img, th)
                feat = np.array(feat)  # shape: (N, 7)
                Features.append(feat)
                Labels.extend([letter] * len(feat))
                i+=1
            
    print(f'Number of images read: {i}')
    #print(f'Labels: {Labels}')

    # Convert and normalize
    Features = np.vstack(Features)
    mean = np.mean(Features, axis=0)  # shape (7,)
    std = np.std(Features, axis=0)    # shape (7,)
    normalized_features = np.vstack([(f - mean) / std for f in Features])

    # Evaluate training performance using a nearest neighbor approach (ignoring self-match)
    D = cdist(normalized_features, normalized_features)
    predicted_labels = []
    for i in range(D.shape[0]):
        indices = np.argsort(D[i])
        second_index = indices[1]  # First is self (distance zero)
        predicted_labels.append(Labels[second_index])
    
    D_index = np.argsort(D, axis=1)
    #np.savetxt('D_index1.txt', D_index, fmt='%d')
    
    # Compute and print confusion matrix
    confM = confusion_matrix(Labels, predicted_labels)
    #np.savetxt('CM.txt', confM, fmt='%d')

    #print("Confusion Matrix on Training Data:")
    #print(confM)
    return Features, Labels, normalized_features, mean, std

    

def test_image_components(image_path, threshold, dialation): 
    img = read_img(image_path)
    img_binary = binarization(img, threshold)
    #image morphology
    img_binary = closing(img_binary, disk(dialation))
    display(img_binary, "new image binary")
    # get components and level each
    img_label = label(img_binary, background=0)
    regions = regionprops(img_label)
    io.imshow(img_binary)
    ax = plt.gca()
    Features = [] # Storing Features 
    count = 1
    for props in regions:
        minr, minc, maxr, maxc = props.bbox 
        if (maxr - minr) < 10 or (maxc - minc) < 10 or (maxr - minr) > 500 or (maxc - minc) > 500:
            continue
        # computing hu moments and removing small components 
        roi = img_binary[minr:maxr, minc:maxc]
        m = moments(roi)
        cc = m[0, 1] / m[0, 0]
        cr = m[1, 0] / m[0, 0]
        mu = moments_central(roi, center=(cr, cc))
        nu = moments_normalized(mu)
        hu = moments_hu(nu)
        # Storing features
        Features.append(hu)  
        ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
    
    print(f'Number of Connected Components: {len(Features)}')
    ax.set_title("Bounding Boxes")
    io.show()


def bounding_boxes_1(img, threshold, dialation):
    img_binary = binarization(img, threshold)
    #image morphology
    img_binary = closing(img_binary, disk(dialation))
    # get components and level each
    img_label = label(img_binary, background=0)
    regions = regionprops(img_label)
    io.imshow(img_binary)    
    ax = plt.gca()
    Features = [] # Storing Features 
    count = 1
    for props in regions:
        minr, minc, maxr, maxc = props.bbox 
        if (maxr - minr) < 15 or (maxc - minc) < 15 or (maxr - minr) > 200 or (maxc - minc) > 200:
            continue
        # computing hu moments and removing small components 
        roi = img_binary[minr:maxr, minc:maxc]
        m = moments(roi)
        cc = m[0, 1] / m[0, 0]
        cr = m[1, 0] / m[0, 0]
        mu = moments_central(roi, center=(cr, cc))
        nu = moments_normalized(mu)
        hu = moments_hu(nu)
        # Storing features
        Features.append(hu)  
        ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
    
    print(f'Number of Connected Components: {len(Features)}')
    ax.set_title("Bounding Boxes")
    io.show()
    return Features, img_binary, img_label, regions

def train_data_1(th=205, disks=6, path='images'): 
    Features = []
    Labels = []
    i = 0
    
    for image in os.listdir(path):
        if image.endswith('.bmp'):
            print(f'Processing: {image}')
            img = read_img(f'images/{image}')
            
            # Get letter
            letter = image.split(".")[0]
            if len(letter) == 1:
                # Extract Hu features
                feat, _, _, _= bounding_boxes_1(img, th, disks)
                feat = np.array(feat)  # shape: (N, 7)
                Features.append(feat)
                Labels.extend([letter] * len(feat))
                i+=1
            
    print(f'Number of images read: {i}')
    #print(f'Labels: {Labels}')

    # Convert and normalize
    Features = np.vstack(Features)
    mean = np.mean(Features, axis=0)  # shape (7,)
    std = np.std(Features, axis=0)    # shape (7,)
    normalized_features = np.vstack([(f - mean) / std for f in Features])

    # Evaluate training performance using a nearest neighbor approach (ignoring self-match)
    D = cdist(normalized_features, normalized_features)
    predicted_labels = []
    for i in range(D.shape[0]):
        indices = np.argsort(D[i])
        second_index = indices[1]  # First is self (distance zero)
        predicted_labels.append(Labels[second_index])
    
    D_index = np.argsort(D, axis=1)
    #np.savetxt('D_index1.txt', D_index, fmt='%d')
    
    # Compute and print confusion matrix
    confM = confusion_matrix(Labels, predicted_labels)
    #np.savetxt('CM.txt', confM, fmt='%d')

    #print("Confusion Matrix on Training Data:")
    #print(confM)
    return Features, Labels, normalized_features, mean, std