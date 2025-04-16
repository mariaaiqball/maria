# Save final training set
import train as t
import numpy as np
import pickle 
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist

'''
Trains the data using test_data(). Image path is hardcoded, but if needed can be changed. 
Updated train_data() with closing uses the function tests_data_1(). 
'''

def test_data(Labels, normalized_train, mean, std, th, file): 
    normalized_train = np.vstack(normalized_train)
    img = t.read_img(f'images/{file}.bmp')
    features_test, _ , _ , regions = t.bounding_boxes(img, th) 
    features_test = np.array(features_test)
    normalized_test = (features_test - mean) / std  

    dist = cdist(normalized_test, normalized_train)  
    D_index = np.argsort(dist, axis=1)
    print(f'Distance Matrix:\n{D_index}')
    np.savetxt('D_index.txt', D_index, fmt='%d')

    pkl_file = open('test_gt_py3.pkl', 'rb')
    mydict = pickle.load(pkl_file)
    pkl_file.close()
    classes = mydict[b'classes']
    gt_locations = mydict[b'locations']

    ans = []
    for i in range((len(D_index))):
        ans.append(Labels[D_index[i][1]])


    correct = 0
    total_matches = 0
    tolerance = 100
    for i, props in enumerate(regions):
        minr, minc, maxr, maxc = props.bbox
        for j, gt in enumerate(gt_locations):
            gt_r, gt_c = gt
            if (minr - tolerance) <= gt_r <= (maxr + tolerance) and (minc - tolerance) <= gt_c <= (maxc + tolerance):
                total_matches += 1
                if classes[i].lower() == ans[i].lower():
                    correct += 1
                break  
    recognition_rate = (correct/total_matches) * 100 if total_matches > 0 else 0.0
    print("FINAL RESULTS")
    print(f'Recognition rate: {recognition_rate}')
    print(f'Correct Count: {correct}/70')
    print(f'Total Matches: {total_matches}')

def test_data_1(Labels, normalized_train, mean, std, th, dialation, file): 
    normalized_train = np.vstack(normalized_train)
    img = t.read_img(f'images/{file}.bmp')
    features_test, _ , _ , regions = t.bounding_boxes_1(img, th, dialation) 
    features_test = np.array(features_test)
    normalized_test = (features_test - mean) / std  

    dist = cdist(normalized_test, normalized_train)  
    D_index = np.argsort(dist, axis=1)
    print(f'Distance Matrix:\n{D_index}')
    np.savetxt('D_index.txt', D_index, fmt='%d')

    pkl_file = open('test_gt_py3.pkl', 'rb')
    mydict = pickle.load(pkl_file)
    pkl_file.close()
    classes = mydict[b'classes']
    gt_locations = mydict[b'locations']

    ans = []
    for i in range((len(D_index))):
        ans.append(Labels[D_index[i][1]])


    correct = 0
    total_matches = 0
    tolerance = 100
    for i, props in enumerate(regions):
        minr, minc, maxr, maxc = props.bbox
        for j, gt in enumerate(gt_locations):
            gt_r, gt_c = gt
            if (minr - tolerance) <= gt_r <= (maxr + tolerance) and (minc - tolerance) <= gt_c <= (maxc + tolerance):
                total_matches += 1
                if classes[i].lower() == ans[i].lower():
                    correct += 1
                break  
    recognition_rate = (correct/total_matches) * 100 if total_matches > 0 else 0.0
    print("FINAL RESULTS")
    print(f'Recognition rate: {recognition_rate}/70')
    print(f'Correct Count: {correct}/70')
    print(f'Total Matches: {total_matches}')



