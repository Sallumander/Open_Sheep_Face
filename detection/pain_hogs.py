import cv2
import numpy as np
from skimage.feature import hog
import math

def extract_hog_features(image, regions,rotate=False, max_length=7488):
    hog_features = []
    for region in regions:
        
        x1, y1, x2, y2 = region
        x1, x2=x1*image.shape[1], x2*image.shape[1]
        y1, y2=y1*image.shape[0], y2*image.shape[0]
        
        x1, x2 = np.clip([x1, x2], 0, image.shape[1])
        y1, y2 = np.clip([y1, y2], 0, image.shape[0])
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        if x1>=x2 :
            x1, x2=x2, x1
        if y1>=y2 :
            y1, y2=y2, y1
        
        crop_img = image[y1:y2, x1:x2]
        crop_img = cv2.resize(crop_img, (100, 100))
        if rotate:
            crop_img=cv2.flip(crop_img,1)
            
        hog_feature, __= hog(crop_img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                                  visualize=True, channel_axis=-1)
        
        if len(hog_feature) > max_length:
            hog_feature = hog_feature[:max_length]
        else:
            hog_feature = np.pad(hog_feature, (0, max_length - len(hog_feature)), 'constant')

        hog_features.append(hog_feature)
  

    return np.concatenate(hog_features)
def extract_geometric_features(landmarks):

    left_ear=landmarks["left_ear"]
    right_ear=landmarks["right_ear"]
    eyes =  landmarks["eyes"]
    nose_mouth = landmarks["nose_mouth"]
    forehead = landmarks["forehead"]
    ears=np.concatenate([left_ear,right_ear])
    # Calculate distances and angles
    left_ear_angle ,right_ear_angle,ear_distance= calc_angles(ears)
    eye_distance = np.linalg.norm(eyes[0] - eyes[1])
    nose_mouth_distance = np.linalg.norm(nose_mouth[0] - nose_mouth[1])
    forehead_distance = np.linalg.norm(forehead[0] - forehead[2])

    return np.array([ear_distance, left_ear_angle,right_ear_angle, eye_distance, nose_mouth_distance, forehead_distance])

def calc_angles(npy_data):
    # angles between tip and root of ears, distance between ear roots
    
    angles = [math.degrees(math.atan2(npy_data[0][1] - npy_data[1][1], npy_data[0][0] - npy_data[1][0])),
              math.degrees(math.atan2(npy_data[5][1] - npy_data[4][1], npy_data[5][0] - npy_data[4][0])),
              math.dist(npy_data[1], npy_data[4])]

    return np.array(angles)
    
def extract_features(image, landmarks,regions, rotate):
    hog_features = extract_hog_features(image, regions, rotate)
    geometric_features = extract_geometric_features(landmarks)
    return hog_features, geometric_features


def get_hogs(image, region_list, landmarks_list,rotate=False):

    hog_features_list = []
    geometric_features_list = []
    for regions, landmarks in zip(region_list, landmarks_list):
       
        hog_features, geometric_features = extract_features(image, landmarks,regions, rotate)
        hog_features_list.append(hog_features)
        geometric_features_list.append(geometric_features)

    hog_features_list = np.array(hog_features_list)
    geometric_features_list = np.array(geometric_features_list)
    return hog_features_list, geometric_features_list

