
import numpy as np
import detection.pain_hogs as pain_hogs
import joblib

class Pain:
    def __init__(self):
         print("[ModelLoader] Loading pain model...")
         self.pain_model = joblib.load("models/pain/svm_models_new.pkl")

    def get_pain(self, original_img, landmarks, regions ):
            # Pain model
            
            if landmarks is None :
                return [0]
        
            hogs, geo = pain_hogs.get_hogs(original_img, regions, landmarks, rotate=False)
            rotated_hogs, rotated_geo = pain_hogs.get_hogs(original_img, regions, landmarks, rotate=True)
        
            geo = np.nan_to_num(geo, nan=0.0, posinf=100.0, neginf=-100.0)
            rotated_geo = np.nan_to_num(rotated_geo, nan=0.0, posinf=1000.0, neginf=-100.0)
          

            combined = np.concatenate((
                np.concatenate((hogs, geo), axis=1),
                np.concatenate((rotated_hogs, rotated_geo), axis=1)
            ), axis=0)

            predictions = []
           
            for image_hog in combined:
                image_hog = image_hog.astype(np.float32)
                if not np.all(np.isfinite(image_hog)):
                    continue
                prob = self.pain_model[2].predict_proba(image_hog.reshape(1, -1))[0][1]  # probability of class 1
                predictions.append(prob * 100)  # scale to percentage

            face_predictions = []

            for i in range(0, len(predictions), 2):
                pair = predictions[i:i+2]
                avg = float(np.mean(pair)) if pair else 0.0
                face_predictions.append(avg)

            return face_predictions
