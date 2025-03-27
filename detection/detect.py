import cv2
import torch
import numpy as np
from models.define_models import ModelLoader

class FaceLandmarkDetector:
    def __init__(self):
        loader = ModelLoader()
        self.device = loader.get_device()
        self.yolo_model = loader.get_yolo()
        self.region_detector = loader.get_region_detector()
        self.landmark_models = loader.get_landmark_models()
        self.original_img = None
        self.img_gray = None
        self.is_cuda = self.device.type == "cuda"

        
      
    def get_boxes(self, img):
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_process = np.stack([self.img_gray] * 3, axis=-1)
        
        results = self.yolo_model(img_process, imgsz=640, conf=0.5,verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        return boxes

    def detect_faces_and_landmarks(self, img, video_mode=False):

        if img.dtype == np.float16:
            img = (img * 255).astype(np.uint8)
        self.original_img = img.copy()


        # Faster YOLO inference
        boxes = self.get_boxes(img)
        
        if not boxes.size:
            return self.original_img, self.original_img, [], [], [],[]
        
        all_pred_landmarks = []
        all_pred_regions = []
        closeups = []

        for box in boxes:
            pred_landmarks, pred_regions, face_closeup = self.process_face(
                box, img, self.img_gray,video_mode=video_mode
            )

            if pred_landmarks is not None:
                all_pred_landmarks.append(pred_landmarks)
                all_pred_regions.append(pred_regions)
                if not video_mode:
                    closeups.append(face_closeup)

        return img, self.original_img, all_pred_landmarks, all_pred_regions,boxes, closeups

    def process_face(self,box, img,img_gray, video_mode=False):
        x1, y1, x2, y2 = map(int, box[:4])
        x1, y1 = max(0, x1 - 10), max(0, y1 - 10)
        x2, y2 = min(img_gray.shape[1], x2 + 10), min(img_gray.shape[0], y2 + 10)

        
        face_crop = img_gray[y1:y2, x1:x2]  # BGR cropsss
        if face_crop.size == 0:
            return None, None, None

        face_crop_resized = cv2.resize(face_crop, (224, 224))
      

        face_tensor = torch.tensor(face_crop_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        face_tensor = face_tensor.repeat(1, 3, 1, 1)  

        if self.is_cuda:
            face_tensor = face_tensor.half()

        with torch.no_grad():
            pred_regions = self.region_detector(face_tensor).cpu().numpy().reshape(-1, 4)

        if pred_regions[0, [2]].max() > pred_regions[2, [2]].max():
            pred_regions[[0, 2]] = pred_regions[[2, 0]]

        # Scale regions to 224x224
        scaled_regions = pred_regions.copy()
        scaled_regions[:, [0, 2]] *= 224
        scaled_regions[:, [1, 3]] *= 224
        scaled_regions[:, [0, 1]] = np.clip(scaled_regions[:, [0, 1]] - 10, 0, 224)
        scaled_regions[:, [2, 3]] = np.clip(scaled_regions[:, [2, 3]] + 10, 0, 224)

        pred_landmarks = {}

        if not video_mode:
            face_disp = self.original_img[y1:y2, x1:x2]  # RGB crop
            face_disp_resized = cv2.resize(face_disp, (224, 224))

        for i, (name, model) in enumerate(self.landmark_models.items()):
            lx1, ly1, lx2, ly2 = scaled_regions[i].astype(int)
            region_crop = face_crop_resized[ly1:ly2, lx1:lx2]

            if region_crop.size == 0:
                continue

            region_resized = cv2.resize(region_crop, (112, 112))
            region_tensor = torch.from_numpy(region_resized).float().unsqueeze(0).unsqueeze(0).to(self.device)

            if self.is_cuda:
                region_tensor = region_tensor.half()

            region_tensor = region_tensor.repeat(1, 3, 1, 1) / 255.0
           
            with torch.no_grad():
                pred = model(region_tensor).cpu().numpy().reshape(-1, 2)
   
            pred *= 112

            width_region = lx2 - lx1
            height_region = ly2 - ly1

            # Scale to 224 face space
            pred[:, 0] = pred[:, 0] * (width_region / 112) + lx1
            pred[:, 1] = pred[:, 1] * (height_region / 112) + ly1

            # Draw landmarks on close-up
            if not video_mode:
                    for (px, py) in pred:
                        cv2.circle(face_disp_resized, (int(px), int(py)), 2, (0, 255, 0), -1)

            # Scale to original image space
            face_width = x2 - x1
            face_height = y2 - y1
  
            pred[:, 0] = pred[:, 0] * (face_width / 224) + x1
            pred[:, 1] = pred[:, 1] * (face_height / 224) + y1

            pred_landmarks[name] = pred

            # Draw landmarks on full image
            if not video_mode:
                for (x, y) in pred:
                     cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

        if video_mode:
            return pred_landmarks, pred_regions, None   

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return pred_landmarks, pred_regions, face_disp_resized

    def detect_batch(self, images, video_mode=False):
        
        batch_size = len(images)
        processed_results = []

        inputs = []
        grays = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            inputs.append(np.stack([gray] * 3, axis=-1))
            grays.append(gray)

        results = self.yolo_model(inputs, imgsz=640, conf=0.5, verbose=False)  # ← use original images here

        for idx, result in enumerate(results):
          
            orig_img = images[idx].copy()  
            self.original_img = images[idx] 
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else np.array([])
            if boxes.size == 0:
                processed_results.append((orig_img, images[idx], [], [], [], []))
                continue

            landmarks_list = []
            regions_list = []
            closeups_list = []
            for box in boxes:
                pred_landmarks, pred_regions, face_closeup = self.process_face(
                    box, orig_img, grays[idx], True
                )


                if pred_landmarks is not None:
                    landmarks_list.append(pred_landmarks)
                    regions_list.append(pred_regions)
                    if not video_mode:
                        closeups_list.append(face_closeup)


            processed_results.append((images[idx], images[idx], landmarks_list, regions_list, boxes, closeups_list))

        return processed_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='test.jpg', help='source image path')
    parser.add_argument('--video', action='store_true', help='run in video mode')
    opt = parser.parse_args()
    detect=FaceLandmarkDetector()
    img = cv2.imread(opt.source)
    result_img, _, _, _, close = detect.detect_faces_and_landmarks(img, video_mode=opt.video)
    cv2.imshow("Landmark Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("Closeup", close[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
