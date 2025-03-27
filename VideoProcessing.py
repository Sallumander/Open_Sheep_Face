
import numpy as np

from PyQt5.QtCore import pyqtSignal, QObject, QRunnable, pyqtSlot

class DetectionWorkerSignals(QObject):
    result_ready = pyqtSignal(int, list, object,object)  # frame_num, pain_score, boxes


class DetectionWorker(QRunnable):
    def __init__(self, frame_batch, detector,pain):
        """
        frame_batch: list of (frame_num, frame)
        """
        super().__init__()
        self.frame_batch = frame_batch
        self.signals = DetectionWorkerSignals()
        self.detector = detector
        self.pain=pain

    def run(self):
        

        frames = [frame for (_, frame) in self.frame_batch]
        frame_nums = [num for (num, _) in self.frame_batch]

        batch_results = self.detector.detect_batch(frames, video_mode=True)


        for i, (img, _, landmarks, regions, boxes, _) in enumerate(batch_results):
         
            if boxes is not None and len(boxes) > 0:
    
                score = self.pain.get_pain(img, landmarks, regions)

            else:
                score = []

            self.signals.result_ready.emit(frame_nums[i], score, boxes, frames[i])


   