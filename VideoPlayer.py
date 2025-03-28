from PyQt5.QtWidgets import QProgressBar, QWidget, QSizePolicy, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QSlider, QFileDialog, QMessageBox
from PyQt5.QtCore import QThreadPool, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont
import cv2
import numpy as np
import threading
import csv
import os
from VideoProcessing import DetectionWorker

  

class VideoPlayerWidget(QWidget):
    def __init__(self, detector, pain, video=None):
        super().__init__()
        self.setWindowTitle("Video Review Player")
        self.setMinimumSize(800, 600)

        self.video_path = video
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.frame_index = 0
        self.total_frames = 0
        self.video_finished = False

        self.overlay_cache = {}
        self.latest_detection = None

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Processing: %p%")
        self.progress_bar.setStyleSheet("height: 20px;")

        # Video label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background-color: black;")

        # Slider
        self.slider = DetectionSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.seek)

        # Buttons
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)

        self.export_btn = QPushButton("Download Pain Scores")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_to_csv)
        
        self.download_btn = QPushButton("Download Annotated Video")
        self.download_btn.setEnabled(False)
        self.download_btn.clicked.connect(self.export_annotated_video)

        # Help button
        self.help_btn = QPushButton("?")
        self.help_btn.setFixedSize(80, 80)  # Slightly larger button
        self.help_btn.setToolTip("Click for help")
        self.help_btn.clicked.connect(self.show_help_message)

        # Layouts
        layout = QVBoxLayout(self)

        # Top layout with progress bar and help button
        top_layout = QHBoxLayout()
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.progress_bar.setMinimumHeight(25)
        top_layout.addWidget(self.progress_bar, stretch=5)
        top_layout.addWidget(self.help_btn, stretch=1)

        layout.addLayout(top_layout)
        layout.addWidget(self.video_label)
        layout.addWidget(self.slider)

        # Button layout
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.play_btn)
        btn_layout.addWidget(self.export_btn)
        btn_layout.addWidget(self.download_btn)
        layout.addLayout(btn_layout)

        self.rendered_frame_cache = {}
        self.thread_pool = QThreadPool()
        self.detector = detector
        self.pain = pain


        self.load_video()

    def load_video(self):
        self.initializing_video = True

        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.resizeEvent(None)
        
        self.slider.setMaximum(self.total_frames - 1)
        self.frame_index = 0
        self.video_finished = False
        self.overlay_cache.clear()
        self.latest_detection = None
        self.slider.detected_frames.clear()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Reset the progress bar
        self.progress_bar.setValue(0)

        # Display a "Processing video..." message
        self.video_label.setText("Processing video...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_interval = int(1000 / self.fps)
        self.skip_val = int(self.fps / 4) 
        self.expected_detections = max(1, self.total_frames // max(1, self.skip_val))

        self.detection_frames_processed = 0
         # Start background detection
        self.start_background_detection()

        QTimer.singleShot(10000, self.finish_loading_video)  
      
        

       

    def finish_loading_video(self):
        # Clear the "Processing video..." message and enable the play button
        self.video_label.clear()
        self.play_btn.setEnabled(True)

        self.initializing_video = False

        # Display the first frame of the video
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame, self.frame_index)

        self.resizeEvent(None)

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            if not self.video_finished:
                self.timer.start(self.frame_interval)
                self.play_btn.setText("Pause")

    def next_frame(self):
        if self.cap is None or self.video_finished:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.video_finished = True
            return

        self.frame_index += 1
        self.slider.blockSignals(True)
        self.slider.setValue(self.frame_index)
        self.slider.blockSignals(False)

        self.display_frame(frame, self.frame_index)

    def seek(self, value):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, value)
            self.frame_index = int(value)  # ensure it's an int
            self.video_finished = False
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame, self.frame_index)


    def display_frame(self, frame, frame_number):

        if getattr(self, "last_drawn_frame_index", -1) == frame_number or getattr(self, "initializing_video", False):
            return
        self.last_drawn_frame_index = frame_number

        # Try using cached frame if available
        if frame_number in self.rendered_frame_cache:
            pixmap = self.rendered_frame_cache[frame_number]
        else:
            # Fallback to latest detection if this frame has no detection yet
            pixmap = self.render_pixmap_with_overlay(frame, frame_number, use_latest_if_missing=True)
            self.rendered_frame_cache[frame_number] = pixmap  # ✅ cache the rendered pixmap

        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)


    def handle_detection_result(self, frame_number, pain_scores, boxes, frame):

        self.detection_frames_processed += 1
        progress = int((self.detection_frames_processed / self.expected_detections) * 100)
        self.progress_bar.setValue(progress)

        if self.detection_frames_processed >= self.expected_detections * 0.90:
            self.download_btn.setEnabled(True)
            self.download_btn.setStyleSheet("background-color: #4CAF50; color: white;")
            self.download_btn.setToolTip("Download the fully annotated video.")
            self.export_btn.setEnabled(True)
            self.export_btn.setStyleSheet("background-color: #4CAF50; color: white;")
            self.export_btn.setToolTip("Export pain scores to CSV.")



        if boxes is None or len(boxes) == 0 or pain_scores is None or len(pain_scores) != len(boxes):
            return

        self.overlay_cache[frame_number] = (pain_scores, boxes)
        self.latest_detection = (pain_scores, boxes)
        self.last_detection_frame = frame_number

        pixmap = self.render_pixmap_with_overlay(frame, frame_number)
        self.rendered_frame_cache[frame_number] = pixmap

        self.slider.detected_frames.add(frame_number)
        self.slider.update()

    def render_pixmap_with_overlay(self, frame, frame_number, use_latest_if_missing=False):
        scale_factor = 0.5
        resized_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Default to no overlays
        pain_scores, boxes = None, None

        sorted_detected_frames = sorted(self.overlay_cache.keys())
        previous_detected_frame = None

        # Find the last processed frame before or at the current frame
        for detected_frame in sorted_detected_frames:
            if detected_frame <= frame_number:
                previous_detected_frame = detected_frame
            else:
                break

        # Use the bounding box from the last processed frame if it exists and is within skip_val
        if previous_detected_frame is not None:
            if frame_number - previous_detected_frame <= self.skip_val:
                previous_data = self.overlay_cache.get(previous_detected_frame, ([], []))
                prev_has_faces = len(previous_data[0]) > 0 and len(previous_data[1]) > 0
                if prev_has_faces:
                    pain_scores, boxes = previous_data

        # Draw bounding boxes if available
        if boxes is not None and pain_scores is not None and len(boxes) == len(pain_scores):
            painter = QPainter(pixmap)
            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)
            font = QFont()
            font.setPointSize(8)
            painter.setFont(font)

            for score, box in zip(pain_scores, boxes):
                scaled_box = np.array(box) * scale_factor
                x1, y1, x2, y2 = map(int, scaled_box.flatten())
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                label = f"Pain Probability: {score:.1f}%"
                text_x = int((x1 + x2) / 3)
                text_y = y1 + 25
                painter.drawText(text_x, text_y, label)

            painter.end()

        return pixmap

    def start_background_detection(self):
        BATCH_SIZE = 4
        frame_batch = []

        def enqueue_frames():
            cap = cv2.VideoCapture(self.video_path)
            frame_num = 0
            frame_batch = []

            while frame_num < self.total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % self.skip_val == 0:
                    frame_batch.append((frame_num, frame.copy()))

                    if len(frame_batch) >= BATCH_SIZE:  # BATCH_SIZE
                        worker = DetectionWorker(frame_batch.copy(), self.detector, self.pain)
                        worker.signals.result_ready.connect(self.handle_detection_result)
                        self.thread_pool.start(worker)
                        frame_batch.clear()

                frame_num += 1

            # Handle any remaining frames in the batch
            if frame_batch:
                worker = DetectionWorker(frame_batch.copy(), self.detector, self.pain)
                worker.signals.result_ready.connect(self.handle_detection_result)
                self.thread_pool.start(worker)

            cap.release()

        threading.Thread(target=enqueue_frames, daemon=True).start()



    def export_to_csv(self):
        if not self.overlay_cache:
            QMessageBox.information(self, "Export", "No detection data to export.")
            return

        vpath = self.video_path
        base_name = os.path.splitext(os.path.basename(vpath))[0]  # Extract file name without extension
        default_name = f"{base_name}_output.csv"
        save_path, _ = QFileDialog.getSaveFileName(self, "Save CSV Date", default_name, "CSV Files (*.csv)")
        if not save_path:
            return
        
        with open(save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            writer.writerow(["Timestamp (s)", "Face Index", "Pain Score (%)"])
            for frame_num, (pain_scores, boxes) in sorted(self.overlay_cache.items()):
                timestamp = frame_num / fps
                for i, score in enumerate(pain_scores):
                    writer.writerow([f"{timestamp:.2f}", i+1, f"{score:.1f}"])

        wrapped_path = '\n'.join([save_path[i:i+80] for i in range(0, len(save_path), 80)])
        QMessageBox.information(self, "Download Complete", f"Pain scores downloaded to:\n{wrapped_path}")

    def export_annotated_video(self):
        if not self.cap:
            return

        vpath = self.video_path
        base_name = os.path.splitext(os.path.basename(vpath))[0]  # Extract file name without extension
        default_name = f"{base_name}_annotated.mp4"

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Annotated Video", default_name, "Videos (*.mp4 *.avi)")
        if not save_path:
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        expected_detections = max(1, self.total_frames // max(1, self.skip_val))
    

        # Get frame size from any cached pixmap
        sample_pixmap = next(iter(self.rendered_frame_cache.values()), None)
        if sample_pixmap is None:
            QMessageBox.warning(self, "Error", "No cached frames available for export.")
            return

        width = sample_pixmap.width()
        height = sample_pixmap.height()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        cap = cv2.VideoCapture(self.video_path)
        frame_idx = 0

        while frame_idx < self.total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Use cached pixmap if available
            if frame_idx in self.rendered_frame_cache:
                pixmap = self.rendered_frame_cache[frame_idx]
            else:
                # fallback: render new one if not cached (optional)
                pixmap = self.render_pixmap_with_overlay(frame, frame_idx, use_latest_if_missing=True)

            # Convert pixmap to cv2 image
            qimg = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
            ptr = qimg.bits()
            ptr.setsize(qimg.byteCount())
            img = np.array(ptr).reshape((qimg.height(), qimg.width(), 4))
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            writer.write(bgr_img)

            frame_idx += 1

        cap.release()
        writer.release()

        wrapped_path = '\n'.join([save_path[i:i+80] for i in range(0, len(save_path), 80)])
        QMessageBox.information(self, "Export Complete", f"Annotated video saved to:\n{wrapped_path}")

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if self.initializing_video:
           
            return

        # Try to redraw the current cached frame if it exists
        if self.frame_index in self.rendered_frame_cache:
            pixmap = self.rendered_frame_cache[self.frame_index]
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
        else:
            # Fallback: try to read and show the raw frame at current index
            if self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
                ret, frame = self.cap.read()
                if ret:
                    self.display_frame(frame, self.frame_index)


    def show_help_message(self):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Help")
        msg_box.setText(
            "Video will process while playing so during the first playthrough all faces may not be detected.\n\n "
            "Red ticks along the slider indicate frames with detected faces.\n\n"
            "Once the video is finished processing, you can:\n\n"
            "- Download the fully annotated video by clicking the 'Download Annotated Video' button.\n"
            "- Download the pain scores to a CSV file for further analysis of pain estimations by clicking the 'Download Pain Scores' button.\n\n"
            "Make sure the progress bar reaches 100% before attempting to download or export."
        )
        msg_box.setStyleSheet("QLabel{font-size: 24px;}")  # Adjust size and font
        msg_box.exec_()
        

class DetectionSlider(QSlider):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.detected_frames = set()

    def paintEvent(self, event):
        super().paintEvent(event)

        if not self.detected_frames or self.maximum() == 0:
            return

        painter = QPainter(self)
        pen = QPen(QColor("red"))

        # Dynamically adjust the tick width based on the slider's width
        tick_width = max(1, self.width() // 300)  # Minimum width of 2, scales with slider width
        pen.setWidth(tick_width)
        painter.setPen(pen)

        w = self.width()
        h = self.height()

        for frame in self.detected_frames:
            x = int((frame / self.maximum()) * w)
            painter.drawLine(x, h - 10, x, h)
