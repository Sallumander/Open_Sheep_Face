import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, QFrame, QPushButton, QMessageBox, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from detection.detect import FaceLandmarkDetector
from detection.pain import Pain
from VideoPlayer import VideoPlayerWidget

class LandmarkDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
       
        self.setWindowTitle('Face, Landmark, and Pain Detection')
        self.setGeometry(100, 100, 1500, 900)
        self.setMinimumSize(800, 500)
      
        self._background_cv_img = cv2.imread("background.jpg")
        self.background_label = QLabel(self)
        self.background_label.setScaledContents(True)
        self.background_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.background_label.lower() 
        self.display_background_image() 

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        self.instruction_label = QLabel("Please upload an image or video.")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setStyleSheet("font-size: 60px; color: black; font-weight: bold;")
        self.main_layout.addWidget(self.instruction_label)

        self.image_display_layout = QHBoxLayout()
        self.main_layout.addLayout(self.image_display_layout)

        self.full_image_frame = QFrame()
        self.full_image_frame.setFrameShape(QFrame.StyledPanel)
        self.full_image_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.full_image_layout = QVBoxLayout(self.full_image_frame)

        self.full_image_label_qt = QLabel()
        self.full_image_label_qt.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.full_image_label_qt.setAlignment(Qt.AlignCenter)
        self.full_image_layout.addWidget(self.full_image_label_qt)

        self.image_display_layout.addWidget(self.full_image_frame, stretch=10)

        self.closeup_frame = QFrame()
        self.closeup_frame.setFrameShape(QFrame.StyledPanel)
        self.closeup_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.closeup_frame.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.closeup_layout = QVBoxLayout(self.closeup_frame)

        self.closeup_label = QLabel()
        self.closeup_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.closeup_label.setAlignment(Qt.AlignCenter)
        self.closeup_layout.addWidget(self.closeup_label)

        self.pain_probability_label = QLabel()
        self.pain_probability_label.setAlignment(Qt.AlignCenter)
        self.pain_probability_label.setVisible(False)
        self.closeup_layout.addWidget(self.pain_probability_label)

        self.arrow_layout = QHBoxLayout()
        self.left_arrow_button = QPushButton("←")
        self.left_arrow_button.clicked.connect(self.previous_closeup)
        self.left_arrow_button.setVisible(False)
        self.left_arrow_button.setStyleSheet("background-color: rgba(255, 255, 255, 255);")
        self.arrow_layout.addWidget(self.left_arrow_button)

        self.right_arrow_button = QPushButton("→")
        self.right_arrow_button.clicked.connect(self.next_closeup)
        self.right_arrow_button.setVisible(False)
        self.right_arrow_button.setStyleSheet("background-color: rgba(255, 255, 255, 255);")
        self.arrow_layout.addWidget(self.right_arrow_button)

        self.closeup_layout.addLayout(self.arrow_layout)
        self.image_display_layout.addWidget(self.closeup_frame, stretch=6)

       

        self.full_image_frame.setVisible(False)
        self.closeup_frame.setVisible(False)

        self.button_layout = QHBoxLayout()
        self.upload_button = QPushButton('Upload Image or Video')
        self.upload_button.setMinimumSize(250, 80)
        self.upload_button.clicked.connect(self.open_file_dialog)
        self.button_layout.addWidget(self.upload_button)

        self.help_button = QPushButton('How to Use')
        self.help_button.setMinimumSize(250, 80)
        self.help_button.clicked.connect(self.show_help_dialog)
        self.button_layout.addWidget(self.help_button)
        
        self.video_player = None
        self.main_layout.addLayout(self.button_layout)

        self.pain_scores = None
        self.detector = FaceLandmarkDetector()
        self.pain=Pain()
        print("Models loaded")

    def show_help_dialog(self):
        instructions = (
            "Instructions:\n\n"
            "1. Click 'Upload Image or Video' to load a file.\n"
            "2. If you upload an image, it will display detections and close-ups.\n"
            "3. If you upload a video, the video player will open for review.\n"
            "4. Use ← / → to cycle through close-ups (images only).\n"
            "5. Pain probability is shown both on the full image and underneath each closeup.\n\n"
            "Supported file types: PNG, JPG, JPEG, MP4, AVI, MOV."
        )

        QMessageBox.information(self, "How to Use", instructions)

    def clear_previous_display(self):
        # Stop previous video if it exists
        if self.video_player:
            self.video_player.setParent(None)
            self.video_player.deleteLater()
            self.video_player = None


        # Remove image
        self.full_image_label_qt.clear()
        self.closeup_label.clear()
        self.pain_scores = None

        # Hide frames
        self.full_image_frame.setVisible(False)
        self.closeup_frame.setVisible(False)

        

    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, 'Open Image or Video', '',
            'Images/Videos (*.png *.jpg *.jpeg *.mp4 *.avi *.mov)'
        )

        if not file_name:
            return

        self.clear_previous_display()
        
        ext = os.path.splitext(file_name)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg']:
            
            self.load_image(file_name)
        elif ext in ['.mp4', '.avi', '.mov']:
            self.load_video(file_name)
        else:
            QMessageBox.warning(self, "Unsupported File", "Unsupported file type.")

    def load_image(self, file_name):
        if self.video_player:
            return
        self.instruction_label.setText("Processing image...")
        self.full_image_frame.setVisible(True)
        self.closeup_frame.setVisible(True)
        self.instruction_label.setVisible(False)

        image = cv2.imread(file_name)
        processed_img, original_img, landmarks, regions, _, closeups = self.detector.detect_faces_and_landmarks(image)

        # Display the full image regardless of detection results
        self.display_full_image(processed_img)

        if len(closeups) == 0:
            # Show a warning dialog box
            QMessageBox.warning(self, "No Faces Detected", "No faces were detected in the image.")
            
            # Update the UI to reflect no detections
            self.pain_probability_label.setText("No faces detected.")
            self.pain_probability_label.setStyleSheet("font-size: 60px; color: black; font-weight: bold;")
            self.pain_probability_label.setVisible(True)
            self.closeup_label.clear()
            self.left_arrow_button.setVisible(False)
            self.right_arrow_button.setVisible(False)
        else:
            
            self.pain_scores = self.pain.get_pain(original_img, landmarks, regions)
           

            # Draw pain score per face
            for score, box in zip(self.pain_scores, _): 
                box = np.array(box).flatten()
                x1, y1, x2, y2 = map(int, box)
                # Ensure the text is within the image boundaries
                text_x = (x1 + x2) // 3
                text_y = y2 + 30  # Default position below the bounding box
                if text_y > processed_img.shape[0]:  # If the text goes outside the image height
                    text_y = y2 - 5  # Move the text above the bounding box
                label = f"{score:.1f}%"
                cv2.putText(processed_img, label, (text_x, text_y),  # below the bounding box
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            self.display_full_image(processed_img)

            self.closeups = closeups
            self.current_closeup_index = 0
            self.instruction_label.setVisible(False)

            self.display_closeup(self.closeups[self.current_closeup_index])
            self.left_arrow_button.setVisible(len(self.closeups) > 1)
            self.right_arrow_button.setVisible(len(self.closeups) > 1)

    def load_video(self, file_name):
        self.instruction_label.setVisible(False)
        self.full_image_frame.setVisible(True)
        self.closeup_frame.setVisible(False)
        self.pain_probability_label.setVisible(False)

        self.video_player = VideoPlayerWidget(self.detector,self.pain,file_name)
        self.full_image_layout.addWidget(self.video_player)
        self.video_player.resize(self.full_image_frame.size())
        self.video_player.updateGeometry()

    def display_background_image(self):
        cv_img=self._background_cv_img
        if cv_img is None:
            print(f"Failed to load background")
            return

        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w = self.size().height(), self.size().width()

        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(w, h, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        self.background_label.setPixmap(pixmap)

        
    def display_full_image(self, cv_img):
        self._last_full_image = cv_img
        h, w = self.full_image_label_qt.height(), self.full_image_label_qt.width()
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.full_image_label_qt.setPixmap(pixmap)

    def display_closeup(self, cv_img):
        self._last_closeup = cv_img
        h, w = self.closeup_label.height(), self.closeup_label.width()
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.closeup_label.setPixmap(pixmap)

        # Update label with the corresponding pain score
        if hasattr(self, "pain_scores") and self.current_closeup_index < len(self.pain_scores):
            score = self.pain_scores[self.current_closeup_index]
            self.pain_probability_label.setText(f"Probability Sheep \n is in Pain: {score:.1f}%")
            self.pain_probability_label.setStyleSheet("font-size: 42px; color: red; font-weight: bold; background-color: white;")
            self.pain_probability_label.setVisible(True)
        else:
            self.pain_probability_label.setText("Pain: N/A")
            self.pain_probability_label.setVisible(True)

    def next_closeup(self):
        if hasattr(self, "closeups") and self.closeups:
            self.current_closeup_index = (self.current_closeup_index + 1) % len(self.closeups)
            self.display_closeup(self.closeups[self.current_closeup_index])

    def previous_closeup(self):
        if hasattr(self, "closeups") and self.closeups:
            self.current_closeup_index = (self.current_closeup_index - 1) % len(self.closeups)
            self.display_closeup(self.closeups[self.current_closeup_index])

    def resizeEvent(self, event):

        if self.video_player:
            self.video_player.resize(self.full_image_frame.size())
        else:
            if self.full_image_label_qt.pixmap():
                self.display_full_image(self._last_full_image)

            if hasattr(self, "closeups") and self.closeups:
                self.display_closeup(self.closeups[self.current_closeup_index])

        self.background_label.resize(self.size())
        self.display_background_image()
        super().resizeEvent(event)


if __name__ == '__main__':
    print("App is starting... please wait.")
    app = QApplication(sys.argv)
    window = LandmarkDetectionApp()
    window.show()
    sys.exit(app.exec_())
