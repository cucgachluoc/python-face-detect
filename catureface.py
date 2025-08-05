
import sys
import cv2
from facenet_pytorch import MTCNN
import torch
from datetime import datetime
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QWidget, QPushButton, QInputDialog, QMessageBox)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Sử dụng thiết bị: {device}")

IMG_PATH_BASE = './data/test_images/' 

class VideoThread(QThread):
  
    change_pixmap_signal = pyqtSignal(QImage)
    
    finished_signal = pyqtSignal(str)
    
    update_count_signal = pyqtSignal(int)

    def __init__(self, usr_name, parent=None):
        super().__init__(parent)
        self.usr_name = usr_name
        self.running = True
        self.count = 150 
        self.mtcnn = MTCNN(margin=20, keep_all=False, select_largest=True, post_process=False, device=device)
        self.USR_PATH = os.path.join(IMG_PATH_BASE, self.usr_name)
        if not os.path.exists(self.USR_PATH):
            try:
                os.makedirs(self.USR_PATH)
            except OSError as e:
                self.finished_signal.emit(f"Lỗi tạo thư mục: {e}")
                self.running = False
                return
        self.leap = 1 

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.finished_signal.emit("Lỗi: Không thể mở camera.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.running and self.count > 0 and cap.isOpened():
            isSuccess, frame = cap.read()
            if isSuccess:
               
                if self.leap % 2 == 0: 
                    boxes, _ = self.mtcnn.detect(frame)
                    if boxes is not None:
                        try:
                           
                            temp_frame_for_mtcnn = frame.copy() 
                            face_img_tensor = self.mtcnn(temp_frame_for_mtcnn) 

                            if face_img_tensor is not None: 
                                
                                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
                                path = os.path.join(self.USR_PATH, f'{timestamp}_{self.count}.jpg')

                               
                                mtcnn_save_path_frame = frame.copy()
                                self.mtcnn(mtcnn_save_path_frame, save_path=path) 

                                self.count -= 1
                                self.update_count_signal.emit(self.count)
                                print(f"Đã lưu: {path}, còn lại: {self.count}")
                        except Exception as e:
                            print(f"Lỗi khi lưu ảnh: {e}")


                self.leap += 1

               
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                p = convert_to_qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                self.change_pixmap_signal.emit(p)
            else:
                self.finished_signal.emit("Lỗi: Không thể đọc khung hình từ camera.")
                self.running = False 
                break

           

        cap.release()
        if self.count == 0:
            self.finished_signal.emit(f"Hoàn tất! Đã lưu")
        elif not self.running: 
             self.finished_signal.emit("Đã dừng chụp ảnh.")
        

    def stop(self):
        self.running = False
        self.wait() 


class FaceCaptureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chụp Ảnh Khuôn Mặt")
        self.setGeometry(0, 0, 680, 580) 
        self.center_window()

       
        self.usr_name = None
        self.video_thread = None

        
        main_layout = QVBoxLayout()

        
        self.image_label = QLabel("Camera sẽ hiển thị ở đây", self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(640, 480) 
        main_layout.addWidget(self.image_label)

        
        self.start_button = QPushButton("Bắt Đầu Chụp Ảnh", self)
        self.start_button.clicked.connect(self.start_capture)
        main_layout.addWidget(self.start_button)

        
        self.stop_button = QPushButton("Dừng Chụp Ảnh", self)
        self.stop_button.clicked.connect(self.stop_capture)
        self.stop_button.setEnabled(False) 
        main_layout.addWidget(self.stop_button)

        
        self.count_label = QLabel("Số ảnh cần chụp: 150", self)
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.count_label)

        
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def center_window(self):
        
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def start_capture(self):
        if not self.usr_name:
            
            text, ok = QInputDialog.getText(self, "Nhập Tên", "Nhập tên của bạn:")
            if ok and text:
                self.usr_name = text.strip().replace(" ", "_") 
                if not self.usr_name:
                    QMessageBox.warning(self, "Lỗi", "Tên không được để trống.")
                    self.usr_name = None 
                    return
            else:
                QMessageBox.information(self, "Thông báo", "Cần nhập tên để tiếp tục.")
                return 

       
        if not os.path.exists(IMG_PATH_BASE):
            try:
                os.makedirs(IMG_PATH_BASE)
                print(f"Đã tạo thư mục cơ sở: {IMG_PATH_BASE}")
            except OSError as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể tạo thư mục cơ sở {IMG_PATH_BASE}: {e}")
                return

        self.image_label.setText("Đang khởi tạo camera...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.count_label.setText("Số ảnh cần chụp: 150")

        
        self.video_thread = VideoThread(self.usr_name, self)
        self.video_thread.change_pixmap_signal.connect(self.update_image_label)
        self.video_thread.finished_signal.connect(self.capture_finished)
        self.video_thread.update_count_signal.connect(self.update_capture_count)
        self.video_thread.start()

    def stop_capture(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.stop_button.setEnabled(False)
            

    def update_image_label(self, q_image):
        self.image_label.setPixmap(QPixmap.fromImage(q_image))

    def update_capture_count(self, count):
        self.count_label.setText(f"Số ảnh cần chụp: {count}")

    def capture_finished(self, message):
        self.image_label.setText(message)
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)
        self.usr_name = None 
        QMessageBox.information(self, "Thông Báo", message)
        if self.video_thread:
            self.video_thread.quit() 
            self.video_thread.wait()
            self.video_thread = None


    def closeEvent(self, event):
       
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = FaceCaptureApp()
    main_window.show()
    sys.exit(app.exec())