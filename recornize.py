
import sys
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import time
import os
import csv
from datetime import datetime

from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QMessageBox, QFileDialog)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal


FRAME_SIZE = (640, 480)
DATA_PATH = './data'
ATTENDANCE_FILE = 'attendance.csv'

RECOGNITION_THRESHOLD = 1.3
POWER_FACTOR = pow(10, 6) 

try:
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
except Exception as e:
    print(f"Lỗi khi kiểm tra CUDA, sử dụng CPU: {e}")
    DEVICE = torch.device('cpu')
print(f"Sử dụng thiết bị: {DEVICE}")

def trans(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    return transform(img)

def load_faceslist(data_path, device):
    try:
        if device.type == 'cpu':
            embeds_path = os.path.join(data_path, 'faceslistCPU.pth')
            if not os.path.exists(embeds_path):
                 embeds_path_gpu = os.path.join(data_path, 'faceslist.pth')
                 if os.path.exists(embeds_path_gpu):
                      print("Không tìm thấy faceslistCPU.pth'")
                      embeds = torch.load(embeds_path_gpu, map_location='cpu')
                 else:
                      raise FileNotFoundError(f"Không tìm thấy tệp embedding: {embeds_path} hoặc {embeds_path_gpu}")
            else:
                 embeds = torch.load(embeds_path, map_location='cpu')

        else:
            embeds_path = os.path.join(data_path, 'faceslist.pth')
            if not os.path.exists(embeds_path):
                embeds_path_cpu = os.path.join(data_path, 'faceslistCPU.pth')
                if os.path.exists(embeds_path_cpu):
                    print("Không tìm thấy faceslist.pth")
                    embeds = torch.load(embeds_path_cpu).to(device) 
                else:
                    raise FileNotFoundError(f"Không tìm thấy tệp embedding: {embeds_path} hoặc {embeds_path_cpu}")
            else:
                embeds = torch.load(embeds_path).to(device)

        names_path = os.path.join(data_path, 'usernames.npy')
        if not os.path.exists(names_path):
            raise FileNotFoundError(f"Không tìm thấy tệp usernames: {names_path}")
        names = np.load(names_path)
        print(f"Đã tải {len(names)} tên và embeddings tương ứng.")
        return embeds, names
    except FileNotFoundError as e:
        print(f"Lỗi tải dữ liệu khuôn mặt: {e}")
        return None, None
    except Exception as e:
        print(f"Lỗi không xác định khi tải dữ liệu khuôn mặt: {e}")
        return None, None


def inference(model, face_tensor, local_embeds, device, threshold=RECOGNITION_THRESHOLD, power_factor=POWER_FACTOR):
    if local_embeds is None or local_embeds.nelement() == 0:
        return -1,

    model.eval() 
    with torch.no_grad():
        face_embedding = model(face_tensor.to(device).unsqueeze(0))

    norm_diff = face_embedding.unsqueeze(-1) - torch.transpose(local_embeds.to(device), 0, 1).unsqueeze(0)
    
    norm_score = torch.sum(torch.pow(norm_diff, 2), dim=1)

    min_dist_raw, embed_idx = torch.min(norm_score, dim=1)
    min_dist_scaled = min_dist_raw.item() * power_factor 

    

    if min_dist_raw.item() > threshold:

        return -1, min_dist_raw.item() 
    else:
        return embed_idx.item(), min_dist_raw.item() 

def extract_face(box, img, margin=20, face_size=160):

    img_size = img.shape[1], img.shape[0]


    box_width = box[2] - box[0]
    box_height = box[3] - box[1]

    if box_width <= 0 or box_height <= 0:
        return None 

    margin_x = int(box_width * (margin / 100.0))
    margin_y = int(box_height * (margin / 100.0))



    x1 = int(max(box[0] - margin_x, 0))
    y1 = int(max(box[1] - margin_y, 0))
    x2 = int(min(box[2] + margin_x, img_size[0]))
    y2 = int(min(box[3] + margin_y, img_size[1]))


    cropped_img = img[y1:y2, x1:x2]

    if cropped_img.size == 0: 
        return None


    resized_face = cv2.resize(cropped_img, (face_size, face_size), interpolation=cv2.INTER_AREA)

    pil_face = Image.fromarray(cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB))
    return pil_face


class RecognitionThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    recognition_update_signal = pyqtSignal(list)
    status_update_signal = pyqtSignal(str) 

    def __init__(self, data_path, device):
        super().__init__()
        self.data_path = data_path
        self.device = device
        self.running = False
        self.mtcnn = None
        self.model = None
        self.embeddings = None
        self.names = None
        self._recognized_names_in_frame = []

    def load_models_and_data(self):
        try:
            self.mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
                               thresholds=[0.9, 0.9, 0.9], factor=0.709, post_process=True,
                               select_largest=True, 
                               keep_all=False,
                               device=self.device)
            if self.mtcnn is None:
                 self.status_update_signal.emit("Lỗi: Không thể khởi tạo MTCNN.")
                 return False

            self.model = InceptionResnetV1(
                classify=False,
                pretrained="casia-webface" 
            ).to(self.device)
            self.model.eval()
            if self.model is None:
                self.status_update_signal.emit("Lỗi: Không thể khởi tạo InceptionResnetV1.")
                return False

            self.embeddings, self.names = load_faceslist(self.data_path, self.device)
            if self.embeddings is None or self.names is None:
                self.status_update_signal.emit("Lỗi: Không thể tải embeddings hoặc tên người dùng.")
                return False
            self.status_update_signal.emit("Các model và dữ liệu đã được tải thành công.")
            return True
        except Exception as e:
            self.status_update_signal.emit(f"Lỗi nghiêm trọng khi tải model: {e}")
            return False

    def run(self):
        if not self.load_models_and_data():
            self.running = False
            return 

        self.running = True
        cap = cv2.VideoCapture(0) 
        if not cap.isOpened():
            self.status_update_signal.emit("Lỗi: Không thể mở camera.")
            self.running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])

        prev_frame_time = time.time()

        while self.running and cap.isOpened():
            is_success, frame = cap.read()
            if not is_success:
                self.status_update_signal.emit("Lỗi: Mất kết nối với camera.")
                break

            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

           
            boxes, probs = self.mtcnn.detect(pil_image) 

            self._recognized_names_in_frame = [] 

            if boxes is not None:
                for i, box in enumerate(boxes):
                   
                    pil_face = extract_face(box, frame, margin=20, face_size=160)

                    if pil_face is not None:
                        face_tensor = trans(pil_face) 
                        idx, score = inference(self.model, face_tensor, self.embeddings, self.device)

                        name_display = "Unknown"
                        color = (0, 0, 255) 

                        if idx != -1: 
                            name_display = f"{self.names[idx]} "
                            self._recognized_names_in_frame.append(self.names[idx])
                            color = (0, 255, 0)

                     
                        b = [int(bi) for bi in box]
                        frame = cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 2)
                        cv2.putText(frame, name_display, (b[0], b[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
            prev_frame_time = new_frame_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)
            self.status_update_signal.emit(f"FPS: {int(fps)}") 

    
            self.recognition_update_signal.emit(self._recognized_names_in_frame)

        
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            scaled_qt_image = qt_image.scaled(FRAME_SIZE[0], FRAME_SIZE[1], Qt.AspectRatioMode.KeepAspectRatio)
            self.change_pixmap_signal.emit(scaled_qt_image)

            if cv2.waitKey(1) & 0xFF == 27: 
                break
       
        cap.release()
        self.running = False
        self.status_update_signal.emit("Đã dừng nhận diện.")
        

    def stop(self):
        self.running = False
        self.status_update_signal.emit("Đang dừng...")

    def get_recognized_names(self):
        return self._recognized_names_in_frame


class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Điểm Danh Nhận Diện Khuôn Mặt")
        self.setGeometry(0, 0, FRAME_SIZE[0] + 40, FRAME_SIZE[1] + 150) 
        self.center_window()

        self.recognition_thread = None
        self.current_recognized_names = [] 

  
        self.image_label = QLabel("Camera sẽ hiển thị ở đây", self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(FRAME_SIZE[0], FRAME_SIZE[1])
        self.image_label.setStyleSheet("border: 1px solid black;")

        self.status_label = QLabel("Trạng thái: Sẵn sàng", self)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.start_button = QPushButton("Bắt Đầu Nhận Diện", self)
        self.start_button.clicked.connect(self.start_recognition)

        self.stop_button = QPushButton("Dừng Nhận Diện", self)
        self.stop_button.clicked.connect(self.stop_recognition)
        self.stop_button.setEnabled(False)

        self.attendance_button = QPushButton("Điểm Danh", self)
        self.attendance_button.clicked.connect(self.record_attendance)
        self.attendance_button.setEnabled(False)

       
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.status_label)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.attendance_button)
        main_layout.addLayout(button_layout)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        if not os.path.isdir(DATA_PATH):
            QMessageBox.critical(self, "Lỗi Thư Mục",
                                 f"Thư mục '{DATA_PATH}' không tồn tại. Vui lòng tạo và đặt các tệp cần thiết vào đó.")
            self.start_button.setEnabled(False)


    def center_window(self):
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def check_data_files_exist(self):
        cpu_embed_path = os.path.join(DATA_PATH, 'faceslistCPU.pth')
        gpu_embed_path = os.path.join(DATA_PATH, 'faceslist.pth')
        names_path = os.path.join(DATA_PATH, 'usernames.npy')

        if not (os.path.exists(cpu_embed_path) or os.path.exists(gpu_embed_path)):
            QMessageBox.warning(self, "Thiếu Tệp", "Không tìm thấy tệp 'faceslist.pth' hoặc 'faceslistCPU.pth' trong thư mục 'data'.")
            return False
        if not os.path.exists(names_path):
            QMessageBox.warning(self, "Thiếu Tệp", "Không tìm thấy tệp 'usernames.npy' trong thư mục 'data'.")
            return False
        return True

    def start_recognition(self):
        if not self.check_data_files_exist():
            return

        self.image_label.setText("Đang khởi tạo camera và model...")
        self.status_label.setText("Trạng thái: Đang khởi tạo...")
        self.start_button.setEnabled(False)

        self.recognition_thread = RecognitionThread(data_path=DATA_PATH, device=DEVICE)
        self.recognition_thread.change_pixmap_signal.connect(self.update_video_frame)
        self.recognition_thread.recognition_update_signal.connect(self.update_recognized_list)
        self.recognition_thread.status_update_signal.connect(self.update_status_message)
        self.recognition_thread.finished.connect(self.thread_finished) 

        self.recognition_thread.start()
        self.stop_button.setEnabled(True)
        self.attendance_button.setEnabled(True) 

    def stop_recognition(self):
        if self.recognition_thread and self.recognition_thread.isRunning():
            self.status_label.setText("Trạng thái: Đang dừng...")
            self.recognition_thread.stop()
            self.stop_button.setEnabled(False)
            self.attendance_button.setEnabled(False) 

    def thread_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.attendance_button.setEnabled(False)
        self.image_label.setText("Đã dừng. Nhấn 'Bắt Đầu' để chạy lại.")
        if self.recognition_thread:
            self.recognition_thread.quit()
            self.recognition_thread.wait()
        self.recognition_thread = None
        print("Luồng nhận diện đã kết thúc.")


    def update_video_frame(self, q_image):
        if q_image and not q_image.isNull():
            self.image_label.setPixmap(QPixmap.fromImage(q_image))
        else:
            
            self.image_label.clear()
            self.image_label.setText("Camera không hoạt động hoặc đã dừng.")


    def update_recognized_list(self, names_list):
        self.current_recognized_names = names_list
        if names_list:
            self.status_label.setText(f"Trạng thái: Đang nhận diện - Thấy: {', '.join(names_list)}")
        else:
             self.status_label.setText("Trạng thái: Đang nhận diện - Không thấy ai quen.")

    def update_status_message(self, message):
        
        if not self.current_recognized_names or "FPS" in message or "Lỗi" in message or "Đang dừng" in message or "tải thành công" in message:
            
            if not ("FPS" in message and self.current_recognized_names) :
                self.status_label.setText(f"Trạng thái: {message}")
        print(f"Status: {message}") 

    def record_attendance(self):
        if not self.current_recognized_names:
            QMessageBox.information(self, "Thông Báo", "Không có ai được nhận diện để điểm danh.")
            return

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        file_exists = os.path.isfile(ATTENDANCE_FILE)
        try:
            with open(ATTENDANCE_FILE, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists or os.path.getsize(ATTENDANCE_FILE) == 0: 
                    writer.writerow(['Tên', 'Ngày', 'Giờ']) 

                recorded_count = 0
                for name in self.current_recognized_names:
                    writer.writerow([name, date_str, time_str])
                    recorded_count += 1
                QMessageBox.information(self, "Điểm Danh Thành Công",
                                        f"Đã ghi nhận {recorded_count} người .")
        except IOError:
            QMessageBox.critical(self, "Lỗi Ghi File", f"Không thể ghi vào tệp '{ATTENDANCE_FILE}'.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi Không Xác Định", f"Đã xảy ra lỗi: {e}")


    def closeEvent(self, event):
        self.stop_recognition() 
        if self.recognition_thread:
            self.recognition_thread.wait() 
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = FaceRecognitionApp()
    main_window.show()
    sys.exit(app.exec())