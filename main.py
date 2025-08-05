import sys
import subprocess
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt

class AttendanceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('VUI ESchool')
        self.setGeometry(300, 300, 400, 200)

        layout = QVBoxLayout()

        
        title_label = QLabel('Ứng dụng điểm danh')
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = title_label.font()
        font.setPointSize(20)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)

        layout.addStretch(1)

        register_button = QPushButton('Đăng ký')
        register_button.clicked.connect(self.open_capture_face)
        layout.addWidget(register_button)


        attendance_button = QPushButton('Điểm danh')
        attendance_button.clicked.connect(self.open_recornize) 
        layout.addWidget(attendance_button)

        layout.addStretch(1)

        attendance_button = QPushButton('Cập nhật')
        attendance_button.clicked.connect(self.open_capnhat) 
        layout.addWidget(attendance_button)

        self.setLayout(layout)
        self.show()

    def open_capture_face(self):
        try:
            print("Attempting to open catureface.py")

            subprocess.Popen([sys.executable, 'catureface.py'])
        except FileNotFoundError:
            print("Error")
        except Exception as e:
            print(f"An error occurred: {e}")

    def open_recornize(self):
        try:
            print("Attempting to open recornize.py")
            subprocess.Popen([sys.executable, 'recornize.py'])
        except FileNotFoundError:
            print("Error")
        except Exception as e:
            print(f"An error occurred: {e}")
    def open_capnhat(self):
        try:
            print("Attempting to open updateface.py")
            subprocess.Popen([sys.executable, 'updateface.py'])
        except FileNotFoundError:
            print("Error")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AttendanceApp()
    sys.exit(app.exec())