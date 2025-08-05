import glob
import torch 
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import os
from PIL import Image
import numpy as np
import sys 
from PyQt6.QtWidgets import QApplication, QMessageBox 

IMG_PATH = './data/test_images'
DATA_PATH = './data'
def show_completion_message():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Icon.Information)
    msg_box.setWindowTitle("Thông Báo Hoàn Tất")
    msg_box.setText(f"Cập nhật hoàn tất!")
    msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg_box.exec()
device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor(),
            fixed_image_standardization
        ])
    return transform(img)
    
model = InceptionResnetV1(
    classify=False,
    pretrained="casia-webface"
).to(device)

model.eval()

embeddings = []
names = []

for usr in os.listdir(IMG_PATH):
    embeds = []
    for file in glob.glob(os.path.join(IMG_PATH, usr)+'/*.jpg'):
        try:
            img = Image.open(file)
        except:
            continue
        with torch.no_grad():
            embeds.append(model(trans(img).to(device).unsqueeze(0))) 
    if len(embeds) == 0:
        continue
    embedding = torch.cat(embeds).mean(0, keepdim=True)
    embeddings.append(embedding) 
    names.append(usr)
    
embeddings = torch.cat(embeddings)
names = np.array(names)

if device == 'cpu':
    torch.save(embeddings, DATA_PATH+"/faceslistCPU.pth")
else:
    torch.save(embeddings, DATA_PATH+"/faceslist.pth")
np.save(DATA_PATH+"/usernames", names)
print('Update Completed! There are {0} people in FaceLists'.format(names.shape[0]))
show_completion_message()