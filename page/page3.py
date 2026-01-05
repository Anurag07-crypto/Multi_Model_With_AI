import streamlit as st 
import torch.nn.functional as F
import torch.nn as nn 
from torchvision.transforms import transforms
from PIL import Image
import torch
st.title("Indentify your image is Ai generated or Not")
class ai_vs_real_model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32, kernel_size=2,stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(32,64,kernel_size=2,stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(64, 128, kernel_size=2,stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
img_width, img_len = 64,64
val_transform = transforms.Compose([
    transforms.Resize((img_width, img_len)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
num_classes = 2
def predict(image_path):
    class_names = ["AI","Real"]
    pred_model = ai_vs_real_model(num_classes)
    state_dict = torch.load(r"models\ai_vs_real.pth", map_location="cpu")
    pred_model.load_state_dict(state_dict)
    pred_model.eval()
    image = Image.open(image_path).convert("RGB")
    image_tensor = val_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = pred_model(image_tensor)
        probs = F.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    label = class_names[pred.item()]
    conf_score = conf.item()*100
    st.write(f"This is {label} | With The Confidence score of {conf_score}")
    st.image(image)
uploaded_image = st.file_uploader("Upload image")
if uploaded_image is not None:
    predict(uploaded_image)
    