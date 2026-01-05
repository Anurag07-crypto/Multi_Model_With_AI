import streamlit as st 
import torch.nn as nn 
from torchvision.transforms import transforms
import torch.nn.functional as F
from PIL import Image
import torch
st.title("Welcome to find Different Types of Leaf")
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
# ---------------------------------------------------------------------------
class cnn_model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3),
            
            nn.Conv2d(32,64,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3),
            
            nn.Conv2d(64,128,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self,x):
        x = self.features(x)
        output = self.classifier(x)
        return output
# ---------------------------------------------------------------------------
img_width, img_len = 244, 244
val_transform = transforms.Compose([
    transforms.Resize((img_width, img_len)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
num_classes = 38
def predict(image_path):
    pred_model = cnn_model(num_classes)
    state_dict = torch.load("models/plant_leaf_classification.pth", map_location="cpu")
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

    
