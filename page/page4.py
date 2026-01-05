import streamlit as st 
import torch.nn as nn 
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import transforms
import torch
st.title("Classify Indian Birds Name by thier images")
val_transform = transforms.Compose(transforms=[
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
class MyCnn_model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,padding=1),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,padding=1),
            
            nn.Conv2d(128,256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, padding=1),
            
            nn.Conv2d(256,512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2,padding=1)
            
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),                 
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
# -------------------------------------------------------
class_names = [
        "Asian_Green_Bee_eater",
        "Brown_Headed_Barbet",
        "Cattle_Egret",
        "Common_Kingfisher",
        "Common_Myna",
        "Common_Rosefinch",
        "Common_Tailorbird",
        "Coppersmith_Barbet",
        "Forest_Wagtail",
        "Gray_Wagtail",
        "Hoopoe",
        "House_Crow",
        "Indian_Grey_Hornbill",
        "Indian_Peacock",
        "Indian_Pitta",
        "Indian_Roller",
        "Jungle_Babbler",
        "Northern_Lapwing",
        "Red_Wattled_Lapwing",
        "Ruddy_Shelduck",
        "Rufous_Treepie",
        "Sarus_Crane",
        "White_Breasted_Kingfisher",
        "White_Breasted_Waterhen",
        "White_Wagtail"
    ]

def predict(image_path):
    pred_model = MyCnn_model(len(class_names))
    state_dict = torch.load("models/Indian_Bird_Indentifier_model.pth", map_location="cpu")
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

    
