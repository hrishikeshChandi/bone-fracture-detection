import torch
import streamlit as st
from torchvision import transforms
from PIL import Image

model = torch.load(
    "./results/model.pth", weights_only=False, map_location=torch.device("cpu")
)

st.set_page_config(layout="centered")
st.title("Bone Fracture Detection 🔍")
st.write("Please upload an X-ray image to check for bone fracture...")

upload_image = st.file_uploader(
    label="Upload X-ray Image", accept_multiple_files=False, type=["jpg", "png", "jpeg"]
)

# For binary clasification: 0 = No Fracture, 1 = Fracture
class_names = ["No Fracture", "Fracture"]

img_trans = transforms.Compose(
    [
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

if upload_image:
    img = Image.open(upload_image).convert("RGB")
    transformed_img = img_trans(img).unsqueeze(dim=0)

    model.eval()
    with torch.inference_mode():
        logits = model(transformed_img)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).int().item()
        confidence = probs.item() if pred == 1 else 1 - probs.item()
        label = class_names[pred]

        st.image(
            img,
            caption=f"Prediction: {label} | Confidence: {confidence * 100:.2f}%",
            width=300,
        )
