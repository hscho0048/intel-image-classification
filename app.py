import os
from typing import List

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gdown

# 🔥 여기만 바꾸면 됨
MODEL_URL = "https://drive.google.com/uc?id=1uAscWYWxPMT3pw9xnmZ30ubGs8pFWRvD"
MODEL_PATH = "best_model_intel.pt"

CLASS_NAMES: List[str] = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]

IMG_SIZE = 224


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔥 모델 없으면 자동 다운로드
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, device


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def predict_image(image: Image.Image, model, device):
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().squeeze(0)

    pred_idx = int(torch.argmax(probs).item())
    pred_label = CLASS_NAMES[pred_idx]
    pred_conf = float(probs[pred_idx].item())

    prob_dict = {
        CLASS_NAMES[i]: float(probs[i].item())
        for i in range(len(CLASS_NAMES))
    }
    return pred_label, pred_conf, prob_dict


st.set_page_config(page_title="Intel Image Classifier", layout="centered")

st.title("Intel Image Classification")
st.write("이미지를 업로드하면 장면 클래스를 예측합니다.")

try:
    model, device = load_model()
    st.caption(f"현재 실행 장치: {device}")
except Exception as e:
    st.error(str(e))
    st.stop()

uploaded_file = st.file_uploader(
    "이미지를 업로드하세요",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드한 이미지", use_container_width=True)

    pred_label, pred_conf, prob_dict = predict_image(image, model, device)

    st.subheader("예측 결과")
    st.write(f"예측 클래스: **{pred_label}**")
    st.write(f"확신도: **{pred_conf:.2%}**")

    st.subheader("클래스별 확률")
    for label, prob in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
        st.write(f"{label}: {prob:.2%}")
        st.progress(prob)
