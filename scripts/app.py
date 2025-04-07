# app.py
import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import transforms
from train import CNN  # Importa tu modelo desde train.py

# === CONFIGURACIÓN ===
MODEL_PATH = "model_fold4.pth"
NUM_CLASSES = 8
CLASS_NAMES = [
    "calce_para_manguera",
    "cubierta_para_cerradura",
    "manija_para_ventana",
    "perilla_aire_acondicionado",
    "perilla_control_de_radio",
    "soporte_para_motor",
    "tapa_de_camara_de_aire",
    "tope_para_frenos"
]

# === TRANSFORMACIONES (igual que en entrenamiento) ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((360, 640)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === CARGAR MODELO ===
@st.cache_resource
def load_model():
    model = CNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# === STREAMLIT UI ===
st.title("🧠 Identificador de Piezas 3D en Tiempo Real")
run = st.checkbox("✅ Activar cámara")

FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("No se pudo abrir la cámara.")
    else:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("No se pudo capturar imagen.")
                break

            # Convertir imagen y hacer predicción
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = transform(img_rgb).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                label = CLASS_NAMES[predicted.item()]

            # Mostrar etiqueta en la imagen
            cv2.putText(img_rgb, f"Pieza: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            FRAME_WINDOW.image(img_rgb)

        cap.release()
else:
    st.write("Activa la cámara para empezar.")
