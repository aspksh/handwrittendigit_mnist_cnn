import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2

from PIL import Image

# -----------------------------
# Page
# -----------------------------
st.set_page_config(
    page_title="MNIST Handwritten Digit Predictor",
    page_icon="✍️",
    layout="wide"
)

st.title("✍️ MNIST Handwritten Digit Predictor")
st.write("Upload one or more handwritten digit images.")

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# CNN Model
# -----------------------------
class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(32*7*7,128)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(128,10)

    def forward(self,x):

        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = x.view(x.size(0),-1)

        x = self.relu3(self.fc1(x))

        x = self.fc2(x)

        return x


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():

    model = CNN().to(device)

    model.load_state_dict(
        torch.load(
            "mnist_cnn.pth",
            map_location=device
        )
    )

    model.eval()

    return model


model = load_model()

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(uploaded_file):

    # Read image
    img = Image.open(uploaded_file).convert("L")
    img = np.array(img)

    # Blur
    img = cv2.GaussianBlur(img, (5,5), 0)

    # Otsu Threshold
    _, thresh = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Dilate
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # Largest Connected Component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        thresh,
        connectivity=8
    )

    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        digit = np.zeros_like(thresh)
        digit[labels == largest] = 255
    else:
        digit = thresh

    # Bounding Box
    coords = cv2.findNonZero(digit)

    if coords is None:
        return None, None

    x, y, w, h = cv2.boundingRect(coords)

    digit = digit[y:y+h, x:x+w]

    # Make Square
    size = max(h, w)

    square = np.zeros((size, size), dtype=np.uint8)

    yoff = (size-h)//2
    xoff = (size-w)//2

    square[yoff:yoff+h, xoff:xoff+w] = digit

    # Resize to 20x20
    digit20 = cv2.resize(square, (20,20))

    # 28x28 Canvas
    canvas = np.zeros((28,28), dtype=np.uint8)
    canvas[4:24,4:24] = digit20

    processed = Image.fromarray(canvas)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,),
            (0.3081,)
        )
    ])

    tensor = transform(processed).unsqueeze(0)

    return processed, tensor
# -----------------------------
# Upload Images
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload handwritten digit images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_files:

    for uploaded_file in uploaded_files:

        st.markdown("---")
        st.subheader(f"📄 {uploaded_file.name}")

        # Original Image
        original = Image.open(uploaded_file).convert("L")

        # Preprocess
        processed, tensor = preprocess_image(uploaded_file)

        if processed is None:
            st.error("No digit detected in the image.")
            continue

        # Preview
        col1, col2 = st.columns(2)

        with col1:
            st.image(original, caption="Original Image", use_container_width=True)

        with col2:
            st.image(processed, caption="Processed Image", use_container_width=True)

        # Prediction
        tensor = tensor.to(device)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]

            pred = output.argmax(dim=1).item()
            confidence = probs[pred].item() * 100

        st.success(f"🎯 Prediction : **{pred}**")
        st.info(f"Confidence : **{confidence:.2f}%**")

        # Probability Chart
        st.subheader("Prediction Probabilities")

        prob_dict = {
            str(i): float(probs[i].cpu()) * 100
            for i in range(10)
        }

        st.bar_chart(prob_dict)

        # Detailed Probabilities
        with st.expander("Show All Probabilities"):

            for i in range(10):

                st.write(
                    f"Digit {i}: {probs[i].item()*100:.2f}%"
                )

st.markdown("---")
st.caption("Developed using PyTorch + Streamlit")
