import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from streamlit_cropper import st_cropper

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="MNIST Handwritten Digit Predictor",
    page_icon="✍️",
    layout="wide"
)

st.title("✍️ MNIST Handwritten Digit Predictor")
st.write("Upload one or more handwritten digit images (0–9).")

# ----------------------------------
# Device
# ----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------
# CNN Model
# ----------------------------------
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


# ----------------------------------
# Load Model
# ----------------------------------
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

# ----------------------------------
# Image Preprocessing
# ----------------------------------
def preprocess_image(image, iterations=4):

    if isinstance(image, Image.Image):
        img = image.convert("L")
    else:
        img = Image.open(image).convert("L")

    img = ImageOps.invert(img)

    img_array = np.array(img)

    img_array[img_array < 100] = 0
    img_array[img_array >= 100] = 255

    binary = img_array > 0

    dilated = binary_dilation(binary, iterations=iterations)

    img_array = (dilated * 255).astype(np.uint8)
    
    coords = np.where(img_array > 0)

    top = coords[0].min()
    bottom = coords[0].max()
    
    left = coords[1].min()
    right = coords[1].max()

    img_array = img_array[top:bottom+1, left:right+1]
    
    h, w = img_array.shape

    size = max(h, w)

    square = np.zeros((size, size), dtype=np.uint8)

    y_offset = (size-h)//2
    x_offset = (size-w)//2

    square[
        y_offset:y_offset+h,
        x_offset:x_offset+w
    ] = img_array

    img20 = Image.fromarray(square).resize((20,20))

    canvas = np.zeros((28,28), dtype=np.uint8)

    canvas[4:24,4:24] = np.array(img20)

    processed = Image.fromarray(canvas)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    tensor = transform(processed).unsqueeze(0)

    return processed, tensor


# ----------------------------------
# Upload Images
# ----------------------------------
uploaded_files = st.file_uploader(
    "Upload handwritten digit images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:

    for uploaded_file in uploaded_files:

        st.subheader(uploaded_file.name)

        img = Image.open(uploaded_file)
        img.thumbnail((800, 800))

        cropped = st_cropper(
            img,
            realtime_update= True,
            box_color="#00FF00",
            aspect_ratio=(1,1)
        )
        
        processed, tensor = preprocess_image(cropped)

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(img, caption="Original Image", use_container_width=True)
        
        with col2:
            st.image(cropped, caption="Cropped Image", use_container_width=True)
        
        with col3:
            st.image(processed, caption="Processed", use_container_width=True)
            

        # --------------------------
        # Predict Button
        # --------------------------
        if st.button(f"Predict {uploaded_file.name}"):
        
            tensor = tensor.to(device)
        
            with torch.no_grad():
        
                output = model(tensor)
        
                probs = torch.softmax(output, dim=1)[0]
        
                pred = torch.argmax(output, dim=1).item()
        
                confidence = probs[pred].item() * 100
        
            st.success(f"🎯 Prediction : {pred}")
        
            st.info(f"Confidence : {confidence:.2f}%")
        
            st.subheader("Prediction Probabilities")
        
            prob_dict = {}
        
            for i in range(10):
                prob_dict[str(i)] = float(probs[i].cpu()) * 100
        
            st.bar_chart(prob_dict)
        
            with st.expander("Show All Probabilities"):
        
                for i in range(10):
                    st.write(f"Digit {i}: {probs[i].item()*100:.2f}%")

st.markdown("---")
st.caption("Developed using PyTorch + Streamlit")
