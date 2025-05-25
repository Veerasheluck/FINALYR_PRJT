
import os
os.environ["STREAMLIT_PREVENT_FILE_WATCHER"] = "1"

import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM
import cv2  # Required for Grad-CAM overlay

# Load the trained model
model = models.efficientnet_b3(weights=None)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(model.classifier[1].in_features, 3)
)
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
model.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Define prediction function
def predict(image):
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item(), torch.nn.functional.softmax(outputs, dim=1).detach().numpy()

# Define LIME explanation
def lime_explanation(image):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        np.array(image),
        lambda x: model(torch.stack([transform(Image.fromarray(i)) for i in x])).detach().numpy(),
        top_labels=1,
        hide_color=0,
        num_samples=200
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False
    )
    return mark_boundaries(temp / 255.0, mask)

# Define Grad-CAM explanation
def gradcam_explanation(image):
    cam_extractor = GradCAM(model, target_layer="features.7")
    input_tensor = transform(image).unsqueeze(0)
    out = model(input_tensor)
    class_idx = out.argmax(dim=1).item()
    activation_map = cam_extractor(class_idx, out)[0].squeeze().detach().numpy()

    # Normalize the activation map
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
    activation_map = np.uint8(255 * activation_map)

    # Resize heatmap to match original image size
    heatmap = cv2.resize(activation_map, image.size)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert original image to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Overlay heatmap on the original image
    overlayed = cv2.addWeighted(heatmap, 0.4, image_cv, 0.6, 0)

    # Convert back to PIL format for Streamlit display
    overlayed = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)
    result_img = Image.fromarray(overlayed)
    return result_img

# Streamlit interface
st.title("Diabetic Retinopathy Detection with XAI")
uploaded_file = st.file_uploader("Upload a DR image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    prediction, probabilities = predict(image)
    class_names = ['NO_DR', 'NPDR', 'PDR']
    
    st.write(f"Predicted Class: **{class_names[prediction]}**")
    st.write("Class Probabilities:")
    for i, prob in enumerate(probabilities[0]):
        st.write(f"{class_names[i]}: {prob:.4f}")

    xai_method = st.selectbox("Choose XAI method", ["LIME", "Grad-CAM"])
    
    if xai_method == "LIME":
        explanation = lime_explanation(image)
        st.image(explanation, caption="LIME Explanation", use_column_width=True)
    else:
        explanation = gradcam_explanation(image)
        st.image(explanation, caption="Grad-CAM Explanation", use_column_width=True)

#python -m venv dr_xai_app
#dr_xai_app\Scripts\activate
#streamlit run app.py
#set STREAMLIT_WATCH_USE_POLLING=true
#pip install opencv-python
