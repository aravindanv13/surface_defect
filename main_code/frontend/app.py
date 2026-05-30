"""
Streamlit app for Surface Defect Detection.

Allows users to:
- Upload images for defect classification
- Select from trained models (Baseline CNN, ResNet18, ResNet50, VGG16)
- View predictions with confidence scores
- Visualize Grad-CAM heatmaps
"""

import os
import sys
import torch
import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

# ─────────────────────────────────────────────
# Path Configuration
# ─────────────────────────────────────────────
# Get parent directory (main_code) and backend path
SCRIPT_DIR = Path(__file__).parent  # frontend/
MAIN_CODE_DIR = SCRIPT_DIR.parent   # main_code/
BACKEND_DIR = MAIN_CODE_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from src.data_loader import CLASS_NAMES, get_val_transforms
from src.utils import load_checkpoint, get_device
from src.model_baseline import build_baseline_cnn
from src.model_transfer import build_resnet18, build_resnet50, build_vgg16
from src.gradcam import GradCAM, get_target_layer

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Surface Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Surface Defect Detection")
st.markdown("AI-powered detection of manufacturing defects using deep learning")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

MODELS_DIR = str(MAIN_CODE_DIR / "backend" / "models")
AVAILABLE_MODELS = {
    "Baseline CNN": "baseline_cnn",
    "ResNet18 (Transfer)": "transfer_resnet18",
    "ResNet50 (Transfer)": "transfer_resnet50",
    "VGG16 (Transfer)": "transfer_vgg16",
}

# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

@st.cache_resource
def load_model(model_name):
    """Load model from checkpoint with caching."""
    device = get_device()

    checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}_last.pth")

    if not os.path.exists(checkpoint_path):
        return None, device, f"❌ Checkpoint not found: {checkpoint_path}"

    try:
        # Create model based on name
        if model_name == "baseline_cnn":
            model = build_baseline_cnn(num_classes=4)
        elif model_name == "transfer_resnet18":
            model = build_resnet18(num_classes=4)
        elif model_name == "transfer_resnet50":
            model = build_resnet50(num_classes=4)
        elif model_name == "transfer_vgg16":
            model = build_vgg16(num_classes=4)
        else:
            return None, device, f"❌ Unknown model: {model_name}"

        # Load checkpoint
        model, epoch, metrics = load_checkpoint(model, checkpoint_path, device=device)
        model = model.to(device)
        model.eval()

        return model, device, f"✅ Loaded: {model_name} (Epoch {epoch})"

    except Exception as e:
        return None, device, f"❌ Error loading model: {str(e)}"


def predict_defect(image_tensor, model, device):
    """Run inference on a single image."""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)

    pred_class = probs.argmax(dim=1).item()
    confidence = probs[0, pred_class].item()
    all_probs = probs[0].cpu().numpy()

    return pred_class, confidence, all_probs


def generate_gradcam(image_tensor, model, device, class_idx):
    """Generate Grad-CAM heatmap."""
    try:
        target_layer = get_target_layer(model)
        gradcam = GradCAM(model, target_layer)
        heatmap, _ = gradcam.generate(image_tensor.to(device), class_idx)
        return heatmap
    except Exception as e:
        st.error(f"Error generating Grad-CAM: {str(e)}")
        return None


# ─────────────────────────────────────────────
# Sidebar Configuration
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    # Mode selection
    st.subheader("Classification Mode")
    classification_mode = st.radio(
        "Choose mode:",
        options=["Binary (Defect / No Defect)", "Multi-Class (Detailed)"],
        index=0
    )

    # Model selection
    st.subheader("Model Selection")
    selected_model_display = st.selectbox(
        "Choose a model:",
        options=list(AVAILABLE_MODELS.keys()),
        index=0  # Default to Baseline CNN (more robust to real-world variation)
    )
    selected_model = AVAILABLE_MODELS[selected_model_display]

    # Grad-CAM toggle
    st.subheader("Visualization")
    show_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=False)

    # Confidence threshold
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    st.markdown("---")
    st.info(
        "**How to use:**\n"
        "1. Select a model from the dropdown\n"
        "2. Upload an image\n"
        "3. View the prediction and confidence\n"
        "4. Toggle Grad-CAM to see model attention"
    )

# ─────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────

col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Supported formats: JPG, PNG, BMP, TIFF"
    )

with col2:
    st.subheader("📊 Prediction Results")

    if uploaded_file is None:
        st.info("👆 Upload an image to get started")
    else:
        # Load and display the uploaded image
        image_pil = Image.open(uploaded_file).convert("RGB")

        # Preprocess for model using transforms
        transform = get_val_transforms(img_size=224)
        image_tensor = transform(image_pil).unsqueeze(0)  # add batch dimension

        # Load model
        model, device, load_msg = load_model(selected_model)

        if model is None:
            st.error(load_msg)
        else:
            st.success(load_msg)

            # Run inference
            pred_class, confidence, all_probs = predict_defect(image_tensor, model, device)
            pred_label = CLASS_NAMES[pred_class]
            
            # Binary Mode Logic
            binary_mode = classification_mode == "Binary (Defect / No Defect)"
            
            if binary_mode:
                no_defect_idx = CLASS_NAMES.index("no_defect")
                prob_no_defect = float(all_probs[no_defect_idx])
                prob_defect = 1.0 - prob_no_defect
                
                is_defect = prob_defect > prob_no_defect
                display_label = "DEFECT" if is_defect else "NO DEFECT"
                display_conf = prob_defect if is_defect else prob_no_defect
                
                # Colors: Red for defect, Green for no defect (if confidence exceeds threshold)
                if display_conf < confidence_threshold:
                    color = "🟡"
                else:
                    color = "🔴" if is_defect else "🟢"
                
                confidence_dict = {
                    "Defect": prob_defect,
                    "No Defect": prob_no_defect
                }
                
                results_data = {
                    "Class": ["Defect", "No Defect"],
                    "Confidence": [f"{prob_defect:.4f}", f"{prob_no_defect:.4f}"],
                    "Percentage": [f"{prob_defect*100:.2f}%", f"{prob_no_defect*100:.2f}%"]
                }
            else:
                display_label = pred_label.upper()
                display_conf = confidence
                color = "🟢" if display_conf >= confidence_threshold else "🟡"
                
                confidence_dict = {
                    CLASS_NAMES[i]: float(all_probs[i])
                    for i in range(len(CLASS_NAMES))
                }
                
                results_data = {
                    "Class": CLASS_NAMES,
                    "Confidence": [f"{p:.4f}" for p in all_probs],
                    "Percentage": [f"{p*100:.2f}%" for p in all_probs]
                }

            # Display prediction
            st.markdown("---")

            # Large prediction display
            st.markdown(f"## {color} Predicted: **{display_label}**")
            st.markdown(f"### Confidence: **{display_conf:.2%}**")

            # Confidence bar chart
            st.markdown("**Confidence Scores by Class:**")
            st.bar_chart(confidence_dict)

            # Table view
            with st.expander("📋 Detailed Scores"):
                st.dataframe(results_data, use_container_width=True)

# ─────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────

if uploaded_file is not None and model is not None:
    st.markdown("---")
    st.subheader("🎨 Visualizations")

    # Original image
    col_orig, col_viz = st.columns([1, 1], gap="medium")

    with col_orig:
        st.markdown("**Original Image**")
        st.image(image_pil, use_column_width=True)

    with col_viz:
        if show_gradcam:
            st.markdown(f"**Grad-CAM: {pred_label}**")

            with st.spinner("Generating Grad-CAM..."):
                heatmap = generate_gradcam(image_tensor, model, device, pred_class)

            if heatmap is not None:
                # Convert heatmap to RGB for display
                heatmap_colored = cv2.applyColorMap(
                    (heatmap * 255).astype(np.uint8),
                    cv2.COLORMAP_JET
                )
                heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                st.image(heatmap_rgb, use_column_width=True)
        else:
            st.info("ℹ️ Enable Grad-CAM in the sidebar to see model attention")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "<small>Surface Defect Detection | Deep Learning Model | PyTorch</small>"
    "</div>",
    unsafe_allow_html=True
)
