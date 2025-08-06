import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO

# ------------------------ Signature Cleanup Function ------------------------
def extract_signature_white_bg(cropped_image: Image.Image) -> Image.Image:
    img_cv = np.array(cropped_image)
    
    # Convert to grayscale if it's a color image
    if img_cv.ndim == 3:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_cv

    # Blur and threshold
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleaning
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # === Removed contour filtering step ===
    # Use the cleaned mask directly
    mask = cleaned

    # Prepare white background image
    white_bg = np.ones_like(img_cv) * 255

    # If image is grayscale, convert to 3-channel RGB
    if img_cv.ndim == 2 or img_cv.shape[2] == 1:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)

    mask_3ch = cv2.merge([mask, mask, mask])
    
    # Apply the mask: keep original where mask is 255, else keep white
    result = np.where(mask_3ch == 255, img_cv, white_bg)

    return Image.fromarray(result)



# ------------------------ Load YOLO Model ------------------------
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

# ------------------------ Streamlit UI ------------------------
st.title("🖊️ Signature Extractor")
st.markdown("Upload an ID card or document image containing a signature.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Extract original file name without extension
    original_filename = os.path.splitext(uploaded_file.name)[0]

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with st.spinner("🔍 Detecting signature..."):
        results = model(image_rgb)

    if results and len(results[0].boxes) > 0:
        box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        cropped_img = image_rgb[y1:y2, x1:x2]
        cropped_pil = Image.fromarray(cropped_img)
        cleaned_pil = extract_signature_white_bg(cropped_pil)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✂️ Cropped Signature")
            st.image(cropped_pil, use_container_width=True)

            with st.spinner("📦 Preparing download..."):
                cropped_buf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                cropped_pil.save(cropped_buf.name)
                st.download_button(
                    "Download Cropped",
                    cropped_buf.read(),
                    file_name=f"{original_filename}.png",
                    mime="image/png"
                )

        with col2:
            st.subheader("🧾 Cleaned Signature (White Background)")
            st.image(cleaned_pil, use_container_width=True)

            with st.spinner("📦 Preparing download..."):
                cleaned_buf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                cleaned_pil.save(cleaned_buf.name)
                st.download_button(
                    "Download Cleaned",
                    cleaned_buf.read(),
                    file_name=f"{original_filename}.png",
                    mime="image/png"
                )

    else:
        st.warning("⚠️ No signature detected. Try another image.")
