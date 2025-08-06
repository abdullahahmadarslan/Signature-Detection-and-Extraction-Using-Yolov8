# 🖊️ Signature Detection and Extraction

This project uses a YOLOv8 model to **automatically detect, crop, and clean signatures** from document or ID card images. It also provides a **Streamlit web interface** for users to interactively upload files and download extracted signature images with a white background.

---

## 📌 Features

- 🔍 Detect signature regions using YOLOv8.
- ✂️ Automatically crop detected signatures.
- 🧼 Clean the signature on a **pure white background** using OpenCV.
- 📥 Download the cropped and cleaned signatures.
- 💻 Simple and interactive **Streamlit UI**.
- 📚 Jupyter notebook included for model training and experiments.

---

## 🚀 Demo

> Upload a document containing a signature, and the app will:
> - Detect and crop the signature.
> - Remove noisy background.
> - Allow download of both versions.

---

## 📓 Model Training (YOLOv8)

Yolov8 was trained on a custom labelled id cards dataset.

---

## 🧠 How Signature Cleanup Works

- Converts to grayscale.

- Applies Gaussian Blur + Otsu Thresholding.

- Uses morphological operations for noise removal.

- Creates a white canvas and overlays the signature using the mask.

- Returns a clean signature on white background.

