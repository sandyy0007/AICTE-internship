# 🌳 Tree Species Classification using Leaf Images

This project uses Deep Learning to automatically identify tree species based on leaf images. It is designed to assist students, researchers, and environmentalists in recognizing trees quickly and accurately using computer vision.

---

## 🌿 Project Introduction

Tree species identification is critical for biodiversity conservation, forest management, and ecological studies. Traditional identification methods are manual and slow. This project provides an automated, scalable solution using machine learning.

We built a CNN-based model leveraging **MobileNetV2** architecture, capable of classifying tree species using visual features like shape, venation, and texture.

---

## 🧠 Features

- 🌲 Recommend Trees by Location  
- 📍 Find Locations for a Tree  
- 📷 Identify Tree from Image  
- 📊 Dataset Description  
- 🗂️ Tree Metadata  
- 🖼️ Tree Image Dataset  
- 🔍 Recommender System  
- 🧠 CNN Classifier  
- 🧪 Preprocessing & Encoding  

---

## ✅ Key Improvements Over Previous Version

- Upgraded from rule-based/manual techniques to deep learning
- Added preprocessing (grayscale, resizing, denoising)
- Used diverse and larger dataset
- Visualized training (accuracy/loss plots)
- Suitable for deployment on apps/web tools

---

## 🔍 Project Overview

We aim to classify tree species using leaf image data.

### 🔨 Work Done So Far

1. Selected *Tree Species Classification* as project topic  
2. Collected a leaf image dataset with multiple species  
3. Preprocessed images (resize, grayscale, denoise)  
4. Extracted shape, texture, and vein features  
5. Built and trained a CNN classifier  
6. Validated and improved accuracy  
7. Analyzed misclassified outputs  
8. Generated accuracy/loss plots  
9. Drafted project documentation  
10. Started GitHub upload and presentation prep  

---



## 📁 Project Structure

├── CNN_Tree_Species.py # Train basic CNN model
├── Transfer_Learning.py # Train EfficientNetB0 model
├── predict.py # Predict species using MobileNetV2
├── dataset_loader.py # Dataset loading/visualization
├── tree_transfer_mobilenetv2.h5 # Trained MobileNetV2 model
├── README.md # Project documentation

## 🔗 Trained Model File

> ⚠️ The model file `model.h5` is over **400 MB** and cannot be uploaded to GitHub directly.

📥 Download it from Google Drive:  
[📂 Click here to download model.h5](https://drive.google.com/file/d/1NV2m4_emdZ0qozZQCLvzILuO-mhS6jhW/view?usp=drive_link)

Place the downloaded file in your project directory (e.g., `/models/` or root).

---

---

## ⚙️ Requirements

Install required Python libraries:

```bash
pip install tensorflow matplotlib pillow


🚀 How to Use
1. Train CNN from Scratch
                        python CNN_Tree_Species.py

2. Train Using Transfer Learning (EfficientNetB0)
                                    python Transfer_Learning.py


3. Predict Tree Species from an Image
                                  python predict.py "path/to/your_image.jpg"


📝 Ensure the tree_transfer_mobilenetv2.h5 model is in the same folder as predict.py.



🧪 Example Output

✅ Loaded model from tree_transfer_mobilenetv2.h5
📚 Class Names: ['neem', 'mango', 'pipal', ..., 'vad']
🌳 Predicted Tree Species: neem (93.20% confidence)
✅ Steps to Run via Command Line (Optional)
 

                         cd D:\Final_tree_species_project
                        python predict.py "test_images\leaf_01.jpg"


📈 Performance & Visualization
Model Accuracy: High (based on validation results)

Misclassifications were analyzed and improved

Accuracy and loss curves plotted during training

📅 Next Steps
✅ Final report proofreading

✅ PPT preparation

✅ Upload final project to GitHub

🤖 Tree Intelligence Assistant (App Preview)
An AI assistant designed to:

Recommend tree species based on location

Identify trees from images

Visualize metadata

Launchable using:


streamlit run streamlit_integrated.py


📢 Acknowledgments
This project is part of the Tree Species Classification assignment for the "AI for Environmental Applications" course.

