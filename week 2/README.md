# 🌳 Tree Species Detection Using Deep Learning

This project detects tree species using deep learning techniques, including Convolutional Neural Networks (CNN) and Transfer Learning with MobileNetV2 and EfficientNetB0.

## 📁 Project Structure

- `CNN_Tree_Species.py` — Train a custom CNN from scratch.
- `Transfer_Learning.py` — Train an EfficientNetB0-based model.
- `predict.py` — Predict species of a new tree image using the pretrained MobileNetV2 model.
- `dataset_loader.py` — Load and visualize the dataset.
- `tree_transfer_mobilenetv2.h5` — Pretrained MobileNetV2 model file.
- `README.md` — Instructions and overview.

## 📦 Requirements

Install the following Python libraries:
```bash
pip install tensorflow matplotlib pillow



🚀 How to Use
1. Train a CNN from Scratch
python CNN_Tree_Species.py

2. Train Using Transfer Learning (EfficientNetB0)
python Transfer_Learning.py

3. Predict Tree Species from an Image
python predict.py "path/to/image.jpg"
📝 Make sure the model file tree_transfer_mobilenetv2.h5 is present in the same folder as predict.py.

🧪 Example Output

✅ Loaded model from tree_transfer_mobilenetv2.h5
📚 Class Names: ['neem', 'mango', 'pipal', ..., 'vad']
🌳 Predicted Tree Species: neem (93.20% confidence)