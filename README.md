# Bone Fracture Detection

## Overview

This project helps you automatically detect bone fractures in X-ray images using deep learning. Upload an X-ray, and the system will predict if a fracture is present or not. The solution is powered by a custom-trained image classification model, and everything runs locally on your system with a user-friendly Streamlit web app.

## Features

- **Fracture Detection:**  
  The model classifies X-ray images as either **fractured** or **not fractured**.
- **Custom-Built CNN:**  
  A lightweight convolutional neural network designed specifically for bone fracture detection.
- **Visual Feedback:**  
  After training, you'll see a confusion matrix and a classification report showing exactly how the model performed.
- **Metrics & Logging:**  
  Training progress and results are saved, and you can visualize everything with TensorBoard.
- **Streamlit Web App:**  
  Easily upload X-ray images and get instant predictions using the included Streamlit app (`app.py`).

## How It Works

1. **Dataset Download:** The dataset is fetched automatically from Kaggle when you run the notebook locally.
2. **Preprocessing & Data Loading:** Images are resized, normalized, and split into training/validation/test sets.
3. **Model Training:** Our CNN learns to spot the difference between fractured and non-fractured bones.
4. **Evaluation:** You'll get a detailed confusion matrix and performance metrics.
5. **Instant Predictions:** Use the Streamlit app to test new X-ray images right away.

## Dataset

No need to download anything manually! When you run the notebook, the dataset is automatically downloaded into your system using KaggleHub.

- **Source:** [Kaggle - Fracture Multi Region X-ray Data](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data)

## Technologies Used

- **PyTorch** – For building and training the model.
- **TorchVision** – For image transforms and data loading.
- **TensorBoard** – To visualize the training process.
- **Other libraries:** Pillow, tqdm, mlxtend, torchmetrics, scikit-learn.

## Local Setup and Usage

**You don’t need to download the dataset manually! The notebook will handle it for you.**

### 1. Clone the repository and install dependencies

This project includes a `requirements.txt` file listing all necessary Python packages. Install them with:

```bash
git clone https://github.com/hrishikeshChandi/bone-fracture-detection
cd bone-fracture-detection
pip install -r requirements.txt
```

### 2. Run the notebook for training and evaluation

Open `main.ipynb` in Jupyter Notebook or VS Code and run all cells. The dataset will be downloaded automatically, the model will be trained, and results will be saved in the `results/` folder.

### 3. Run the Streamlit app for predictions

After training, launch the web app:

```bash
streamlit run app.py
```

Upload an X-ray image and get instant fracture predictions in your browser.

## Project Structure

- `main.ipynb` – Notebook for training and evaluation (run locally).
- `app.py` – Streamlit web app for fracture prediction.
- `requirements.txt` – List of all required Python packages.
- `results/` – Stores trained model, logs, and evaluation metrics.

## Results

- **Test Set Performance:**
  - **Loss:** 0.0975
  - **Accuracy:** 96.84%

These results were obtained on the test set after training the model using the steps above.

## Notes

- **No need to manually download the dataset**—the notebook will do it for you.
- The pipeline works best with a local GPU, but will also run on CPU (slower).
- To make predictions on your own images, use the Streamlit app (`app.py`).

## License

This project is licensed under the MIT License.
