# Bone Fracture Detection

## Overview

This project helps you automatically detect bone fractures in X-ray images using deep learning. Upload an X-ray, and the system will predict if a fracture is present or not. The solution is powered by a custom-trained image classification model, and everything runs directly in a Google Colab notebook—no manual setup or dataset download required.

## Features

- **Fracture Detection:**  
  The model classifies X-ray images as either **fractured** or **not fractured**.
- **Custom-Built CNN:**  
  A lightweight convolutional neural network designed specifically for bone fracture detection.
- **Visual Feedback:**  
  After training, you'll see a confusion matrix and a classification report showing exactly how the model performed.
- **Metrics & Logging:**  
  Training progress and results are saved, and you can visualize everything with TensorBoard.

## How It Works

1. **No download:** The dataset is fetched automatically from Kaggle when you run the notebook in Colab.
2. **Preprocessing & Data Loading:** Images are resized, normalized, and split into training/validation/test sets.
3. **Model Training:** Our CNN learns to spot the difference between fractured and non-fractured bones.
4. **Evaluation:** You'll get a detailed confusion matrix and performance metrics.
5. **Instant Predictions:** Use the notebook to test new X-ray images right away.

## Dataset

No need to download anything manually! When you run the notebook on Google Colab, the dataset is automatically downloaded into your session.

- **Source:** [Kaggle - Fracture Multi Region X-ray Data](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data)
- **Structure after download:**

  ```
  Bone_Fracture_Binary_Classification/
  ├── train/
  ├── val/
  └── test/
  ```

## Technologies Used

- **PyTorch** – For building and training the model.
- **TorchVision** – For image transforms and data loading.
- **TensorBoard** – To visualize the training process.
- **Other libraries:** Pillow, tqdm, mlxtend, torchmetrics, scikit-learn.

## Setup and Usage

**You don’t need to download the dataset manually or worry about dependencies!**  
Just use the notebook in Google Colab and follow these steps:

1. **Open the Colab notebook:**  
   [Open Bone Fracture Detection Colab Notebook](https://colab.research.google.com/drive/1unevzAdW9EbpodrRGIezCiAvnF_LENB4?authuser=1)

2. **Save a copy to your Google Drive:**

   - Go to `File` > `Save a copy in Drive` in the Colab menu bar.
   - This ensures you have your own editable version and full access to saving results.

3. **Run the notebook:**

   - Open your saved copy in Colab.
   - Each cell is ready to execute. The dataset download, Google Drive mounting, and everything else is automated.

4. **Install dependencies (if not previously installed):**

   ```python
   !pip install kagglehub torchmetrics mlxtend
   ```

5. **Dataset setup happens automatically:**

   ```python
   import kagglehub
   path = kagglehub.dataset_download("bmadushanirodrigo/fracture-multi-region-x-ray-data")
   print("Path to dataset files:", path)
   ```

6. **Train the model:**

   - Just run through the training cells. Your model and all results are saved to Google Drive (under `/content/drive/My Drive/results/`).

7. **View metrics and logs:**

   - Logs and the confusion matrix are saved automatically.
   - To view logs in TensorBoard:
     ```bash
     tensorboard --logdir "/content/drive/My Drive/results/runs"
     ```

8. **Make predictions:**
   - Use the prediction cell at the end of the notebook to test with your own X-rays.

> **Note:**  
> There is **no web frontend or Streamlit app** for this project. Everything runs in the Colab notebook.

## Project Structure

- `bone.ipynb` – The Colab notebook for training and evaluation ([open here](https://colab.research.google.com/drive/1unevzAdW9EbpodrRGIezCiAvnF_LENB4?authuser=1)).
- `Bone_Fracture_Binary_Classification/` – Dataset folder (auto-downloaded).
- `results/` – Stores trained model, logs, and evaluation metrics.

## Results

- **Test Set Performance:**
  - **Loss:** 0.0975
  - **Accuracy:** 96.84%

These results were obtained on the test set after training the model using the steps above.

## Notes

- **No need to manually download the dataset**—Colab takes care of it.
- For persistent storage, connect your Google Drive in Colab.
- The whole pipeline works best with a Colab GPU runtime.
- To make predictions on your own images, just use the last cell in the notebook.

## License

This project is licensed under the MIT License.
