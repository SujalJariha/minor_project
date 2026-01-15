# Dry Fruit Grade Classification

This project uses a deep learning model to classify images of dry fruits (such as almonds and cashews) into different quality grades (for example, Grade A and Grade B).

The model is built using **TensorFlow / Keras** and leverages **transfer learning with ResNet50** for high-accuracy image classification.

---

## Results

- **Training:** The final fine-tuned model achieved a peak **validation accuracy of ~99.7%**.
- **Inference:** Real-world testing on individual images shows strong performance, with confidence scores frequently above **99%**. Some images may produce lower confidence (around **75%**) depending on image quality and similarity to the training distribution.

---

## Dataset

- **Source:** 850 original images at 720×720 resolution covering multiple dry fruit types.
- **Augmentation:** Expanded offline (on-disk) to **42,600 images** using rotations, brightness/contrast adjustments, and noise injection.
- **Structure:**  
  Original structure:  
  `DryFruits_Dataset/Fruit/Grade/`

  Reorganized into a flat structure for Keras compatibility:  
  `dataset_flat/Fruit_Grade/`

---

## Model and Training

The model consists of a pre-trained **ResNet50** backbone with a custom classification head:

- Global Average Pooling
- Dense layer with 128 units
- Final Softmax output layer

Training was performed in **Google Colab** using a **T4 GPU**, following a two-stage training strategy.

### Stage 1: Feature Extraction

- The ResNet50 base was frozen.
- Only the classification head was trained for 10 epochs.
- Purpose: Warm up the newly added layers.
- **Result:** ~99.4% validation accuracy.

### Stage 2: Fine-Tuning

- The entire model was unfrozen.
- Recompiled with a **very low learning rate (1e-5)** to preserve pre-trained weights.
- Training continued until `EarlyStopping` (monitoring `val_loss`) halted training.
- **Final Result:** ~99.7% validation accuracy.

---

## Deployment

The trained model has been deployed as an interactive web application using **Hugging Face Spaces**.

- [**Live Demo**](https://huggingface.co/spaces/DSCmatter/deployment_model)
- Users can upload a dry fruit image directly in the browser and receive the predicted grade along with the confidence score.
- The deployment uses the same fine-tuned model (`resnet50_dryfruits_best.keras`) as described above, ensuring consistency between local inference and the hosted application.

---

## Dataset Access

The full augmented dataset used for training and evaluation is publicly available.

- [**Dataset**](https://huggingface.co/datasets/DSCmatter/Dryfruit_Image_Dataset)
- **Contents:**
  - 42,600 augmented images derived from 850 original high-resolution images.
  - Organized by dry fruit type and quality grade.
- Suitable for reuse in training, benchmarking, or transfer learning experiments.

---

## How to Use

### 1. Training the Model

1. **Setup**

   - Upload the project notebook to Google Colab.
   - Upload the dataset archive (for example, `dryfruitsDataset.rar`) to Google Drive.

2. **Run the Training Cells**
   - **Cell 1 (Setup):** Mounts Google Drive and extracts the dataset.
   - **Cell 2 (Reorganize):** Converts the nested folder structure into a flat structure required by Keras.
   - **Cell 3 (Data Generators):** Loads all images using `ImageDataGenerator` with ResNet50 preprocessing.
   - **Cell 4 (Stage 1 Training):** Trains the frozen classification head.
   - **Cell 5 (Stage 2 Training):** Fine-tunes the full model and saves the best version as `resnet50_dryfruits_best.keras`.

### 2. Running Inference (Predicting New Images)

1. **Load the Model**

```python
from tensorflow.keras.models import load_model

model = load_model('resnet50_dryfruits_best.keras')

class_indices = train_generator.class_indices
class_names = {v: k for k, v in class_indices.items()}
```

2. **Upload and Predict**

```python
from google.colab import files
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

uploaded = files.upload()
test_image_path = list(uploaded.keys())[0]

img = image.load_img(test_image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_preprocessed = preprocess_input(img_batch)

prediction = model.predict(img_preprocessed)
predicted_index = np.argmax(prediction[0])
predicted_class_name = class_names[predicted_index]
confidence = np.max(prediction[0])

print(f"Prediction: {predicted_class_name} | Confidence: {confidence*100:.2f}%")
```
