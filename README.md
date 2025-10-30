# Dry Fruit Grade Classification

This project uses a deep learning model to classify images of dry fruits (like almonds and cashews) into different quality grades (e.g., Grade A, Grade B).

The model is built using TensorFlow/Keras and employs transfer learning with the **ResNet50** architecture.

---

## Results

* **Training:** The final fine-tuned model achieved a peak **validation accuracy of ~99.7%** during training.
* **Inference:** Real-world testing on individual images shows strong performance, with confidence scores often exceeding **99%**. Some images may yield lower confidence (e.g., ~75%) depending on quality and similarity to the training data.

---

## Dataset

* **Source:** 850 original 720x720 images of various dry fruits.
* **Augmentation:** The dataset was expanded "offline" (on-disk) to 42,600 images, including rotations, brightness/contrast changes, and noise.
* **Classes:** The folder structure `DryFruits_Dataset/Fruit/Grade/` was reorganized into a flat structure (`dataset_flat/Fruit_Grade/`) for training.

---

## Model and Training

The model is a pre-trained ResNet50 base with a new classification head (Global Average Pooling, a 128-node Dense layer, and a final Softmax output).

The training was performed in a Google Colab notebook using a T4 GPU, following a crucial **two-stage process**:

1.  **Stage 1: Feature Extraction**
    * The ResNet50 base was frozen.
    * Only the new classification head was trained for 10 epochs. This quickly "warms up" the new layers.
    * **Result:** ~99.4% validation accuracy.

2.  **Stage 2: Fine-Tuning**
    * The entire model (including the ResNet50 base) was unfrozen.
    * The model was re-compiled with a **very low learning rate** (`1e-5`) to prevent destroying the pre-trained weights.
    * Training continued until `EarlyStopping` (monitoring `val_loss`) stopped the process.
    * **Final Result:** ~99.7% validation accuracy.

---

## How to Use

### 1. Training the Model

1.  **Setup:**
    * Upload the project notebook to Google Colab.
    * Upload your dataset (e.g., `dryfruitsDataset.rar`) to Google Drive.

2.  **Run the Training Cells:**
    * **Cell 1 (Setup):** Mounts your Google Drive and un-RARs the dataset.
    * **Cell 2 (Reorganize):** Runs a script to convert the nested folder structure `(Almond/Grade_A)` into the flat structure `(Almond_Grade_A)` required by Keras.
    * **Cell 3 (Data Generators):** Loads the 42.6k images using `ImageDataGenerator`. It applies the mandatory ResNet50 preprocessing.
    * **Cell 4 (Stage 1 Training):** Trains the frozen model head.
    * **Cell 5 (Stage 2 Training):** Unfreezes and fine-tunes the full model, saving the best version as `resnet50_dryfruits_best.keras`.

### 2. Running Inference (Predicting New Images)

1.  **Load Model:** In a new cell (ideally in the same notebook), load the saved model.
    ```python
    from tensorflow.keras.models import load_model
    
    model = load_model('resnet50_dryfruits_best.keras')
    
    # Get the class mapping from the training generator
    # (This requires 'train_generator' to still be in memory)
    class_indices = train_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    ```

2.  **Upload and Predict:** Use the provided inference code to upload a single image, preprocess it, and see the model's prediction.
    ```python
    from google.colab import files
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.resnet50 import preprocess_input
    import numpy as np
    
    # Upload an image
    uploaded = files.upload()
    test_image_path = list(uploaded.keys())[0]
    
    # Load and preprocess the image
    img = image.load_img(test_image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    
    # Make prediction
    prediction = model.predict(img_preprocessed)
    predicted_index = np.argmax(prediction[0])
    predicted_class_name = class_names[predicted_index]
    confidence = np.max(prediction[0])
    
    print(f"Prediction: {predicted_class_name} | Confidence: {confidence*100:.2f}%")
    ```