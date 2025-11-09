import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2

MODEL_PATH = 'resnet50_dryfruits.keras'
IMG_PATH   = 'almond_A_001.jpg'
IMG_SIZE   = 224

# --- load model and image ---
model = load_model(MODEL_PATH, compile=False)

img = image.load_img(IMG_PATH, target_size=(IMG_SIZE, IMG_SIZE))
x = image.img_to_array(img).astype("float32")
x = np.expand_dims(x, axis=0)

from tensorflow.keras.applications.resnet50 import preprocess_input
x = preprocess_input(x.copy())

def prepare_inputs(m, arr):
    try:
        name = m.input_names[0]
        return {name: arr}
    except Exception:
        return arr

x_in = prepare_inputs(model, x)

# --- pick conv layers: 1st, 4th, 17th (fallback to last if fewer) ---
conv_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
if not conv_layers:
    raise ValueError("No Conv2D layers found.")

idxs = [0, 3, 16]
idxs = [min(i, len(conv_layers)-1) for i in idxs]
sel_layers = [conv_layers[i] for i in idxs]

# --- build activation model and run ---
act_model = tf.keras.Model(inputs=model.inputs, outputs=[l.output for l in sel_layers])
acts = act_model(x_in, training=False)  # list of tensors [1,H,W,C]

def to_featuremap(t):
    t = tf.convert_to_tensor(t)
    t = tf.reduce_mean(t, axis=-1)        # [1,H,W]
    t = t[0]
    t = t - tf.reduce_min(t)
    t = t / (tf.reduce_max(t) + 1e-8)
    return t.numpy()

fmaps = [to_featuremap(a) for a in acts]

# --- render like the sample (no predictions shown) ---
# Load original (BGR->RGB) only for display
orig = cv2.imread(IMG_PATH)
orig = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

titles = ["INPUT", "FIRST LAYER FEATURES", "FOURTH LAYER FEATURES", "SEVENTEEN LAYER FEATURES"]

plt.figure(figsize=(14,4))

plt.subplot(1,4,1)
plt.imshow(orig); plt.title(titles[0]); plt.axis('off')

for i, fmap in enumerate(fmaps, start=2):
    plt.subplot(1,4,i-0)  # positions: 2,3,4
    plt.imshow(fmap, cmap='viridis')
    plt.title(titles[i-1])
    plt.axis('off')

plt.tight_layout()
plt.show()
