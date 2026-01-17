import tensorflow as tf
import os
import sys

# --- Configuration ---
# You can either set the file name here...
DEFAULT_H5_FILE = 'resnet50_dryfruits.h5'
# --- OR provide it as a command-line argument ---

def convert(h5_file_name):
    # Check if the file exists
    if not os.path.exists(h5_file_name):
        print(f"Error: File not found at '{h5_file_name}'")
        print("Please make sure the .h5 file is in the same directory.")
        return

    # 1. Define the new file name
    # This will turn 'my_model.h5' into 'my_model.keras'
    new_keras_file = os.path.splitext(h5_file_name)[0] + '.keras'

    try:
        # 2. Load the old model
        print(f"\nLoading '{h5_file_name}'...")
        model = tf.keras.models.load_model(h5_file_name)

        # 3. Save in the new format
        print(f"Saving new model as '{new_keras_file}'...")
        model.save(new_keras_file)

        print(f"\nConversion successful! New file saved as '{new_keras_file}'")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    # This checks if you provided a filename when running the script
    if len(sys.argv) > 1:
        h5_file_name = sys.argv[1]
    else:
        h5_file_name = DEFAULT_H5_FILE
    
    convert(h5_file_name)