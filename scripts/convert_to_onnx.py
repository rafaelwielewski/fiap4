
import os
import tf2onnx
import tensorflow as tf
import onnx
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "" 

def convert_to_onnx():
    artifacts_dir = 'artifacts'
    keras_model_path = os.path.join(artifacts_dir, 'final_model.keras')
    saved_model_dir = os.path.join(artifacts_dir, 'temp_saved_model')
    onnx_model_path = os.path.join(artifacts_dir, 'final_model.onnx')

    print(f"Loading Keras model from {keras_model_path}...")
    model = tf.keras.models.load_model(keras_model_path)

    print(f"Exporting to SavedModel at {saved_model_dir}...")
    if os.path.exists(saved_model_dir):
        shutil.rmtree(saved_model_dir)
    
    model.export(saved_model_dir)

    print("Converting SavedModel to ONNX...")
    command = f"python -m tf2onnx.convert --saved-model {saved_model_dir} --output {onnx_model_path} --opset 13"
    print(f"Running: {command}")
    exit_code = os.system(command)
    
    if exit_code == 0:
        print("Conversion successful!")
        shutil.rmtree(saved_model_dir)
    else:
        print("Conversion failed!")

if __name__ == "__main__":
    convert_to_onnx()
