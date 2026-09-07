import tensorflow as tf
import numpy as np
import os

input_shape = (50,50,3) # Change for different models
color_mode = 'rgb'
num_classes = 2

path_to_training_set = './wake_vision/train_quality'
path_to_validation_set = './wake_vision/validation'
path_to_test_set = './wake_vision/test'
model_path = "model_zoo/wv_quality_anas_benalla.tflite"

#Test quantized model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory= path_to_validation_set, # change later
    labels='inferred',
    label_mode='categorical',
    color_mode=color_mode,
    batch_size=1,
    image_size=input_shape[0:2],
    shuffle=True,
    seed=11
)

correct = 0
wrong = 0

for image, label in test_ds :
    image = image / 255
    # Check if the input type is quantized, then rescale input data to uint8
    if input['dtype'] == tf.uint8:
       input_scale, input_zero_point = input["quantization"]
       image = image / input_scale + input_zero_point
       input_data = tf.dtypes.cast(image, tf.uint8)
       interpreter.set_tensor(input['index'], input_data)
       interpreter.invoke()
       if label.numpy().argmax() == interpreter.get_tensor(output['index']).argmax() :
           correct = correct + 1
       else :
           wrong = wrong + 1
print(f"\n\nTflite model test accuracy: {(correct/(correct+wrong)*100)}\n\n")
