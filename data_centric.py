# SOURCE: https://github.com/Anasb73/wake_vision_challenge_2_data_centric_track/blob/main/data_centric.py

import tensorflow as tf
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from enhacing_data.random_eraser import get_random_eraser

model_name = 'wv_quality_mcunet-320kb-1mb_vww'

input_shape = (144,144,3)
color_mode = 'rgb'
num_classes = 2

batch_size = 128
epochs = 50
learning_rate = 0.001

path_to_training_set = './wake_vision/train_quality'
path_to_validation_set = './wake_vision/validation'
path_to_test_set = './wake_vision/test'

inputs = tf.keras.Input(shape=input_shape)
#
x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(inputs)
x = tf.keras.layers.Conv2D(16, (3,3), padding='valid', strides=(2,2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(8, (1,1), padding='valid')(x)
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(24, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid', strides=(2,2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = tf.keras.layers.Conv2D(16, (1,1), padding='valid')(x)
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(80, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(16, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(96, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(16, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(48, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(16, (1,1), padding='valid')(x)
# add
x = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(80, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(3,3))(x)
x = tf.keras.layers.DepthwiseConv2D((7,7),  padding='valid', strides=(2,2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = tf.keras.layers.Conv2D(24, (1,1), padding='valid')(x)
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(96, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(24, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(96, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(24, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(144, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(x)
x = tf.keras.layers.DepthwiseConv2D((7,7),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(24, (1,1), padding='valid')(x)
# add
x = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(144, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid', strides=(2,2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = tf.keras.layers.Conv2D(40, (1,1), padding='valid')(x)
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(240, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(3,3))(x)
x = tf.keras.layers.DepthwiseConv2D((7,7),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(40, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(160, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(40, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(200, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(40, (1,1), padding='valid')(x)
# add
x = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(200, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = tf.keras.layers.Conv2D(48, (1,1), padding='valid')(x)
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(144, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(48, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(192, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(48, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(144, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(48, (1,1), padding='valid')(x)
# add
x = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(192, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid', strides=(2,2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = tf.keras.layers.Conv2D(96, (1,1), padding='valid')(x)
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(480, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(96, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(384, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(96, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(384, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(96, (1,1), padding='valid')(x)
# add
x = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(480, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(160, (1,1), padding='valid')(x)
#
x = tf.keras.layers.AveragePooling2D(5)(x)
x = tf.keras.layers.Conv2D(2, (1,1), padding='valid')(x)
outputs = tf.keras.layers.Reshape((num_classes,))(x)

model = tf.keras.Model(inputs, outputs)

#compile model
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=opt,
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

#load dataset
width_shift = 0.1
height_shift = 0.1
img_h = 144
img_w = 144
shuffle = True
seed = 11
class_mode = 'categorical' 

width_shift = 0.1
height_shift = 0.1
img_h = 144
img_w = 144
shuffle = True
seed = 11
class_mode = 'categorical' 

data_generator_training = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=width_shift,  
    height_shift_range=height_shift,
    shear_range=10,
    zoom_range=[0.8, 1.2],
    brightness_range=(0.5, 1.5),
    channel_shift_range=20.0,
    horizontal_flip=True,
    preprocessing_function=get_random_eraser(pixel_level=True)
)

data_generator_test =  ImageDataGenerator(
    rescale=1./255
    )

training_iterator = tf.keras.preprocessing.image.DirectoryIterator(
    path_to_training_set, data_generator_training, target_size= (img_h, img_w),
    color_mode= color_mode, classes=None, class_mode= class_mode,
    batch_size= batch_size, shuffle=shuffle, seed=seed, data_format='channels_last',follow_links=False,
    interpolation='bilinear', dtype="float32"
)

validation_iterator = tf.keras.preprocessing.image.DirectoryIterator(
    path_to_validation_set, data_generator_test, target_size= (img_h, img_h),
    color_mode= color_mode, classes=None, class_mode=class_mode,
    batch_size= batch_size, seed=seed, data_format='channels_last',follow_links=False,
    interpolation='bilinear', dtype="float32"
)


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= model_name + ".keras",
    monitor='val_accuracy',
    mode='max', save_best_only=True)

model.fit(training_iterator, epochs=epochs, validation_data=validation_iterator, callbacks=[model_checkpoint_callback])
 
model = tf.keras.models.load_model(model_name + ".keras")

def representative_dataset():
    num_samples = 150
    count = 0

    while count < num_samples:
        batch_images, _ = training_iterator.next()
        batch_size = batch_images.shape[0]

        for i in range(batch_size):
            if count >= num_samples:
                break

            image = tf.cast(batch_images[i], tf.float32)
            image = tf.expand_dims(image, axis=0)

            yield [image]
            count += 1

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.int8
converter.inference_output_type = tf.uint8  # or tf.int8
tflite_quant_model = converter.convert()

with open(model_name + ".tflite", 'wb') as f:
    f.write(tflite_quant_model)
        
#Test quantized model
interpreter = tf.lite.Interpreter(model_name + ".tflite")
interpreter.allocate_tensors()

output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory= path_to_test_set,
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
