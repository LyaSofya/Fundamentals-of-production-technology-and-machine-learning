# BiT-LR-91-83 Model

This repository contains the BiT-LR-91-83 model, a state-of-the-art image classification model based on the Big Transfer (BiT) approach. Due to GitHub's file size limitations, the model has been split into four parts.

## Model Description

**BiT-LR-91-83** is a model name in the context of Big Transfer (BiT). Big Transfer is an approach where a pre-trained model on a large dataset is used and then applied to tasks with smaller datasets through transfer learning.

### Specifically:
- **BiT**: Big Transfer
- **LR**: Large ResNet (a large variant of ResNet, where ResNet is a convolutional neural network architecture designed for computer vision tasks)
- **91**: The number of layers in the ResNet
- **83**: The version or revision of the model

BiT-LR-91-83 means that this model is based on a ResNet architecture with 91 layers, modified and optimized for specific tasks. The BiT approach leverages pre-training on large datasets to improve performance on downstream tasks with smaller datasets via transfer learning.

## Model Files

The model is split into four parts due to GitHub's file size restrictions. You will need to download all parts and combine them to use the model.

- `BiT-LR-91-83.part1.h5`
- `BiT-LR-91-83.part2.h5`
- `BiT-LR-91-83.part3.h5`
- `BiT-LR-91-83.part4.h5`

## Usage

### Combining the Model Files

To combine the model parts into a single file, use the following command:

```sh
cat BiT-LR-91-83.part* > BiT-LR-91-83.h5
```

### Loading the Model

Once you have combined the parts, you can load the model in Python using TensorFlow or Keras:

```python
from tensorflow.keras.models import load_model

model = load_model('BiT-LR-91-83.h5')
```

### Example Code

Here is an example of how to load and use the model:

```python
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model = load_model('BiT-LR-91-83.h5')

# Load and preprocess an image
img_path = 'path/to/your/image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.resnet.preprocess_input(img_array)

# Predict
predictions = model.predict(img_array)
print(predictions)
```

## Acknowledgements

The BiT-LR-91-83 model is based on the research by the Google Brain team. For more details, please refer to the original [Big Transfer (BiT) paper](https://arxiv.org/abs/1912.11370).
