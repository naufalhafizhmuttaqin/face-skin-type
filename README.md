### Skin Type Classification
This dataset contains images of faces with different skin types, such as:
- Normal 😊
- Oily 💦
- Dry 🌵
- Acne 😓

Kaggle Dataset = https://www.kaggle.com/datasets/muttaqin1113/face-skin-type

I employed a Convolutional Neural Network (CNN) model to classify skin types and provide skincare recommendations based on the classification results. The methodology includes data preprocessing, model training, and evaluation.
The training set is used to train the model, while the test set is used to evaluate the model's final performance. Next, the training set is further divided into a training set and a validation set. The model is then fine-tuned using the validation set. This process repeats until the model reaches optimal performance. Finally, the predictive model is evaluated using the test set to estimate its performance on new data.

### Model Architecture
A Convolutional Neural Network (CNN) model is used to classify the dataset categories. The model architecture can be customized based on requirements, but for this project, the following architecture is used:
- **Input Layer**
- **Convolutional Layers**: Used for feature extraction from the images.
- **Max Pooling Layers**: Used for dimensionality reduction of the features.
- **Flatten Layer**: Flattens the features into a vector.
- **Fully Connected Layers**: Perform classification tasks.
- **Output Layer**: Outputs the classification predictions based on labels.

### Results
The model was evaluated using a testing generator, resulting in the following accuracy and loss metrics:

| Metric       | Value   |
|--------------|---------|
| Test Loss    | 0.3207  |
| Test Accuracy| 0.9034  |

### Requirements
- **Python**
- **Keras** 
- **TensorFlow** 
- **NumPy** 
- **Matplotlib** 
- **sckit-learn**
- **pandas**
- **seaborn**
