## **Driver's Drowsiness Detection System**

This project presents a basic model for a Driver’s Drowsiness Detection System, designed to determine whether a driver’s eyes are open or closed. If the eyes remain closed for a specified duration, the system activates an alert sound to warn the driver, helping to prevent accidents caused by drowsiness.
The model leverages Convolutional Neural Networks (CNNs) to extract features from video frames captured by a webcam. The extracted features are passed through dense layers to classify the frames, identifying whether the eyes are open or closed. The steps involved in this project are as follows:

1. Capture Input Images:
- Use a webcam to collect real-time images.
2. Detect Face and Eyes:
- Utilize pre-trained HAAR model (machine learning object detection model) for face and eye detection in the captured frames.
3. Create Region of Interest (ROI):
- Extract the ROI for the detected face and eyes.
4. Classify Using Deep Learning Model:
- Pass the ROI to the trained deep learning model to classify whether the eyes are open or closed.
5. Monitor Driver’s State:
- Calculate the time duration for which the eyes remain closed and trigger an alert if this duration exceeds a safe threshold.

**Dataset and Training**
The model is trained on the **YAWNN Dataset**, which provides labeled images of eyes in both open and closed states. The training process involved fine-tuning a CNN architecture to achieve high accuracy in real-time classification.

**Model Architecture**
The model architecture consists of the following layers:

1. Convolutional Layers:
- 3 layers, each with 64 nodes and a kernel size of 3x3.
2. Fully Connected Layers:
- One dense layer with 128 nodes for feature interpretation.
- An output layer with 2 nodes for binary classification (open/closed).

The ReLU activation function is used in all layers except the output layer, where the Softmax activation function is applied to produce probabilities for the two classes.

When the driver's eyes are detected as closed, a timer will start counting. If the eyes remain closed for more than 10 seconds, an alarm will activate to alert the driver!

<img width="627" alt="image" src="https://github.com/user-attachments/assets/aaee4fd4-3a57-4315-9c1d-7f199d80e621" />

The timer will start decreasing once the driver returns to an active state, i.e., when their eyes are open.

<img width="628" alt="image" src="https://github.com/user-attachments/assets/831d33cb-343a-461d-9de6-50c4cfe909f0" />

