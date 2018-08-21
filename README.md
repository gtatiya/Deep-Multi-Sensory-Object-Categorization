# Deep Multi-Sensory Object Categorization System Using Interactive Exploration For Humanoid Robot

**Abstract:**

> For a robot to operate in a human-inhabited environment, it is very important for it to understand key features of the objects in interacts with.
When identifying objects, humans typically rely not only on visual perception, but also perform physical interactions with them that allow understanding of the objects' non-visual properties.
Motivated by this cognitive process, we propose a method that enabled a robot to classify the category of 100 different household objects, each belonging to one of 20 categories by performing 9 exploratory interactions with the objects.
In this paper, we present deep learning techniques that perform better than previous methods by training Tensor-Train Gated Recurrent Unit for video data and Convolutional Neural Networks for audio and haptic data produced during object manipulation by a robot.
In most cases, purely video neural networks outperform audio and haptic neural networks suggesting that vision is the most important modality for category recognition.
Furthermore, we also propose a novel strategy to train a single neural network that inputs video, audio and haptic data, and demonstrate that its performance is better than separate neural networks for video, audio and haptic data.
Our approach enables the robot to recognize object category within only 0.1 second of interaction with high accuracy and outperforms the previous published baseline for the dataset used in our experiments.

<img src="pics/System.jpg" alt="drawing" width="600px"/>

## Dependencies

`Python 3.5.4` is used for development and following packages are required to run the code:<br><br>
`pip install tensorflow-gpu==1.8.0`<br>
`pip install keras==2.2.0`<br>
`pip install matplotlib==2.2.2`<br>
`pip install numpy==1.14.5`<br>
`pip install googledrivedownloader`

## Dataset

- [Visualization of each modalities](DatasetVisualization.ipynb)

## Video Classification

- [Training](VideoNetworkTraining.ipynb)
- [Testing](VideoNetworkTesting.ipynb)

## Sound Classification

- [Training](SoundNetworkTraining.ipynb)
- [Testing](SoundNetworkTesting.ipynb)

## Haptic Classification

- [Training](HapticNetworkTraining.ipynb)
- [Testing](HapticNetworkTesting.ipynb)

## Multimodal Classification

- [Training](MultimodalNetworkTraining.ipynb)
- [Testing](MultimodalNetworkTesting.ipynb)
