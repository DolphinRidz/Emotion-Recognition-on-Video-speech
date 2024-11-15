
# Emotion Recognition On Video Speech

This project aims to To build a Machine Learning model that can process video speech signals and detect human emotions like Happiness, Anger, Sadness, Surprise, etc. The main processes involved are Preprocessing audio chunks from You Tube Videos, Extract the acoustic features and classify them discretely using a pre-trained model with the help of a pre-trained dataset.

## Table of Contents
- [Methadology](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Workflow](#project-workflow)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Methadology
- Creating a dataset : A dataset of audio chunks is created from video files downloaded.
- Training a pre – trained dataset : RAVDESS dataset is used to train the model.
- Applying trained model: The trained model is applied to the dataset created to get the emotion distribution in videos.
- MLP:  Multi-layer Perceptron Classifier is used in this project for more accuracy.


## Dataset
The dataset contains audio chunks prepared from YouTube videos.

RAVDESS dataset is also used to train the model.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/DolphinRidz/Emotion-Recognition-on-Video-speech.git
   cd rainfall-prediction
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Project Workflow
The project is divided into the following steps:
1. **Data Preparation**:
   - Download YouTube Videos
   - Convert to .wav format
   - Clip Audio
   - Extract Audio Chunks
   - Cleaning and masking Audio Chunks
2. **Data Visualization**:
   - Play Audio
   - Plotting Graphs
3. **Training Model**:
   - Extract features like MFCC, MEL, CHROMA from audios.
   - Select Emotions
   - Split into train and test data
   - Apply MLP model
   - Predict Test Set
   - Check Accuracy
   - Save the model
4. **User Interface**:
   - Use Flask
   - Upload Audio chunk
   - Predict Data
5. **Model Evaluation**:
   - Emotion Visualization

## Usage
The main purpose of this application is to identify the emotions of the speaker in
the video from their speeches. Instead of predicting emotions on acted datasets
present online, a self-made dataset has been created for prediction and analysis of
video speech. This shall allow to find out the underlying emotions of the speaker
at a given point of time during entire video. This will also lead to get the
dominating emotions of the Speakers in a video.

## Results
The accuracy rate of 81.82% is achieved with MLP model and 4 emotions i.e... Happy, Calm, Angry and Disgust. The accuracy rate differs for the combination of different Emotions. The emotion of the speaker changes in audio chunks over a period of time in video. Emotions like Happy and Angry is dominated in most videos.

## License
This project is licensed under the MIT License.
