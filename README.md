# Yoga Pose Recognition 

An AI-powered solution to identify yoga poses and provide feedback on alignment and accuracy. This project was developed as part of an AI-themed proof of concept for enhancing the yoga experience.

---

## Objective 
This project implements **Pose Detection & Correction**, which recognizes yoga poses and lays the foundation for providing feedback for alignment and accuracy. The system leverages deep learning and computer vision to identify various yoga postures from images.

---

## Dataset 

The dataset for this project was sourced from Kaggle:
[Yoga Posture Dataset](https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset)

The dataset contains images of various yoga poses categorized into folders for each pose:
- **Adho Mukha Svanasana**
- **Halasana**
- **Trikonasana**
- **Other Poses**

To use this dataset:
1. Download the dataset from the Kaggle link.
2. Extract it and structure it as follows:
   ```
   dataset_path/
   ├── Adho Mukha Svanasana/
   ├── Halasana/
   ├── Trikonasana/
   └── Other_Poses/
   ```

---

## Getting Started 

### Virtual Environment Setup
1. Create a virtual environment using `venv` or `conda`:
   ```bash
   python -m venv /path/to/your/environment/
   ```

2. Activate the environment:
   - **Windows**:
     ```bash
     .\path\to\your\environment\Scripts\activate
     ```
   - **Mac/Linux**:
     ```bash
     source /path/to/your/environment/bin/activate
     ```

3. Install the required libraries:
   ```bash
   pip install notebook tensorflow opencv-python numpy matplotlib scikit-learn
   ```

---

### Installation of Dependencies
Run the following command to install all dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Running the Project 

### Jupyter Notebook Workflow
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open `Yoga_Pose_Recognition.ipynb` in your browser.

3. Follow the notebook workflow step by step:
   - **Data Preprocessing**: Load the dataset and preprocess the images.
   - **Model Training**: Train a Convolutional Neural Network (CNN) model for pose classification.
   - **Pose Prediction**: Test the trained model on new images to identify yoga poses.
   - **Model Evaluation**: Evaluate model performance using validation data.

---

## Features 
- **Pose Detection**: Accurately classifies yoga poses like:
  - Adho Mukha Svanasana
  - Halasana
  - Trikonasana
  - Other Poses
- **Feedback**: Lays groundwork for providing feedback on posture alignment and accuracy.
- **Interactive Workflow**: Jupyter Notebook-based interface for easy execution and visualization.

---

## Results 

- **Model Accuracy**: Achieved an accuracy of **XX%** on the validation dataset.
- **Sample Predictions**: Successfully identified yoga poses from test images with high confidence.

---

## Future Enhancements 
- **Real-Time Detection**: Integrate with a webcam for real-time yoga pose recognition.
- **Feedback System**: Add functionality to provide alignment and improvement suggestions.
- **App Integration**: Build an API to integrate this feature into a mobile yoga application.
- **Expanded Dataset**: Incorporate more yoga poses and variations for a comprehensive model.

---

## How to Use 

### Training the Model
Train the model using the following command in Jupyter Notebook:
1. Prepare your dataset path in the notebook.
2. Run the cells under the "Training" section.

### Predicting Yoga Poses
Use a sample image for prediction:
1. Place the image in the appropriate folder.
2. Run the cells under the "Pose Prediction" section in the notebook.

---

## Contributing 
Contributions are welcome! Feel free to fork the repository and submit pull requests to:
- Improve the model's accuracy.
- Add new features like real-time pose detection.
- Enhance the documentation.

---

## License 
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments 
- **TensorFlow** and **Keras** for their powerful deep learning tools.
- **OpenCV** for efficient image processing.
- **Kaggle** for providing the [Yoga Posture Dataset](https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset).
- Yoga practitioners for inspiring this innovative AI project.

---
