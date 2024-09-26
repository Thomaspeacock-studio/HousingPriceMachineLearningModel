**Housing Price Machine Learning project.**
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
This model has been developed to provide predictive capabilities for the Australian housing market. It takes in two datasets merges them and applies data cleaning and preprocessing techniques. 
From there it splits the dataset into training and test models and uses the pandas and sklearn and other libraries to select an appropritate model and build the model.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Table of Contents**
Project Overview
Environment Setup
Dataset Description
Model Training
Prediction
Results
Contributing

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Project Overview**

Provide a high-level summary of the project, including the main goal, model architecture, and key features. Mention any specific libraries or technologies used (e.g., TensorFlow, PyTorch, etc.).

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Environment Setup**

Prerequisites
List all necessary software and dependencies:

Python >= 3.7
Libraries: numpy, pandas, tensorflow, scikit-learn, etc.
Installation
Provide step-by-step instructions on how to install dependencies and configure the environment. You can add a requirements.txt file if necessary.


1. Open IDE and create Project Directory or Navigate to Project Directory
cd [/Users/path/to/your/directory]

2. Clone the repository into IDE
git clone https://github.com/Thomaspeacock-studio/HousingPriceMachineLearningModel.git

3. Install virtualenv if you don't have it
pip install virtualenv

4. Create a virtual environment
virtualenv venv

5. Activate the environment (Linux/Mac)
source venv/bin/activate

6. Activate the environment (Windows)
.\venv\Scripts\activate

7. Install the required packages
python>=3.7
numpy>=1.18.0
pandas>=1.0.0
scikit-learn>=0.24.0
jupyterlab>=3.0.0
matplotlib>=3.2.0
seaborn>=0.10.0



------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Dataset Description**

Describe the dataset used for training, testing, and validation. Include information such as:

The model was trained on two datasets obtained from Kaggle, the links are below. 
Dataset 1 - https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market
Dataset 2 - https://www.kaggle.com/datasets/amalab182/property-salesmelbourne-city

Both datasets are in a CSV format and can be downloaded from the links provided. The authors of the original dataset claim that the data was scarped from domain.com and other real estate website API's. 
Data was merged and then processed to remove duplicates, NaN's and irrelevant columns were dropped. The method for this is shown in the source code.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Model Training**

Training the Model
Provide a detailed explanation of how to train the model, including all required parameters and commands.

bash
Copy code
# Command to run the training script
python train.py --data_dir ./data --epochs 50 --batch_size 32 --learning_rate 0.001
Explain what each command and parameter does:

--data_dir: Directory where the dataset is stored
--epochs: Number of training epochs
--batch_size: Size of the training batch
--learning_rate: Learning rate for the optimizer
Model Checkpoints
Mention if the model saves checkpoints during training and where users can find them.


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Prediction**

How to Use the Model for Prediction
Provide instructions on how to use the trained model for making predictions. Include example commands:

bash
Copy code
# Command to run the prediction script
python predict.py --model_path ./models/best_model.pth --input_data ./data/test.csv
Explain the parameters:

--model_path: Path to the saved model file
--input_data: Path to the input data for predictions
Include an example of the expected output format.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Results**

Briefly discuss how the model performed, including metrics (accuracy, F1-score, etc.) and any charts or graphs if applicable. You can link to a more detailed report or results page if necessary.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Contact**
- Thomas - 102173577@student.swin.edu.au
- -Nitesh - 104484695@student.swin.edu.au
- Alex - 104268899@student.swin.edu.au
