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
Dataset 3 - https://data.melbourne.vic.gov.au/explore/dataset/house-prices-by-small-area-transfer-year/export/

Both datasets are in a CSV format and can be downloaded from the links provided. The authors of the original dataset claim that the data was scarped from domain.com and other real estate website API's. 
Data was merged and then processed to remove duplicates, NaN's and irrelevant columns were dropped. The method for this is shown in the source code.Third dataset is also in CSV format and is license by CC BY, published by data.melbourne.vic.gov.au and data provided by City of Melbourne. The third dataset is median prices for dwellings/townhouses, and apartments by their year of settlement for the City of Melbourne by CLUE Small area.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Model Training**

Training the Model
The model was trained using functions from the sklearn library to firstly split data into training and test sets and then relevant models were imported in. We used two models firstly a simple linear regression model and then a random forest algorithm. 

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Results**

To evaluate the linear regression model we used the scorer function from sk learn. It returns an R^2 value (coefficient of determintation) which shows how well the regression fit. The result was a r2 of 0.59 which is pretty average. 

Because the model didnt return a desirable r^2 score I chose to build a new random forest model that should evaluate better. The score for this model was 0.82


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Contact**
- Thomas - 102173577@student.swin.edu.au
- -Nitesh - 104484695@student.swin.edu.au
- Alex - 104268899@student.swin.edu.au
