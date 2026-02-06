# Car Price Prediction App

A machine learning-powered web application that estimates the selling price of a used car based on its features. This project uses a K-Nearest Neighbors (KNN) regression model and is deployed using Streamlit.

## Overview
This application provides users with an instant valuation of a car by analyzing factors such as purchase year, present market price, mileage, and fuel type. It is designed for both buyers and sellers to understand the fair market value of a vehicle.


## Project Demo

<video src="https://github.com/user-attachments/assets/c644103d-59c1-42a2-af83-7be2ffaab098" width="400" controls>" width="600" controls muted>
</video>

## Check Out My WebApp
[Sample App](https://predict-my-car01.streamlit.app/)



## Tech Stack

Frontend: Streamlit.


Machine Learning: Scikit-learn (KNN Regressor).


Data Processing: NumPy and Pandas.


Model Serialization: Pickle.



## Project Structure

├── app.py                     
├── best_knn_model.pkl        
├── scaler.pkl               
├── requirements.txt         
└── README.md    



## Getting Started
Prerequisites
Ensure you have Python 3.8+ installed on your system.



## Installation
#### 1. Clone the repository:

#### Bash

git clone https://github.com/Devadetharu/CPA_Streamlit.git

cd CPA_Streamlit



#### 2. Install dependencies:

#### Bash
pip install -r requirements.txt



#### 3. Run the application:

#### Bash
streamlit run app.py



## How It Works

The app requires the following inputs from the user:

Year of Purchase: Used to calculate the age of the car.

Present Price: Current showroom price of the car (in ₹ Lakhs).

Kilometers Driven: Total distance covered by the vehicle.

Transmission: Manual or Automatic.

Fuel Type: Petrol, Diesel, or CNG.

Seller Type: Dealer or Individual.

The backend scales the numerical features using StandardScaler before passing them to the KNN model for prediction.



## Model Details

Model Type: K-Nearest Neighbors Regressor.


Scaling: Features are normalized to ensure the distance-based KNN algorithm performs optimally.


Features Included: Year, Present Price, Kms Driven, Owner, Fuel Type, Seller Type, and Transmission.
