# Health-ml-app
This project is an interactive web-based application built using Streamlit that predicts a person's health condition based on various medical and lifestyle inputs. It integrates multiple machine learning models to provide insights into different health risks including heart disease, diabetes, lung health, thyroid conditions.
Human Health Prediction System
📌 Overview

The Human Health Prediction System is an interactive web application built using Streamlit that leverages multiple machine learning models to analyze and predict various health conditions.

This project combines several trained ML models into a single unified interface, allowing users to input their health and lifestyle data and receive predictive insights for different diseases and overall health risk.

It demonstrates how machine learning can be applied in healthcare to assist in early risk detection and decision-making.

🚀 Live Demo

🔗 https://health-ml-app-w3trjt7crpjzw8h4of74zv.streamlit.app/
🧠 Problem Statement

Healthcare risk prediction often requires analyzing multiple factors such as lifestyle habits, medical history, and physiological parameters.

This project aims to:

Integrate multiple ML models into one platform
Provide a simple interface for health data input
Deliver quick and understandable health predictions


⚙️ Tech Stack
Category	Technology
Frontend	Streamlit
Backend	Python
Libraries	Pandas, NumPy, Scikit-learn, Pickle
Deployment	Streamlit Cloud / Local
ML Models	Multiple trained classification/regression models


🏗️ Project Structure
├── app.py                      # Main Streamlit application
├── heart_model (1).pkl        # Heart disease model
├── diabetes_model (1).pkl     # Diabetes prediction model
├── lung_model (1).pkl         # Lung health model
├── model.pkl                  # Thyroid model
├── model (1).pkl              # Cardiovascular risk model
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation


✨ Features
🔍 Multi-Model Predictions
Heart Health Prediction
Diabetes Risk Prediction
Lung Health Analysis
Thyroid Condition Prediction
General Cardiovascular Risk
📊 Interactive User Interface
Sliders, checkboxes, and input fields
Clean and simple UI
Real-time input handling
⚡ Instant Results
Predictions generated instantly
Output displayed with interpretation:
✅ Good health
⚠️ Moderate risk
🚨 High risk
🧩 Robust Model Handling
Automatically loads .pkl models
Uses fallback dummy models if files are missing
Handles errors gracefully
