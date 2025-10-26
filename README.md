GDP Prediction Project

This project implements a machine learning model to predict GDP per capita for countries based on various socio-economic indicators. The project includes a deployed Streamlit app for interactive predictions.

🌐 Live Demo

Try the GDP Predictor on Streamlit:
https://mzb2nv6jxmrvcft5zbarkt.streamlit.app/

📂 Project Structure
endtoend(mlflow)/
├── app.py              # Streamlit app for predictions
├── models.pkl          # Trained machine learning models
├── scaler.pkl          # Feature scaler
├── pt.pkl              # PowerTransformer to reduce skew
├── requirements.txt    # Project dependencies
└── countries of the world.csv  # Dataset

⚡ Features

Predicts GDP per capita using a pre-trained machine learning model

Handles data preprocessing, including numeric conversion, skew reduction, and scaling

Interactive web interface for real-time predictions

Supports cloud deployment via Streamlit

📊 Dataset

The project uses the "Countries of the World" dataset, which includes:

Population metrics

Geographic data (Area, Coastline)

Economic indicators (GDP, Net migration)

Social statistics (Literacy, Infant mortality, Phones per 1000)

Agriculture, Industry, and Service contributions

💻 Local Setup

Install dependencies:

pip install -r requirements.txt
pip install streamlit


Run the Streamlit app:

streamlit run app.py


The app will open in your default browser.

🧠 Model Information

Pre-trained RandomForestRegressor stored in models.pkl

Feature scaling handled with scaler.pkl

Skew reduction using PowerTransformer stored in pt.pkl

Input features are automatically processed and scaled before prediction

🌍 Deployment

The project is deployed using Streamlit Cloud:
https://mzb2nv6jxmrvcft5zbarkt.streamlit.app/

🛠 Requirements

Python 3.7+

pandas

numpy

scikit-learn

streamlit

✅ Conclusion

This GDP Prediction project provides a simple and interactive way to estimate GDP per capita for any country based on socio-economic and geographic indicators. By leveraging pre-trained machine learning models, data preprocessing, and an easy-to-use Streamlit interface, users can quickly explore predictions and understand the impact of different factors on GDP.