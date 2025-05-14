🌾 Crop Recommendation System
Developed by: Noel Ninan Sheri
🔗 GitHub Repository: CropRecommendationModel
Streamlit App: https://crop-recommendation-model-noel-ninan.streamlit.app
📌 Introduction
Agriculture is the backbone of our economy, and selecting the right crop for given soil and environmental conditions is crucial for maximizing yield. The Crop Recommendation System is a machine learning–powered web application that intelligently predicts the most suitable crop to grow based on real-time environmental and soil parameters.

This system enables smart farming by helping farmers and agricultural researchers make informed decisions, reducing risks and increasing productivity.

🔍 Problem Statement
"Given specific input parameters such as soil type, pH level, temperature, humidity, NPK values, and other environmental conditions, recommend the most suitable crop to cultivate."

🚀 Key Features
✅ Predicts the best crop using ML algorithms

🌐 User-friendly web app interface (built with Streamlit)

📊 Accepts soil type, NPK values, pH, temperature, humidity, wind speed, crop yield, and soil quality as inputs

🔁 Model trained on real-world dataset from Kaggle

📈 Scaled features and label encoding handled

📄 Transparent output and intuitive UI with Times New Roman font styling

🧠 Machine Learning Techniques
Model Used: Random Forest Classifier

Label Encoding: For categorical crop labels

Feature Scaling: StandardScaler from Scikit-learn

Model File: Serialized using joblib for easy deployment

📦 Tech Stack
Layer	Tools/Tech Used
🧠 ML Model	Scikit-learn, Pandas, NumPy
🌐 Frontend	Streamlit (Python web framework)
🧰 Deployment	Streamlit Cloud
📊 Visualization	Matplotlib, Seaborn (optional for EDA)
📁 Version Control	Git & GitHub

📁 File Structure
CropRecommendationModel/
├── app.py                    # Main Streamlit application
├── crop_predictor.pkl        # Trained ML model
├── scaler.pkl                # StandardScaler object
├── crop_label_encoder.pkl    # LabelEncoder object for crop names
├── requirements.txt          # Required libraries
├── README.md                 # Project overview
└── sample_dataset.csv        # Sample dataset (optional for testing)
🧪 Input Parameters
Parameter	Type	Description
Soil Type	Categorical (Loamy, Sandy, etc.)	Nature of the soil
Soil pH	Float	Acidity/alkalinity of soil
Temperature	Integer	°C
Humidity	Integer	%
Wind Speed	Integer	km/h
Nitrogen (N)	Integer	ppm
Phosphorus (P)	Integer	ppm
Potassium (K)	Integer	ppm
Crop Yield	Integer	kg/ha
Soil Quality	Float	Calculated index (0–100 scale)

📊 Dataset Used
📁 Kaggle Source: Crop Prediction Dataset

Number of features: 10
Number of Rows: 32,620
Number of classes: Multiple crops like wheat, rice, sugarcane, cotton, etc.

🎯 How It Works
User enters environmental and soil data

Inputs are scaled using a trained scaler

Model predicts the crop index

Encoded label is converted back to crop name

Output displayed with recommendation

🖥 Deployment Instructions
🔹 Local Deployment
bash
Copy
Edit
git clone https://github.com/NoelNinanSheri1307/CropRecommendationModel.git
cd CropRecommendationModel
pip install -r requirements.txt
streamlit run app.py
🔹 Deploy on Streamlit Cloud
Push project to GitHub

Go to Streamlit Cloud

Link your GitHub repo

Set app.py as entry point

Add dependencies in requirements.txt

⚠️ Don't forget to include joblib, streamlit, scikit-learn, numpy, and pandas in your requirements.txt.

✨ UI Highlights
Clean and intuitive interface

Times New Roman font styling

Centralized prediction button

“Developed by Noel Ninan Sheri” signature under the title

📜 Future Enhancements
📱 Mobile responsive design

🌐 Multi-language support

🛰 Integration with live weather APIs

📍 Nearby fertilizer store & soil testing center locator (via Google Maps API)

🧠 Transfer learning models for higher accuracy

🙌 Acknowledgements
Kaggle Datasets

Streamlit for deployment

Scikit-learn documentation

VS Code and Python community

👨‍💻 About the Developer
Noel Ninan Sheri
M.Tech Integrated Software Engineering Student, VIT Vellore
📍 Passionate about AI/ML, Web Development, and solving real-world problems
🔗 LinkedIn Profile: https://www.linkedin.com/in/noel-ninan-sheri/
💻 GitHub

