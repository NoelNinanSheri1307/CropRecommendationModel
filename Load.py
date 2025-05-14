import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

df = pd.read_csv("Crop_Dataset.csv")  

print(df.info())
print(df.head())
print(df.describe())
print("Missing values:\n", df.isnull().sum())

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

df_numeric = df.select_dtypes(include=['float64', 'int64'])


correlation_matrix = df_numeric.corr()


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

le_crop = LabelEncoder()
df['Crop_Type'] = le_crop.fit_transform(df['Crop_Type'])

le_soil = LabelEncoder()
df['Soil_Type'] = le_soil.fit_transform(df['Soil_Type'])

sns.countplot(x='Crop_Type', data=df)
plt.xticks(rotation=90)
plt.title("Crop Type Distribution")
plt.show()

df.hist(figsize=(12, 8))
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

X = df.drop(['Crop_Type','Date'], axis=1)
y = df['Crop_Type']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"\n{name} model trained.")

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"{name} Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

rf_scores = cross_val_score(models["Random Forest"], X_scaled, y, cv=5)
print("\nRandom Forest Cross-Validation Accuracy:", rf_scores.mean())

joblib.dump(models["Random Forest"], "crop_predictor.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_crop, "crop_label_encoder.pkl")
print("\n✅ Model saved successfully!")
def predict_crop():
    print("\nEnter Environmental Conditions:")
    
    features = []
    for col in X.columns:
        while True:
            try:
                val = float(input(f"{col}: "))
                features.append(val)
                break  
            except ValueError:
                print("Invalid input. Please enter a valid number.")
    features_scaled = scaler.transform([features])
    
    model = joblib.load("crop_predictor.pkl")
    crop_pred = model.predict(features_scaled)
    
    le = joblib.load("crop_label_encoder.pkl")
    predicted_crop = le.inverse_transform(crop_pred)
    
    print(f"\n🌾 Recommended Crop: {predicted_crop[0]}")
predict_crop()

