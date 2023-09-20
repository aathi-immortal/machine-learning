import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Sample input data
data = [
    {"text": "Both these are decoctions advised in Jvara (fever) and Kasa (cough).", "medicine": "Punarnavadi Kashaya", "disease": "Fever, Cough"},
    {"text": "But in a patient with upper respiratory tract infections like common cold (Pinasa), Vyaghryadi would suit better than Punarnavadi.", "medicine": "Vyaghryadi Kashaya", "disease": "Common Cold"},
    {"text": "Whereas in a condition associated with inflammatory changes all over the body, Punarnavadi would be the appropriate choice.", "medicine": "Punarnavadi Kashaya", "disease": "Inflammatory Conditions"}
]

# Create a DataFrame from the input data
df = pd.DataFrame(data)

# Split the data into features (X) and labels (y)
X = df['text']
y = df['disease']

# Encode the labels into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Create an XGBoost classifier
model = xgb.XGBClassifier(objective="multi:softmax", num_classes=len(df['disease'].unique()), random_state=42)

# Train the model on the TF-IDF vectors and encoded labels
model.fit(X_tfidf, y_encoded)

# Now, you can use the trained model to predict the disease for new text descriptions of medicines
new_text = ["common "]
new_text_tfidf = vectorizer.transform(new_text)
predicted_disease_encoded = model.predict(new_text_tfidf)

# Decode the predicted disease back to its original label
predicted_disease = label_encoder.inverse_transform(predicted_disease_encoded)

# Create a mapping of medicines to diseases
medicine_to_disease = dict(zip(df['medicine'], predicted_disease))

# Print the mapping
print("Medicine to Disease Mapping:")
for medicine, disease in medicine_to_disease.items():
    print(f"Medicine: {medicine} => Disease: {disease}")
