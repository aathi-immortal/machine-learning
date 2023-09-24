import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

# Load your labeled dataset from the CSV file
data = pd.read_csv("E:/challenge/SIH/machine-learning/DATASET.csv")  # Replace "your_dataset.csv" with the actual path to your CSV file

# Split the data into features (X) and labels (y)
X = data['Symptoms']
y = data['Name of Medicine']

# Encode the labels into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Create a Gradient Boosting classifier
model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train the model on the TF-IDF vectors and encoded labels
model.fit(X_tfidf, y_encoded)

# Define a function to extract medicines and diseases from a text
def extract_medicines_and_diseases(text):
    # Vectorize the input text
    text_tfidf = vectorizer.transform([text])
    print(text_tfidf)
    # Predict the medicine for the input text
    # print(type(text_tfidf))
    predicted_medicine_encoded = model.predict(text_tfidf)
    tem = label_encoder.inverse_transform(text_tfidf)
    print(tem)
    
    print(predicted_medicine_encoded)
    predicted_medicine = label_encoder.inverse_transform(predicted_medicine_encoded)
    print(predicted_medicine)
    # Find the diseases associated with the predicted medicine
    diseases_with_medicine = data[data['Name of Medicine'] == predicted_medicine[0]]
    
    # Extract the list of diseases and their corresponding medicine
    disease_medicine_list = []
    for index, row in diseases_with_medicine.iterrows():
        disease_medicine_list.append({"Disease": row['Symptoms'], "Recommended Medicine": row['Name of Medicine']})
    
    return predicted_medicine[0], disease_medicine_list

# Provide a new text for extraction
new_text =  "Ayurveda has a large database of single herbs, minerals, and formulations that have been tailormade to suit each individual, his/her psychosomatic constitution, clinical condition, comorbidities, age, region, etc. These data are spread over more than 150 texts, amidst manuscripts in multiple languages and scripts. With the rise of transcriptional and translational facilities, several traditional medicinal texts are now available in their digitized forms. But for an Ayurvedic student or practitioner, exploring this multitude of literature for identifying their drug of choice' often becomes tedious and impractical. Here is the need of a custom software that can identify the apt formulation that has been designed to treat a constellation of symptoms and present it to the student/practitioner along with its reference and other desired properties. For example, the two formulations Punarnavadi Kashaya and Vyaghryadi Kashaya are mentioned in texts as follows: Both these are decoctions advised in Jvara (fever) and Kasa (cough). But in a patient with upper respiratory tract infections like common cold (Pinasa), Vyaghryadi would suit better than Punarnavadi. Whereas in a condition associated with inflammatory changes all over the body. I Punarnavadi would be the appropriate choice. The objective of the proposed software is to identify the single drugs and formulations that suit a set of symptoms. Certain ingredients (eg. jaggery) are unsuitable for certain categories of patients (e.g. diabetics). There are also medicine mediums that are unsuitable for specific diseases (e.g. fermented/alcoholic preparations in diabetes). Such information is also expected to be conveyed to the learner or practitioner who uses the software. The same disease has been mentioned in different names (E.g. Jvara and Santapa for fever) and the same word has been used to denote different (Eg. Abhaya generally means Terminalia chebula but in the context of Jatyadi ghrita, it means Vetiveria zizanioides. The multiple names of same diseases are expected to be included in the tags of each formulation. The sources for the formulations, and synonyms and similar words have been included in the data section. It is also desirable to include the Ayurvedic pharmacological properties of the single drugs, and the compound formulation (called Rasa, Guna, Virya, Vipaka, etc.) as and where available."
# Extract the medicine and list of diseases
medicine, disease_medicine_list = extract_medicines_and_diseases(new_text)

# Print the extracted medicine and list of diseases
print("Extracted Medicine:", medicine)
print("List of Diseases and Recommended Medicines:")
for item in disease_medicine_list:
    print(f"Disease: {item['Disease']}, Recommended Medicine: {item['Recommended Medicine']}")
