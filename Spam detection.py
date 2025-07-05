# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.externals import joblib  # for saving the model to disk
import os

# Function to load the dataset (you can replace this with your actual dataset path)
def load_dataset():
    # Using a sample dataset (replace with your actual file path)
    # Example: df = pd.read_csv("spam_dataset.csv")
    data = {
        'text': [
            'Free entry in 2 a wkly comp to win a £1000 gift card',
            'Call me when you get this message',
            'Congratulations, you’ve won a free ticket to the Bahamas!',
            'Hey, are you coming to the party tonight?',
            'Get paid to take surveys online',
            'Happy birthday! Enjoy your special day!',
            'Limited offer: Free iPhone, act fast!',
            'Hey, I need to talk to you about the project.',
            'Special offer on your credit card, don’t miss out!',
            'I am looking forward to our dinner tomorrow.',
            'Free vacation to Hawaii – click here to claim!',
            'Let’s discuss the meeting notes when you get a chance.',
            'Congratulations, you’ve won a lottery of $1000!'
        ],
        'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'spam', 'ham', 'spam']
    }
    
    # Convert dictionary into DataFrame
    df = pd.DataFrame(data)
    return df

# Function to preprocess the dataset
def preprocess_data(df):
    # Split the dataset into features and labels
    X = df['text']
    y = df['label']
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to convert text to numerical features using CountVectorizer
def vectorize_data(X_train, X_test):
    vectorizer = CountVectorizer(stop_words='english')  # Remove common stopwords like 'the', 'is', etc.
    
    # Fit the vectorizer on the training data and transform both train and test data
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)
    
    return vectorizer, X_train_vect, X_test_vect

# Function to train the model using Naive Bayes
def train_model(X_train_vect, y_train):
    model = MultinomialNB()  # Naive Bayes classifier for text classification
    model.fit(X_train_vect, y_train)  # Train the model on the training data
    return model

# Function to evaluate the model
def evaluate_model(model, X_test_vect, y_test):
    y_pred = model.predict(X_test_vect)  # Predict labels for test data
    
    # Accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Function to save the trained model and vectorizer to disk
def save_model(model, vectorizer):
    if not os.path.exists('model'):
        os.makedirs('model')  # Create 'model' folder if it doesn't exist
    
    # Save the model and vectorizer using joblib
    joblib.dump(model, 'model/spam_classifier_model.pkl')
    joblib.dump(vectorizer, 'model/spam_vectorizer.pkl')
    print("Model and vectorizer saved successfully!")

# Function to load a saved model and vectorizer
def load_saved_model():
    model = joblib.load('model/spam_classifier_model.pkl')
    vectorizer = joblib.load('model/spam_vectorizer.pkl')
    print("Model and vectorizer loaded successfully!")
    return model, vectorizer

# Main function to execute the program
def main():
    # Step 1: Load the dataset
    df = load_dataset()
    
    # Step 2: Preprocess the data (split it into train/test)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Step 3: Vectorize the text data
    vectorizer, X_train_vect, X_test_vect = vectorize_data(X_train, X_test)
    
    # Step 4: Train the Naive Bayes model
    model = train_model(X_train_vect, y_train)
    
    # Step 5: Evaluate the model
    evaluate_model(model, X_test_vect, y_test)
    
    # Step 6: Save the trained model and vectorizer for future use
    save_model(model, vectorizer)
    
    # Optional: Load saved model and vectorizer (for testing)
    # model, vectorizer = load_saved_model()

# Run the program
if __name__ == "__main__":
    main()
