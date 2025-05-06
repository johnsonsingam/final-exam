# final-exam

700766438   singamshetty johnson
video link: https://drive.google.com/drive/folders/1eZgmEJFvR8a6TB0Cdxty2LDSg0J58MSH?usp=sharing


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv('Exam2_Code.ipynb')  # Adjust the path if needed

# Preprocess: drop rows with missing text or sentiment
df_clean = df[['text', 'sentiment']].dropna()

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(
    df_clean['text'], df_clean['sentiment'], test_size=0.2, random_state=42
)

# Define pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "sentiment_model.joblib")

# Predict on new text
example_text = ["A lot of good things are happening. We are respected again throughout the world, and that's a great thing.@realDonaldTrump"]
loaded_model = joblib.load("sentiment_model.joblib")
prediction = loaded_model.predict(example_text)
print("Prediction for example tweet:", prediction[0])

# GridSearchCV for hyperparameter tuning
param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__max_df': [0.9, 1.0],
    'clf__C': [0.1, 1, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Print best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

# Optional: Evaluate best model on test set
y_pred = grid_search.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
