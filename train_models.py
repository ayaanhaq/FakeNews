import pandas as pd
import numpy as np
import re
import string
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def preprocess_text(text):
    """Text preprocessing function"""
    text = text.lower()
    text = re.sub(r'\.*?\[]','',text)
    text = re.sub(r'[()]','',text)
    text = re.sub(r'\\W',' ',text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    return text

def train_and_save_models():
    """Train models and save them to disk"""
    print("🚀 Starting model training...")
    
    try:
        # Load data
        print("📂 Loading data...")
        fake = pd.read_csv("Fake.csv")
        true = pd.read_csv("True.csv")
        
        # Clean data
        print("🧹 Cleaning data...")
        true["text"] = true["text"].replace("(Reuters)","",regex=True)
        fake['target'] = 0
        true['target'] = 1
        
        # Drop unnecessary columns
        fake = fake.drop(['title','subject','date'], axis=1)
        true = true.drop(['title','subject','date'], axis=1)
        
        # Combine datasets
        news = pd.concat([fake, true], axis=0)
        
        # Sample a smaller subset for faster training
        sample_size = min(8000, len(news))
        news = news.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        print(f"📊 Using {len(news)} samples for training")
        
        # Apply preprocessing
        print("🔧 Preprocessing text...")
        news['text'] = news['text'].apply(preprocess_text)
        
        # Prepare features and target
        X = news['text']
        Y = news['target']
        
        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
        
        # Vectorize text with optimized parameters
        print("📝 Vectorizing text...")
        vectorization = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        xv_train = vectorization.fit_transform(X_train)
        xv_test = vectorization.transform(X_test)
        
        # Train models with optimized parameters
        print("🤖 Training models...")
        
        # Logistic Regression
        print("  - Training Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=100)
        lr.fit(xv_train, Y_train)
        lr_score = lr.score(xv_test, Y_test)
        print(f"    Accuracy: {lr_score:.3f}")
        
        # Decision Tree
        print("  - Training Decision Tree...")
        dtc = DecisionTreeClassifier(random_state=42, max_depth=10)
        dtc.fit(xv_train, Y_train)
        dtc_score = dtc.score(xv_test, Y_test)
        print(f"    Accuracy: {dtc_score:.3f}")
        
        # Gradient Boosting
        print("  - Training Gradient Boosting...")
        gbc = GradientBoostingClassifier(random_state=42, n_estimators=50)
        gbc.fit(xv_train, Y_train)
        gbc_score = gbc.score(xv_test, Y_test)
        print(f"    Accuracy: {gbc_score:.3f}")
        
        # Random Forest
        print("  - Training Random Forest...")
        rfc = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10)
        rfc.fit(xv_train, Y_train)
        rfc_score = rfc.score(xv_test, Y_test)
        print(f"    Accuracy: {rfc_score:.3f}")
        
        # Create models dictionary
        models_dict = {
            'vectorization': vectorization,
            'lr': lr,
            'dtc': dtc,
            'gbc': gbc,
            'rfc': rfc,
            'X_test': X_test,
            'Y_test': Y_test,
            'xv_test': xv_test
        }
        
        # Save models
        print("💾 Saving models to disk...")
        with open('models.pkl', 'wb') as f:
            pickle.dump(models_dict, f)
        
        print("✅ Models trained and saved successfully!")
        print(f"📈 Model Accuracies:")
        print(f"   - Logistic Regression: {lr_score:.3f}")
        print(f"   - Decision Tree: {dtc_score:.3f}")
        print(f"   - Gradient Boosting: {gbc_score:.3f}")
        print(f"   - Random Forest: {rfc_score:.3f}")
        print(f"   - Average: {(lr_score + dtc_score + gbc_score + rfc_score) / 4:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔍 Fake News Detector - Model Training")
    print("=" * 50)
    
    if train_and_save_models():
        print("\n🎉 Ready to run the Streamlit app!")
        print("Run: streamlit run app.py")
    else:
        print("\n❌ Training failed. Please check your data files.") 