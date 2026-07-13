import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

def load_and_preprocess_data(filepath):
    print(f"Loading dataset from: {filepath}...")
    # 1. Load Dataset
    df = pd.read_csv(filepath)
    
    # Clean data by dropping rows with missing values
    df = df.dropna() 
    
    return df

def train_credit_model(data_path):
    df = load_and_preprocess_data(data_path)
    
    # 2. Separate Features (income, debts, payment_history) and Target (default)
    X = df.drop(columns=['default']) 
    y = df['default']
    
    # 3. Split data into Training and Testing sets (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Feature Scaling (Standardizes the range of financial variables)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Initialize and Train the Random Forest Classifier
    print("Training the Credit Scoring Model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 6. Predict on the Test Set
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] # Target probabilities for ROC-AUC
    
    # --- CodeAlpha Performance Assessment Metrics ---
    print("\n================ CLASSIFICATION REPORT ================")
    print(classification_report(y_test, y_pred, target_names=['Good Credit (0)', 'Bad Credit/Default (1)']))
    
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("=======================================================")
    
    # 7. Generate and Save the ROC Curve Graph
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.2f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    # Save chart to disk
    plt.savefig('roc_curve.png')
    print("\n[SUCCESS] Saved validation graph as 'roc_curve.png' in your folder.")

if __name__ == "__main__":
    # Execute the training pipeline using our data folder path
    train_credit_model("data/credit_data.csv")