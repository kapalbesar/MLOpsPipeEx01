import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train():
    df = pd.read_csv('processed_titanic.csv')
    X = df.drop(columns=['Survived', 'Name', 'Ticket', 'PassengerId'])
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Validation accuracy: {acc:.4f}")
    
    joblib.dump(model, 'titanic_model.joblib')
    print("Model saved.")

if __name__ == '__main__':
    train()
