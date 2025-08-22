import pandas as pd

def process_data():
    df = pd.read_csv('../raw_titanic.csv')  # Adjust path as needed
    
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[^0], inplace=True)
    df.drop(columns=['Cabin'], inplace=True)
    
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    
    df.to_csv('processed_titanic.csv', index=False)
    print("Processed data saved.")

if __name__ == '__main__':
    process_data()
