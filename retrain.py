import sqlite3
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def train():
    print("🚀 Starting Retraining System (SQL Integration)...")
    
    db_path = 'student_stress.db'
    if not os.path.exists(db_path):
        print("❌ Database missing.")
        return

    conn = sqlite3.connect(db_path)
    
    # 1. Check for New Data
    try:
        new_count = pd.read_sql("SELECT COUNT(*) as count FROM user_feedback", conn)['count'][0]
        print(f"📊 Found {new_count} new user feedback records in SQL.")
        
        # 2. Fetch Combined Data
        query = """
        SELECT * FROM training_data
        UNION ALL
        SELECT Study_Hours_Per_Day, Sleep_Hours_Per_Day, Social_Hours_Per_Day, 
               Physical_Activity_Hours_Per_Day, Extracurricular_Hours_Per_Day, 
               GPA, Stress_Level 
        FROM user_feedback
        """
        df = pd.read_sql_query(query, conn)
        
    except Exception as e:
        print(f"Error: {e}")
        return
    finally:
        conn.close()

    # 3. Process & Train
    df['Lifestyle_Score'] = (df['Social_Hours_Per_Day'] + df['Physical_Activity_Hours_Per_Day'] + df['Extracurricular_Hours_Per_Day'])
    df['Academic_Pressure'] = df['GPA'] * df['Study_Hours_Per_Day']

    features = ['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Lifestyle_Score', 'Academic_Pressure', 'GPA']
    X = df[features]
    y = df['Stress_Level']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y_encoded)
    
    # Using the "Fixed" parameters (max_depth=2)
    model = RandomForestClassifier(n_estimators=50, max_depth=2, min_samples_split=10, random_state=42)
    model.fit(X_bal, y_bal)
    
    joblib.dump(model, 'student_stress_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    print("✅ Retraining Complete. Model Updated.")

if __name__ == "__main__":
    train()