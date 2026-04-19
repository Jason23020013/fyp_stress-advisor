import sqlite3
import pandas as pd
import os

def init_db():
    print("🔌 Initializing SQL Database...")
    conn = sqlite3.connect('student_stress.db')
    cursor = conn.cursor()

    # 1. Create Main Training Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Study_Hours_Per_Day REAL,
            Sleep_Hours_Per_Day REAL,
            Social_Hours_Per_Day REAL,
            Physical_Activity_Hours_Per_Day REAL,
            Extracurricular_Hours_Per_Day REAL,
            GPA REAL,
            Stress_Level TEXT
        )
    ''')

    # 2. Create User Feedback Table (For your Survey Logic)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Study_Hours_Per_Day REAL,
            Sleep_Hours_Per_Day REAL,
            Social_Hours_Per_Day REAL,
            Physical_Activity_Hours_Per_Day REAL,
            Extracurricular_Hours_Per_Day REAL,
            GPA REAL,
            Stress_Level TEXT,
            Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 3. Import CSV Data (If it exists)
    if os.path.exists('student_lifestyle_dataset.csv'):
        try:
            df = pd.read_csv('student_lifestyle_dataset.csv')
            # Select only relevant columns
            cols = ['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Social_Hours_Per_Day', 
                    'Physical_Activity_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 
                    'GPA', 'Stress_Level']
            # Write to SQL
            df[cols].to_sql('training_data', conn, if_exists='replace', index=False)
            print(f"✅ Imported {len(df)} records from CSV to SQL.")
        except Exception as e:
            print(f"⚠️ Error importing CSV: {e}")
    
    conn.commit()
    conn.close()
    print("🚀 Database Ready: student_stress.db")

if __name__ == "__main__":
    init_db()