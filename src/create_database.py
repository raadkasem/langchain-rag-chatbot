import sqlite3
import pandas as pd
import os

def create_database():
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'company.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create employees table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        position TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        hire_date DATE NOT NULL,
        salary INTEGER NOT NULL
    )
    ''')
    
    # Create customers table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY,
        company_name TEXT NOT NULL,
        contact_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        phone TEXT,
        industry TEXT,
        subscription_tier TEXT NOT NULL,
        monthly_revenue INTEGER NOT NULL
    )
    ''')
    
    # Load data from CSV files
    employees_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'employees.csv')
    customers_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'customers.csv')
    
    if os.path.exists(employees_csv):
        employees_df = pd.read_csv(employees_csv)
        employees_df.to_sql('employees', conn, if_exists='replace', index=False)
    
    if os.path.exists(customers_csv):
        customers_df = pd.read_csv(customers_csv)
        customers_df.to_sql('customers', conn, if_exists='replace', index=False)
    
    conn.commit()
    conn.close()
    print(f"Database created successfully at {db_path}")

if __name__ == "__main__":
    create_database()