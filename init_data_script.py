import pandas as pd
import sqlite3

file_paths = [
    '/home/dombrzalskip/hackology/distr1.csv',
    '/home/dombrzalskip/hackology/distr2.csv',
    '/home/dombrzalskip/hackology/distr3.csv'
]


column_names = ['index', 'date', 'products_sold', 'price', 'trends']


dataframes = []


for idx, file_path in enumerate(file_paths, start=1):
    df = pd.read_csv(file_path, names=column_names, header=0, index_col=None)
    df['dist_id'] = idx
    dataframes.append(df)


combined_df = pd.concat(dataframes, ignore_index=True)


combined_df.drop(columns=['index'], inplace=True, errors='ignore')


print(combined_df)

def create_database():
    conn = sqlite3.connect('my_database.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sales_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date DATE,
        products_sold INTEGER,
        price REAL,
        trends INTEGER,
        dist_id INTEGER
    )
    ''')
    conn.commit()
    conn.close()

def load_data_to_sqlite(df):
    conn = sqlite3.connect('my_database.db')
    df.to_sql('sales_data', conn, if_exists='append', index=False)
    conn.commit()
    conn.close()

# Main function
def main():
    create_database()
    load_data_to_sqlite(combined_df)

if __name__ == "__main__":
    main()
