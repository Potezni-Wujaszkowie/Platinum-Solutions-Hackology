from sklearn.ensemble import IsolationForest
import sqlite3
import logging
import pandas as pd


class AnomalyDetector:
    def __init__(self, db_path):
        self.db_path = db_path

    def connect_db(self):
        return sqlite3.connect(self.db_path)

    def load_data(self):
        conn = self.connect_db()
        data = pd.read_sql_query("SELECT * FROM sales_data", conn)
        conn.close()
        return data

    def preprocess_data(self, data):
        original_dates = data['date'].copy()
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        invalid_dates = original_dates[data['date'].isna()]

        if not invalid_dates.empty:
            logging.warning(f"Invalid dates encountered: {invalid_dates.tolist()}")

        data = data.dropna(subset=['date'])

        # Use .loc to avoid SettingWithCopyWarning
        data.loc[:, 'year'] = data['date'].dt.year
        data.loc[:, 'month'] = data['date'].dt.month
        data.loc[:, 'day'] = data['date'].dt.day
        data.loc[:, 'day_of_week'] = data['date'].dt.dayofweek
        return data

    def detect_anomalies(self, data):
        features = data[['products_sold', 'price', 'trends', 'dist_id', 'year', 'month', 'day', 'day_of_week']]
        model = IsolationForest(contamination=0.00001)
        model.fit(features)
        data['anomaly'] = model.predict(features)
        anomalies = data[data['anomaly'] == -1]
        return anomalies

    def run(self):
        logging.info("Anomaly detection started.")
        data = self.load_data()
        data = self.preprocess_data(data)
        anomalies = self.detect_anomalies(data)

        if not anomalies.empty:
            logging.info(f"Anomalies found:\n{anomalies}")
        else:
            logging.info("No anomalies found.")
