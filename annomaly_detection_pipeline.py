import logging
from annomaly_detector import AnomalyDetector

class AnomalyDetectionPipeline:
    def __init__(self, db_path):
        self.db_path = db_path
        self.detector = AnomalyDetector(db_path)

    def run(self):
        logging.info("Starting anomaly detection pipeline.")
        data = self.detector.load_data()
        data = self.detector.preprocess_data(data)
        anomalies = self.detector.detect_anomalies(data)

        self.handle_anomalies(anomalies)

    def handle_anomalies(self, anomalies):
        if not anomalies.empty:
            logging.info(f"Anomalies detected:\n{anomalies}")
            # Here you can implement additional handling, like sending alerts or storing the anomalies in another table
        else:
            logging.info("No anomalies detected.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = AnomalyDetectionPipeline('my_database.db')
    pipeline.run()
