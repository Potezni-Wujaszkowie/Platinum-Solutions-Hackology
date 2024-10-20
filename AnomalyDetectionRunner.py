import logging
import time
from AnomalyDetector import AnomalyDetector  

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
            # Add some actions
        else:
            logging.info("No anomalies detected.")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("anomaly_detection.log"),
            logging.StreamHandler()
        ]
    )

    pipeline = AnomalyDetectionPipeline('my_database.db')

    while True:
        try:
            logging.info("Running anomaly detection...")
            pipeline.run()
            logging.info("Sleeping for 1 hour.")
            time.sleep(3600)  
        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)
            logging.info("Retrying in 1 hour.")
            time.sleep(3600)  
