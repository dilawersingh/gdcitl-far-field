# Class to handle the full lifecycle of an experiment with logging
import os
import json
from datetime import datetime


class ExperimentManager:
    def __init__(self, name, base_dir="runs", overwrite=False):
        self.base_name = name
        self.overwrite = overwrite

        if overwrite:
            self.dir = os.path.join(base_dir, name)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.dir = os.path.join(base_dir, f"{name}_{timestamp}")

        os.makedirs(self.dir, exist_ok=True)

        self.log_path = os.path.join(self.dir, "experimental_log.txt")
        self.metrics_path = os.path.join(self.dir, "metrics.json")
        self.summary_path = os.path.join(base_dir, "summary.log")
        self.metrics = []

        if overwrite:
            # Clear the experiment log when overwriting
            if os.path.exists(self.log_path):
                with open(self.log_path, "w", encoding="utf-8") as f:
                    f.truncate(0)

    def log(self, msg):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def prepend(self, header_msg):
        if os.path.exists(self.log_path):
            with open(self.log_path, "r", encoding="utf-8") as f:
                existing = f.read()
        else:
            existing = ""
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(header_msg + "\n" + existing)

    def write_summary(self, msg):
        with open(self.summary_path, "a", encoding="utf-8") as f:
            f.write(msg)

    def save_metrics(self, entry: dict):
        """Append one record to metrics.json."""
        self.metrics.append(entry)
        with open(self.metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
