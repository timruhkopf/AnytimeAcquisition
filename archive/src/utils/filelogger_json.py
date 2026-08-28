import atexit
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os


class BufferedDictLogger:
    def __init__(self, file_path, buffer_size=10, ):
        self.file_path = file_path
        self.buffer = []
        self.logs = []
        self.buffer_size = buffer_size
        self.postfix = {}

        # If file exists, load logs (optional, for append mode)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                self.logs = [json.loads(line) for line in f]
        atexit.register(self._atexit_handler)

    def __repr__(self):
        return f"BufferedDictLogger(file_path={self.file_path}, buffer_size={self.buffer_size}, logs_count={len(self)})"

    def __len__(self):
        """Returns the total number of logged entries."""
        return len(self.logs) + len(self.buffer)

    def log(self, data_dict):
        if not isinstance(data_dict, dict):
            raise ValueError("Input must be a dictionary")

        if bool(self.postfix):
            # Append postfix to each key in data_dict
            data_dict.update(self.postfix)

        self.buffer.append(data_dict)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        # Append buffer to logs and file
        self.logs.extend(self.buffer)

        with open(self.file_path, "a") as f:
            for row in self.buffer:
                f.write(json.dumps(row) + "\n")

        self.buffer.clear()

    @property
    def df(self) -> pd.DataFrame:
        # Reads all logs so far (including buffer, and re-reads file for full history)
        # read file in case of external writes/old logs
        if self.file_path.exists():
            with open(self.file_path, "r") as f:
                all_logs = [json.loads(line) for line in f]
        else:
            all_logs = []
        all_logs += self.buffer  # Unflushed (in-memory) logs
        return pd.DataFrame(all_logs)

    @property
    def dfs(self) -> list:
        all_logs = self.logs + self.buffer
        if not all_logs:
            return []
        groups = defaultdict(list)
        for d in all_logs:
            key_tuple = tuple(sorted(d.keys()))
            groups[key_tuple].append(d)
        return [pd.DataFrame(g) for g in groups.values()]

    def plot_metric(self, metric, subset=None, stratify_by=None, **plt_kwargs):
        """Generate a line plot for debugging.
        metric: string, column to plot on Y axis. X axis uses DataFrame index.
        subset: pandas query string, e.g. "lr > 0.001"
        stratify_by: column name or None (for hue/line splitting)
        **plt_kwargs: passes to plt.plot
        """
        df = self.df
        if subset is not None:
            df = df.query(subset)
        if stratify_by is None:
            plt.plot(df[metric], **plt_kwargs)
        else:
            for key, subdf in df.groupby(stratify_by):
                plt.plot(subdf[metric].values, label=str(key), **plt_kwargs)
            plt.legend(title=stratify_by)
        plt.title(f"{metric} over steps")
        plt.xlabel("step/index")
        plt.ylabel(metric)
        plt.show()

    def _atexit_handler(self):
        self.flush()


if __name__ == '__main__':
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = BufferedDictLogger(Path(tmpdir) / "runs.jsonl", buffer_size=5)
        logger.log({"epoch": 1, "acc": 0.90, "lr": 1e-3, "group": "A"})
        logger.log({"epoch": 2, "acc": 0.92, "lr": 1e-4, "group": "A"})
        logger.log({"epoch": 2, "lr": 1e-4, "group": "B"})
        logger.log({"epoch": 2, "lr": 1e-3, "group": "B"})

        logger.df
        logger.flush()  # Writes to file
        logger.dfs

        logger.plot_metric("lr", stratify_by="group")
