import json
import joblib
from pathlib import Path
import os

class ModelLoader:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.level0_models = {}
        self.level1_models = {}
        self.model_info = {}
        self.feature_columns = []
        self.feature_stats = {}
        self.loaded = False

    def load_models(self):
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory tidak ditemukan: {self.model_dir}")
        # Load Level 0 Models
        self.level0_models = {
            "gb": joblib.load(self.model_dir / "gb_regressor.pkl"),
            "rf": joblib.load(self.model_dir / "rf_regressor.pkl"),
        }
        # Load Level 1 Models
        self.level1_models = {
            "meta_ridge": joblib.load(self.model_dir / "meta_ridge.pkl")
        }
        # Load Model Info
        with open(self.model_dir / "model_info.json", "r") as f:
            self.model_info = json.load(f)
        self.feature_columns = self.model_info["feature_columns"]
        self.feature_stats = self.model_info["feature_stats"]
        self.loaded = True

    def get_level0_models(self):
        return self.level0_models
    def get_level1_models(self):
        return self.level1_models
    def get_model_info(self):
        return self.model_info
    def get_feature_columns(self):
        return self.feature_columns
    def get_feature_stats(self):
        return self.feature_stats
