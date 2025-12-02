import numpy as np
from datetime import datetime

def build_base_features(tanggal, nominal):
    # Contoh: ekstrak fitur dari tanggal dan nominal
    dt = datetime.strptime(tanggal, "%Y-%m-%d")
    features = {
        "Bulan": dt.month,
        "Hari": dt.day,
        "Nominal": nominal
    }
    return features

def prepare_features(base_dict, feature_columns):
    # Urutkan fitur sesuai urutan yang diharapkan model
    return np.array([[base_dict.get(col, 0) for col in feature_columns]])
