import numpy as np
from .model_loader import ModelLoader
from .feature_builder import build_base_features, prepare_features

class Predictor:
    def __init__(self, model_loader):
        self.model_loader = model_loader

    def predict(self, tanggal, nominal, target_type, rt_number=None, verbose=False):
        # Build features
        base_dict = build_base_features(tanggal, nominal)
        X = prepare_features(base_dict, self.model_loader.get_feature_columns())
        result = {
            "risk_score": 0.0,
            "details": {} if verbose else None
        }
        # Get models
        level0_models = self.model_loader.get_level0_models()
        level1_models = self.model_loader.get_level1_models()
        # Level 0 predictions
        level0_preds = []
        level0_details = {}
        for name, model in level0_models.items():
            pred = model.predict(X)[0]
            level0_preds.append(pred)
            level0_details[name] = float(pred)
        level0_array = np.array(level0_preds).reshape(1, -1)
        level0_avg = float(np.mean(level0_preds))
        if verbose:
            result["details"]["level0"] = level0_details
        # Level 1 prediction (meta model)
        meta_pred = level1_models["meta_ridge"].predict(level0_array)[0]
        result["risk_score"] = round(float(meta_pred), 2)
        # Risk category logic (contoh sederhana)
        if meta_pred < 20:
            status = "RENDAH"; emoji = "✅"; rekomendasi = "Risiko rendah. Transaksi aman."; tindakan = ["Transaksi aman"]
        elif meta_pred < 50:
            status = "SEDANG"; emoji = "⚠️"; rekomendasi = "Perlu monitoring berkala."; tindakan = ["Monitor pembayaran secara berkala", "Kirim reminder H-3 jatuh tempo"]
        elif meta_pred < 75:
            status = "TINGGI"; emoji = "🔴"; rekomendasi = "Aktifkan reminder & follow-up intensif."; tindakan = ["Aktifkan reminder", "Follow-up intensif"]
        else:
            status = "SANGAT TINGGI"; emoji = "🚨"; rekomendasi = "Tunda transaksi & siapkan prosedur penagihan."; tindakan = ["Tunda transaksi", "Siapkan penagihan"]
        result["risk_category"] = {
            "status": status,
            "emoji": emoji,
            "rekomendasi": rekomendasi,
            "tindakan": tindakan
        }
        return result
