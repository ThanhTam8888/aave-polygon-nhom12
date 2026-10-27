"""Pure inference, metrics and validation. No training, RPC or future-label inputs."""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix
from xgboost import DMatrix

ROOT = Path(__file__).resolve().parent
SCORE_COLUMNS = {"Logistic": "score_lr", "Random Forest": "score_rf", "XGBoost": "score_xgb"}
DERIVED = {"activity_spike_7_vs_previous_23", "repay_borrow_ratio_30d", "withdraw_supply_ratio_30d", "net_borrow_count_30d"}
LABELS = {
    "borrow": "Vay", "repay": "Trả nợ", "supply": "Nạp tài sản", "withdraw": "Rút tài sản",
    "collateral_toggle": "Đổi thế chấp", "emode": "Đổi eMode", "rate_change": "Đổi kiểu lãi", "activity": "Hoạt động",
    "days_since_last_borrow": "Ngày từ lần vay gần nhất", "days_since_last_repay": "Ngày từ lần trả gần nhất",
    "days_since_last_activity": "Ngày từ hoạt động gần nhất", "borrower_history_days": "Tuổi lịch sử vay quan sát",
    "activity_spike_7_vs_previous_23": "Mức tăng nhịp hoạt động 7/23 ngày", "repay_borrow_ratio_30d": "Tỷ số số lần trả/vay 30 ngày",
    "withdraw_supply_ratio_30d": "Tỷ số số lần rút/nạp 30 ngày", "net_borrow_count_30d": "Chênh số lần vay–trả 30 ngày",
    "active_days_30d": "Số ngày hoạt động 30 ngày", "past_liquidation_count_30d": "Thanh lý trước · 30 ngày",
    "past_liquidation_count_90d": "Thanh lý trước · 90 ngày", "is_v3": "Cờ phiên bản V3",
}


def label(name):
    if name in LABELS:
        return LABELS[name]
    for action in ["collateral_toggle", "rate_change", "activity", "borrow", "repay", "supply", "withdraw", "emode"]:
        prefix = action + "_count_"
        if name.startswith(prefix):
            return LABELS[action] + " · " + name[len(prefix):-1] + " ngày"
    return name


@lru_cache(maxsize=1)
def summary():
    return json.loads((ROOT / "data/summary.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=3)
def load_model(name):
    meta = summary()["models"][name]
    path = ROOT / "models" / meta["filename"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("Tệp mô hình khác bản đã kiểm chứng. Hãy khôi phục từ gói demo.")
    obj = joblib.load(path)
    if obj["features"] != summary()["features"]:
        raise ValueError("Bộ biến không khớp mô hình.")
    return obj


def hf_status(value):
    try:
        hf = float(value)
    except (TypeError, ValueError):
        return {"key": "unknown", "color": "#A6B1C2", "label": "Chưa có HF hợp lệ"}
    if not np.isfinite(hf) or hf < 0:
        return {"key": "unknown", "color": "#A6B1C2", "label": "Chưa có HF hợp lệ"}
    if hf < 1:
        return {"key": "red", "color": "#FF7D90", "label": "Dưới ngưỡng HF"}
    if hf <= 1.3:
        return {"key": "amber", "color": "#F5C777", "label": "Cần theo dõi"}
    return {"key": "green", "color": "#69DED0", "label": "Xa ngưỡng hơn"}


def recompute_derived(row):
    row = dict(row)
    row["activity_spike_7_vs_previous_23"] = (row["activity_count_7d"] / 7) / ((row["activity_count_30d"] - row["activity_count_7d"] + 1) / 23)
    row["repay_borrow_ratio_30d"] = row["repay_count_30d"] / (row["borrow_count_30d"] + 1)
    row["withdraw_supply_ratio_30d"] = row["withdraw_count_30d"] / (row["supply_count_30d"] + 1)
    row["net_borrow_count_30d"] = row["borrow_count_30d"] - row["repay_count_30d"]
    return row


def validate_features(row, check_consistency=True):
    errors = []
    for f in summary()["features"]:
        try:
            val = float(row[f])
            if not np.isfinite(val):
                errors.append(label(f) + ": cần số hữu hạn.")
            elif f != "net_borrow_count_30d" and val < 0:
                errors.append(label(f) + ": không được âm.")
        except (ValueError, TypeError, KeyError):
            errors.append(label(f) + ": thiếu hoặc không phải số.")
    if errors or not check_consistency:
        return errors
    if row["is_v3"] not in (0, 1):
        errors.append("Cờ V3 phải bằng 0 hoặc 1.")
    if row["days_since_last_borrow"] > 90:
        errors.append("Ngoài cohort nghiên cứu: quá 90 ngày từ lần vay gần nhất.")
    if row["active_days_30d"] > 30 or row["active_days_30d"] > row["activity_count_30d"]:
        errors.append("Số ngày hoạt động phải <= 30 và <= số hoạt động 30 ngày.")
    for a in ["borrow", "repay", "activity"]:
        if not row[f"{a}_count_1d"] <= row[f"{a}_count_7d"] <= row[f"{a}_count_30d"] <= row[f"{a}_count_90d"]:
            errors.append(LABELS[a] + ": cần số lần 1 ngày <= 7 ngày <= 30 ngày <= 90 ngày.")
    for a in ["supply", "withdraw", "collateral_toggle", "emode", "rate_change", "past_liquidation"]:
        short, long = (30, 90) if a == "past_liquidation" else (7, 30)
        if row[f"{a}_count_{short}d"] > row[f"{a}_count_{long}d"]:
            errors.append(label(f"{a}_count_{short}d") + ": lớn hơn cửa sổ dài.")
    for w in [7, 30]:
        total = sum(row[f"{a}_count_{w}d"] for a in ["borrow", "repay", "supply", "withdraw", "collateral_toggle", "emode", "rate_change"])
        if abs(total - row[f"activity_count_{w}d"]) > .01:
            errors.append(f"Tổng các hành vi {w} ngày phải bằng số hoạt động {w} ngày.")
    for f in summary()["features"]:
        if "_count_" in f or f == "active_days_30d":
            if abs(row[f] - round(row[f])) > 1e-4:
                errors.append(label(f) + ": số đếm phải là số nguyên.")
    if errors:
        return list(dict.fromkeys(errors))
    expected = recompute_derived(row)
    for f in DERIVED:
        if not np.isclose(row[f], expected[f], rtol=1e-5, atol=1e-5):
            errors.append(label(f) + ": không khớp các số đếm; cần tính lại.")
    return list(dict.fromkeys(errors))


def predict(row, name="XGBoost"):
    errors = validate_features(row)
    if errors:
        raise ValueError("\n".join(errors))
    obj = load_model(name)
    values = np.array([[row[f] for f in obj["features"]]], dtype=np.float32)
    score = float(obj["pipeline"].predict_proba(values)[0, 1])
    return {"model": name, "score": score, "threshold": obj["threshold"], "alert": bool(score >= obj["threshold"]),
            "score_is_calibrated_probability": False}


def predict_frame(incoming):
    if not 1 <= len(incoming) <= 5000:
        raise ValueError("Cần từ 1 đến 5.000 dòng.")
    missing = [f for f in summary()["features"] if f not in incoming]
    if missing:
        raise ValueError("Thiếu cột: " + ", ".join(missing))
    clean = incoming[summary()["features"]].apply(pd.to_numeric, errors="raise")
    for index, (_, row) in enumerate(clean.iterrows(), start=2):
        errors = validate_features(row.to_dict())
        if errors:
            raise ValueError(f"Dòng {index}: " + errors[0])
    if "total_debt_usdc_equivalent" in incoming:
        debt = pd.to_numeric(incoming.total_debt_usdc_equivalent, errors="raise")
        if not np.isfinite(debt).all() or not debt.gt(0).all():
            raise ValueError("Cohort demo chỉ dành cho vị thế có nợ hữu hạn > 0.")
    result = incoming.copy()
    for name, meta in summary()["models"].items():
        obj = load_model(name)
        scores = obj["pipeline"].predict_proba(clean.to_numpy(np.float32))[:, 1]
        result[name+" · điểm"] = scores
        result[name+" · cảnh báo"] = scores >= meta["threshold"]
    return result


def metrics(y, scores, threshold):
    tn, fp, fn, tp = confusion_matrix(y, np.asarray(scores) >= threshold, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if tp + fp else 0.
    recall = float(tp / (tp + fn)) if tp + fn else 0.
    return {"n": len(y), "positive": int(np.sum(y)), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "alerts": int(tp+fp), "precision": precision, "recall": recall,
            "F2": float(5*tp/(5*tp+4*fn+fp)) if 5*tp+4*fn+fp else 0.,
            "AP": float(average_precision_score(y, scores)) if np.sum(y) else None}


def xgb_contributions(row):
    obj = load_model("XGBoost")
    x = np.array([[row[f] for f in obj["features"]]], dtype=np.float32)
    x = obj["pipeline"].named_steps["impute"].transform(x)
    booster = obj["pipeline"].named_steps["clf"].get_booster()
    dm = DMatrix(x, feature_names=None)
    contrib = booster.predict(dm, pred_contribs=True)[0]
    margin = float(booster.predict(dm, output_margin=True)[0])
    assert np.isclose(contrib.sum(), margin, atol=1e-4)
    return pd.DataFrame({"feature": obj["features"], "label": [label(f) for f in obj["features"]], "value": x[0], "contribution": contrib[:-1]}), float(contrib[-1]), margin


def hf_scenario(hf, collateral_change_pct, debt_change_pct):
    if debt_change_pct <= -100 or collateral_change_pct < -100:
        raise ValueError("Biến động nằm ngoài miền hợp lệ.")
    return float(hf * (1+collateral_change_pct/100)/(1+debt_change_pct/100))
