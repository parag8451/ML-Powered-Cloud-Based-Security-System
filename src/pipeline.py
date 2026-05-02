from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils import DATA_DIR, MODELS_DIR, REPORTS_DIR, ensure_directories, save_json, utc_now_iso


TOP_FEATURE_COUNT = 35
TRAINING_SAMPLE_ROWS = 180_000
DEFAULT_PIPELINE_PATH = MODELS_DIR / "intrusion_pipeline.joblib"
DEFAULT_REPORT_PATH = REPORTS_DIR / "training_report.json"
DEFAULT_SCHEMA_PATH = REPORTS_DIR / "input_schema.json"

LABEL_MAPPING = {
    "benign": "BENIGN",
    "bot": "Bot",
    "ddos": "DDoS",
    "dos goldeneye": "DoS",
    "dos hulk": "DoS",
    "dos slowhttptest": "DoS",
    "dos slowloris": "DoS",
    "ftp-patator": "BruteForce",
    "heartbleed": "DoS",
    "infiltration": "Infiltration",
    "portscan": "PortScan",
    "ssh-patator": "BruteForce",
    "web attack - brute force": "WebAttack",
    "web attack - sql injection": "WebAttack",
    "web attack - xss": "WebAttack",
}

SEVERITY_MAP = {
    "BENIGN": ("Low", 8),
    "Bot": ("Medium", 60),
    "BruteForce": ("High", 76),
    "DDoS": ("Critical", 95),
    "DoS": ("High", 82),
    "Infiltration": ("Critical", 98),
    "PortScan": ("Medium", 48),
    "WebAttack": ("High", 84),
}


def normalize_label(value: Any) -> str:
    text = str(value).strip()
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\ufffd": "-",
        "Ã¯Â¿Â½": "-",
        "ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“": "-",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return " ".join(text.split())


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()
    return cleaned


def load_raw_dataset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {Path(data_dir).resolve()}")

    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path, low_memory=False)
        frames.append(clean_columns(frame))
    return pd.concat(frames, ignore_index=True)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    dataset = clean_columns(df)
    if "Label" not in dataset.columns:
        raise KeyError("Expected a Label column in the dataset.")

    dataset["Label"] = dataset["Label"].apply(normalize_label).str.lower().map(LABEL_MAPPING)
    dataset = dataset.dropna(subset=["Label"]).drop_duplicates()
    dataset = dataset.replace([np.inf, -np.inf], np.nan)

    feature_columns = [column for column in dataset.columns if column != "Label"]
    
    # Convert columns individually to avoid memory issues with large datasets
    numeric_data = {}
    for col in feature_columns:
        numeric_data[col] = pd.to_numeric(dataset[col], errors="coerce")
    numeric_frame = pd.DataFrame(numeric_data)
    
    valid_columns = numeric_frame.columns[numeric_frame.notna().mean().ge(0.85)].tolist()
    if not valid_columns:
        raise ValueError("No sufficiently valid numeric columns found in the dataset.")

    prepared = pd.concat([numeric_frame[valid_columns], dataset["Label"]], axis=1)
    return prepared.reset_index(drop=True)


def stratified_sample(
    df: pd.DataFrame,
    max_total_rows: int = TRAINING_SAMPLE_ROWS,
    benign_ratio_cap: float = 0.42,
    rare_class_floor: int = 3_500,
    random_state: int = 42,
) -> pd.DataFrame:
    counts = df["Label"].value_counts()
    allocations: dict[str, int] = {}
    remaining_budget = max_total_rows

    for label, count in counts.items():
        if label == "BENIGN":
            continue
        target = min(count, max(rare_class_floor, int(max_total_rows * 0.11)))
        allocations[label] = target
        remaining_budget -= target

    benign_available = counts.get("BENIGN", 0)
    benign_cap = min(
        benign_available,
        max(10_000, int(max_total_rows * benign_ratio_cap), remaining_budget),
    )
    if benign_available:
        allocations["BENIGN"] = benign_cap

    sampled_frames: list[pd.DataFrame] = []
    for label, count in counts.items():
        target = min(count, allocations.get(label, count))
        group = df[df["Label"] == label]
        if target < count:
            group = group.sample(n=target, random_state=random_state)
        sampled_frames.append(group)

    sampled = pd.concat(sampled_frames, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def select_top_features(X: pd.DataFrame, y: np.ndarray, feature_count: int = TOP_FEATURE_COUNT) -> list[str]:
    selector = LGBMClassifier(
        objective="multiclass",
        num_class=len(np.unique(y)),
        class_weight="balanced",
        learning_rate=0.06,
        n_estimators=220,
        num_leaves=64,
        max_depth=12,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
    )
    selector.fit(X.fillna(X.median(numeric_only=True)), y)
    importance = pd.Series(selector.feature_importances_, index=X.columns)
    return importance.sort_values(ascending=False).head(feature_count).index.tolist()


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    return {int(label): float(total / (len(classes) * count)) for label, count in zip(classes, counts)}


def build_feature_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_search_model(num_classes: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )


def threat_pattern_from_probabilities(probabilities: dict[str, float]) -> str:
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:3]
    summary = ", ".join(f"{label} {score:.1f}%" for label, score in ranked if score > 0)
    return summary or "No dominant threat pattern detected"


@dataclass
class IntrusionDetectionPipeline:
    feature_names: list[str]
    label_encoder: LabelEncoder
    feature_pipeline: Pipeline
    calibrated_model: CalibratedClassifierCV
    metadata: dict[str, Any]

    def align_input_frame(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        frame = clean_columns(df).drop(columns=["Label"], errors="ignore")
        extra_features = [column for column in frame.columns if column not in self.feature_names]
        missing_features = [column for column in self.feature_names if column not in frame.columns]

        for feature in missing_features:
            frame[feature] = np.nan

        aligned = frame.reindex(columns=self.feature_names)
        numeric = aligned.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        if numeric.isna().all(axis=None):
            raise ValueError("Input schema does not match the trained feature set.")

        schema_report = {
            "missing_features": missing_features,
            "extra_features": extra_features,
            "invalid_value_count": int(numeric.isna().sum().sum()),
        }
        return numeric, schema_report

    def preprocess_frame(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        numeric, schema_report = self.align_input_frame(df)
        transformed = self.feature_pipeline.transform(numeric)
        transformed_frame = pd.DataFrame(transformed, columns=self.feature_names, index=numeric.index)
        return transformed_frame, numeric, schema_report

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed, numeric, schema_report = self.preprocess_frame(df)
        probabilities = self.calibrated_model.predict_proba(transformed)
        predicted_indices = probabilities.argmax(axis=1)
        predicted_labels = self.label_encoder.inverse_transform(predicted_indices)
        class_names = list(self.label_encoder.classes_)

        results: list[dict[str, Any]] = []
        for idx, (label, row_probabilities) in enumerate(zip(predicted_labels, probabilities)):
            probability_map = {
                class_name: round(float(probability * 100.0), 2)
                for class_name, probability in zip(class_names, row_probabilities)
            }
            ranked_features = (
                numeric.iloc[idx].fillna(0.0).abs().sort_values(ascending=False).head(5).index.tolist()
            )
            severity, severity_score = self.get_severity(label, probability_map[label])
            results.append(
                {
                    "predicted_label": label,
                    "confidence": round(probability_map[label], 2),
                    "severity": severity,
                    "severity_score": severity_score,
                    "probabilities": probability_map,
                    "top_features": ranked_features,
                    "threat_pattern": threat_pattern_from_probabilities(probability_map),
                    "schema_validation": schema_report,
                }
            )
        return pd.DataFrame(results)

    def explain_row(self, df: pd.DataFrame) -> dict[str, float]:
        transformed, _, _ = self.preprocess_frame(df)
        calibrated_models = getattr(self.calibrated_model, "calibrated_classifiers_", None)
        if not calibrated_models:
            return {}
        estimator = calibrated_models[0].estimator

        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(transformed[:1])
        values = np.abs(np.asarray(shap_values))
        if values.ndim == 3:
            values = values[0].mean(axis=1)
        elif values.ndim == 2:
            values = values[0]

        importance = pd.Series(values, index=self.feature_names).sort_values(ascending=False).head(10)
        return {feature: round(float(score), 5) for feature, score in importance.items()}

    def get_severity(self, label: str, confidence: float) -> tuple[str, int]:
        _, base_score = SEVERITY_MAP.get(label, ("Low", 10))
        adjusted = min(100, int(round(base_score * 0.7 + confidence * 0.3)))
        if adjusted >= 90:
            level = "Critical"
        elif adjusted >= 75:
            level = "High"
        elif adjusted >= 45:
            level = "Medium"
        else:
            level = "Low"
        if label == "BENIGN":
            level = "Low"
        return level, adjusted

    def input_schema(self) -> dict[str, Any]:
        return {
            "features": self.feature_names,
            "feature_count": len(self.feature_names),
            "classes": self.metadata.get("classes", []),
        }

    def save(self, path: Path = DEFAULT_PIPELINE_PATH) -> None:
        ensure_directories()
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path = DEFAULT_PIPELINE_PATH) -> "IntrusionDetectionPipeline":
        return joblib.load(path)


def train_pipeline(
    data_dir: Path = DATA_DIR,
    pipeline_path: Path = DEFAULT_PIPELINE_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> dict[str, Any]:
    ensure_directories()
    raw_df = load_raw_dataset(data_dir)
    prepared_df = prepare_dataset(raw_df)
    sampled_df = stratified_sample(prepared_df)

    X = sampled_df.drop(columns=["Label"])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(sampled_df["Label"])

    selected_features = select_top_features(X, y)
    X_selected = X[selected_features]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X_selected, y))
    X_train = X_selected.iloc[train_idx]
    X_test = X_selected.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    feature_pipeline = build_feature_pipeline()
    X_train_prepared = feature_pipeline.fit_transform(X_train)
    X_test_prepared = feature_pipeline.transform(X_test)

    class_weights = compute_class_weights(y_train)
    sample_weights = np.array([class_weights[int(label)] for label in y_train], dtype=float)

    search = RandomizedSearchCV(
        estimator=build_search_model(len(label_encoder.classes_)),
        param_distributions={
            "learning_rate": [0.03, 0.05, 0.07],
            "n_estimators": [180, 240, 320],
            "num_leaves": [32, 48, 64, 80],
            "max_depth": [8, 10, 12, -1],
            "min_child_samples": [20, 30, 40],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "reg_alpha": [0.0, 0.1, 0.3],
            "reg_lambda": [0.0, 0.2, 0.5, 0.8],
        },
        n_iter=10,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(X_train_prepared, y_train, sample_weight=sample_weights)

    calibrator = CalibratedClassifierCV(
        estimator=search.best_estimator_,
        method="sigmoid",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    )
    calibrator.fit(X_train_prepared, y_train, sample_weight=sample_weights)

    probabilities = calibrator.predict_proba(X_test_prepared)
    predictions = probabilities.argmax(axis=1)
    report = classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    pipeline = IntrusionDetectionPipeline(
        feature_names=selected_features,
        label_encoder=label_encoder,
        feature_pipeline=feature_pipeline,
        calibrated_model=calibrator,
        metadata={
            "trained_at": utc_now_iso(),
            "feature_count": len(selected_features),
            "classes": label_encoder.classes_.tolist(),
            "source_rows": int(len(raw_df)),
            "training_rows": int(len(sampled_df)),
            "best_params": search.best_params_,
            "macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
            "weighted_f1": round(float(f1_score(y_test, predictions, average="weighted")), 4),
            "report_path": str(report_path),
        },
    )
    pipeline.save(pipeline_path)

    metrics_payload = {
        "metadata": pipeline.metadata,
        "class_distribution": sampled_df["Label"].value_counts().to_dict(),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classes": label_encoder.classes_.tolist(),
    }
    save_json(report_path, metrics_payload)
    save_json(schema_path, pipeline.input_schema())

    legacy_paths = {
        MODELS_DIR / "final_model.pkl": pipeline.calibrated_model,
        MODELS_DIR / "scaler.pkl": pipeline.feature_pipeline.named_steps["scaler"],
        MODELS_DIR / "label_encoder.pkl": pipeline.label_encoder,
        MODELS_DIR / "selected_features.pkl": pipeline.feature_names,
    }
    for path, artifact in legacy_paths.items():
        joblib.dump(artifact, path)

    return metrics_payload


def load_pipeline(path: Path = DEFAULT_PIPELINE_PATH) -> IntrusionDetectionPipeline:
    if path.exists():
        return IntrusionDetectionPipeline.load(path)

    legacy_model = MODELS_DIR / "final_model.pkl"
    legacy_scaler = MODELS_DIR / "scaler.pkl"
    legacy_encoder = MODELS_DIR / "label_encoder.pkl"
    legacy_features = MODELS_DIR / "selected_features.pkl"
    if all(artifact.exists() for artifact in (legacy_model, legacy_scaler, legacy_encoder, legacy_features)):
        feature_names = joblib.load(legacy_features)
        label_encoder = joblib.load(legacy_encoder)
        imputer = SimpleImputer(strategy="constant", fill_value=0.0)
        imputer.fit(pd.DataFrame([{feature: 0.0 for feature in feature_names}]))
        return IntrusionDetectionPipeline(
            feature_names=feature_names,
            label_encoder=label_encoder,
            feature_pipeline=Pipeline(
                steps=[
                    ("imputer", imputer),
                    ("scaler", joblib.load(legacy_scaler)),
                ]
            ),
            calibrated_model=joblib.load(legacy_model),
            metadata={
                "trained_at": "legacy-artifact",
                "feature_count": len(feature_names),
                "classes": label_encoder.classes_.tolist(),
                "source_rows": None,
                "training_rows": None,
                "best_params": {},
                "macro_f1": None,
                "weighted_f1": None,
                "report_path": str(DEFAULT_REPORT_PATH),
            },
        )

    raise FileNotFoundError(
        "No trained pipeline artifact found. Run `python src/train.py` to create models/intrusion_pipeline.joblib."
    )
