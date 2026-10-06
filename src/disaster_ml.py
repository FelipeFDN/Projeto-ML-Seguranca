"""Preparação inicial dos modelos de classificação de desastres."""

import io
import re
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


MODEL_FILE_NAME = "best_model.joblib"
DISASTERS = ("enxurradas", "alagamentos", "inundacoes", "deslizamentos")
TARGET_COLUMNS = [f"desastre_{name}" for name in DISASTERS]
WEATHER_COLUMNS = {
    "precip": "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)",
    "temp": "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)",
    "humidity": "UMIDADE RELATIVA DO AR, HORARIA (%)",
    "wind": "VENTO, VELOCIDADE HORARIA (m/s)",
    "pressure": "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)",
}


def read_inmet_csv(path: str | Path) -> pd.DataFrame:
    with Path(path).open(encoding="latin1") as file:
        metadata = dict(line.strip().split(";", 1) for line in [next(file) for _ in range(8)])

    frame = pd.read_csv(path, skiprows=8, sep=";", decimal=",", encoding="latin1")
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")].copy()
    frame["estacao"] = metadata["ESTACAO:"]
    frame["datetime"] = pd.to_datetime(
        frame["Data"] + " " + frame["Hora UTC"].str.replace(" UTC", "", regex=False),
        format="%Y/%m/%d %H%M",
    )
    return frame


def read_inmet_station(path: str | Path) -> dict[str, str]:
    with Path(path).open(encoding="latin1") as file:
        metadata = dict(line.strip().split(";", 1) for line in [next(file) for _ in range(8)])
    return {
        "estacao": metadata["ESTACAO:"],
        "latitude": metadata["LATITUDE:"].replace(",", "."),
        "longitude": metadata["LONGITUDE:"].replace(",", "."),
    }


def read_s2id_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, skiprows=4, sep=";", encoding="latin1")
    match = re.search(r"\d{4}", str(path))
    frame["ano"] = int(match.group()) if match else pd.NA
    return frame


def read_ibge_kml(path: str | Path) -> dict[str, str] | None:
    namespace = {"k": "http://www.opengis.net/kml/2.2"}
    root = ET.parse(path).getroot()
    for placemark in root.findall(".//k:Placemark", namespace):
        values = {
            item.attrib["name"]: item.text
            for item in placemark.findall(".//k:SimpleData", namespace)
        }
        if values.get("CT_LOCALIDADE") == "Cidade":
            return {
                "cidade": values.get("NM_LOCALIDADE"),
                "latitude": values.get("LAT_LOCALIDADE"),
                "longitude": values.get("LONG_LOCALIDADE"),
            }
    return None


def relate_cities_stations(cities: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    city_points = cities.assign(_key=1)
    station_points = stations.assign(_key=1)
    distances = city_points.merge(station_points, on="_key", suffixes=("_cidade", "_estacao"))
    distances["distancia"] = (
        (distances["latitude_cidade"] - distances["latitude_estacao"]) ** 2
        + (distances["longitude_cidade"] - distances["longitude_estacao"]) ** 2
    )
    closest = distances.groupby("cidade")["distancia"].idxmin()
    return (
        distances.loc[closest, ["cidade", "estacao"]]
        .sort_values("cidade")
        .reset_index(drop=True)
    )


def impute_weather_values(
    inmet: pd.DataFrame,
    stations: pd.DataFrame,
) -> pd.DataFrame:
    """Preenche clima pelo ano anterior e, depois, pela estação mais próxima."""
    frame = inmet.copy()
    for column in WEATHER_COLUMNS.values():
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    coordinates = stations.drop_duplicates("estacao").set_index("estacao")[["latitude", "longitude"]]
    frame["latitude"] = frame["estacao"].map(coordinates["latitude"])
    frame["longitude"] = frame["estacao"].map(coordinates["longitude"])
    frame["_previous_datetime"] = frame["datetime"] - pd.DateOffset(years=1)

    for column in WEATHER_COLUMNS.values():
        previous = frame[["estacao", "datetime", column]].dropna(subset=[column])
        previous = previous.rename(
            columns={"datetime": "_previous_datetime", column: "_previous_value"}
        )
        frame = frame.merge(
            previous,
            on=["estacao", "_previous_datetime"],
            how="left",
        )
        missing = frame[column].isna()
        frame.loc[missing, column] = frame.loc[missing, "_previous_value"]
        frame = frame.drop(columns="_previous_value")

        values = frame.pivot_table(
            index="datetime", columns="estacao", values=column, aggfunc="first"
        ).reindex(columns=coordinates.index)
        if values.empty:
            continue
        station_coordinates = coordinates.reindex(values.columns)
        distance_matrix = (
            station_coordinates.to_numpy()[:, None, :]
            - station_coordinates.to_numpy()[None, :, :]
        )
        distance_matrix = (distance_matrix ** 2).sum(axis=2)
        for target_index, station in enumerate(values.columns):
            source_order = np.argsort(distance_matrix[target_index])
            nearest_values = values.iloc[:, source_order].bfill(axis=1).iloc[:, 0]
            values[station] = values[station].fillna(nearest_values)
        lookup_index = pd.MultiIndex.from_arrays(
            [frame["datetime"], frame["estacao"]]
        )
        replacements = values.stack(future_stack=True)
        replacement_values = pd.Series(lookup_index.map(replacements), index=frame.index)
        frame[column] = frame[column].fillna(replacement_values)

    return frame.drop(columns=["_previous_datetime", "latitude", "longitude"])


def load_data(
    inmet_dir: str | Path,
    s2id_dir: str | Path,
    ibge_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    inmet_paths = sorted(
        path for path in Path(inmet_dir).rglob("*") if path.suffix.lower() == ".csv"
    )
    s2id_paths = sorted(
        path for path in Path(s2id_dir).glob("*") if path.suffix.lower() == ".csv"
    )
    ibge_paths = sorted(Path(ibge_dir).rglob("*.kml"))

    inmet = pd.concat((read_inmet_csv(path) for path in inmet_paths), ignore_index=True)
    s2id = pd.concat((read_s2id_csv(path) for path in s2id_paths), ignore_index=True)
    ibge = pd.DataFrame(
        filter(None, (read_ibge_kml(path) for path in ibge_paths)),
        columns=["cidade", "latitude", "longitude"],
    )
    ibge[["latitude", "longitude"]] = ibge[["latitude", "longitude"]].apply(
        pd.to_numeric, errors="coerce"
    )
    ibge = ibge.dropna(subset=["cidade", "latitude", "longitude"]).drop_duplicates()

    stations = pd.DataFrame(read_inmet_station(path) for path in inmet_paths).drop_duplicates()
    stations[["latitude", "longitude"]] = stations[["latitude", "longitude"]].apply(
        pd.to_numeric, errors="coerce"
    )
    stations = stations.dropna()
    inmet = impute_weather_values(inmet, stations)

    return {
        "inmet": inmet,
        "s2id": s2id,
        "ibge": ibge,
        "cidade_estacao": relate_cities_stations(ibge, stations),
    }


def _normalise_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char)).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def prepare_training_data(
    data: dict[str, pd.DataFrame],
    window_days: int = 3,
    positive_ratio: float = 1.0,
    noise_scale: float = 0.01,
) -> pd.DataFrame:
    if window_days < 1:
        raise ValueError("window_days deve ser maior ou igual a 1.")
    if positive_ratio <= 0:
        raise ValueError("positive_ratio deve ser maior que zero.")
    if noise_scale < 0:
        raise ValueError("noise_scale deve ser maior ou igual a zero.")
    inmet = data["inmet"].copy()
    for column in WEATHER_COLUMNS.values():
        inmet[column] = pd.to_numeric(inmet[column], errors="coerce")
    inmet["data"] = inmet["datetime"].dt.floor("D")
    daily = inmet.groupby(["estacao", "data"], as_index=False).agg(
        precip_total=(WEATHER_COLUMNS["precip"], "sum"),
        precip_max=(WEATHER_COLUMNS["precip"], "max"),
        temp_mean=(WEATHER_COLUMNS["temp"], "mean"),
        humidity_mean=(WEATHER_COLUMNS["humidity"], "mean"),
        wind_max=(WEATHER_COLUMNS["wind"], "max"),
        pressure_mean=(WEATHER_COLUMNS["pressure"], "mean"),
    )
    daily = daily.merge(data["cidade_estacao"], on="estacao", how="left")
    daily["cidade_normalizada"] = daily["cidade"].map(_normalise_text)

    events = data["s2id"].copy()
    events["data"] = pd.to_datetime(events["Registro"], dayfirst=True, errors="coerce").dt.floor("D")
    events["cidade_normalizada"] = events["Município"].map(_normalise_text)
    events["desastre"] = events["COBRADE"].map(_normalise_text)
    events = events.dropna(subset=["data"])
    labels = events.assign(valor=1).pivot_table(
        index=["cidade_normalizada", "data"],
        columns="desastre",
        values="valor",
        aggfunc="max",
        fill_value=0,
    ).reset_index()
    for name, column in zip(DISASTERS, TARGET_COLUMNS):
        matching = next((source for source in labels.columns if name in source), None)
        labels[column] = labels[matching] if matching else 0

    daily = daily.merge(
        labels[["cidade_normalizada", "data", *TARGET_COLUMNS]],
        on=["cidade_normalizada", "data"],
        how="left",
    ).fillna({column: 0 for column in TARGET_COLUMNS})
    daily["periodo"] = (
        daily["data"] - daily.groupby("estacao")["data"].transform("min")
    ).dt.days // window_days
    aggregations = {
        "data": "min",
        "cidade": "first",
        "precip_total": "mean",
        "precip_max": "max",
        "temp_mean": "mean",
        "humidity_mean": "mean",
        "wind_max": "max",
        "pressure_mean": "mean",
        **{column: "max" for column in TARGET_COLUMNS},
    }
    training_data = (
        daily.groupby(["estacao", "periodo"], as_index=False)
        .agg(aggregations)
        .drop(columns="periodo")
    )
    positive_rows = training_data[TARGET_COLUMNS].any(axis=1)
    positive = training_data.loc[positive_rows]
    negative = training_data.loc[~positive_rows]
    if positive.empty:
        return training_data
    synthetic_count = max(0, int(len(negative) * positive_ratio) - len(positive))
    if synthetic_count:
        rng = np.random.default_rng(42)
        synthetic = positive.sample(
            n=synthetic_count,
            replace=True,
            random_state=42,
        ).reset_index(drop=True)
        numeric_columns = [
            column for column in [
                "precip_total", "precip_max", "temp_mean", "humidity_mean",
                "wind_max", "pressure_mean",
            ] if column in synthetic
        ]
        for column in numeric_columns:
            deviation = positive[column].std()
            if pd.notna(deviation) and deviation > 0:
                synthetic[column] += rng.normal(0, deviation * noise_scale, len(synthetic))
        for column in ["precip_total", "precip_max", "wind_max"]:
            if column in synthetic:
                synthetic[column] = synthetic[column].clip(lower=0)
        if "humidity_mean" in synthetic:
            synthetic["humidity_mean"] = synthetic["humidity_mean"].clip(0, 100)
        training_data = pd.concat([positive, synthetic, negative], ignore_index=True)
    return training_data.sample(frac=1, random_state=42).reset_index(drop=True)


def train_models(
    training_data: pd.DataFrame,
    random_state: int = 42,
    positive_ratio: float = 1.0,
    noise_scale: float = 0.01,
) -> dict[str, dict]:
    training_data = training_data.copy()
    if "estacao" not in training_data:
        training_data["estacao"] = "desconhecida"
    wind_column = "wind_max" if "wind_max" in training_data else "wind_mean"
    feature_columns = [
        "estacao", "precip_total", "temp_mean", "humidity_mean", wind_column,
        "pressure_mean",
    ]
    if "precip_max" in training_data:
        feature_columns.insert(2, "precip_max")
    sampled_data = training_data.sample(frac=1, random_state=random_state)
    features = pd.get_dummies(sampled_data[feature_columns], columns=["estacao"]).fillna(0)
    feature_columns = features.columns.tolist()
    targets = sampled_data[TARGET_COLUMNS].astype(int)
    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            MultiOutputClassifier(LogisticRegression(max_iter=500, class_weight="balanced")),
        ),
        "random_forest": MultiOutputClassifier(
            RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight="balanced")
        ),
        "hist_gradient_boosting": MultiOutputClassifier(
            HistGradientBoostingClassifier(random_state=random_state)
        ),
    }
    return {
        name: {"model": model.fit(features, targets), "features": feature_columns}
        for name, model in candidates.items()
    }


def evaluate_models(
    training_data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    noise_scale: float = 0.01,
) -> dict[str, dict]:
    train_data, test_data = train_test_split(
        training_data, test_size=test_size, random_state=random_state
    )
    models = train_models(
        train_data,
        random_state=random_state,
        noise_scale=noise_scale,
    )
    targets = test_data[TARGET_COLUMNS].astype(int)

    for result in models.values():
        test_data = test_data.copy()
        if "estacao" not in test_data:
            test_data["estacao"] = "desconhecida"
        features = pd.get_dummies(test_data, columns=["estacao"]).reindex(
            columns=result["features"], fill_value=0
        ).fillna(0)
        predictions = result["model"].predict(features)
        result["metrics"] = {
            "accuracy": float(accuracy_score(targets, predictions)),
            "precision_macro": float(
                precision_score(targets, predictions, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(targets, predictions, average="macro", zero_division=0)
            ),
            "f1_macro": float(
                f1_score(targets, predictions, average="macro", zero_division=0)
            ),
        }
    return models


def predict_disaster_probabilities(
    models: dict[str, dict],
    inputs: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Calcula a probabilidade de cada desastre para cada entrada."""
    values = inputs.copy()
    if "estacao" not in values:
        values["estacao"] = "desconhecida"
    predictions = {}
    for name, result in models.items():
        features = pd.get_dummies(values, columns=["estacao"]).reindex(
            columns=result["features"], fill_value=0
        ).fillna(0)
        probabilities = result["model"].predict_proba(features)
        predictions[name] = pd.DataFrame(
            {
                disaster: probability[:, 1]
                for disaster, probability in zip(DISASTERS, probabilities)
            },
            index=inputs.index,
        )
    return predictions


def export_best_model(
    training_data: pd.DataFrame,
    evaluated_models: dict[str, dict],
    output_path: str | Path,
    city_station: pd.DataFrame | None = None,
    selection_metric: str = "f1_macro",
) -> dict:
    """Seleciona pelo resultado de avaliação e retreina o vencedor com todos os dados."""
    scores = {
        name: result.get("metrics", {}).get(selection_metric)
        for name, result in evaluated_models.items()
    }
    scores = {
        name: float(score)
        for name, score in scores.items()
        if score is not None and np.isfinite(score)
    }
    if not scores:
        raise ValueError(f"Nenhum modelo possui a métrica válida {selection_metric!r}.")

    best_name = max(scores, key=scores.get)
    fitted_models = train_models(training_data)
    best_fit = fitted_models[best_name]
    artifact = {
        "model_name": best_name,
        "model": best_fit["model"],
        "features": best_fit["features"],
        "metrics": evaluated_models[best_name]["metrics"],
        "selection_metric": selection_metric,
        "selection_score": scores[best_name],
        "trained_at": datetime.now().astimezone().isoformat(),
        "training_rows": len(training_data),
        "city_station": (
            city_station.to_dict(orient="records") if city_station is not None else []
        ),
    }
    save_model_archive(artifact, output_path)
    return artifact


def save_model_archive(artifact: dict, output_path: str | Path) -> Path:
    """Grava o artefato compactado em ZIP, pronto para versionar no GitHub."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer, compress=0)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr(MODEL_FILE_NAME, buffer.getvalue())
    return target


def load_model_archive(path: str | Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        return joblib.load(io.BytesIO(archive.read(MODEL_FILE_NAME)))
