"""Interface de consulta municipal para o classificador de desastres."""

from __future__ import annotations

import io
import re
from pathlib import Path
import sys
import math

import joblib
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
municipality_svg_component = components.declare_component(
    "sc_municipality_svg_map",
    path=str(PROJECT_ROOT / "app" / "components" / "municipality_map"),
)

from src.disaster_ml import (
    _normalise_text,
    load_model_archive,
    predict_disaster_probabilities,
    read_ibge_kml,
)

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.zip"
IBGE_MUNICIPALITIES_URL = (
    "https://servicodados.ibge.gov.br/api/v3/malhas/estados/42"
    "?formato=application%2Fvnd.geo%2Bjson&intrarregiao=municipio"
)
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "America/Sao_Paulo"

def _risk_summary(probabilities: pd.Series) -> tuple[str, str | None]:
    highest_disaster = str(probabilities.idxmax())
    highest_probability = float(probabilities[highest_disaster])
    if highest_probability > 0.70:
        return f"{highest_probability:.1%}", highest_disaster
    return "Baixa chance de desastres", None


st.set_page_config(page_title="Risco de desastres | SC", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.8rem; max-width: 1500px; }
    h1 { color: #173b35; font-size: 2rem; }
    [data-testid="stMetric"] {
        background: #f1f5f2;
        padding: 12px;
        border-left: 3px solid #c24f3d;
        border-radius: 4px;
    }
    [data-testid="stMetric"] *,
    [data-testid="stMetric"] label {
        color: #172b25 !important;
        opacity: 1 !important;
    }
    [data-testid="stMetricValue"] { font-size: 1.45rem !important; line-height: 1.25; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, persist="disk")
def load_municipality_map() -> tuple[dict, pd.DataFrame]:
    """Obtém polígonos do IBGE e associa nomes/coordenadas das localidades locais."""
    places = []
    names_by_code = {}
    places_by_code = {}
    for path in (PROJECT_ROOT / "data" / "IBGE" / "SC").glob("*.kml"):
        match = re.search(r"_(\d{7})_localidades_", path.name)
        place = read_ibge_kml(path)
        if not match or not place:
            continue
        code = match.group(1)
        place.update(
            {
                "codigo_ibge": code,
                "latitude": pd.to_numeric(place["latitude"], errors="coerce"),
                "longitude": pd.to_numeric(place["longitude"], errors="coerce"),
            }
        )
        places.append(place)
        names_by_code[code] = place["cidade"]
        places_by_code[code] = place

    response = requests.get(IBGE_MUNICIPALITIES_URL, timeout=30)
    response.raise_for_status()
    geojson = response.json()
    for feature in geojson["features"]:
        code = str(feature["properties"].get("codarea", ""))
        feature["properties"]["municipio"] = names_by_code.get(code, code)
        place = places_by_code.get(code)
        if place:
            feature["properties"]["sede_latitude"] = float(place["latitude"])
            feature["properties"]["sede_longitude"] = float(place["longitude"])
    place_frame = pd.DataFrame(places).dropna(subset=["latitude", "longitude"])
    return geojson, place_frame


@st.cache_resource(show_spinner=False)
def load_model_artifact(path: str, modified_at: float) -> dict:
    del modified_at
    return load_model_archive(path)


@st.cache_data(ttl=43200, show_spinner=False)
def fetch_recent_weather(latitude: float, longitude: float) -> tuple[dict, str]:
    """Agrega as 24 horas mais recentes retornadas pelo modelo meteorológico."""
    response = requests.get(
        OPEN_METEO_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": (
                "precipitation,temperature_2m,relative_humidity_2m,"
                "wind_speed_10m,surface_pressure"
            ),
            "past_days": 1,
            "forecast_days": 1,
            "timezone": TIMEZONE,
            "wind_speed_unit": "ms",
        },
        timeout=30,
    )
    response.raise_for_status()
    hourly = pd.DataFrame(response.json()["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    now = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None).floor("h")
    recent = hourly.loc[
        (hourly["time"] <= now) & (hourly["time"] > now - pd.Timedelta(hours=24))
    ].tail(24)
    if recent.empty:
        raise ValueError("A fonte não retornou observações horárias recentes para este município.")

    weather = {
        "precip_total": float(recent["precipitation"].sum(min_count=1)),
        "precip_max": float(recent["precipitation"].max()),
        "temp_mean": float(recent["temperature_2m"].mean()),
        "humidity_mean": float(recent["relative_humidity_2m"].mean()),
        "wind_max": float(recent["wind_speed_10m"].max()),
        "pressure_mean": float(recent["surface_pressure"].mean()),
    }
    if any(pd.isna(value) for value in weather.values()):
        raise ValueError("A fonte retornou variáveis meteorológicas incompletas.")
    return weather, str(recent["time"].max())


def _coordinate_pairs(value: object):
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and all(
            isinstance(coordinate, (int, float)) for coordinate in value[:2]
        ):
            yield value
        else:
            for child in value:
                yield from _coordinate_pairs(child)


@st.cache_data(show_spinner=False, persist="disk")
def build_municipality_svg_data(geojson: dict) -> dict:
    width, height, padding = 1000.0, 700.0, 24.0
    positions = [
        position
        for feature in geojson["features"]
        for position in _coordinate_pairs(feature["geometry"]["coordinates"])
    ]
    min_longitude = min(position[0] for position in positions)
    max_longitude = max(position[0] for position in positions)
    min_latitude = min(position[1] for position in positions)
    max_latitude = max(position[1] for position in positions)
    longitude_factor = math.cos(
        math.radians((min_latitude + max_latitude) / 2)
    )
    min_projected_longitude = min_longitude * longitude_factor
    max_projected_longitude = max_longitude * longitude_factor
    longitude_span = max(max_projected_longitude - min_projected_longitude, 1e-6)
    latitude_span = max(max_latitude - min_latitude, 1e-6)
    scale = min(
        (width - padding * 2) / longitude_span,
        (height - padding * 2) / latitude_span,
    )
    drawn_width = longitude_span * scale
    drawn_height = latitude_span * scale
    offset_x = (width - drawn_width) / 2
    offset_y = (height - drawn_height) / 2

    def project(longitude: float, latitude: float) -> tuple[float, float]:
        return (
            offset_x + (longitude * longitude_factor - min_projected_longitude) * scale,
            offset_y + (max_latitude - latitude) * scale,
        )

    result = []
    for feature in geojson["features"]:
        geometry = feature["geometry"]
        polygons = (
            [geometry["coordinates"]]
            if geometry["type"] == "Polygon"
            else geometry["coordinates"]
        )
        path_parts = []
        feature_points = []
        for polygon in polygons:
            for ring in polygon:
                points = [project(point[0], point[1]) for point in ring]
                if len(points) < 3:
                    continue
                feature_points.extend(points)
                path_parts.append(
                    "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in points) + "Z"
                )
        if not path_parts:
            continue
        x_values = [point[0] for point in feature_points]
        y_values = [point[1] for point in feature_points]
        result.append(
            {
                "id": str(feature["properties"].get("codarea", "")),
                "name": feature["properties"]["municipio"],
                "path": "".join(path_parts),
                "bounds": [
                    min(x_values),
                    min(y_values),
                    max(x_values) - min(x_values),
                    max(y_values) - min(y_values),
                ],
            }
        )
    return {
        "version": "ibge-sc-municipalities-2022-v2-equirectangular",
        "features": result,
        "state_viewbox": f"0 0 {width:.0f} {height:.0f}",
    }


def main() -> None:
    st.title("Risco de desastres | Santa Catarina")
    st.caption("Estimativa exploratória baseada em clima recente e registros históricos municipais.")

    try:
        geojson, places = load_municipality_map()
    except (requests.RequestException, ValueError, KeyError) as error:
        st.error(f"Não foi possível carregar a malha municipal do IBGE: {error}")
        st.stop()

    artifact = None
    if MODEL_PATH.exists():
        artifact = load_model_artifact(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
        metric_value = artifact["metrics"].get(artifact["selection_metric"], 0)
        st.caption(
            f"Modelo: {artifact['model_name']} | "
            f"{artifact['selection_metric']}: {metric_value:.3f} | "
            f"treinado em {artifact['trained_at'][:10]}"
        )
    else:
        st.warning(
            "Modelo não encontrado em `models/best_model.zip`. "
            "Gere-o executando `src/main.ipynb`."
        )

    city_names = sorted(places["cidade"].dropna().unique())
    st.session_state["map_city_names"] = city_names
    map_data = build_municipality_svg_data(geojson)
    map_event = st.session_state.get("mapa_municipios_sc_svg")
    if isinstance(map_event, dict):
        event_id = map_event.get("event_id")
        event_type = map_event.get("event_type")
        if event_type == "map_ready" and map_event.get("version") == map_data["version"]:
            st.session_state["svg_map_loaded_version"] = map_data["version"]
        elif event_type == "features_needed":
            st.session_state.pop("svg_map_loaded_version", None)
        elif event_type == "selection" and event_id is not None and event_id != st.session_state.get("last_map_event_id"):
            clicked_city = map_event.get("city")
            st.session_state["last_map_event_id"] = event_id
            st.session_state["selected_city"] = clicked_city
            if clicked_city:
                st.session_state["risk_requested_city"] = clicked_city
            else:
                st.session_state.pop("risk_requested_city", None)

    selected_city = st.session_state.get("selected_city")
    map_features = (
        map_data["features"]
        if st.session_state.get("svg_map_loaded_version") != map_data["version"]
        else []
    )
    left, right = st.columns([1.8, 1], gap="large")
    with left:
        st.subheader("Municípios")
        st.caption("Passe o mouse para ver o município; clique nele para consultar o risco.")
        map_event = municipality_svg_component(
            features=map_features,
            state_viewbox=map_data["state_viewbox"],
            selected_city=selected_city,
            version=map_data["version"],
            key="mapa_municipios_sc_svg",
        )

    with right:
        st.subheader("Consulta municipal")
        selected_city = st.selectbox(
            "Município",
            city_names,
            index=None,
            placeholder="Selecione no mapa ou na lista",
            key="selected_city",
        )
        analyze = st.button("Consultar risco", type="primary", use_container_width=True)
        risk_requested = bool(
            selected_city
            and st.session_state.get("risk_requested_city") == selected_city
        )
        if risk_requested:
            st.session_state.pop("risk_requested_city", None)
        if selected_city and (analyze or risk_requested):
            if not artifact:
                st.info("Exporte o modelo pelo notebook para habilitar as previsões.")
            else:
                place = places.loc[places["cidade"] == selected_city].iloc[0]
                city_key = _normalise_text(selected_city)
                station_map = {
                    _normalise_text(item["cidade"]): item["estacao"]
                    for item in artifact.get("city_station", [])
                }
                station = station_map.get(city_key)
                if not station:
                    st.error("Não há estação meteorológica associada a este município no treino.")
                else:
                    try:
                        with st.spinner("Consultando clima recente e calculando estimativa..."):
                            weather, weather_time = fetch_recent_weather(
                                float(place["latitude"]), float(place["longitude"])
                            )
                            inputs = pd.DataFrame([{"estacao": station, **weather}])
                            model = {
                                artifact["model_name"]: {
                                    "model": artifact["model"],
                                    "features": artifact["features"],
                                }
                            }
                            probabilities = predict_disaster_probabilities(model, inputs)[
                                artifact["model_name"]
                            ].iloc[0]
                        risk_value, highest = _risk_summary(probabilities)
                        st.metric("Maior estimativa", risk_value, highest)
                        table = (
                            probabilities.rename("Probabilidade")
                            .rename(index={
                                "enxurradas": "Enxurradas",
                                "alagamentos": "Alagamentos",
                                "inundacoes": "Inundações",
                                "deslizamentos": "Deslizamentos",
                            })
                            .sort_values(ascending=False)
                            .mul(100)
                            .to_frame()
                        )
                        st.bar_chart(table, horizontal=True, x_label="Probabilidade (%)")
                        st.caption(f"Clima agregado até {weather_time} ({TIMEZONE}).")
                        st.dataframe(
                            pd.DataFrame(
                                {"Variável": list(weather), "Valor": list(weather.values())}
                            ),
                            hide_index=True,
                            use_container_width=True,
                        )
                        st.caption(
                            "As porcentagens são saídas do classificador, não probabilidades "
                            "calibradas nem um alerta oficial."
                        )
                    except (requests.RequestException, KeyError, ValueError) as error:
                        st.error(f"Falha ao obter o clima ou calcular a estimativa: {error}")

    st.caption(
        "Clima: Open-Meteo (cache de 12 h). Fronteiras: malha municipal do IBGE. "
        "Não substitui monitoramento ou alertas da Defesa Civil."
    )


if __name__ == "__main__":
    main()