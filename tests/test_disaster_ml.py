import pandas as pd

from src.disaster_ml import (
	DISASTERS,
	TARGET_COLUMNS,
	WEATHER_COLUMNS,
	read_inmet_csv,
	read_s2id_csv,
	relate_cities_stations,
	impute_weather_values,
	_normalise_text,
	evaluate_models,
	export_best_model,
	prepare_training_data,
	predict_disaster_probabilities,
	train_models,
)


def test_model_metadata_is_ready():
	assert len(DISASTERS) == len(TARGET_COLUMNS)
	assert all(column.startswith("desastre_") for column in TARGET_COLUMNS)
	assert set(WEATHER_COLUMNS) == {"precip", "temp", "humidity", "wind", "pressure"}
	assert _normalise_text("Herval D`Oeste") == _normalise_text("Herval d'Oeste")


def test_readers_and_city_station_relation(tmp_path):
	inmet_path = tmp_path / "INMET_2025.CSV"
	inmet_path.write_text(
		"ESTACAO:;TESTE\n"
		"LATITUDE:;-27,0\n"
		"LONGITUDE:;-50,0\n"
		"META3:;valor\nMETA4:;valor\nMETA5:;valor\nMETA6:;valor\nMETA7:;valor\n"
		"Data;Hora UTC;PRECIPITAÇÃO TOTAL, HORÁRIO (mm)\n"
		"2025/01/01;1200 UTC;10,5\n",
		encoding="latin1",
	)
	s2id_path = tmp_path / "Danos_Informados (2025).csv"
	s2id_path.write_text(
		"ignorado\nignorado\nignorado\nignorado\nMunicípio;COBRADE\nChapecó;Enxurrada\n",
		encoding="latin1",
	)

	inmet = read_inmet_csv(inmet_path)
	s2id = read_s2id_csv(s2id_path)
	relation = relate_cities_stations(
		pd.DataFrame({"cidade": ["Norte", "Sul"], "latitude": [0, 10], "longitude": [0, 10]}),
		pd.DataFrame({"estacao": ["A", "B"], "latitude": [0, 10], "longitude": [0, 10]}),
	)

	assert inmet.loc[0, "estacao"] == "TESTE"
	assert inmet.loc[0, "datetime"] == pd.Timestamp("2025-01-01 12:00")
	assert s2id.loc[0, "ano"] == 2025
	assert relation.set_index("cidade")["estacao"].to_dict() == {"Norte": "A", "Sul": "B"}


def test_impute_weather_uses_previous_year_then_nearest_station():
	date = pd.to_datetime(["2020-01-01 12:00", "2021-01-01 12:00", "2021-01-01 12:00"])
	columns = {column: [None, None, None] for column in WEATHER_COLUMNS.values()}
	columns[WEATHER_COLUMNS["temp"]] = [20.0, None, None]
	columns[WEATHER_COLUMNS["wind"]] = [None, None, 5.0]
	inmet = pd.DataFrame({"estacao": ["A", "A", "B"], "datetime": date, **columns})
	stations = pd.DataFrame({
		"estacao": ["A", "B"],
		"latitude": [0.0, 1.0],
		"longitude": [0.0, 0.0],
	})

	result = impute_weather_values(inmet, stations)

	assert result.loc[1, WEATHER_COLUMNS["temp"]] == 20.0
	assert result.loc[1, WEATHER_COLUMNS["wind"]] == 5.0


def test_train_models_returns_fitted_models():
	rows = []
	for index in range(8):
		rows.append({
			"precip_total": float(index),
			"temp_mean": 20 + index,
			"humidity_mean": 70 + index,
			"wind_mean": 3 + index,
			"pressure_mean": 1000 - index,
			**{column: index % 2 for column in TARGET_COLUMNS},
		})

	models = train_models(pd.DataFrame(rows))

	assert set(models) == {"logistic_regression", "random_forest", "hist_gradient_boosting"}
	assert all(result["features"] for result in models.values())


def test_evaluate_models_returns_metrics():
	rows = []
	for index in range(20):
		rows.append({
			"precip_total": float(index),
			"temp_mean": 20 + index,
			"humidity_mean": 70 + index,
			"wind_mean": 3 + index,
			"pressure_mean": 1000 - index,
			**{column: index % 2 for column in TARGET_COLUMNS},
		})

	models = evaluate_models(pd.DataFrame(rows), test_size=0.25)

	assert all(set(result["metrics"]) == {
		"accuracy", "precision_macro", "recall_macro", "f1_macro"
	} for result in models.values())


def test_export_best_model_uses_f1_and_saves_city_station_mapping(tmp_path):
	rows = []
	for index in range(8):
		rows.append({
			"estacao": "TESTE",
			"precip_total": float(index),
			"precip_max": float(index),
			"temp_mean": 20 + index,
			"humidity_mean": 70 + index,
			"wind_max": 3 + index,
			"pressure_mean": 1000 - index,
			**{column: index % 2 for column in TARGET_COLUMNS},
		})
	training_data = pd.DataFrame(rows)
	evaluated = {
		"logistic_regression": {"metrics": {"f1_macro": 0.4}},
		"random_forest": {"metrics": {"f1_macro": 0.8}},
		"hist_gradient_boosting": {"metrics": {"f1_macro": 0.6}},
	}
	city_station = pd.DataFrame({"cidade": ["Chapecó"], "estacao": ["TESTE"]})
	output_path = tmp_path / "best_model.zip"

	artifact = export_best_model(
		training_data,
		evaluated,
		output_path,
		city_station=city_station,
	)

	assert artifact["model_name"] == "random_forest"
	assert artifact["selection_score"] == 0.8
	assert artifact["city_station"] == [{"cidade": "Chapecó", "estacao": "TESTE"}]
	assert output_path.exists()


def test_predict_disaster_probabilities_returns_each_disaster():
	rows = []
	for index in range(8):
		rows.append({
			"estacao": "TESTE",
			"precip_total": float(index),
			"precip_max": float(index),
			"temp_mean": 20 + index,
			"humidity_mean": 70 + index,
			"wind_max": 3 + index,
			"pressure_mean": 1000 - index,
			**{column: index % 2 for column in TARGET_COLUMNS},
		})
	models = train_models(pd.DataFrame(rows))
	inputs = pd.DataFrame([rows[0], rows[1]]).drop(columns=TARGET_COLUMNS)

	predictions = predict_disaster_probabilities(models, inputs)

	assert set(predictions) == set(models)
	assert all(list(result.columns) == list(DISASTERS) for result in predictions.values())
	assert all(result.shape == (2, len(DISASTERS)) for result in predictions.values())
	assert all(result.apply(lambda column: column.between(0, 1).all()).all() for result in predictions.values())
