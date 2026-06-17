from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "knn_depression_model.joblib"

USER_FEATURES = [
    "Gender",
    "Age",
    "Profession",
    "Academic Pressure",
    "CGPA",
    "Study Satisfaction",
    "Job Satisfaction",
    "Sleep Duration",
    "Dietary Habits",
    "Degree",
    "Have you ever had suicidal thoughts ?",
    "Work/Study Hours",
    "Financial Stress",
    "Family History of Mental Illness",
    "City_freq_City",
]

MODEL_FEATURES = [
    "Gender",
    "Age",
    "Profession",
    "Academic Pressure",
    "Work Pressure",
    "CGPA",
    "Study Satisfaction",
    "Job Satisfaction",
    "Sleep Duration",
    "Dietary Habits",
    "Degree",
    "Have you ever had suicidal thoughts ?",
    "Work/Study Hours",
    "Financial Stress",
    "Family History of Mental Illness",
    "City_freq_City",
]

BINARY_OPTIONS = {
    "Tidak": 0,
    "Ya": 1,
}

GENDER_OPTIONS = {
    "Gender 0": 0,
    "Gender 1": 1,
}

PROFESSION_OPTIONS = {
    "Profession 0": 0,
    "Profession 1": 1,
}

SLEEP_DURATION_OPTIONS = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
}

DIETARY_HABITS_OPTIONS = {
    "0": 0,
    "1": 1,
    "2": 2,
}


st.set_page_config(
    page_title="Prediksi Depresi Mahasiswa",
    page_icon=":brain:",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def prediction_label(prediction):
    if int(prediction) == 1:
        return "Depresi"

    if int(prediction) == 0:
        return "Tidak depresi"

    return str(prediction)


def read_uploaded_file(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    raise ValueError("Format file tidak didukung. Gunakan CSV atau Excel.")


def validate_columns(df):
    return [column for column in USER_FEATURES if column not in df.columns]


def prepare_input_data(df):
    input_df = df.copy()

    if "Work Pressure" not in input_df.columns:
        input_df["Work Pressure"] = 0

    input_df = input_df[MODEL_FEATURES].copy()

    for column in MODEL_FEATURES:
        input_df[column] = pd.to_numeric(input_df[column], errors="coerce")

    return input_df


def build_manual_input():
    input_data = {}

    st.subheader("Data Identitas dan Akademik")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", list(GENDER_OPTIONS.keys()))
        input_data["Gender"] = GENDER_OPTIONS[gender]

        input_data["Academic Pressure"] = st.slider(
            "Academic Pressure",
            min_value=0,
            max_value=5,
            value=3,
        )

        input_data["Study Satisfaction"] = st.slider(
            "Study Satisfaction",
            min_value=0,
            max_value=5,
            value=3,
        )

        sleep_duration = st.selectbox(
            "Sleep Duration",
            list(SLEEP_DURATION_OPTIONS.keys()),
        )
        input_data["Sleep Duration"] = SLEEP_DURATION_OPTIONS[sleep_duration]

        suicidal_thoughts = st.selectbox(
            "Have you ever had suicidal thoughts ?",
            list(BINARY_OPTIONS.keys()),
        )
        input_data["Have you ever had suicidal thoughts ?"] = BINARY_OPTIONS[
            suicidal_thoughts
        ]

    with col2:
        input_data["Age"] = st.number_input(
            "Age",
            min_value=18,
            max_value=59,
            value=21,
            step=1,
        )

        input_data["CGPA"] = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=7.50,
            step=0.01,
            format="%.2f",
        )

        input_data["Job Satisfaction"] = st.slider(
            "Job Satisfaction",
            min_value=0,
            max_value=5,
            value=0,
        )

        dietary_habits = st.selectbox(
            "Dietary Habits",
            list(DIETARY_HABITS_OPTIONS.keys()),
        )
        input_data["Dietary Habits"] = DIETARY_HABITS_OPTIONS[dietary_habits]

        family_history = st.selectbox(
            "Family History of Mental Illness",
            list(BINARY_OPTIONS.keys()),
        )
        input_data["Family History of Mental Illness"] = BINARY_OPTIONS[
            family_history
        ]

    with col3:
        profession = st.selectbox("Profession", list(PROFESSION_OPTIONS.keys()))
        input_data["Profession"] = PROFESSION_OPTIONS[profession]

        input_data["Degree"] = st.number_input(
            "Degree",
            min_value=0,
            max_value=26,
            value=10,
            step=1,
        )

        input_data["Work/Study Hours"] = st.number_input(
            "Work/Study Hours",
            min_value=0,
            max_value=12,
            value=6,
            step=1,
        )

        input_data["Financial Stress"] = st.slider(
            "Financial Stress",
            min_value=1,
            max_value=5,
            value=3,
        )

        input_data["City_freq_City"] = st.number_input(
            "City_freq_City",
            min_value=0.0,
            max_value=1.0,
            value=0.03,
            step=0.001,
            format="%.6f",
        )

    input_data["Work Pressure"] = 0

    return pd.DataFrame([input_data])[MODEL_FEATURES]


def show_prediction_result(model, input_df, source_name):
    prediction = model.predict(input_df)[0]
    label = prediction_label(prediction)

    st.subheader("Hasil Prediksi")

    if int(prediction) == 1:
        st.error(f"Hasil prediksi {source_name}: {label}")
    else:
        st.success(f"Hasil prediksi {source_name}: {label}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        class_names = [prediction_label(class_value) for class_value in model.classes_]

        probability_df = pd.DataFrame(
            {
                "Class": class_names,
                "Probability": probabilities,
            }
        )

        st.subheader("Probabilitas Prediksi")
        st.dataframe(probability_df, use_container_width=True)


def show_batch_prediction(model, df, input_df):
    predictions = model.predict(input_df)

    result_df = df.copy()
    result_df["depresion_prediction"] = predictions
    result_df["depresion_label"] = [
        prediction_label(prediction) for prediction in predictions
    ]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)

        for index, class_value in enumerate(model.classes_):
            label = prediction_label(class_value).lower().replace(" ", "_")
            column_name = f"probability_{label}"
            result_df[column_name] = probabilities[:, index]

    st.subheader("Hasil Prediksi")
    st.dataframe(result_df, use_container_width=True)

    csv_result = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Hasil Prediksi",
        data=csv_result,
        file_name="hasil_prediksi_depresi_mahasiswa.csv",
        mime="text/csv",
    )


st.title("Prediksi Depresi Mahasiswa")
st.write(
    "Aplikasi ini menggunakan model KNN dari file `knn_depression_model.joblib` "
    "untuk memprediksi label `depresion` berdasarkan data mahasiswa."
)
st.warning(
    "Hasil aplikasi ini adalah prediksi model machine learning untuk kebutuhan pembelajaran, "
    "bukan diagnosis medis."
)

try:
    model = load_model()
    st.sidebar.success("Model aktif: knn_depression_model.joblib")
except Exception as error:
    st.sidebar.error(f"Model gagal dimuat: {error}")
    st.stop()

st.sidebar.header("Menu")
menu = st.sidebar.radio("Pilih Menu", ["Prediksi Manual", "Prediksi CSV/Excel"])

if menu == "Prediksi Manual":
    st.header("Prediksi Manual")
    manual_input_df = build_manual_input()

    st.subheader("Data yang Dikirim ke Model")
    st.dataframe(manual_input_df, use_container_width=True)

    if st.button("Prediksi Manual", type="primary"):
        try:
            show_prediction_result(model, manual_input_df, "data manual")
        except Exception as error:
            st.error(f"Terjadi error saat prediksi: {error}")

if menu == "Prediksi CSV/Excel":
    st.header("Prediksi CSV/Excel")
    st.write(
        "Upload file CSV atau Excel yang memiliki kolom fitur sesuai model. "
        "Kolom `Work Pressure` boleh tidak ada karena akan diisi otomatis dengan nilai 0."
    )

    uploaded_file = st.file_uploader(
        "Upload file CSV atau Excel",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file is not None:
        try:
            uploaded_df = read_uploaded_file(uploaded_file)

            st.subheader("Preview Data")
            st.dataframe(uploaded_df.head(), use_container_width=True)

            missing_columns = validate_columns(uploaded_df)

            if missing_columns:
                st.error("Kolom pada file belum lengkap.")
                st.write("Kolom yang belum ada:")
                st.write(missing_columns)
                st.write("Kolom yang dibutuhkan:")
                st.write(USER_FEATURES)
            else:
                batch_input_df = prepare_input_data(uploaded_df)

                invalid_values = batch_input_df.isnull().sum()
                invalid_values = invalid_values[invalid_values > 0]

                if not invalid_values.empty:
                    st.error(
                        "Ada nilai kosong atau data non-numerik pada kolom fitur. "
                        "Silakan periksa kembali file CSV/Excel."
                    )
                    st.write("Jumlah nilai bermasalah per kolom:")
                    st.write(invalid_values)
                else:
                    st.success("Format data valid. Data siap diprediksi.")
                    st.subheader("Data yang Dikirim ke Model")
                    st.dataframe(batch_input_df.head(), use_container_width=True)

                    if st.button("Prediksi File", type="primary"):
                        try:
                            show_batch_prediction(model, uploaded_df, batch_input_df)
                        except Exception as error:
                            st.error(f"Terjadi error saat prediksi: {error}")

        except Exception as error:
            st.error(f"File gagal dibaca: {error}")
