import streamlit as st
import pandas as pd
import pickle

# basic page setup - centered layout looks cleaner for a form like this
st.set_page_config(
    page_title="Student Health Risk Predictor",
    page_icon="🩺",
    layout="centered",
)

# some small css tweaks so it doesn't look like a default streamlit app - just spacing, font size, and button colour - nothing fancy
st.markdown("""
    <style>
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 2.5rem;
            max-width: 700px;
        }
        h1 {
            font-size: 1.6rem;
            font-weight: 600;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #6b7280;
            font-size: 0.95rem;
            margin-bottom: 2rem;
        }
        div[data-testid="stForm"] {
            border: none;
            padding: 0;
        }
        div.stButton > button {
            width: 100%;
            border-radius: 6px;
            padding: 0.6rem 0;
            font-weight: 500;
            background-color: #111827;
            color: white;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #374151;
            color: white;
        }
        .result-box {
            border-radius: 8px;
            padding: 1.2rem 1.5rem;
            margin-top: 1.5rem;
            font-size: 1.05rem;
        }
    </style>
""", unsafe_allow_html=True)


# loading the model and the encoders once, so the app doesn't reload them
# every time someone clicks the button
@st.cache_resource
def load_artifacts():
    # using gradient boosting now instead of random forest
    # it came out slightly ahead when i compared 4 models
    with open("gb_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)  # these turn the text columns into numbers

    with open("columns.pkl", "rb") as f:
        columns = pickle.load(f)  # column order the model expects

    return model, encoders, columns


model, encoders, columns = load_artifacts()

# leaving this here in case the columns don't line up again later
# st.write("Expected columns:", columns)

# page title and short description
st.title("Student Health Risk Predictor")
st.markdown('<p class="subtitle">Enter lifestyle details to estimate health risk category.</p>', unsafe_allow_html=True)

# the actual input form
with st.form("input_form"):

    st.markdown("**Lifestyle**")
    c1, c2 = st.columns(2)
    with c1:
        sleep_duration = st.slider("Sleep duration (hrs)", 3.0, 10.0, 7.0, 0.5)
        sleep_quality = st.selectbox("Sleep quality", encoders["sleep_quality"].classes_)
        stress_level = st.selectbox("Stress level", encoders["stress_level"].classes_)
    with c2:
        physical_activity_level = st.selectbox("Activity level", encoders["physical_activity_level"].classes_)
        step_count = st.number_input("Daily steps", 0, 30000, 6000, 500)
        exercise_duration = st.number_input("Exercise (mins/day)", 0, 180, 20, 5)

    st.markdown("**Profile**")
    c3, c4 = st.columns(2)
    with c3:
        bmi = st.number_input("BMI", 12.0, 45.0, 22.0, 0.1)
        gender = st.selectbox("Gender", encoders["gender"].classes_)
    with c4:
        diet_type = st.selectbox("Diet type", encoders["diet_type"].classes_)
        smoking_alcohol = st.selectbox("Smoking / alcohol", encoders["smoking_alcohol"].classes_)

    # these two were missing before and caused a KeyError when predicting
    # the model needs every column it was trained on, so adding them here
    st.markdown("**Other**")
    c5, c6 = st.columns(2)
    with c5:
        calorie_expenditure = st.number_input("Calorie expenditure (kcal/day)", 800, 5000, 2200, 50)
    with c6:
        water_intake = st.number_input("Water intake (litres/day)", 0.0, 6.0, 2.0, 0.1)

    heart_rate = st.number_input("heart rate (bpm)", 40, 140, 72, 1)

    submitted = st.form_submit_button("Predict health risk")

# runs only after the button is clicked
if submitted:

    # putting everything into one row, encoding the text fields the same
    # way they were encoded during training
    raw_input = {
        "sleep_duration": sleep_duration,
        "sleep_quality": encoders["sleep_quality"].transform([sleep_quality])[0],
        "stress_level": encoders["stress_level"].transform([stress_level])[0],
        "physical_activity_level": encoders["physical_activity_level"].transform([physical_activity_level])[0],
        "step_count": step_count,
        "exercise_duration": exercise_duration,
        "bmi": bmi,
        "gender": encoders["gender"].transform([gender])[0],
        "diet_type": encoders["diet_type"].transform([diet_type])[0],
        "smoking_alcohol": encoders["smoking_alcohol"].transform([smoking_alcohol])[0],
        "heart_rate": heart_rate,
        "calorie_expenditure": calorie_expenditure,
        "water_intake": water_intake,
    }

    # putting the columns back in the same order the model was trained with
    input_df = pd.DataFrame([raw_input])[columns]

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    proba_dict = dict(zip(model.classes_, proba))

    # simple colour per result instead of the default streamlit success/error boxes
    colour_map = {
        "at-risk": ("#fef2f2", "#991b1b"),
        "unhealthy": ("#fffbeb", "#92400e"),
        "fit": ("#f0fdf4", "#166534"),
    }
    bg, text = colour_map.get(prediction, ("#f3f4f6", "#111827"))

    st.markdown(f"""
        <div class="result-box" style="background-color:{bg}; color:{text};">
            <strong>Predicted category: {prediction}</strong><br>
            Confidence: {proba_dict[prediction]*100:.1f}%
        </div>
    """, unsafe_allow_html=True)

    # show the other probabilities too, in case someone wants to see the full picture
    with st.expander("View full probability breakdown"):
        for cls, p in sorted(proba_dict.items(), key=lambda x: -x[1]):
            st.write(f"{cls}: {p*100:.1f}%")