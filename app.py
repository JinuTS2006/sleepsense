import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SleepSense",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ================================
       MAIN APP
    ================================= */

    .stApp {
        background-color: #080d1c;
    }

    .main .block-container {
        max-width: 1350px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ================================
       TEXT
    ================================= */

    h1, h2, h3, h4 {
        color: #f8fafc !important;
    }

    p {
        color: #94a3b8;
    }


    /* ================================
       SIDEBAR
    ================================= */

    section[data-testid="stSidebar"] {
        background-color: #0d1426;
        border-right: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #cbd5e1 !important;
    }


    /* ================================
       SELECTBOX
    ================================= */

    div[data-baseweb="select"] > div {
        background-color: #111827;
        border-color: #334155;
        color: #f8fafc;
    }


    /* ================================
       METRIC CARDS
    ================================= */

    div[data-testid="stMetric"] {
        background-color: #101827;
        border: 1px solid #26344e;
        border-radius: 16px;
        padding: 18px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }


    /* ================================
       BUTTON
    ================================= */

    .stButton > button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 20px;
        font-size: 16px;
        font-weight: 700;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }


    /* ================================
       INPUTS
    ================================= */

    div[data-baseweb="input"] > div {
        background-color: #111827;
        border-color: #334155;
    }

    input {
        color: #f8fafc !important;
    }


    /* ================================
       SLIDER
    ================================= */

    div[data-testid="stSlider"] {
        color: #cbd5e1;
    }


    /* ================================
       EXPANDER
    ================================= */

    div[data-testid="stExpander"] {
        background-color: #0f1727;
        border: 1px solid #26344e;
        border-radius: 14px;
    }


    /* ================================
       INFO / WARNING
    ================================= */

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }


    /* ================================
       DIVIDER
    ================================= */

    hr {
        border-color: #1e293b;
    }


    /* ================================
       RESULT BOX
    ================================= */

    .result-box {
        background-color: #101827;
        border: 1px solid #31506e;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 25px;
    }

    .result-number {
        font-size: 52px;
        font-weight: 800;
        color: #f8fafc;
    }

    .result-text {
        font-size: 18px;
        color: #94a3b8;
    }


    /* ================================
       RECOMMENDATION
    ================================= */

    .recommendation {
        background-color: #101827;
        border: 1px solid #26344e;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
        color: #cbd5e1;
    }


    /* ================================
       FOOTER
    ================================= */

    .footer {
        text-align: center;
        color: #53627a;
        font-size: 12px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #1e293b;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL METRICS
# ============================================================

MODEL_METRICS = {

    "Ridge Regression": {
        "R2": 0.7806,
        "MAE": 0.6288,
        "RMSE": 0.7989
    },

    "Gradient Boosting": {
        "R2": 0.8040,
        "MAE": 0.5991,
        "RMSE": 0.7552
    },

    "CatBoost": {
        "R2": 0.8044,
        "MAE": 0.5989,
        "RMSE": 0.7545
    }

}


# ============================================================
# LOAD MODELS
# ============================================================

# Backward-compatibility for models trained with scikit-learn <= 1.6
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass


@st.cache_resource
def load_models():

    ridge_model = joblib.load(
        "sleepsense_ridge_model.pkl"
    )

    gradient_model = joblib.load(
        "sleepsense_gradient_boosting_model.pkl"
    )

    catboost_model = CatBoostRegressor()

    catboost_model.load_model(
        "sleepsense_catboost_model.cbm"
    )

    config = joblib.load(
        "sleepsense_config.pkl"
    )

    return (
        ridge_model,
        gradient_model,
        catboost_model,
        config
    )


# ============================================================
# LOAD MODELS
# ============================================================

try:

    (
        ridge_model,
        gradient_model,
        catboost_model,
        config
    ) = load_models()

except Exception as e:

    st.error(
        f"❌ Model loading failed: {e}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🌙 SleepSense")

    st.caption(
        "AI-Based Sleep Health Assessment"
    )

    st.divider()

    st.subheader("🤖 Prediction Model")

    selected_model = st.selectbox(
        "Choose a model",
        [
            "CatBoost",
            "Gradient Boosting",
            "Ridge Regression"
        ],
        index=0
    )

    st.divider()

    st.subheader("📌 About")

    st.write(
        "SleepSense uses machine learning to "
        "estimate sleep quality from phone usage, "
        "lifestyle and related factors."
    )

    st.divider()

    st.caption(
        "3 trained regression models available"
    )


# ============================================================
# HERO SECTION
# ============================================================

st.title("🌙 SleepSense")

st.subheader(
    "AI-Based Phone Usage & Sleep Health Assessment"
)

st.write(
    "Machine Learning • Sleep Quality Prediction"
)

st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("📊 Model Performance")

st.caption(
    "Comparison of the three trained regression models."
)

col1, col2, col3 = st.columns(3)


# ---------------- RIDGE ----------------

with col1:

    st.subheader("🟣 Ridge Regression")

    st.metric(
        "R² Score",
        "0.7806"
    )

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:

        st.metric(
            "MAE",
            "0.6288"
        )

    with metric_col2:

        st.metric(
            "RMSE",
            "0.7989"
        )


# ---------------- GRADIENT BOOSTING ----------------

with col2:

    st.subheader("🔵 Gradient Boosting")

    st.metric(
        "R² Score",
        "0.8040"
    )

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:

        st.metric(
            "MAE",
            "0.5991"
        )

    with metric_col2:

        st.metric(
            "RMSE",
            "0.7552"
        )


# ---------------- CATBOOST ----------------

with col3:

    st.subheader("🟢 CatBoost ⭐")

    st.metric(
        "R² Score",
        "0.8044"
    )

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:

        st.metric(
            "MAE",
            "0.5989"
        )

    with metric_col2:

        st.metric(
            "RMSE",
            "0.7545"
        )


# ============================================================
# SELECTED MODEL
# ============================================================

st.divider()

selected_metrics = MODEL_METRICS[selected_model]

st.info(
    f"🤖 **Selected Model: {selected_model}**  |  "
    f"R²: **{selected_metrics['R2']:.4f}**  |  "
    f"MAE: **{selected_metrics['MAE']:.4f}**  |  "
    f"RMSE: **{selected_metrics['RMSE']:.4f}**"
)


# ============================================================
# SLEEP ASSESSMENT
# ============================================================

st.header("📝 Sleep Assessment")

st.caption(
    "Enter the information below to estimate sleep quality."
)


# ============================================================
# PERSONAL INFORMATION
# ============================================================

with st.expander(
    "👤 Personal Information",
    expanded=True
):

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.slider(
            "Age",
            18,
            59,
            25
        )

    with col2:

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Other"
            ]
        )

    with col3:

        occupation = st.selectbox(
            "Occupation",
            [
                "Manager",
                "Doctor",
                "Software Engineer",
                "Freelancer",
                "Student",
                "Researcher",
                "Designer",
                "Teacher"
            ]
        )


# ============================================================
# PHONE USAGE
# ============================================================

with st.expander(
    "📱 Phone & Digital Usage",
    expanded=True
):

    col1, col2, col3 = st.columns(3)

    with col1:

        screen_time = st.slider(
            "Daily Screen Time (hours)",
            1.0,
            10.0,
            5.0,
            0.1
        )

    with col2:

        phone_before_sleep = st.slider(
            "Phone Usage Before Sleep (minutes)",
            0,
            119,
            30
        )

    with col3:

        notifications = st.slider(
            "Notifications Per Day",
            0,
            300,
            100
        )


# ============================================================
# SLEEP & LIFESTYLE
# ============================================================

with st.expander(
    "🛏️ Sleep & Lifestyle",
    expanded=True
):

    col1, col2, col3 = st.columns(3)

    with col1:

        sleep_duration = st.slider(
            "Sleep Duration (hours)",
            4.0,
            9.0,
            7.0,
            0.1
        )

    with col2:

        caffeine = st.slider(
            "Caffeine Intake (cups/day)",
            0,
            4,
            1
        )

    with col3:

        physical_activity = st.slider(
            "Physical Activity (minutes/day)",
            20,
            299,
            60
        )


# ============================================================
# STRESS & FATIGUE
# ============================================================

with st.expander(
    "🧠 Stress & Mental Fatigue",
    expanded=True
):

    col1, col2 = st.columns(2)

    with col1:

        stress = st.slider(
            "Stress Level",
            0.0,
            10.0,
            5.0,
            0.1
        )

    with col2:

        mental_fatigue = st.slider(
            "Mental Fatigue Score",
            1.0,
            10.0,
            5.0,
            0.1
        )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

screen_stress = (
    screen_time * stress
)

screen_fatigue = (
    screen_time * mental_fatigue
)

phone_stress = (
    phone_before_sleep * stress
)

phone_fatigue = (
    phone_before_sleep * mental_fatigue
)

stress_fatigue = (
    stress * mental_fatigue
)

digital_exposure = (
    screen_time
    + phone_before_sleep / 60
    + notifications / 100
)


# ============================================================
# INPUT DATA
# ============================================================

input_data = pd.DataFrame({

    "age": [age],

    "gender": [str(gender)],

    "occupation": [str(occupation)],

    "daily_screen_time_hours": [
        screen_time
    ],

    "phone_usage_before_sleep_minutes": [
        phone_before_sleep
    ],

    "sleep_duration_hours": [
        sleep_duration
    ],

    "stress_level": [
        stress
    ],

    "caffeine_intake_cups": [
        caffeine
    ],

    "physical_activity_minutes": [
        physical_activity
    ],

    "notifications_received_per_day": [
        notifications
    ],

    "mental_fatigue_score": [
        mental_fatigue
    ],

    "screen_stress": [
        screen_stress
    ],

    "screen_fatigue": [
        screen_fatigue
    ],

    "phone_stress": [
        phone_stress
    ],

    "phone_fatigue": [
        phone_fatigue
    ],

    "stress_fatigue": [
        stress_fatigue
    ],

    "digital_exposure": [
        digital_exposure
    ]

})


# ============================================================
# PREDICT BUTTON
# ============================================================

st.divider()

col1, col2, col3 = st.columns(
    [1, 2, 1]
)

with col2:

    predict_button = st.button(
        "🌙 Predict Sleep Quality",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # --------------------------------------------
        # RIDGE
        # --------------------------------------------

        if selected_model == "Ridge Regression":

            prediction = ridge_model.predict(
                input_data
            )[0]


        # --------------------------------------------
        # GRADIENT BOOSTING
        # --------------------------------------------

        elif selected_model == "Gradient Boosting":

            prediction = gradient_model.predict(
                input_data
            )[0]


        # --------------------------------------------
        # CATBOOST
        # --------------------------------------------

        else:

            catboost_input = input_data.copy()

            catboost_input["gender"] = (
                catboost_input["gender"].astype(str)
            )

            catboost_input["occupation"] = (
                catboost_input["occupation"].astype(str)
            )

            prediction = catboost_model.predict(
                catboost_input
            )[0]


        # --------------------------------------------
        # LIMIT SCORE
        # --------------------------------------------

        prediction = float(
            np.clip(
                prediction,
                1,
                10
            )
        )


        # ====================================================
        # SCORE INTERPRETATION
        # ====================================================

        if prediction >= 7:

            status = "Good Sleep Quality"
            emoji = "🟢"

        elif prediction >= 5:

            status = "Moderate Sleep Quality"
            emoji = "🟡"

        else:

            status = "Low Sleep Quality"
            emoji = "🔴"


        # ====================================================
        # RESULT
        # ====================================================

        st.subheader("🎯 Prediction Result")

        st.success(
            f"### {emoji} {status}"
        )

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.metric(
                "Sleep Quality Score",
                f"{prediction:.2f} / 10"
            )

        with result_col2:
            st.metric(
                "Model Used",
                selected_model
            )

        with result_col3:
            st.metric(
                "R² Score",
                f"{selected_metrics['R2']:.4f}"
            )

        
        st.caption(
            f"Prediction generated using **{selected_model}**."
        )
        st.caption(
    f"Prediction generated using **{selected_model}**."
)


        # ====================================================
        # RECOMMENDATIONS
        # ====================================================

        st.header("💡 Recommendations")

        recommendations = []


        if screen_time > 7:

            recommendations.append(
                "📱 Your daily screen time is relatively high. "
                "Consider reducing unnecessary screen usage, "
                "especially during the evening."
            )

        elif screen_time > 5:

            recommendations.append(
                "📱 Consider gradually reducing recreational "
                "screen time, particularly later in the day."
            )


        if phone_before_sleep > 60:

            recommendations.append(
                "🌙 Consider reducing phone usage during the "
                "hour before bedtime."
            )

        elif phone_before_sleep > 30:

            recommendations.append(
                "🌙 Try keeping your phone away from your bed "
                "during the final part of your evening."
            )


        if stress >= 7:

            recommendations.append(
                "🧠 Your reported stress level is high. "
                "Relaxation activities such as breathing "
                "exercises, light stretching or walking may "
                "be useful."
            )

        elif stress >= 5:

            recommendations.append(
                "🧠 Managing daily stress may help support "
                "better sleep quality."
            )


        if mental_fatigue >= 7:

            recommendations.append(
                "🧠 Your reported mental fatigue is high. "
                "Consider taking regular breaks from "
                "mentally demanding activities."
            )


        if caffeine >= 3:

            recommendations.append(
                "☕ Consider limiting caffeine later in the "
                "day, especially close to bedtime."
            )


        if physical_activity < 45:

            recommendations.append(
                "🏃 Your reported physical activity is relatively "
                "low. Consider adding regular light or moderate "
                "activity during the day."
            )


        if sleep_duration < 6:

            recommendations.append(
                "🛏️ Your reported sleep duration is relatively "
                "short. Consider maintaining a consistent "
                "sleep schedule."
            )

        elif sleep_duration >= 7:

            recommendations.append(
                "🛏️ Your reported sleep duration is within a "
                "generally healthy range. Maintaining consistency "
                "can be useful."
            )


        if notifications > 200:

            recommendations.append(
                "🔔 You receive a high number of notifications. "
                "Consider using notification limits or "
                "Do Not Disturb during the evening."
            )


        if len(recommendations) == 0:

            recommendations.append(
                "✅ Your current inputs do not indicate major "
                "areas for adjustment. Maintaining consistent "
                "sleep and lifestyle habits is a good approach."
            )


        for recommendation in recommendations:

            st.markdown(
                f"""
                <div class="recommendation">
                    {recommendation}
                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # ASSESSMENT SUMMARY
        # ====================================================

        st.header("📋 Assessment Summary")

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Daily Screen Time",
                f"{screen_time:.1f} hrs"
            )

            st.metric(
                "Sleep Duration",
                f"{sleep_duration:.1f} hrs"
            )


        with col2:

            st.metric(
                "Stress Level",
                f"{stress:.1f}/10"
            )

            st.metric(
                "Mental Fatigue",
                f"{mental_fatigue:.1f}/10"
            )


        with col3:

            st.metric(
                "Phone Before Sleep",
                f"{phone_before_sleep} min"
            )

            st.metric(
                "Physical Activity",
                f"{physical_activity} min"
            )


    except Exception as e:

        st.error(
            f"❌ Prediction failed: {e}"
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    "⚠️ **Disclaimer:** SleepSense provides an ML-based "
    "estimate for educational and informational purposes only. "
    "It is not a medical diagnosis and should not replace "
    "professional medical advice."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        SleepSense • AI-Based Phone Usage & Sleep Health Assessment
        <br>
        Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)