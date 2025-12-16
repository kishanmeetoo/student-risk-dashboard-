import streamlit as st
import pandas as pd
from ucimlrepo import fetch_ucirepo
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# -----------------------
# PAGE SETUP
# -----------------------
st.title("🚨 Student Risk Prediction & Early Intervention")
st.caption(
    "This page proactively identifies students at risk of failing, enabling earlier and more targeted academic support."
)

# -----------------------
# LOAD DATA
# -----------------------
@st.cache_data
def load_student_data():
    student = fetch_ucirepo(id=320)
    X = student.data.features
    y = student.data.targets
    df = pd.concat([X, y], axis=1)
    df.columns = [c.replace(" ", "_").replace("-", "_") for c in df.columns]

    df["school_name"] = df["school"].map({
        "GP": "Gabriel Pereira",
        "MS": "Mousinho da Silveira"
    })
    return df

df = load_student_data()

# -----------------------
# SIDEBAR FILTERS
# -----------------------
st.sidebar.header("🔎 Filters")

selected_school = st.sidebar.multiselect(
    "Select School:",
    sorted(df["school_name"].unique()),
    default=sorted(df["school_name"].unique())
)

selected_gender = st.sidebar.multiselect(
    "Select Gender:",
    sorted(df["sex"].unique()),
    default=sorted(df["sex"].unique())
)

selected_studytime = st.sidebar.slider(
    "Minimum Study Time:",
    min_value=int(df["studytime"].min()),
    max_value=int(df["studytime"].max()),
    value=int(df["studytime"].min())
)

filtered = df[
    (df["school_name"].isin(selected_school)) &
    (df["sex"].isin(selected_gender)) &
    (df["studytime"] >= selected_studytime)
]

# -----------------------
# AT-RISK DEFINITION
# -----------------------
st.info("**At-Risk Definition:** Students predicted to score **below 10** in the final grade (G3).")

# -----------------------
# MODEL TRAINING
# -----------------------
features = filtered.drop(columns=["G3", "school_name"])
target = filtered["G3"]

X_encoded = pd.get_dummies(features, drop_first=True)

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)
rf.fit(X_encoded, target)

# -----------------------
# PREDICTIONS
# -----------------------
filtered["predicted_G3"] = rf.predict(X_encoded)
filtered["risk_probability"] = (
    (10 - filtered["predicted_G3"]).clip(lower=0) / 10
).clip(0, 1)

filtered["at_risk"] = filtered["predicted_G3"] < 10

# -----------------------
# KPI ROW
# -----------------------
st.subheader("📌 Predicted Risk Overview")

total_students = filtered.shape[0]
at_risk_count = filtered["at_risk"].sum()
at_risk_pct = (at_risk_count / total_students) * 100
high_risk_count = (filtered["risk_probability"] >= 0.7).sum()

c1, c2, c3 = st.columns(3)
c1.metric("Predicted At-Risk (%)", f"{at_risk_pct:.1f}%")
c2.metric("At-Risk Students", int(at_risk_count))
c3.metric("High-Risk Cases (≥70%)", int(high_risk_count))

# -----------------------
# RISK BY SCHOOL (SINGLE STRATEGIC VISUAL)
# -----------------------
st.subheader("🏫 Risk Distribution by School")

risk_by_school = (
    filtered.groupby("school_name")["at_risk"]
    .mean()
    .reset_index()
)
risk_by_school["at_risk_pct"] = risk_by_school["at_risk"] * 100

fig_school = px.bar(
    risk_by_school,
    x="school_name",
    y="at_risk_pct",
    labels={
        "school_name": "School",
        "at_risk_pct": "Predicted At-Risk (%)"
    },
    title="Predicted At-Risk Students by School"
)
st.plotly_chart(fig_school, use_container_width=True)

# -----------------------
# INDIVIDUAL STUDENT RISK SIMULATOR
# -----------------------
st.subheader("🧪 Individual Student Risk Assessment")

with st.form("risk_form"):
    col1, col2 = st.columns(2)

    with col1:
        school = st.selectbox("School", df["school_name"].unique())
        gender = st.selectbox("Gender", df["sex"].unique())
        studytime = st.slider("Weekly Study Time", 1, 4, 2)
        absences = st.number_input("Number of Absences", 0, 100, 5)

    with col2:
        G1 = st.slider("Grade Period 1 (G1)", 0, 20, 10)
        G2 = st.slider("Grade Period 2 (G2)", 0, 20, 10)
        failures = st.number_input("Past Class Failures", 0, 4, 0)
        health = st.slider("Health (1–5)", 1, 5, 3)

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    student_dict = {
        "studytime": studytime,
        "absences": absences,
        "G1": G1,
        "G2": G2,
        "failures": failures,
        "health": health,
        "sex": gender,
        "school": "GP" if school == "Gabriel Pereira" else "MS"
    }

    student_df = pd.DataFrame([student_dict])
    student_encoded = pd.get_dummies(student_df, drop_first=True)
    student_encoded = student_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    predicted_grade = rf.predict(student_encoded)[0]
    risk_prob = max(0, min((10 - predicted_grade) / 10, 1))

    if predicted_grade < 10:
        st.warning(
            f"**At Risk**\n\n"
            f"Predicted Final Grade: **{predicted_grade:.1f}**  \n"
            f"Risk Probability: **{risk_prob:.0%}**  \n\n"
            "Primary risk drivers are low prior grades and attendance patterns."
        )
    else:
        st.success(
            f"**Not At Risk**\n\n"
            f"Predicted Final Grade: **{predicted_grade:.1f}**  \n"
            f"Risk Probability: **{risk_prob:.0%}**"
        )

# -----------------------
# EXECUTIVE TAKEAWAYS
# -----------------------
st.subheader("🎯 Key Takeaways for Early Intervention")

st.markdown(
    """
- Prior academic performance is the strongest indicator of future risk.
- Attendance-based interventions provide high potential impact.
- Early predictive screening enables targeted support before failure occurs.
"""
)
