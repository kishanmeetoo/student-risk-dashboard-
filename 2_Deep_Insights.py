import streamlit as st
import pandas as pd
from ucimlrepo import fetch_ucirepo
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

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
    
    # Map school codes to names
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
# COLOR MAP
# -----------------------
school_colors = {
    "Gabriel Pereira": "blue",
    "Mousinho da Silveira": "orange"
}

# -----------------------
# ATTENDANCE VS FINAL GRADE
# -----------------------
st.subheader("📉 Attendance vs Final Grade")
fig_attendance = px.scatter(
    filtered,
    x="absences",
    y="G3",
    color="school_name",
    trendline="ols",
    labels={
        "absences": "Number of Absences",
        "G3": "Final Grade",
        "school_name": "School Name"
    },
    color_discrete_map=school_colors,
    title="Absences vs Final Grade with Trend Line"
)
st.plotly_chart(fig_attendance, use_container_width=True)

# -----------------------
# STUDY TIME VS FINAL GRADE
# -----------------------
st.subheader("⏱️ Study Time vs Final Grade")
fig_study = px.box(
    filtered,
    x="studytime",
    y="G3",
    color="school_name",
    labels={
        "studytime": "Weekly Study Time",
        "G3": "Final Grade",
        "school_name": "School Name"
    },
    color_discrete_map=school_colors,
    title="Study Time vs Final Grade"
)
st.plotly_chart(fig_study, use_container_width=True)

# -----------------------
# SOCIOECONOMIC FACTORS
# -----------------------
st.subheader("🏠 Socioeconomic Factors vs Final Grade")

# Example 1: Parents' education
fig_medufedu = px.box(
    filtered,
    x="Medu",
    y="G3",
    color="school_name",
    labels={
        "Medu": "Mother's Education (0-4)",
        "G3": "Final Grade",
        "school_name": "School Name"
    },
    color_discrete_map=school_colors,
    title="Mother's Education vs Final Grade"
)
st.plotly_chart(fig_medufedu, use_container_width=True)

fig_fedufedu = px.box(
    filtered,
    x="Fedu",
    y="G3",
    color="school_name",
    labels={
        "Fedu": "Father's Education (0-4)",
        "G3": "Final Grade",
        "school_name": "School Name"
    },
    color_discrete_map=school_colors,
    title="Father's Education vs Final Grade"
)
st.plotly_chart(fig_fedufedu, use_container_width=True)

# Example 2: Family size
fig_famsize = px.box(
    filtered,
    x="famsize",
    y="G3",
    color="school_name",
    labels={
        "famsize": "Family Size",
        "G3": "Final Grade",
        "school_name": "School Name"
    },
    color_discrete_map=school_colors,
    title="Family Size vs Final Grade"
)
st.plotly_chart(fig_famsize, use_container_width=True)

# -----------------------
# FEATURE IMPORTANCE (Random Forest)
# -----------------------
st.subheader("📊 Feature Importance for Final Grade Prediction")

# Features: drop target & categorical school column
X_features = filtered.drop(columns=["G3", "school_name"])
y_target = filtered["G3"]

# Simple encoding for categorical columns
X_encoded = pd.get_dummies(X_features, drop_first=True)

# Fit Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_encoded, y_target)

# Extract feature importances
importances = pd.Series(rf.feature_importances_, index=X_encoded.columns).sort_values(ascending=True)

# Plot horizontal bar chart
fig_feat = px.bar(
    importances,
    x=importances.values,
    y=importances.index,
    orientation="h",
    labels={
        "x": "Importance",
        "y": "Feature"
    },
    title="Feature Importance for Predicting Final Grade"
)
st.plotly_chart(fig_feat, use_container_width=True)

# -----------------------
# QUICK SUMMARY INSIGHTS
# -----------------------
st.write("### 💡 Quick Summary Insights")

insights = [
    "**Study Time vs Grades:** Median grades are stable across study time levels; study time alone is not a strong predictor. Gabriel Pereira students generally show slightly higher medians.",
    "**Parental Education:** Higher mother’s and father’s education (levels 3–4) is associated with slightly higher student grades; top-level medians approach 15–16.",
    "**Family Size:** Median and distribution of final grades are nearly identical across all family size categories; family size has negligible impact.",
    "**Attendance vs Grades:** Slight negative correlation exists; more absences generally relate to lower grades, but individual variation is high.",
    "**Feature Importance:** Random Forest identifies the most influential factors as previous grades (G1, G2) and absences (≈0.8). Moderate predictors include famrel, age, freetime, health, goout, traveltime, Fedu, studytime, and Medu. Least predictive are family size and job-related factors.",
    "**Overall Conclusion:** Final grade is primarily driven by previous academic performance and attendance. Behavioral and socioeconomic factors have moderate influence, while study time and family size are among the least predictive."
]

for insight in insights:
    st.markdown(f"- {insight}")

