import streamlit as st
import pandas as pd
from ucimlrepo import fetch_ucirepo
import plotly.express as px

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

    # Map school codes to full names
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
# KPIs
# -----------------------
st.subheader("📌 Key Performance Indicators")

total_students = filtered.shape[0]
avg_grade = filtered["G3"].mean()
at_risk_pct = (filtered["G3"] < 10).mean() * 100

col1, col2, col3 = st.columns(3)
col1.metric("Total Students", total_students)
col2.metric("Average Final Grade (out of 20)", f"{avg_grade:.2f}")
col3.metric("At-Risk Students (%)", f"{at_risk_pct:.2f}%")

# -----------------------
# COLOR MAP
# -----------------------
school_colors = {
    "Gabriel Pereira": "blue",
    "Mousinho da Silveira": "orange"
}

# -----------------------
# FINAL GRADE DISTRIBUTION
# -----------------------
st.write("### 📊 Final Grade Distribution")
fig_grade = px.histogram(
    filtered,
    x="G3",
    color="school_name",
    nbins=20,
    labels={
        "G3": "Final Grade",
        "school_name": "School Name"
    },
    color_discrete_map=school_colors,
    title="Distribution of Final Grades"
)
st.plotly_chart(fig_grade, use_container_width=True)

# -----------------------
# ABSENCES VS FINAL GRADE
# -----------------------
st.write("### 📈 Absences vs Final Grade")
fig_abs = px.scatter(
    filtered,
    x="absences",
    y="G3",
    color="school_name",
    labels={
        "absences": "Number of Absences",
        "G3": "Final Grade",
        "school_name": "School Name"
    },
    color_discrete_map=school_colors,
    title="Absences vs Final Grade"
)
st.plotly_chart(fig_abs, use_container_width=True)


# -----------------------
# STUDY TIME VS FINAL GRADE
# -----------------------
st.write("### ⏱️ Study Time vs Final Grade")
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
    color_discrete_map={
        "Gabriel Pereira": "blue",
        "Mousinho da Silveira": "orange"
    },
    title="Study Time vs Final Grade"
)
st.plotly_chart(fig_study, use_container_width=True)

# -----------------------
# QUICK SUMMARY INSIGHTS
# -----------------------
st.write("### 💡 Quick Summary Insights")

insights = [
    f"**Total Students:** {total_students} students are included in this view.",
    f"**Average Final Grade:** The average final grade is {avg_grade:.2f} out of 20, indicating overall moderate academic performance.",
    f"**At-Risk Students:** {at_risk_pct:.2f}% of students are at risk (Final Grade < 10), signaling potential areas for intervention.",
    "Most students’ final grades cluster between 10 and 12, with fewer students at the extremes.",
    "Gabriel Pereira contributes the majority of students across nearly all grade bins, while Mousinho da Silveira has fewer students in each bin.",
    "Absences and study time do not show a strong linear effect on final grades at the aggregate level, suggesting multiple factors influence performance."
]

for insight in insights:
    st.markdown(f"- {insight}")
