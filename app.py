
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

model = joblib.load("credit_risk_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(
    page_title="Credit Risk Analysis",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>
#MainMenu {
    visibility: hidden;
}
.stDeployButton {
    display: none;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

st.title("💳 Credit Risk Analysis System")
st.write("Credit risk prediction and analysis dashboard")

df = pd.read_csv("german_credit_data.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

df["Saving accounts"] = df["Saving accounts"].fillna("unknown")
df["Checking account"] = df["Checking account"].fillna("unknown")

st.header("📊 Credit Risk Dashboard")

total_applicants = len(df)
good_count = (df["Risk"] == "good").sum()
bad_count = (df["Risk"] == "bad").sum()
avg_credit = df["Credit amount"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Applicants", total_applicants)
col2.metric("Good Risk", good_count)
col3.metric("Bad Risk", bad_count)
col4.metric("Average Credit", f"{avg_credit:.0f}")

st.divider()

col1, col2 = st.columns(2)

with col1:
    risk_data = df["Risk"].value_counts().reset_index()
    risk_data.columns = ["Risk", "Count"]

    fig = px.bar(
        risk_data,
        x="Risk",
        y="Count",
        title="Risk Distribution",
        text="Count"
    )

    fig.update_traces(textposition="outside")

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "staticPlot": True,
            "displayModeBar": False
        }
    )

with col2:
    fig = px.box(
        df,
        x="Risk",
        y="Credit amount",
        title="Credit Amount by Risk"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "staticPlot": True,
            "displayModeBar": False
        }
    )

col1, col2 = st.columns(2)

with col1:
    checking_data = pd.crosstab(
        df["Checking account"],
        df["Risk"]
    ).reset_index()

    fig = px.bar(
        checking_data,
        x="Checking account",
        y=["good", "bad"],
        title="Checking Account vs Risk",
        barmode="group"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "staticPlot": True,
            "displayModeBar": False
        }
    )

with col2:
    housing_data = df["Housing"].value_counts().reset_index()
    housing_data.columns = ["Housing", "Count"]

    fig = px.pie(
        housing_data,
        names="Housing",
        values="Count",
        title="Housing Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "staticPlot": True,
            "displayModeBar": False
        }
    )

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        df,
        x="Age",
        color="Risk",
        nbins=20,
        title="Age Distribution by Risk"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "staticPlot": True,
            "displayModeBar": False
        }
    )

with col2:
    fig = px.scatter(
        df,
        x="Duration",
        y="Credit amount",
        color="Risk",
        title="Credit Amount vs Loan Duration"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "staticPlot": True,
            "displayModeBar": False
        }
    )

st.divider()

st.header("🔍 Credit Risk Predictor")
st.write("Enter applicant details below to predict the applicant's credit risk.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    sex = st.selectbox(
        "Sex",
        ["male", "female"]
    )

    job = st.selectbox(
        "Job",
        [0, 1, 2, 3]
    )

    housing = st.selectbox(
        "Housing",
        ["own", "rent", "free"]
    )

    saving_accounts = st.selectbox(
        "Saving accounts",
        ["unknown", "little", "moderate", "quite rich", "rich"]
    )

with col2:
    checking_account = st.selectbox(
        "Checking account",
        ["unknown", "little", "moderate", "rich"]
    )

    credit_amount = st.number_input(
        "Credit amount",
        min_value=100,
        value=2000
    )

    duration = st.number_input(
        "Duration (months)",
        min_value=1,
        max_value=72,
        value=12
    )

    purpose = st.selectbox(
        "Purpose",
        [
            "car",
            "furniture/equipment",
            "radio/TV",
            "domestic appliances",
            "repairs",
            "education",
            "vacation/others",
            "business"
        ]
    )

if st.button(
    "🔍 Predict Credit Risk",
    use_container_width=True
):
    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "Job": [job],
        "Housing": [housing],
        "Saving accounts": [saving_accounts],
        "Checking account": [checking_account],
        "Credit amount": [credit_amount],
        "Duration": [duration],
        "Purpose": [purpose]
    })

    input_data = pd.get_dummies(
        input_data,
        columns=[
            "Sex",
            "Housing",
            "Saving accounts",
            "Checking account",
            "Purpose"
        ],
        drop_first=True,
        dtype=int
    )

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.divider()
    st.subheader("📌 Prediction Result")

    if prediction == 0:
        st.success("✅ Good Credit Risk")
    else:
        st.error("⚠️ Bad Credit Risk")

    good_probability = probability[0] * 100
    bad_probability = probability[1] * 100

    col1, col2 = st.columns(2)

    col1.metric(
        "Good Risk Probability",
        f"{good_probability:.2f}%"
    )

    col2.metric(
        "Bad Risk Probability",
        f"{bad_probability:.2f}%"
    )

    probability_data = pd.DataFrame({
        "Risk": ["Good Risk", "Bad Risk"],
        "Probability": [
            good_probability,
            bad_probability
        ]
    })

    fig = px.bar(
        probability_data,
        x="Risk",
        y="Probability",
        title="Prediction Probability",
        text="Probability"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_range=[0, 100],
        yaxis_title="Probability (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "staticPlot": True,
            "displayModeBar": False
        }
    )
