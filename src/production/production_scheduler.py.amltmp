import streamlit as st
import pandas as pd
import plotly.express as px
from pyprojroot import here
import os
import numpy as np
import joblib
from skimpy import clean_columns

# Set the path and read the data for the initial part of the app
path_data = here("./data")
os.chdir(path_data)
data = pd.read_csv("data_shap_hospital.csv")[["Hospital ID", "Driver", "Impact"]]

# Streamlit title for the initial part of the app
st.title("Hospital Drivers Analysis")

st.markdown("""
This is a demo with synthetic data based on a model predicting an ER visit within the last 30 days. 
The results below show the average impact of the driver on the outcome in percentage.
""")

# Sidebar for hospital selection
hospital_id = st.sidebar.selectbox("Select Hospital ID", options=data["Hospital ID"].unique())

# Sidebar options for adjusting percentages
st.sidebar.markdown("### Adjust Binary Percentages")
age_60_percentage = st.sidebar.slider("Age 60+ Percentage", min_value=0.0, max_value=1.0, value=0.1)
high_cholesterol_percentage = st.sidebar.slider("High Cholesterol Percentage", min_value=0.0, max_value=1.0, value=0.1)
diabetes_percentage = st.sidebar.slider("Diabetes Percentage", min_value=0.0, max_value=1.0, value=0.1)
preventative_services_percentage = st.sidebar.slider("Preventative Services Percentage", min_value=0.0, max_value=1.0, value=0.7)

# Filter data based on selected hospital
df = data[data["Hospital ID"] == hospital_id]
df["AbsImpact"] = df["Impact"].abs()
df = df.sort_values(by="AbsImpact", ascending=False)
df = df.iloc[::-1]
df = df.drop("AbsImpact", axis=1)
df["ImpactText"] = df["Impact"].apply(lambda x: f"{x:.0%}")
df["Color"] = df["Impact"].apply(lambda x: "blue" if x > 0 else "red")

# Create horizontal bar chart using Plotly
fig = px.bar(
    df,
    x="Impact",
    y="Driver",
    orientation="h",
    text="ImpactText",
    color="Color",
    labels={"Impact": "Impact Value", "Driver": "Driver Factor"},
)

# Customizing the layout
fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showticklabels=False, title=None),
    showlegend=False,
)
fig.update_traces(marker_coloraxis=None)
fig.update_traces(texttemplate="%{text}", textposition="inside")

# Display the plot in Streamlit
st.plotly_chart(fig)

# New Section for Hospital Averages and Predictions

# Load additional data and the model
data_er_visits = pd.read_csv("data_er_visits.csv").rename(columns={"Unnamed: 0": "Member ID"})
data_no_member_id = data_er_visits.drop(columns=["Member ID"])
data_predict = data_no_member_id.drop(columns=["Hospital ID", "ER Visit"])
model = joblib.load("model_drivers.joblib")

# Function to adjust binary percentages
import pandas as pd
import numpy as np


def adjust_binary_percentages(df, **column_percentages):
    """
    Adjusts the percentage of 1's in each specified column of a binary DataFrame.

    Args:
    df (pd.DataFrame): DataFrame with binary values.
    column_percentages (dict): A dictionary where keys are column names and values are the new desired percentages of 1's.

    Returns:
    pd.DataFrame: Modified DataFrame.
    """

    for column, percentage in column_percentages.items():
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in DataFrame")

        # Calculate current percentage of 1's
        current_percentage = df[column].mean()

        # Calculate the desired number of 1's
        target_count = int(df.shape[0] * percentage)

        # Find indices where changes are needed
        ones_indices = df[df[column] == 1].index
        zeros_indices = df[df[column] == 0].index

        if target_count > ones_indices.size:  # Need to add more 1's
            change_count = target_count - ones_indices.size
            indices_to_change = np.random.choice(
                zeros_indices, change_count, replace=False
            )
            df.loc[indices_to_change, column] = 1
        else:  # Need to remove some 1's
            change_count = ones_indices.size - target_count
            indices_to_change = np.random.choice(
                ones_indices, change_count, replace=False
            )
            df.loc[indices_to_change, column] = 0

    return df

# Preprocess and adjust data with user-selected values
data_predict_adjust = clean_columns(data_predict.copy())
data_predict_adjust = adjust_binary_percentages(
    df=data_predict_adjust,
    age_60=age_60_percentage,
    high_cholesterol=high_cholesterol_percentage,
    diabetes=diabetes_percentage,
    preventative_services=preventative_services_percentage
)
data_predict_adjust.rename(columns={
    'age_60': 'Age 60+',
    'high_cholesterol': 'High Cholesterol',
    'diabetes': 'Diabetes',
    'preventative_services': 'Preventative Services'
}, inplace=True)

# Make predictions
predictions = model.predict(data_predict_adjust)
predictions_pd = pd.DataFrame(predictions).rename(columns={0: 'Predictions'})

# Combine predictions with Hospital ID
data_predictions_hospital_id = pd.concat(
    [data_no_member_id["Hospital ID"], predictions_pd], axis=1
)

# Calculate and adjust hospital averages
data_predictions_hospital_group = (
    data_predictions_hospital_id.groupby("Hospital ID").mean().reset_index().round(2)
)
noise = np.random.uniform(-0.02, 0.02, data_predictions_hospital_group["Predictions"].shape)
data_predictions_hospital_group["Predictions"] = (
    data_predictions_hospital_group["Predictions"] + noise
).round(2)

# Display Hospital Averages in Streamlit
st.markdown("### Hospital Averages for Selected Hospital")
selected_hospital_avg = data_predictions_hospital_group[data_predictions_hospital_group["Hospital ID"] == hospital_id]
st.write(selected_hospital_avg)

