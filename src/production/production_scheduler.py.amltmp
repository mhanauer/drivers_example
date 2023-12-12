import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pyprojroot import here
import os
from skimpy import clean_columns

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

# Load data and model
def load_data_model():
    path_data = here("./data")
    os.chdir(path_data)
    data = pd.read_csv("data_pmpm.csv")
    model = joblib.load("model_drivers.joblib")
    return data, model

# Streamlit app layout
def main():
    st.title("PMPM Cost Prediction")

    # Sliders for feature adjustment
    bp_percentage = st.slider('High Blood Pressure Percentage', 0.1, 1.0, 0.1, step=0.1)
    chol_percentage = st.slider('High Cholesterol Percentage', 0.1, 1.0, 0.1, step=0.1)
    diabetes_percentage = st.slider('Diabetes Percentage', 0.1, 1.0, 0.1, step=0.1)
    preventive_percentage = st.slider('Preventative Services Percentage', 0.1, 1.0, 0.1, step=0.1)

    data, model = load_data_model()

    if st.button("Generate Predictions"):
        with st.spinner('Processing...'):
            data_predict = data.drop(columns=["Hospital ID", "Per Member Per Month Cost"])
            data_predict_adjust = clean_columns(data_predict.copy())
            data_predict_adjust = adjust_binary_percentages(
                df=data_predict_adjust,
                high_blood_pressure=bp_percentage,
                high_cholesterol=chol_percentage,
                diabetes=diabetes_percentage,
                preventative_services=preventive_percentage,
            )
            data_predict_adjust.rename(
                columns={
                    "high_blood_pressure": "High Blood Pressure",
                    "high_cholesterol": "High Cholesterol",
                    "diabetes": "Diabetes",
                    "preventative_services": "Preventative Services",
                },
                inplace=True,
            )
            predictions = model.predict(data_predict_adjust)
            predictions_pd = pd.DataFrame(predictions).rename(columns={0: "Predictions"})
            data_predictions_hospital_id = pd.concat([data["Hospital ID"], predictions_pd], axis=1)
            data_predictions_hospital_group = (
                data_predictions_hospital_id.groupby("Hospital ID").mean().reset_index().round(2)
            )
            st.write(data_predictions_hospital_group)

if __name__ == "__main__":
    main()
