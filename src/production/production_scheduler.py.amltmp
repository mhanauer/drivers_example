import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pyprojroot import here
import os
from skimpy import clean_columns

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
    data_pmpm = pd.read_csv("data_pmpm.csv")
    data_shap = pd.read_csv("data_shap_hospital.csv")
    model = joblib.load("model_drivers.joblib")
    return data_pmpm, data_shap, model

# Streamlit app layout
def main():
    # Streamlit title for the initial part of the app
    st.title("PMPM Drivers Analysis")

    st.markdown("""
    This demo uses synthetic data based on a model that predicts per member per month (PMPM) costs. The results displayed below represent the average impact of various factors (referred to as 'drivers') on PMPM costs.  We also include a what if analysis allowing the users to evaluate changes in their attributed population on PMPM. 
    """)

    data_pmpm, data_shap, model = load_data_model()

    # Sidebar for hospital selection
    hospital_id = st.sidebar.selectbox("Select Hospital ID", options=data_shap["Hospital ID"].unique())

    # Filter data based on selected hospital
    df = data_shap[data_shap["Hospital ID"] == hospital_id]
    df["AbsImpact"] = df["Impact"].abs()
    df = df.sort_values(by="AbsImpact", ascending=False).iloc[::-1]
    df = df.drop("AbsImpact", axis=1)
    # Round Impact to nearest dollar and format as text
    df["ImpactText"] = df["Impact"].apply(lambda x: f"${round(x):,}")
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
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showticklabels=False, title=None),
        showlegend=False,
    )
    fig.update_traces(marker_coloraxis=None)
    fig.update_traces(texttemplate="%{text}", textposition="inside")
    st.plotly_chart(fig)

    # Streamlit title for the PMPM prediction part
    st.title("PMPM Cost Prediction")

    # Sliders for feature adjustment in the sidebar
    bp_percentage = st.sidebar.slider('High Blood Pressure Percentage', 0.1, 1.0, 0.1, step=0.1)
    chol_percentage = st.sidebar.slider('High Cholesterol Percentage', 0.1, 1.0, 0.1, step=0.1)
    diabetes_percentage = st.sidebar.slider('Diabetes Percentage', 0.1, 1.0, 0.1, step=0.1)
    preventive_percentage = st.sidebar.slider('Preventative Services Percentage', 0.1, 1.0, 0.1, step=0.1)

    if st.sidebar.button("Generate Predictions"):
        with st.spinner('Processing...'):
            data_predict = data_pmpm.drop(columns=["Hospital ID", "Per Member Per Month Cost"])
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
            predictions_rounded = np.round(predictions)  # Round to nearest dollar
            predictions_pd = pd.DataFrame(predictions_rounded).rename(columns={0: "Predictions"})
            data_predictions_hospital_id = pd.concat([data_pmpm["Hospital ID"], predictions_pd], axis=1)
            data_predictions_hospital_group = (
                data_predictions_hospital_id.groupby("Hospital ID").mean().reset_index().round(0)  # Round to nearest dollar
            )

            merged_data = data_pmpm.merge(data_predictions_hospital_group, on="Hospital ID", suffixes=('_Actual', '_Predicted'))

        # Calculate difference and percentage change
        merged_data['Difference'] = merged_data['Per Member Per Month Cost_Actual'] - merged_data['Predictions_Predicted']
        merged_data['Percentage Change'] = (merged_data['Difference'] / merged_data['Per Member Per Month Cost_Actual']) * 100

        # Display the results
        st.write("Comparison of Actual and Predicted PMPM per Hospital ID")
        st.dataframe(merged_data[['Hospital ID', 'Per Member Per Month Cost_Actual', 'Predictions_Predicted', 'Difference', 'Percentage Change']])


if __name__ == "__main__":
    main()