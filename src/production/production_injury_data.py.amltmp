import requests
import pandas as pd
import numpy as np
from pyprojroot import here
from skimpy import clean_columns
import os
from azureml.core import Workspace, Dataset
import argparse

print("Script started: production_injury_data.py")

from azureml.core import Datastore
path_data = here("./")
os.chdir(path_data)

workspace = Workspace.from_config()

datastore = workspace.get_default_datastore()

from bs4 import BeautifulSoup
import datetime

url = "https://www.espn.com/nfl/injuries"
response = requests.get(url)
print("Fetching URL")
soup = BeautifulSoup(response.text, "html.parser")
print("URL fetched and parsed")

# Find all tables containing injury data
print("Finding injury tables")
injury_tables = soup.find_all("table", class_="Table")

# Define the columns for the DataFrame
columns = ["Name", "POS", "Date", "Status"]

# Prepare an empty list to store the data
injury_data = []

# Get the current year
current_year = datetime.datetime.now().year

# Iterate over each table
for injury_table in injury_tables:
    # Get all rows in the table
    rows = injury_table.find_all("tr")

    # Iterate over the rows, excluding the header row
    for row in rows[1:]:
        cells = row.find_all("td")

        # Extract data from the cells
        name = cells[0].text.strip()
        pos = cells[1].text.strip()
        date = cells[2].text.strip() + f" {current_year}"
        status = cells[3].text.strip()

        # Append the data as a tuple to the injury_data list
        injury_data.append((name, pos, date, status))

print("Injury data extracted")

# Create the DataFrame
injury_df = pd.DataFrame(injury_data, columns=columns)

print("Cleaning columns")
data_nfl_injury = clean_columns(injury_df)

print("Converting date column to datetime")
data_nfl_injury["date"] = pd.to_datetime(data_nfl_injury["date"])

print("Registering pandas dataframe")
ds = Dataset.Tabular.register_pandas_dataframe(
    dataframe=data_nfl_injury,
    name="data_nfl_injury_test",
    description="NFL injury data",
    target=datastore,
)

print("Script finished: production_injury_data.py")
