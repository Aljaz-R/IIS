import os

import numpy as np
import pandas as pd
from lxml import etree as ET


def clean_value(value):
    if value is None or value == "":
        return np.nan
    if value == "<1":
        return 1
    if value == "<2":
        return 2
    return value


def preprocess_air_data():
    # Open XML file
    with open("data/raw/air/air_data.xml", "rb") as file:
        tree = ET.parse(file)
        root = tree.getroot()

    # Extract and print metadata
    print(f"Version: {root.attrib['verzija']}")
    print(f"Source: {root.find('vir').text}")
    print(f"Suggested Capture: {root.find('predlagan_zajem').text}")
    print(f"Suggested Capture Period: {root.find('predlagan_zajem_perioda').text}")
    print(f"Preparation Date: {root.find('datum_priprave').text}")

    # Get all station codes
    sifra_vals = sorted(set(tree.xpath("//postaja/@sifra")))

    # Output folder
    output_dir = "data/preprocessed/air"
    os.makedirs(output_dir, exist_ok=True)

    # Process each station
    for sifra in sifra_vals:
        postaja_elements = tree.xpath(f'//postaja[@sifra="{sifra}"]')

        columns = ["date_to", "PM10", "PM2.5"]

        # Load existing CSV if it exists
        csv_path = os.path.join(output_dir, f"{sifra}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(columns=columns)

        # Collect new rows from XML
        new_rows = []
        for postaja in postaja_elements:
            date_to = postaja.findtext("datum_do")
            pm10 = clean_value(postaja.findtext("pm10"))
            pm2_5 = clean_value(postaja.findtext("pm2.5"))

            new_rows.append([date_to, pm10, pm2_5])

        # Append new rows
        if new_rows:
            df_new = pd.DataFrame(new_rows, columns=columns)
            df = pd.concat([df, df_new], ignore_index=True)

        # Clean values
        df = df.replace("", np.nan)
        df = df.replace("<1", 1)
        df = df.replace("<2", 2)

        # Remove duplicates and sort
        df = df.drop_duplicates(subset=["date_to"])
        df = df.sort_values(by="date_to")

        # Save updated CSV
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")


if __name__ == "__main__":
    preprocess_air_data()