import pandas as pd
import json


def mapMarks(EXCEL_FILE,JSON_FILE):
    # Load master Excel
    master_df = pd.read_excel(EXCEL_FILE)
    master_df["REGISTER NO"] = master_df["REGISTER NO"].astype(str).str.strip().str.upper()
    master_df["NAME"] = master_df["NAME"].astype(str).str.strip().str.upper()

    # Load Gemini JSON
    with open(JSON_FILE, "r") as f:
        student_data = json.load(f)

    # Convert to DataFrame for easier matching
    json_df = pd.DataFrame(student_data)
    json_df["Registerno"] = json_df["Registerno"].astype(str).str.strip().str.upper()
    json_df["Name"] = json_df["Name"].astype(str).str.strip().str.upper()

    # Expand marks into columns (Q1, Q2, ...)
    marks_expanded = json_df["Marks"].apply(pd.Series)
    marks_expanded.columns = [f"Q{c}" for c in marks_expanded.columns]

    json_df = pd.concat([json_df.drop(columns=["Marks"]), marks_expanded], axis=1)

    # Add empty columns for marks in master if not exist
    for col in marks_expanded.columns:
        if col not in master_df.columns:
            master_df[col] = None

    # Now map data
    for _, row in json_df.iterrows():
        reg_no = row["Registerno"]
        name = row["Name"]

        # Match by Register No
        idx = master_df[master_df["REGISTER NO"] == reg_no].index

        # If no match, try by Name
        if len(idx) == 0 and name:
            idx = master_df[master_df["NAME"].str.contains(name, na=False, regex=False)].index

        # If match found → update marks
        if len(idx) > 0:
            for q_col in marks_expanded.columns:
                master_df.loc[idx, q_col] = row[q_col]

    # Save final Excel
    master_df.to_excel("mapped_results.xlsx", index=False)

    print("✅ Mapped results saved to mapped_results.xlsx")
    return {"success"}
