import pandas as pd
import os
import numpy as np



def merge_dfs(dfs: list[pd.DataFrame]):
    out_df = dfs[0]

    for i in range(1, len(dfs)):
        out_df = pd.merge(out_df, dfs[i], how="left", on=["Country", "Date"])

    return out_df


def generate_indications_cols(df: pd.DataFrame, indications: list[any], label: str, filler):
    new_cols = pd.DataFrame.from_dict([
        {
            f"{label}_ind{ind}": filler
                for ind in indications
        } 
        for _ in range(len(df))
    ], dtype=float)

    return pd.concat([ df, new_cols ], axis=1)


def fix_date_missing_period(df: pd.DataFrame, col_name="Date"):
    month_map = {
        "janv": "Jan",
        "févr": "Feb",
        "mars": "Mar",
        "avr": "Apr",
        "mai": "May",
        "juin": "Jun",
        "juil": "Jul",
        "août": "Aug",
        "sept": "Sep",
        "oct": "Oct",
        "nov": "Nov",
        "déc": "Dec"
    }
    
    df[col_name] = df[col_name].apply(lambda x: f"{month_map[x.split('-')[0]]}-{x.split('-')[1][-2:]}")
    df[col_name] = pd.to_datetime(df[col_name], format="%b-%y")
    return df



def clean_innovix_ex_factory_vol(df: pd.DataFrame):
    df.drop(columns=["Data type", "Unit of measure", "Measure"], inplace=True, errors="ignore")
    df.rename(columns={"Value": "ex_factory_volumes"}, inplace=True)
    return df


def clean_innovix_demand_volumes(df: pd.DataFrame, products, measure_name="Unit of measure", measure_type="Month of treatment"):
    df.drop( df[ df[measure_name] == measure_type ].index )
    df.drop(columns=["Data type", measure_name], inplace=True, errors="ignore")

    for prod in products:
        df[f"{prod}_months_of_treatment"] = 0.0

    for i, row in df.iterrows():
        df.loc[i, f"{row['Product']}_months_of_treatment"] = row["Value"]

    df.drop(columns=["Value"], inplace=True)
    country = df.iloc[0]["Country"]
    df = df.groupby("Date").sum(numeric_only=True)
    df["Country"] = country

    return df


def clean_innovix_activity(df: pd.DataFrame, indications: list[int], products: list[str]):
    channel_name_map = {
        "Face to face call": "face-to-face-call", 
        "Email": "email", 
        "Remote call": "remote-call", 
        "Meetings": "meetings"
    }
    df["Channel"] = df["Channel"].map(channel_name_map)

    if indications is not None:
        for prod in products:
            for channel in df["Channel"].unique().tolist():
                df = generate_indications_cols(df, indications, f"{prod}_{channel}", 0.0)

        for i, row in df.iterrows():
            df.loc[i, f"{row['Product']}_{row['Channel']}_{row['Indication']}"] = row["Value"]
    else:
        for i, row in df.iterrows():
            df.loc[i, f"{row['Product']}_{row['Channel']}"] = row["Value"]

    df.drop(columns=["Data type", "Product", "Channel", "Indication", "Value"], inplace = True, errors="ignore")
    country = df.iloc[0]["Country"]
    df = df.groupby("Date").sum(numeric_only=True)
    df["Country"] = country

    return df


def clean_innovix_share_of_voice(df: pd.DataFrame, indications: list[int], products):
    df["Indication"] = df["Indication"].map(lambda x: f"ind{x.split(' ')[1]}")
    for prod in products:
        df = generate_indications_cols(df, indications, f"{prod}_share_of_voice", 0.0)

    for i, row in df.iterrows():
        df.loc[i, f"{row['Product']}_share_of_voice_{row['Indication']}"] = row["Value"]

    df.drop(columns=["Data type", "Indication", "Product", "Value", "Product"], inplace=True)
    country = df.iloc[0]["Country"]
    df = df.groupby("Date").sum(numeric_only=True)
    df["Country"] = country
    return df


def clean_innovix_new_patient_share(df: pd.DataFrame, indications: list[int], products):
    df["Indication"] = df["Indication"].map(lambda x: f"ind{x.split(' ')[1]}")
    for prod in products:
        df = generate_indications_cols(df, indications, f"{prod}_new_patient_shape", 0.0)

    for i, row in df.iterrows():
        df.loc[i, f"{row['Product']}_new_patient_shape_{row['Indication']}"] = row["Value"]

    df.drop(columns=["Data type", "Indication", "Product", "Sub-Indication", "Value"], inplace=True)
    country = df.iloc[0]["Country"]
    df = df.groupby("Date").sum(numeric_only=True)
    df["Country"] = country
    return df


def clean_innovix_indication_split(df: pd.DataFrame, indications: list[int], products):
    df["Indication"] = df["Indication"].map(lambda x: f"ind{x.split(' ')[1]}")
    for prod in products:
        df = generate_indications_cols(df, indications, f"{prod}_indication_split", 0.0)

    for i, row in df.iterrows():
        df.loc[i, f"{row['Product']}_indication_split_{row['Indication']}"] = row["Value"]

    df.drop(columns=["Data type", "Indication", "Product", "Sub-Indication", "Value"], inplace=True, errors="ignore")
    country = df.iloc[0]["Country"]
    df = df.groupby("Date").sum(numeric_only=True)
    df["Country"] = country
    return df


def clean_patient_share(df: pd.DataFrame, indications: list[int], products):
    df["Indication"] = df["Indication"].map(lambda x: f"ind{x.split(' ')[1]}")
    for prod in products:
        df = generate_indications_cols(df, indications, f"{prod}_patient_share", 0.0)
    df = df[df["Measure"] == "New Patient %"]

    for i, row in df.iterrows():
        df.loc[i, f"{row['Product']}_patient_share_{row['Indication']}"] = row["Value"]

    df.drop(columns=["Data type", "Indication", "Product", "Sub-Indication", "Value", "Measure"], inplace=True, errors="ignore")
    country = df.iloc[0]["Country"]
    df = df.groupby("Date").sum(numeric_only=True)
    df["Country"] = country
    return df



def load_innovix_floresland(data_root="./data"):
    df_ex_factory_vol = pd.read_excel(os.path.join(data_root, "INNOVIX_Floresland.xlsx"), sheet_name="Ex-Factory volumes")
    df_demand_volumes = pd.read_excel(os.path.join(data_root, "INNOVIX_Floresland.xlsx"), sheet_name="Demand volumes")
    df_activity = pd.read_excel(os.path.join(data_root, "INNOVIX_Floresland.xlsx"), sheet_name="Activity")
    df_share_of_voice = pd.read_excel(os.path.join(data_root, "INNOVIX_Floresland.xlsx"), sheet_name="Share of Voice")
    df_new_patient_share = pd.read_excel(os.path.join(data_root, "INNOVIX_Floresland.xlsx"), sheet_name="New patient share")
    df_new_patient_share = fix_date_missing_period(df_new_patient_share)
    df_indication_split = pd.read_excel(os.path.join(data_root, "INNOVIX_Floresland.xlsx"), sheet_name="Indication split")

    indications = set(
        [int(x.split(" ")[1]) for x in df_activity["Indication"].unique()] +
        [int(x.split(" ")[1]) for x in df_share_of_voice["Indication"].unique()] +
        [int(x.split(" ")[1]) for x in df_new_patient_share["Indication"].unique()] +
        [int(x.split(" ")[1]) for x in df_indication_split["Indication"].unique()]
    )
    products = set(
        df_ex_factory_vol["Product"].unique().tolist() +
        df_demand_volumes["Product"].unique().tolist() +
        df_activity["Product"].unique().tolist() +
        df_share_of_voice["Product"].unique().tolist() +
        df_new_patient_share["Product"].unique().tolist() +
        df_new_patient_share["Product"].unique().tolist() +
        df_indication_split["Product"].unique().tolist()
    )

    df = merge_dfs([
        clean_innovix_ex_factory_vol(df_ex_factory_vol),
        clean_innovix_demand_volumes(df_demand_volumes, products),
        clean_innovix_activity(df_activity, indications, products),
        clean_innovix_share_of_voice(df_share_of_voice, indications, products),
        clean_innovix_new_patient_share(df_new_patient_share, indications, products),
        clean_innovix_indication_split(df_indication_split, indications, products)
    ])

    df.drop(columns=["Country", "Product"], inplace=True)
    df = df.sort_values(by="Date")
    df["Date"] = np.arange(0, len(df["Date"]))
    df.dropna(inplace=True, how="all", axis=1)

    return df



def load_innovix_elbonie(data_root="./data"):
    df_ex_factory_vol = pd.read_excel(os.path.join(data_root, "INNOVIX_Elbonie.xlsx"), sheet_name="Ex-factory volumes")
    df_demand_volumes = pd.read_excel(os.path.join(data_root, "INNOVIX_Elbonie.xlsx"), sheet_name="Demand volumes")
    df_activity = pd.read_excel(os.path.join(data_root, "INNOVIX_Elbonie.xlsx"), sheet_name="Activity")
    df_indication_split = pd.read_excel(os.path.join(data_root, "INNOVIX_Elbonie.xlsx"), sheet_name="Indication split")
    df_indication_split = fix_date_missing_period(df_indication_split)


    indications = set(
        [int(x.split(" ")[1]) for x in df_activity["Indication"].unique()] +
        [int(x.split(" ")[1]) for x in df_indication_split["Indication"].unique()]
    )
    products = set(
        df_ex_factory_vol["Product"].unique().tolist() +
        df_demand_volumes["Product"].unique().tolist() +
        df_activity["Product"].unique().tolist() +
        df_indication_split["Product"].unique().tolist()
    )

    df = merge_dfs([
        clean_innovix_ex_factory_vol(df_ex_factory_vol),
        clean_innovix_demand_volumes(df_demand_volumes, products, measure_type="Milligrams"),
        clean_innovix_activity(df_activity, indications, products),
        clean_innovix_indication_split(df_indication_split, indications, products)
    ])

    df.drop(columns=["Country", "Product"], inplace=True)
    df = df.sort_values(by="Date")
    df["Date"] = np.arange(0, len(df["Date"]))
    df.dropna(inplace=True, how="all", axis=1)

    return df


def load_bristor_zegoland(data_root="./data"):
    df_ex_factory_vol = pd.read_excel(os.path.join(data_root, "BRISTOR_Zegoland.xlsx"), sheet_name="Ex-factory volumes")
    df_demand_volumes = pd.read_excel(os.path.join(data_root, "BRISTOR_Zegoland.xlsx"), sheet_name="Demand volumes")
    df_demand_volumes = fix_date_missing_period(df_demand_volumes)
    df_activity = pd.read_excel(os.path.join(data_root, "BRISTOR_Zegoland.xlsx"), sheet_name="Activity")
    df_activity = fix_date_missing_period(df_activity)
    df_share_of_voice = pd.read_excel(os.path.join(data_root, "BRISTOR_Zegoland.xlsx"), sheet_name="Share of Voice")
    df_share_of_voice.rename(columns={ "Products": "Product" }, inplace=True)
    df_share_of_voice = fix_date_missing_period(df_share_of_voice)
    df_patient_share = pd.read_excel(os.path.join(data_root, "BRISTOR_Zegoland.xlsx"), sheet_name="Patient numbers and share")
    df_patient_share = fix_date_missing_period(df_patient_share)

    indications = set(
        [x.split(" ")[1] for x in df_share_of_voice["Indication"].unique()] +
        [x.split(" ")[1] for x in df_patient_share["Indication"].unique()]
    )

    products = set(
        df_ex_factory_vol["Product"].unique().tolist() +
        df_demand_volumes["Product"].unique().tolist() +
        df_activity["Product"].unique().tolist() +
        df_share_of_voice["Product"].unique().tolist() +
        df_patient_share["Product"].unique().tolist()
    )

    df = merge_dfs([
        clean_innovix_ex_factory_vol(df_ex_factory_vol,),
        clean_innovix_demand_volumes(df_demand_volumes, products, measure_name="Measure", measure_type="Days of Treatment"),
        clean_innovix_activity(df_activity, None, products),
        clean_innovix_share_of_voice(df_share_of_voice, indications, products),
        clean_patient_share(df_patient_share, indications, products)
    ])

    df.drop(columns=["Country", "Product"], inplace=True)
    df = df.sort_values(by="Date")
    df["Date"] = np.arange(0, len(df["Date"]))
    df.dropna(inplace=True, how="all", axis=1)

    return df