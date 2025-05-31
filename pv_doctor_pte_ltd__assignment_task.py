import os
import zipfile
import glob
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Helper function to ignore comment lines
def read_csv_without_comments(file_path: str, comment_str: str = "//") -> pd.DataFrame:
    with open(file_path, 'r', encoding='utf-8') as f:
        filtered_lines = [line for line in f if not line.lstrip().startswith(comment_str)]
    content = "".join(filtered_lines)
    return pd.read_csv(io.StringIO(content))

# Preprocessing data
def preprocess_data(pr_folder: str, ghi_folder: str, output_file: str) -> None:
    pr_files = glob.glob(os.path.join(pr_folder, '**', '*.csv'), recursive=True)
    ghi_files = glob.glob(os.path.join(ghi_folder, '**', '*.csv'), recursive=True)

    pr_dfs = [read_csv_without_comments(file) for file in pr_files]
    ghi_dfs = [read_csv_without_comments(file) for file in ghi_files]

    pr_df = pd.concat(pr_dfs, ignore_index=True) if pr_dfs else pd.DataFrame(columns=["Date", "PR"])
    ghi_df = pd.concat(ghi_dfs, ignore_index=True) if ghi_dfs else pd.DataFrame(columns=["Date", "GHI"])

    pr_df['Date'] = pd.to_datetime(pr_df['Date'])
    ghi_df['Date'] = pd.to_datetime(ghi_df['Date'])

    merged_df = pd.merge(ghi_df, pr_df, on='Date', how='outer')
    merged_df = merged_df.sort_values(by='Date').reset_index(drop=True)
    merged_df = merged_df[['Date', 'GHI', 'PR']]
    merged_df.to_csv(output_file, index=False)

# Visualization
def generate_graph(df: pd.DataFrame, start_date: str = None, end_date: str = None):
    df = df.sort_values('Date').reset_index(drop=True)
    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date'] <= pd.to_datetime(end_date)]

    df['PR_MA30'] = df['PR'].rolling(window=30, min_periods=1).mean()

    def compute_budget(date):
        cycle_year = date.year - 1 if date.month < 7 else date.year
        years_since_start = max(cycle_year - 2019, 0)
        return 73.9 * (0.992)**(years_since_start)

    df['budget'] = df['Date'].apply(compute_budget)

    def ghi_color(ghi):
        if ghi < 2: return 'navy'
        elif ghi < 4: return 'lightblue'
        elif ghi < 6: return 'orange'
        else: return 'brown'

    df['color'] = df['GHI'].apply(ghi_color)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(df['Date'], df['PR'], c=df['color'], edgecolor='k', s=50, label='Daily PR')
    ax.plot(df['Date'], df['PR_MA30'], color='red', linewidth=2, label='30-day Moving Average')
    ax.plot(df['Date'], df['budget'], color='darkgreen', linewidth=2, label='Budget Line')
    ax.set_ylim(0, 90)
    ax.set_xlabel('Date')
    ax.set_ylabel('PR Value')
    ax.set_title('PV Plant Performance: Daily PR, 30-Day MA, and Budget')
    ax.legend()

    # Stats box
    max_date = df['Date'].max()
    periods = [7, 30, 60, 90, 365]
    avg_text_lines = [f"Last {n}d Avg: {df[df['Date'] >= max_date - pd.Timedelta(days=n)]['PR'].mean():.2f}" for n in periods]
    avg_text_lines.append(f"Lifetime Avg: {df['PR'].mean():.2f}")

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.02, "\n".join(avg_text_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=props)

    fig.autofmt_xdate()
    st.pyplot(fig)

# Streamlit UI
st.title("PV Doctor Dashboard")

pr_zip = st.file_uploader("Upload PR.zip", type='zip')
ghi_zip = st.file_uploader("Upload GHI.zip", type='zip')

if pr_zip and ghi_zip:
    base_dir = "temp_data"
    os.makedirs(base_dir, exist_ok=True)

    pr_path = os.path.join(base_dir, "PR")
    ghi_path = os.path.join(base_dir, "GHI")
    os.makedirs(pr_path, exist_ok=True)
    os.makedirs(ghi_path, exist_ok=True)

    with zipfile.ZipFile(pr_zip, 'r') as zip_ref:
        zip_ref.extractall(pr_path)
    with zipfile.ZipFile(ghi_zip, 'r') as zip_ref:
        zip_ref.extractall(ghi_path)

    output_file = os.path.join(base_dir, "merged_data.csv")
    preprocess_data(pr_path, ghi_path, output_file)
    df = pd.read_csv(output_file, parse_dates=['Date'])

    st.success(f"Merged data shape: {df.shape}")
    start_date = st.date_input("Start Date", df['Date'].min().date())
    end_date = st.date_input("End Date", df['Date'].max().date())

    if st.button("Generate Graph"):
        generate_graph(df, str(start_date), str(end_date))
