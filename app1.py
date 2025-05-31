import os
import glob
import io
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# -----------------------------
# Utility: CSV preprocessor functions
# -----------------------------
def read_csv_without_comments(file_path: str, comment_str: str = "//") -> pd.DataFrame:
    """
    Reads a CSV file, filtering out lines that start with the given comment string.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        filtered_lines = [line for line in f if not line.lstrip().startswith(comment_str)]
    content = "".join(filtered_lines)
    return pd.read_csv(io.StringIO(content))

def preprocess_data(pr_folder: str, ghi_folder: str, output_file: str) -> None:
    """
    Preprocess the data from the PR and GHI folders and merge them into one CSV.
    It reads all CSVs recursively, filters out comment lines, converts Date columns,
    performs an outer join on Date, and saves a CSV with columns (Date, GHI, PR).
    """
    pr_files = glob.glob(os.path.join(pr_folder, '**', '*.csv'), recursive=True)
    ghi_files = glob.glob(os.path.join(ghi_folder, '**', '*.csv'), recursive=True)
    
    pr_dfs = []
    for file in pr_files:
        try:
            df = read_csv_without_comments(file, comment_str="//")
            pr_dfs.append(df)
        except Exception as e:
            st.error(f"Failed to process PR file {file}: {e}")
    
    ghi_dfs = []
    for file in ghi_files:
        try:
            df = read_csv_without_comments(file, comment_str="//")
            ghi_dfs.append(df)
        except Exception as e:
            st.error(f"Failed to process GHI file {file}: {e}")
    
    if pr_dfs:
        pr_df = pd.concat(pr_dfs, ignore_index=True)
    else:
        pr_df = pd.DataFrame(columns=["Date", "PR"])
        
    if ghi_dfs:
        ghi_df = pd.concat(ghi_dfs, ignore_index=True)
    else:
        ghi_df = pd.DataFrame(columns=["Date", "GHI"])
    
    pr_df['Date'] = pd.to_datetime(pr_df['Date'])
    ghi_df['Date'] = pd.to_datetime(ghi_df['Date'])
    
    merged_df = pd.merge(ghi_df, pr_df, on='Date', how='outer')
    merged_df = merged_df.sort_values(by='Date').reset_index(drop=True)
    merged_df = merged_df[['Date', 'GHI', 'PR']]
    
    merged_df.to_csv(output_file, index=False)
    st.success(f"Saved merged data with {merged_df.shape[0]} rows to {output_file}")

def unzip_files(zip_path: str, extract_path: str) -> None:
    """
    Unzips a zip file to the target folder.
    """
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# -----------------------------
# Graph generation function (modified to return a matplotlib Figure)
# -----------------------------
def generate_graph(processed_csv_file: str, start_date: str = None, end_date: str = None) -> plt.Figure:
    """
    Reads the processed CSV data (Date, GHI, PR) and generates a graph with:
      - Scatter points: Daily PR plotted as points (color-coded by GHI).
      - A red line for the 30-day moving average of PR.
      - A dark green line for the budget line (computed per July–June cycle).
      - A textbox displaying averages for last 7,30,60,90,365 days plus lifetime average,
        plus budget statistics.
      - Y-axis fixed from 0 to 90.
      - Optionally, only data between start_date and end_date is plotted.
    Returns a matplotlib Figure.
    """
    # Read CSV
    df = pd.read_csv(processed_csv_file, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Filter date range if provided
    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date'] <= pd.to_datetime(end_date)]
    
    # Calculate moving average
    df['PR_MA30'] = df['PR'].rolling(window=30, min_periods=1).mean()
    
    # Compute budget line value based on July-June cycle.
    def compute_budget(date):
        cycle_year = date.year - 1 if date.month < 7 else date.year
        years_since_start = max(cycle_year - 2019, 0)
        return 73.9 * (0.992)**(years_since_start)
    df['budget'] = df['Date'].apply(compute_budget)
    
    # Compute cycle year for reporting.
    df['cycle_year'] = df['Date'].apply(lambda d: d.year - 1 if d.month < 7 else d.year)
    
    # Define color mapping for GHI.
    def ghi_color(ghi):
        if ghi < 2:
            return 'navy'
        elif ghi < 4:
            return 'lightblue'
        elif ghi < 6:
            return 'orange'
        else:
            return 'brown'
    df['color'] = df['GHI'].apply(ghi_color)
    
    # Create the plot.
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(df['Date'], df['PR'], c=df['color'], edgecolor='k', s=50, label='Daily PR')
    ax.plot(df['Date'], df['PR_MA30'], color='red', linewidth=2, label='30-day Moving Average')
    ax.plot(df['Date'], df['budget'], color='darkgreen', linewidth=2, label='Budget Line')
    ax.set_ylim(0, 90)
    ax.set_xlabel('Date')
    ax.set_ylabel('PR Value')
    ax.set_title('PV Plant Performance: Daily PR, 30-Day MA, and Budget')
    ax.legend()
    
    # Compute statistics for display.
    yearly_budgets = df.groupby('cycle_year')['budget'].mean().reset_index()
    yearly_budget_text = ", ".join(f"{int(row['cycle_year'])}: {row['budget']:.1f}" for idx, row in yearly_budgets.iterrows())
    days_above_budget = (df['PR'] > df['budget']).sum()
    total_days = len(df)
    above_budget_pct = (days_above_budget / total_days * 100) if total_days > 0 else 0
    max_date = df['Date'].max()
    periods = [7, 30, 60, 90, 365]
    avg_text_lines = []
    for n in periods:
        recent = df[df['Date'] >= max_date - pd.Timedelta(days=n)]
        avg_pr = recent['PR'].mean() if not recent.empty else np.nan
        avg_text_lines.append(f"Last {n}d Avg: {avg_pr:.2f}")
    lifetime_avg = df['PR'].mean() if not df.empty else np.nan
    avg_text_lines.append(f"Lifetime Avg: {lifetime_avg:.2f}")
    stats_text = (
        f"Target Budget Yield Performance Ratio\n"
        f"[{yearly_budget_text}]\n\n"
        f"Points above Target Budget PR = {days_above_budget}/{total_days} = {above_budget_pct:.1f}%\n\n"
        + "\n".join(avg_text_lines)
    )
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    fig.autofmt_xdate()
    return fig

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="PV Doctor", layout="wide")
st.title("PV Doctor: PV Plant Performance Analysis")

# Sidebar controls:
st.sidebar.header("Input Options")
data_option = st.sidebar.radio("Select Data Input", ("Use merged CSV", "Process from Zip Files"))
start_date = st.sidebar.text_input("Start Date (YYYY-MM-DD)", value="2019-01-01")
end_date = st.sidebar.text_input("End Date (YYYY-MM-DD)", value="2024-06-30")

if data_option == "Use merged CSV":
    uploaded_file = st.sidebar.file_uploader("Upload merged CSV", type=["csv"])
    if uploaded_file is not None:
        merged_csv_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(merged_csv_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    else:
        st.sidebar.info("Please upload a merged CSV file.")
        merged_csv_path = None
else:
    st.sidebar.info("Upload ZIP files for PR and GHI to process.")
    pr_zip = st.sidebar.file_uploader("Upload PR.zip", type=["zip"])
    ghi_zip = st.sidebar.file_uploader("Upload GHI.zip", type=["zip"])
    if pr_zip and ghi_zip:
        base_dir = "temp_extracted"
        pr_extract_path = os.path.join(base_dir, "PR")
        ghi_extract_path = os.path.join(base_dir, "GHI")
        os.makedirs(pr_extract_path, exist_ok=True)
        os.makedirs(ghi_extract_path, exist_ok=True)
        # Save uploaded ZIP files to disk.
        pr_zip_path = os.path.join("temp_extracted", "PR.zip")
        ghi_zip_path = os.path.join("temp_extracted", "GHI.zip")
        with open(pr_zip_path, "wb") as f:
            f.write(pr_zip.getbuffer())
        with open(ghi_zip_path, "wb") as f:
            f.write(ghi_zip.getbuffer())
        # Unzip the files.
        with zipfile.ZipFile(pr_zip_path, 'r') as zip_ref:
            zip_ref.extractall(pr_extract_path)
        with zipfile.ZipFile(ghi_zip_path, 'r') as zip_ref:
            zip_ref.extractall(ghi_extract_path)
        # Process and generate merged CSV.
        merged_csv_path = os.path.join(base_dir, "merged_data.csv")
        preprocess_data(pr_extract_path, ghi_extract_path, merged_csv_path)
    else:
        merged_csv_path = None

# Button to generate and show graph if a merged CSV path is available.
if st.sidebar.button("Generate Graph") and merged_csv_path is not None:
    fig = generate_graph(merged_csv_path, start_date=start_date, end_date=end_date)
    st.pyplot(fig)
else:
    st.info("Please provide the required file(s) and click 'Generate Graph'.")