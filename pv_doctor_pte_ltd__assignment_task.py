# -*- coding: utf-8 -*-
"""PV Doctor Pte. Ltd._Assignment task

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1s-KKXFVNIjrd_1rl1ZiHi0wGusAXMzrA
"""

import os
import zipfile
import glob
import pandas as pd
import io

def read_csv_without_comments(file_path: str, comment_str: str = "//") -> pd.DataFrame:
    with open(file_path, 'r', encoding='utf-8') as f:
        filtered_lines = [line for line in f if not line.lstrip().startswith(comment_str)]
    content = "".join(filtered_lines)
    return pd.read_csv(io.StringIO(content))

def preprocess_data(pr_folder: str, ghi_folder: str, output_file: str) -> None:
    pr_files = glob.glob(os.path.join(pr_folder, '**', '*.csv'), recursive=True)
    ghi_files = glob.glob(os.path.join(ghi_folder, '**', '*.csv'), recursive=True)

    pr_dfs = []
    for file in pr_files:
        try:
            df = read_csv_without_comments(file, comment_str="//")
            pr_dfs.append(df)
        except Exception as e:
            print(f"Failed to process PR file {file}: {e}")

    ghi_dfs = []
    for file in ghi_files:
        try:
            df = read_csv_without_comments(file, comment_str="//")
            ghi_dfs.append(df)
        except Exception as e:
            print(f"Failed to process GHI file {file}: {e}")

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
    print(f"Saved merged data with {merged_df.shape[0]} rows to {output_file}")


base_dir = "/content/Intern"
pr_zip_path = "/content/PR.zip"
ghi_zip_path = "/content/GHI.zip"
pr_extract_path = os.path.join(base_dir, "PR")
ghi_extract_path = os.path.join(base_dir, "GHI")

os.makedirs(pr_extract_path, exist_ok=True)
os.makedirs(ghi_extract_path, exist_ok=True)

with zipfile.ZipFile(pr_zip_path, 'r') as zip_ref:
    zip_ref.extractall(pr_extract_path)

with zipfile.ZipFile(ghi_zip_path, 'r') as zip_ref:
    zip_ref.extractall(ghi_extract_path)

merged_csv = os.path.join(base_dir, "merged_data.csv")
preprocess_data(pr_extract_path, ghi_extract_path, merged_csv)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_graph(processed_csv_file: str, start_date: str = None, end_date: str = None) -> None:
    """
    Reads the processed CSV data (Date, GHI, PR) and generates a graph with:
      - Scatter points: each day's PR is plotted as a point.
          The point color is determined by the daily GHI:
            * GHI < 2: Navy blue
            * 2 <= GHI < 4: Light blue
            * 4 <= GHI < 6: Orange
            * GHI >= 6: Brown
      - A red line representing the 30-day moving average of PR.
      - A dark green line representing the budget line. For each date, the budget is computed:
            • For the first cycle (July 2019 - June 2020) the value is 73.9.
            • For subsequent cycles the value decreases by 0.8% each year.
      - A textbox in the lower-right displaying average PR for the last 7, 30, 60, 90, and 365 days,
        as well as the lifetime average PR and additional budget statistics.
      - The y-axis is set from 0 to 90.
      - Optionally, only data between start_date and end_date is plotted.

    Args:
        processed_csv_file (str): Path to the merged CSV file.
        start_date (str, optional): Start date in 'YYYY-MM-DD' format.
        end_date (str, optional): End date in 'YYYY-MM-DD' format.
    """

    df = pd.read_csv(processed_csv_file, parse_dates=['Date'])
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


    df['cycle_year'] = df['Date'].apply(lambda d: d.year - 1 if d.month < 7 else d.year)

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

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(df['Date'], df['PR'], c=df['color'], edgecolor='k', s=50, label='Daily PR')

    ax.plot(df['Date'], df['PR_MA30'], color='red', linewidth=2, label='30-day Moving Average')

    ax.plot(df['Date'], df['budget'], color='darkgreen', linewidth=2, label='Budget Line')

    ax.set_ylim(0, 90)

    ax.set_xlabel('Date')
    ax.set_ylabel('PR Value')
    ax.set_title('PV Plant Performance: Daily PR, 30-Day MA, and Budget')
    ax.legend()

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

    plt.show()

if __name__ == "__main__":
    processed_csv = "/content/Intern/merged_data.csv"
    start_date = "2019-01-01"
    end_date = "2024-06-30"
    generate_graph(processed_csv, start_date=start_date, end_date=end_date)
