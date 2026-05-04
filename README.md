# PV Doctor - PV Plant Performance Analysis

## Overview

**PV Doctor** is a Python-based application designed to analyze and visualize the performance of Photovoltaic (PV) plants. It processes solar irradiance (GHI - Global Horizontal Irradiance) and Performance Ratio (PR) data from CSV files, merges them by date, and generates comprehensive performance visualizations with budget tracking and historical analytics.

## Project Purpose

This project was developed for PV Doctor Pte. Ltd. as an assignment task to automate the analysis of PV plant performance metrics. The application helps monitor plant efficiency by comparing actual performance against budget targets while accounting for various environmental and operational factors.

## Features

### 1. **Data Preprocessing**
- Reads multiple CSV files from PR (Performance Ratio) and GHI (Global Horizontal Irradiance) folders
- Filters out comment lines (lines starting with `//`)
- Handles encoding properly (UTF-8)
- Merges datasets by date using outer join to preserve all records
- Outputs a unified CSV with columns: `Date`, `GHI`, `PR`

### 2. **Performance Visualization**
The application generates a comprehensive graph featuring:

- **Scatter Plot**: Daily PR values color-coded by GHI levels:
  - Navy Blue: GHI < 2
  - Light Blue: 2 ≤ GHI < 4
  - Orange: 4 ≤ GHI < 6
  - Brown: GHI ≥ 6

- **30-Day Moving Average**: Red line showing smoothed PR trends

- **Budget Line**: Dark green line representing the target budget yield
  - Initial budget (July 2019 - June 2020): 73.9
  - Annual decrease: 0.8% per year
  - Cycle basis: July to June fiscal year

- **Performance Statistics Box** (lower-right):
  - Target budget yield performance ratio by cycle year
  - Percentage of days performing above budget
  - Rolling averages: Last 7, 30, 60, 90, and 365 days
  - Lifetime average PR

### 3. **Interactive Web Application**
Built with **Streamlit**, the app provides:
- Flexible data input options (pre-merged CSV or raw ZIP files)
- Date range filtering
- Real-time graph generation
- File upload capabilities
