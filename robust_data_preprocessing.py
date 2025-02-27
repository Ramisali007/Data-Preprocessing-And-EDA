import os
import pandas as pd
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import track
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize Rich Console for Beautiful Output
console = Console()

# --------------------------- STEP 1: Load and Merge Data --------------------------- #

console.rule("[bold blue]Step 1: Loading and Merging Data[/bold blue]")

# Define base folder paths
base_path = r"D:\Data Science\Assignments\Data Science Assignment 3"
electricity_folder = os.path.join(base_path, "electricity_raw_data")
weather_folder = os.path.join(base_path, "weather_raw_data")

# Load electricity JSON files
console.print("[yellow]Loading Electricity Data...[/yellow]")
electricity_files = glob.glob(os.path.join(electricity_folder, "*.json"))
electricity_dataframes = []

for file in track(electricity_files, description="Processing electricity data..."):
    if os.path.getsize(file) > 0:  
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except UnicodeDecodeError:
            try:
                with open(file, "r", encoding="utf-16") as f:
                    data = json.load(f)
            except Exception as e:
                console.print(f"[red]❌ Skipping {file} due to encoding error: {e}[/red]")
                continue 
        
        # Extract electricity demand data
        if "response" in data and "data" in data["response"]:
            extracted_data = pd.DataFrame(data["response"]["data"])[['period', 'value']]
            extracted_data.rename(columns={'period': 'timestamp', 'value': 'electricity_demand'}, inplace=True)
            extracted_data['timestamp'] = pd.to_datetime(extracted_data['timestamp'], errors='coerce')
            extracted_data['electricity_demand'] = pd.to_numeric(extracted_data['electricity_demand'], errors='coerce')
            extracted_data = extracted_data.groupby('timestamp', as_index=False).sum()
            electricity_dataframes.append(extracted_data)

electricity_df = pd.concat(electricity_dataframes, ignore_index=True) if electricity_dataframes else pd.DataFrame()
electricity_df.dropna(subset=['timestamp'], inplace=True)
electricity_df['timestamp'] = electricity_df['timestamp'].dt.tz_localize('UTC')

console.print("[green]✅ Electricity Data Processed Successfully![/green]")

# Load weather CSV files
console.print("[yellow]Loading Weather Data...[/yellow]")
weather_files = glob.glob(os.path.join(weather_folder, "*.csv"))
weather_dataframes = [pd.read_csv(file, encoding="utf-8") for file in weather_files if os.path.getsize(file) > 0]
weather_df = pd.concat(weather_dataframes, ignore_index=True) if weather_dataframes else pd.DataFrame()

# Rename and convert timestamp column
weather_df.rename(columns={'date': 'timestamp'}, inplace=True)
weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'], errors='coerce')
weather_df.dropna(subset=['timestamp'], inplace=True)
weather_df['timestamp'] = weather_df['timestamp'].dt.tz_convert('UTC')

console.print("[green]✅ Weather Data Processed Successfully![/green]")

# Merge datasets on 'timestamp'
merged_df = pd.merge(electricity_df, weather_df, on="timestamp", how="inner")

if 'electricity_demand' not in merged_df.columns:
    console.print("[red]❌ ERROR: 'electricity_demand' column is missing![/red]")
    exit()

console.print("[cyan]✅ Data Merged Successfully![/cyan]")

# --------------------------- STEP 2: Data Preprocessing --------------------------- #

console.rule("[bold blue]Step 2: Data Preprocessing[/bold blue]")

merged_df.ffill(inplace=True)
merged_df.drop_duplicates(inplace=True)

# Extract useful time-based features
merged_df['hour'] = merged_df['timestamp'].dt.hour
merged_df['day'] = merged_df['timestamp'].dt.day
merged_df['month'] = merged_df['timestamp'].dt.month
merged_df['day_of_week'] = merged_df['timestamp'].dt.dayofweek

console.print("[green]✅ Data Preprocessed Successfully![/green]")

# --------------------------- STEP 3: Exploratory Data Analysis (EDA) --------------------------- #

console.rule("[bold blue]Step 3: Exploratory Data Analysis (EDA)[/bold blue]")

# Summary statistics in a table
table = Table(title="Summary Statistics", show_lines=True)
table.add_column("Feature", style="cyan", justify="left")
table.add_column("Mean", style="yellow")
table.add_column("Min", style="green")
table.add_column("Max", style="red")

for col in ['electricity_demand', 'temperature_2m']:
    table.add_row(col, f"{merged_df[col].mean():.2f}", f"{merged_df[col].min():.2f}", f"{merged_df[col].max():.2f}")

console.print(table)

# Electricity Demand Over Time Plot
plt.figure(figsize=(10,5))
plt.plot(merged_df['timestamp'], merged_df['electricity_demand'], label="Electricity Demand", color="blue")
plt.xlabel('Time')
plt.ylabel('Electricity Demand')
plt.title('Electricity Demand Over Time')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(merged_df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation")
plt.show()

console.print("[green]✅ EDA Completed Successfully![/green]")

corr_matrix = merged_df.corr()
corr_matrix.to_csv("correlation_matrix.csv")


summary_df = merged_df.describe().T[['mean', 'min', 'max']]
summary_df.reset_index(inplace=True)
summary_df.rename(columns={'index': 'Feature'}, inplace=True)
summary_df.to_csv("summary_statistics.csv", index=False)

# --------------------------- STEP 4: Regression Modeling --------------------------- #

console.rule("[bold blue]Step 4: Regression Modeling[/bold blue]")

X = merged_df[['hour', 'day', 'month', 'temperature_2m']]
y = merged_df['electricity_demand']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

console.print(f"\n📌 [yellow]Model Evaluation:[/yellow]")
console.print(f"✅ MSE: [cyan]{mse:.2f}[/cyan]")
console.print(f"✅ RMSE: [cyan]{rmse:.2f}[/cyan]")
console.print(f"✅ R² Score: [cyan]{r2:.4f}[/cyan]")

# --------------------------- STEP 5: Save Processed Data --------------------------- #

merged_df.to_csv("processed_electricity_weather_data.csv", index=False)
console.print("\n✅ [bold green]Assignment Completed Successfully![/bold green]")
