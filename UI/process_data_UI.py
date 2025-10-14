import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
import sys
sys.path.append("c:\\python_workspace")
from process_kline_data import clean_data, add_technical_indicators, add_time_features, add_target_variable

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        df = clean_data(df)
        df = add_technical_indicators(df)
        df = add_time_features(df)
        df = add_target_variable(df)
        
        # Show output buttons
        def save_as_csv():
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], initialfile=f"{os.path.basename(file_path).split('.')[0]}_processed_data.csv")
            if save_path:
                df.to_csv(save_path, index=False)
                print(f"CSV saved to {save_path}")
        
        def save_as_parquet():
            save_path = filedialog.asksaveasfilename(defaultextension=".parquet", filetypes=[("Parquet files", "*.parquet")], initialfile=f"{os.path.basename(file_path).split('.')[0]}_processed_data.parquet")
            if save_path:
                df.to_parquet(save_path, engine='pyarrow')
                print(f"Parquet saved to {save_path}")
        
        csv_button = tk.Button(root, text="Output CSV", command=save_as_csv)
        csv_button.pack(pady=5)
        
        parquet_button = tk.Button(root, text="Output Parquet", command=save_as_parquet)
        parquet_button.pack(pady=5)

# Create UI
root = tk.Tk()
root.title("CSV Processor")

select_button = tk.Button(root, text="Select CSV File", command=select_file)
select_button.pack(pady=20)

root.mainloop()