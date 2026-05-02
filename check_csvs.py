import os
import glob
import pandas as pd

folder_path = 'data/raw'
all_files = glob.glob(os.path.join(folder_path, '*.csv'))

if not all_files:
    print('No CSV files found.')
else:
    for file in all_files:
        try:
            df = pd.read_csv(file, nrows=5)
            print(f"{os.path.basename(file)}: Loaded, shape={df.shape}")
        except Exception as e:
            print(f"{os.path.basename(file)}: Failed to load. Error: {e}")
