from pathlib import Path
import pandas as pd

SECOND_FEATURE_DIR = Path(r"G:\3TD_KLTN_AI\CMOSE\secondFeature")

csv_files = list(SECOND_FEATURE_DIR.rglob("*.csv"))

print("📄 Số file CSV:", len(csv_files))

if len(csv_files) == 0:
    print("❌ Không tìm thấy file CSV nào. Kiểm tra lại cấu trúc folder!")
    exit()

dfs = []

for file in csv_files:
    df = pd.read_csv(file)
    df["source_file"] = file.name
    df["source_path"] = str(file.parent)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

print("✅ Dataset shape:", data.shape)
print(data.head())

