from pathlib import Path

def save_tree(path, file, prefix=""):
    file.write(prefix + path.name + "/\n")
    for p in sorted(path.iterdir()):
        if p.is_dir():
            save_tree(p, file, prefix + "│   ")
        else:
            file.write(prefix + "│   " + p.name + "\n")


# 🔴 SỬA ĐƯỜNG DẪN NÀY CHO ĐÚNG VỚI MÁY BẠN
CMOSE_PATH = Path(r"G:\3TD_KLTN_AI\Dataset\CMOSE")

OUTPUT_FILE = Path("cmose_structure.txt")

if not CMOSE_PATH.exists():
    print("❌ Không tìm thấy thư mục CMOSE:", CMOSE_PATH)
else:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        save_tree(CMOSE_PATH, f)

    print("✅ Đã xuất cấu trúc CMOSE ra file:", OUTPUT_FILE.resolve())
