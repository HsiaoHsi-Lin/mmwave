import os
import shutil
import random
import math

# --- 1. 修改您的路徑 ---

# 包含 8HZ, 6HZ, 4HZ 以及 background 資料夾的「來源」路徑
SOURCE_DIR = r"C:\Users\User\Downloads\RDIPHD\RDIPHD"

# 您想要建立 "train" 和 "val" 資料夾的「目標」路徑
OUTPUT_DIR = r"C:\Users\User\Downloads\RDIPHD\split_dataset"

# --- 2. 修改您的設定 ---

# (可選) 驗證集所佔的比例 (例如 0.2 代表 20%)
VAL_SPLIT_RATIO = 0.2

# [重要] 定義來源資料夾和目標資料夾的名稱對應
# 格式為 "來源資料夾名稱": "目標資料夾名稱"
# 我們在此加入了 "background"
CATEGORY_MAP = {
    "8HZ": "8Hz",
    "6HZ": "6Hz",
    "4HZ": "4Hz",
    "background": "background" 
}

# [重要] 您的最終類別
ALL_CATEGORIES = ['background', '4Hz', '6Hz', '8Hz']

# -------------------------------------------------------------------

def split_dataset():
    print(f"開始分割資料集...")
    print(f"來源資料夾: {SOURCE_DIR}")
    print(f"目標資料夾: {OUTPUT_DIR}\n")

    # 1. 建立目標資料夾 (train/ 和 val/)
    train_dir = os.path.join(OUTPUT_DIR, "train")
    val_dir = os.path.join(OUTPUT_DIR, "val")

    # 為所有類別建立子資料夾
    for category_name in ALL_CATEGORIES:
        os.makedirs(os.path.join(train_dir, category_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category_name), exist_ok=True)
    
    print("已建立目標資料夾結構 (train/ 和 val/)。")

    # 2. 遍歷 CATEGORY_MAP 中的所有項目 (包含 background)
    for source_folder, target_folder in CATEGORY_MAP.items():
        print(f"\n--- 正在處理: {source_folder} ---")
        
        current_source_path = os.path.join(SOURCE_DIR, source_folder)
        
        # 檢查來源資料夾是否存在
        if not os.path.exists(current_source_path):
            print(f"錯誤: 找不到來源資料夾: {current_source_path}，已跳過。")
            # 這裡特別提醒：如果您的資料夾叫 "Background" (大寫B)，請修改 CATEGORY_MAP
            continue

        # 找出所有 .h5 檔案
        try:
            all_files = [f for f in os.listdir(current_source_path) if f.endswith('.h5')]
        except Exception as e:
            print(f"錯誤: 無法讀取資料夾 {current_source_path}: {e}")
            continue
            
        if not all_files:
            print(f"警告: 資料夾 {source_folder} 中沒有找到 .h5 檔案。")
            continue
            
        # [關鍵] 隨機打亂檔案
        random.shuffle(all_files)

        # 3. 計算分割點並複製檔案
        total_files = len(all_files)
        val_count = math.ceil(total_files * VAL_SPLIT_RATIO) 
        train_count = total_files - val_count

        train_files = all_files[:train_count]
        val_files = all_files[train_count:]

        # 複製訓練集檔案
        target_train_folder = os.path.join(train_dir, target_folder)
        for f in train_files:
            shutil.copy2(os.path.join(current_source_path, f), target_train_folder)

        # 複製驗證集檔案
        target_val_folder = os.path.join(val_dir, target_folder)
        for f in val_files:
            shutil.copy2(os.path.join(current_source_path, f), target_val_folder)
            
        print(f"處理完成: {total_files} 個檔案 -> "
              f"{len(train_files)} 個 (train), {len(val_files)} 個 (val)")

    print("\n--- 全數處理完畢 ---")
    print(f"資料已分割並複製到: {OUTPUT_DIR}")

if __name__ == "__main__":
    split_dataset()