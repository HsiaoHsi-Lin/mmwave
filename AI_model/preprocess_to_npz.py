import os
import h5py
import numpy as np
import random
from tqdm import tqdm  # 用於顯示進度條

# ----------------------------------------------------
# preprocess_to_npz.py
#
# 目的：
# 1. 讀取 "train" 和 "val" 資料夾。
# 2. 遍歷 ['background', '4Hz', '6Hz', '8Hz'] 子資料夾。
# 3. 讀取所有 .h5 檔案 (RDI/PHD map)。
# 4. 套用滑動視窗 (sliding window)。
# 5. 將資料和標籤 (label) 處理好。
# 6. 打亂 (shuffle) 資料。
# 7. 儲存為 "train.npz" 和 "val.npz" 兩個檔案。
# ----------------------------------------------------

# --- 1. 修改您的設定 ---

# [!!] 請修改這裡 [!!]
# 指向您 "train" 資料夾 (裡面應包含 4Hz, 6Hz 等子資料夾)
TRAIN_FOLDER = r"C:\Users\User\Downloads\RDIPHD\split_dataset\train" 

# [!!] 請修改這裡 [!!]
# 指向您 "val" 資料夾 (結構同上)
VAL_FOLDER = r"C:\Users\User\Downloads\RDIPHD\split_dataset\val"

# [!!] 請修改這裡 [!!]
# 儲存處理好的 "訓練集" .npz 檔案的名稱
OUTPUT_TRAIN_NPZ = "my_4-6-8Hz_training.npz"

# [!!] 請修改這裡 [!!]
# 儲存處理好的 "驗證集" .npz 檔案的名稱
OUTPUT_VAL_NPZ = "my_4-6-8Hz_validation.npz"

# --- 2. 處理超參數 (應與 training.py 一致) ---

# 滑動視窗的大小 (幀數)
WINDOW_SIZE = 30
# 滑動視窗的步長
STEP_SIZE = 1
# 總類別數
NUM_CLASSES = 4
# 類別名稱 (必須與您的資料夾名稱 "完全一致")
GESTURE_TYPES = ['background', '4Hz', '6Hz', '8Hz']
# .h5 檔案中，儲存 RDI/PHD 數據的鍵 (Key) (根據簡報 slide 53, 58)
H5_DATASET_KEY = 'DS1' 

# ----------------------------------------------------

def to_one_hot(label_index, num_classes):
    """ 輔助函數：將數字標籤 0, 1, 2... 轉換為 one-hot 編碼 """
    one_hot = np.zeros(num_classes, dtype=np.float32)
    one_hot[label_index] = 1.0
    return one_hot

def load_and_process_data(data_folder_path):
    """
    從資料夾結構讀取所有 .h5 檔案，處理並回傳 X 和 y 陣列。
    """
    print(f"\n[Info] 開始處理資料夾: {data_folder_path}")
    
    X_list, y_list = [], []
    
    # 1. 遍歷 GESTURE_TYPES 中定義的每個子資料夾
    for label_index, gesture_name in enumerate(GESTURE_TYPES):
        subfolder_path = os.path.join(data_folder_path, gesture_name)
        
        if not os.path.exists(subfolder_path):
            print(f"  [Warning] 資料夾不存在, 已跳過: {subfolder_path}")
            continue
            
        print(f"  --- 正在處理: {gesture_name} (標籤: {label_index}) ---")
        
        # 獲取資料夾中所有 .h5 檔案
        try:
            h5_files = [f for f in os.listdir(subfolder_path) if f.endswith('.h5')]
            if not h5_files:
                print(f"  [Warning] 在 {gesture_name} 中未找到 .h5 檔案。")
                continue
        except Exception as e:
            print(f"  [Error] 無法讀取資料夾 {subfolder_path}: {e}")
            continue

        # 2. 遍歷該資料夾中的所有 .h5 檔案 (使用 tqdm 顯示進度條)
        for h5_filename in tqdm(h5_files, desc=f"    -> {gesture_name}", unit="file"):
            file_path = os.path.join(subfolder_path, h5_filename)
            
            try:
                with h5py.File(file_path, 'r') as f:
                    if H5_DATASET_KEY not in f:
                        print(f"  [Warning] {h5_filename} 中找不到 '{H5_DATASET_KEY}' 資料, 已跳過。")
                        continue
                        
                    # 讀取 RDI/PHD 資料，格式應為 (2, 32, 32, N)
                    features = f[H5_DATASET_KEY][:] 
                    num_frames = features.shape[-1]

                    # 3. 準備此檔案的標籤 (one-hot 格式)
                    label_soft = to_one_hot(label_index, NUM_CLASSES) # 例如: [0, 1, 0, 0]

                    # 4. 套用滑動視窗
                    for start in range(0, num_frames - WINDOW_SIZE + 1, STEP_SIZE):
                        end = start + WINDOW_SIZE
                        
                        # 取得視窗資料 (2, 32, 32, WINDOW_SIZE)
                        window_feature = features[..., start:end]  
                        
                        # 轉換維度以符合模型輸入 (2, WINDOW_SIZE, 32, 32)
                        window_feature = np.transpose(window_feature, (0, 3, 1, 2))
                        
                        X_list.append(window_feature)
                        y_list.append(label_soft)
                        
            except Exception as e:
                print(f"  [Error] 處理檔案 {file_path} 失敗: {e}")

    if not X_list:
        print(f"[Error] 資料夾 {data_folder_path} 中未讀取到任何有效資料。")
        return None, None

    # 5. 將 list 轉換為 NumPy 陣列
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    # 6. [非常重要!] 打亂資料
    # 因為我們是按順序 (background, 4Hz...) 讀取的，必須打亂
    print("\n  [Info] 正在打亂 (Shuffling) 資料...")
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    X = X[indices]
    y = y[indices]
    
    print("  [Info] 資料處理與打亂完成。")
    print(f"  [Result] X (特徵) shape: {X.shape}")
    print(f"  [Result] y (標籤) shape: {y.shape}")
    return X, y

# -----------------
# 主程式 (Main)
# -----------------
if __name__ == "__main__":
    
    # --- 處理訓練資料 ---
    X_train, y_train = load_and_process_data(TRAIN_FOLDER)
    
    if X_train is not None and y_train is not None:
        print(f"\n[Success] 正在儲存訓練資料至: {OUTPUT_TRAIN_NPZ} (這可能需要一點時間)...")
        # 使用 savez_compressed 來壓縮並儲存
        np.savez_compressed(OUTPUT_TRAIN_NPZ, X=X_train, y=y_train)
        print(f"[Success] 訓練資料 {OUTPUT_TRAIN_NPZ} 儲存完畢。")
    else:
        print("[Error] 未能處理訓練資料，已跳過儲存。")

    # --- 處理驗證資料 ---
    X_val, y_val = load_and_process_data(VAL_FOLDER)
    
    if X_val is not None and y_val is not None:
        print(f"\n[Success] 正在儲存驗證資料至: {OUTPUT_VAL_NPZ} ...")
        np.savez_compressed(OUTPUT_VAL_NPZ, X=X_val, y=y_val)
        print(f"[Success] 驗證資料 {OUTPUT_VAL_NPZ} 儲存完畢。")
    else:
        print("[Error] 未能處理驗證資料，已跳過儲存。")

    print("\n--- 所有處理已完成 ---")