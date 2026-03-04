import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm  # 進度條

# 檢查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# 1. 超參數設定 (Hyperparameters)
# -------------------------------

# [!!] 請確保這裡的檔名與您生成的 .npz 檔名完全一致
TRAINING_DATA_DIR = "my_4-6-8Hz_training.npz"
VAL_DATA_DIR = "my_4-6-8Hz_validation.npz"

BATCH_SIZE = 32       # 批次大小
EPOCHS = 50           # 訓練輪數
LEARNING_RATE = 1e-4  # 學習率
NUM_CLASSES = 4       # background, 4Hz, 6Hz, 8Hz

# 建立儲存模型的資料夾
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
MODEL_SAVE_PATH = os.path.join("output", "models", timestamp)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

gesture_types = ['background', '4Hz', '6Hz', '8Hz']

# -------------------------------
# 2. 資料載入函數 (已簡化)
# -------------------------------
def load_data(data_path):
    """
    直接讀取預處理好的 .npz 檔案中的 'X' (特徵) 和 'y' (標籤)
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到檔案: {data_path}，請確認路徑是否正確。")

    print(f"正在讀取資料: {data_path} ...")
    data = np.load(data_path)
    
    # 讀取 preprocess_to_npz.py 儲存的鍵值 'X' 和 'y'
    X = data['X'] 
    y = data['y']
    
    print(f"  -> X shape: {X.shape}")  # 預期: (N, 2, 30, 32, 32)
    print(f"  -> y shape: {y.shape}")  # 預期: (N, 4)
    return X, y

# -------------------------------
# 3. 3D CNN 模型定義 (保持不變)
# -------------------------------
class Gesture3DCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(Gesture3DCNN, self).__init__()
        self.features = nn.Sequential(
            # 輸入形狀: (Batch, 2, 30, 32, 32)
            nn.Conv3d(in_channels=2, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.BatchNorm3d(32),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.BatchNorm3d(64),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)), # Output: (Batch, 128, 3, 4, 4) approx
            nn.BatchNorm3d(128),
        )
        # 使用 AdaptiveAvgPool3d 強制將特徵圖縮小為 (1, 1, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        return x # 回傳 Logits，因為後面使用 CrossEntropyLoss (內含 Softmax)

# -------------------------------
# 4. 模型訓練函數
# -------------------------------
def train_model(X_train, y_train, X_val, y_val):
    # 轉換為 PyTorch Tensor
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float() 
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).float()
    
    # 建立 Dataset 與 DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 初始化模型、優化器、損失函數
    model = Gesture3DCNN(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss() 
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float('inf')
    
    # [新增] Early Stopping 設定
    patience = 10          # 容忍幾輪沒進步
    counter = 0            # 目前累積幾輪沒進步
    early_stop = False     # 是否觸發停止
    
    print(f"\n開始訓練... 共 {EPOCHS} 個 Epochs")
    print(f"模型與結果將儲存於: {MODEL_SAVE_PATH}")

    for epoch in range(EPOCHS):
        # --- 訓練階段 ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # CrossEntropyLoss 需要類別索引 (0, 1, 2, 3)，而非 one-hot
            # 所以我們用 torch.argmax 轉回索引
            loss = criterion(outputs, torch.argmax(labels, dim=1))
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # 計算準確率
            _, predicted = torch.max(outputs, 1)
            _, targets = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == targets).sum().item()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # --- 驗證階段 ---
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, torch.argmax(labels, dim=1))
                
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                _, targets = torch.max(labels, 1)
                val_total += labels.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_epoch_loss = val_running_loss / len(val_dataset)
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)
        
        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
        
        # [修改] 儲存最佳模型與 Early Stopping 邏輯
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "best_model.pth"))
            print(f"  -> Model Saved! (Best Val Loss: {best_val_loss:.4f})")
            counter = 0  # 有進步，歸零計數器
        else:
            counter += 1 # 沒進步，計數器 +1
            print(f"  -> No improvement. Counter: {counter}/{patience}")
            if counter >= patience:
                print("  -> Early Stopping triggered! Stop training.")
                early_stop = True

        # 如果觸發 Early Stop，就跳出迴圈
        if early_stop:
            break
        
        # 儲存最佳模型
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "best_model.pth"))
            
    # 儲存最終模型
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "final_model.pth"))
    print("\n訓練完成！")
    
    history = {
        'train_loss': train_losses, 'val_loss': val_losses,
        'train_acc': train_accuracies, 'val_acc': val_accuracies
    }
    return model, history

# -------------------------------
# 5. 繪製結果函數 (修改版：支援存檔)
# -------------------------------
def plot_history(history, save_path=None):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.tight_layout()
    
    # 存檔
    if save_path:
        plt.savefig(save_path)
        print(f"訓練歷程圖已儲存至: {save_path}")
        
    plt.show()

# -------------------------------
# 6. [新增] 繪製混淆矩陣函數
# -------------------------------
def plot_confusion_matrix(model, X_val, y_val, save_path=None):
    print("\n正在計算混淆矩陣...")
    
    # 準備資料
    X_tensor = torch.from_numpy(X_val).float().to(device)
    y_tensor = torch.from_numpy(y_val).float().to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            _, targets = torch.max(labels, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    # 計算混淆矩陣 (手動計算以避免依賴 sklearn)
    num_classes = len(gesture_types)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1
        
    # 繪圖
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, gesture_types, rotation=45)
    plt.yticks(tick_marks, gesture_types)
    
    # 在格子裡填數字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
                     
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # 存檔
    if save_path:
        plt.savefig(save_path)
        print(f"混淆矩陣已儲存至: {save_path}")
        
    plt.show()

# -------------------------------
# 主程式進入點
# -------------------------------
if __name__ == "__main__":
    # 1. 載入資料
    X_train, y_train = load_data(TRAINING_DATA_DIR)
    X_val, y_val = load_data(VAL_DATA_DIR)
    
    # 2. 開始訓練
    model, history = train_model(X_train, y_train, X_val, y_val)
    
    # 3. 畫訓練圖並存檔
    history_plot_path = os.path.join(MODEL_SAVE_PATH, "training_history.png")
    plot_history(history, save_path=history_plot_path)
    
    # 4. 畫混淆矩陣並存檔
    cm_plot_path = os.path.join(MODEL_SAVE_PATH, "confusion_matrix.png")
    plot_confusion_matrix(model, X_val, y_val, save_path=cm_plot_path)