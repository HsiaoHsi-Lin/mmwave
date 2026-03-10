# online_inference_gui_copy.py  ─  30 秒統計版
# 改動說明：
#   1. InferenceUpdater.__init__ 接收 gui 的 start_session 不需要主動呼叫，
#      GUI 按鈕自己觸發 start_session()。
#   2. InferenceUpdater.update() 完全不變 ─ 只是把 probs 傳給 gui.update_probabilities()。
#   3. 其餘硬體初始化邏輯完全不動。

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PySide6 import QtWidgets

# ======== 路徑/參數（可直接改） ========
MODEL_PATH   = r"D:\mmwave\mm實作\best_model.pth"
SETTING_FILE = r"D:\mmwave\mmWave\radar-gesture-recognition-chore-update-20250815\TempParam\K60168-Test-00256-008-v0.0.8-20230717_60cm"

WINDOW_SIZE  = 30
CLASS_NAMES  = ["Background", "4Hz", "6Hz", "8Hz"]
ENTER_TH     = 0.40
EXIT_TH      = 0.20
STREAM_TYPE  = "feature_map"
# ======================================

from gesture_gui_pyside_copy import GestureGUI   # 新版 GUI（同資料夾）

from KKT_Module import kgl
from KKT_Module.DataReceive.Core import Results
from KKT_Module.DataReceive.DataReceiver import MultiResult4168BReceiver
from KKT_Module.FiniteReceiverMachine import FRM
from KKT_Module.SettingProcess.SettingConfig import SettingConfigs
from KKT_Module.SettingProcess.SettingProccess import SettingProc
from KKT_Module.GuiUpdater.GuiUpdater import Updater


# -------------------- Kaiku helpers --------------------
def connect_device():
    try:
        device = kgl.ksoclib.connectDevice()
        if device == 'Unknow':
            ret = QtWidgets.QMessageBox.warning(
                None, 'Unknown Device', 'Please reconnect device and try again',
                QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            if ret == QtWidgets.QMessageBox.Ok:
                connect_device()
    except Exception:
        ret = QtWidgets.QMessageBox.warning(
            None, 'Connection Failed', 'Please reconnect device and try again',
            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        if ret == QtWidgets.QMessageBox.Ok:
            connect_device()


def run_setting_script(setting_name: str):
    ksp = SettingProc()
    cfg = SettingConfigs()
    cfg.Chip_ID = kgl.ksoclib.getChipID().split(' ')[0]
    cfg.Processes = [
        'Reset Device', 'Gen Process Script', 'Gen Param Dict', 'Get Gesture Dict',
        'Set Script', 'Run SIC', 'Phase Calibration', 'Modulation On'
    ]
    cfg.setScriptDir(f'{setting_name}')
    ksp.startUp(cfg)


def set_properties(obj: object, **kwargs):
    print(f"==== Set properties in {obj.__class__.__name__} ====")
    for k, v in kwargs.items():
        if not hasattr(obj, k):
            print(f'Attribute "{k}" not in {obj.__class__.__name__}.')
            continue
        setattr(obj, k, v)
        print(f'Attribute "{k}", set "{v}"')


# -------------------- 3D CNN --------------------
class Gesture3DCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(2, 32, 3), nn.ReLU(), nn.MaxPool3d(2), nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, 3), nn.ReLU(), nn.MaxPool3d(2), nn.BatchNorm3d(64),
            nn.Conv3d(64, 128, 3), nn.ReLU(), nn.MaxPool3d(2), nn.BatchNorm3d(128),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def _maybe_remap_keys_to_classifier(state: dict) -> dict:
    if any(k.startswith("fc.") for k in state.keys()):
        new = {}
        for k, v in state.items():
            new["classifier." + k[3:]] = v if k.startswith("fc.") else v
        return new
    return state


# -------------------- 即時推論核心 --------------------
class OnlineInferenceContext:
    def __init__(self, model: nn.Module, device: torch.device, window_size: int):
        self.model = model
        self.device = device
        self.window = window_size
        self.buffer = np.zeros((2, 32, 32, self.window), dtype=np.float32)
        self.collected = 0
        self.active = False
        self.last_pred = "Background"

    @staticmethod
    def to_frame(arr) -> np.ndarray:
        x = np.asarray(arr)
        if x.shape == (2, 32, 32):
            pass
        elif x.shape == (32, 32, 2):
            x = np.transpose(x, (2, 0, 1))
        else:
            raise ValueError(f"Unexpected frame shape: {x.shape}")
        return x.astype(np.float32, copy=True)

    def push_and_infer(self, frame: np.ndarray):
        self.buffer = np.roll(self.buffer, shift=-1, axis=-1)
        self.buffer[..., -1] = frame
        self.collected += 1
        if self.collected < self.window:
            return None
        win = np.expand_dims(self.buffer, axis=0)
        win = np.transpose(win, (0, 1, 4, 2, 3))
        x = torch.from_numpy(win).float().to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            p = F.softmax(logits, dim=1).cpu().numpy()[0]
        return p

    def apply_double_threshold(self, probs: np.ndarray):
        bg, p4, p6, p8 = probs
        nonbg = np.array([p4, p6, p8])
        nonbg_names = CLASS_NAMES[1:]
        top_idx = int(nonbg.argmax())
        top_name = nonbg_names[top_idx]
        top_prob = float(nonbg[top_idx])

        if not self.active:
            current = top_name if top_prob >= ENTER_TH else "Background"
            if top_prob >= ENTER_TH:
                self.active = True
        else:
            if (nonbg < EXIT_TH).all():
                self.active = False
                current = "Background"
            else:
                current = self.last_pred

        changed = (current != self.last_pred)
        self.last_pred = current
        return current, changed, (bg, p4, p6, p8)


# -------------------- InferenceUpdater --------------------
class InferenceUpdater(Updater):
    """
    硬體每幀呼叫 update()，只負責推論並把結果傳給 GUI。
    統計邏輯完全在 GestureGUI 內部（由按鈕觸發 start_session）。
    """
    def __init__(self, ctx: OnlineInferenceContext, gesture_gui: GestureGUI,
                 stream: str = "feature_map"):
        super().__init__()
        self.ctx = ctx
        self.gui = gesture_gui
        self.stream = stream

    def update(self, res: Results):
        try:
            arr = res['raw_data'].data if self.stream == "raw_data" else res['feature_map'].data
            frame = self.ctx.to_frame(arr)
            probs = self.ctx.push_and_infer(frame)
            if probs is None:
                return

            current, changed, (bg, p4, p6, p8) = self.ctx.apply_double_threshold(probs)

            # ── 唯一對 GUI 的呼叫，與舊版完全相同 ──
            self.gui.update_probabilities(float(bg), float(p4), float(p6), float(p8), current)

            if changed:
                print(f"[Pred] {current} | BG:{bg:.2f} 4Hz:{p4:.2f} 6Hz:{p6:.2f} 8Hz:{p8:.2f}")

        except Exception as e:
            print(f"[Updater Error] {e}")


# -------------------- 背景初始化執行緒 --------------------
from PySide6.QtCore import QThread, Signal as QtSignal

class HardwareInitThread(QThread):
    """
    把所有耗時的硬體初始化移到背景執行緒，
    讓 GUI 視窗可以立即顯示，不會卡住。
    """
    ready   = QtSignal()          # 初始化成功
    failed  = QtSignal(str)       # 初始化失敗，帶錯誤訊息

    def __init__(self, gui: GestureGUI):
        super().__init__()
        self.gui = gui

    def run(self):
        try:
            kgl.setLib()
            connect_device()
            run_setting_script(SETTING_FILE)

            if STREAM_TYPE == "raw_data":
                kgl.ksoclib.writeReg(0, 0x50000504, 5, 5, 0)
            else:
                kgl.ksoclib.writeReg(1, 0x50000504, 5, 5, 0)

            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Gesture3DCNN(num_classes=4).to(torch_device)
            state = torch.load(MODEL_PATH, map_location=torch_device, weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            state = _maybe_remap_keys_to_classifier(state)
            model.load_state_dict(state, strict=False)
            model.eval()
            print(f"[INFO] model loaded: {MODEL_PATH} | device: {torch_device}")

            ctx = OnlineInferenceContext(model, torch_device, WINDOW_SIZE)
            updater = InferenceUpdater(ctx, gesture_gui=self.gui, stream=STREAM_TYPE)

            receiver = MultiResult4168BReceiver()
            set_properties(receiver, actions=1, rbank_ch_enable=7,
                           read_interrupt=0, clear_interrupt=0)
            FRM.setReceiver(receiver)
            FRM.setUpdater(updater)
            FRM.trigger()
            FRM.start()

            print("[INFO] Online inference started.")
            self.ready.emit()

        except Exception as e:
            self.failed.emit(str(e))


# -------------------- 主流程 --------------------
def main():
    app = QtWidgets.QApplication(sys.argv)

    gui = GestureGUI()
    gui.show()

    # 初始化執行緒
    init_thread = HardwareInitThread(gui)

    def on_ready():
        print("[INFO] Hardware ready.")
        gui.set_hardware_ready()

    def on_failed(msg):
        QtWidgets.QMessageBox.critical(
            gui, "初始化失敗",
            f"硬體初始化發生錯誤：\n\n{msg}\n\n請確認裝置連線後重新啟動程式。"
        )

    init_thread.ready.connect(on_ready)
    init_thread.failed.connect(on_failed)
    init_thread.start()   # GUI 立即顯示，硬體在背景初始化

    print("[INFO] GUI started. Hardware initializing in background...")
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        pass
    finally:
        try:
            FRM.stop()
        except:
            pass
        try:
            kgl.ksoclib.closeDevice()
        except:
            pass
        print("[INFO] Stopped.")


if __name__ == "__main__":
    main()
