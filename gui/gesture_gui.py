import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar, QHBoxLayout, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPalette


class GestureGUI(QWidget):
    """
    PySide2 手勢辨識 GUI，顯示 4 種手勢的機率條狀圖，並突顯當前辨識結果。
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Recognition")
        self.resize(600, 400)

        # **主要 Layout**
        main_layout = QVBoxLayout()

        # **當前手勢標籤**
        self.current_gesture_label = QLabel("Current gesture: Background")
        self.current_gesture_label.setAlignment(Qt.AlignCenter)
        self.current_gesture_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; padding: 10px; background-color: lightgray; border-radius: 5px;"
        )
        main_layout.addWidget(self.current_gesture_label)

        # **進度條區域**
        self.hbox = QHBoxLayout()

        # **手勢名稱 & 條狀圖** (請確保順序與您的 training.py 一致！)
        self.gesture_names = ["Background", "4Hz", "6Hz", "8Hz"]
        self.bars = {}  # 存放進度條物件

        # **可調整的參數**
        self.BAR_WIDTH = 15  # 進度條寬度
        # 設定您喜歡的顏色
        self.bar_colors = {
            "Background": "green",
            "4Hz": "blue",
            "6Hz": "red",
            "8Hz": "purple"
        }
        # 設定結果標籤的背景色
        self.gesture_colors = {
            "Background": "lightgray",
            "4Hz": "#ADD8E6",  # 淺藍
            "6Hz": "#FFCCCB",  # 淺紅
            "8Hz": "#E6E6FA"   # 淺紫
        }

        # **Spacer 讓條狀圖置中**
        self.hbox.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        for name in self.gesture_names:
            # 建立一個垂直 Layout
            v_layout = QVBoxLayout()

            # 進度條
            bar = QProgressBar()
            bar.setOrientation(Qt.Vertical)
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)  # 不顯示文字
            bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {self.bar_colors[name]}; }}")
            bar.setFixedWidth(self.BAR_WIDTH)  # 設定進度條寬度
            v_layout.addWidget(bar, alignment=Qt.AlignBottom)

            # 手勢標籤
            label = QLabel(name)
            label.setAlignment(Qt.AlignCenter)
            v_layout.addWidget(label, alignment=Qt.AlignCenter)

            self.hbox.addLayout(v_layout)

            # 在每個進度條之間加入 SpacerItem，讓它們等距排列
            self.hbox.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

            self.bars[name] = bar  # 存入字典

        # **Spacer 讓條狀圖置中**
        self.hbox.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        main_layout.addLayout(self.hbox)
        self.setLayout(main_layout)

    # 修改這裡：參數名稱改清楚，並使用正確的 Key (4Hz, 6Hz, 8Hz)
    def update_probabilities(self, prob_bg, prob_4hz, prob_6hz, prob_8hz, current_gesture):
        """
        更新 4 個類別的機率進度條與辨識結果。
        """
        # 轉換為百分比並設定給對應的 Bar
        # [重要] 這裡的 Key 必須跟 self.gesture_names 裡的一模一樣
        if "Background" in self.bars: self.bars["Background"].setValue(int(prob_bg * 100))
        if "4Hz" in self.bars:        self.bars["4Hz"].setValue(int(prob_4hz * 100))
        if "6Hz" in self.bars:        self.bars["6Hz"].setValue(int(prob_6hz * 100))
        if "8Hz" in self.bars:        self.bars["8Hz"].setValue(int(prob_8hz * 100))

        # 更新中央標籤
        self.current_gesture_label.setText(f"Current gesture: {current_gesture}")
        
        # 防呆機制：如果 current_gesture 傳進來的是舊的名稱，給一個默認顏色
        bg_color = self.gesture_colors.get(current_gesture, "lightgray")
        
        self.current_gesture_label.setStyleSheet(
            f"font-size: 20px; font-weight: bold; padding: 10px; background-color: {bg_color}; border-radius: 5px;"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GestureGUI()
    window.show()

    # 測試數據：每秒更新一次
    import random

    def simulate_data():
        # 產生隨機機率，總和為 1
        bg = random.uniform(0, 1)
        prob_4hz = random.uniform(0, 1 - bg)
        prob_6hz = random.uniform(0, 1 - bg - prob_4hz)
        prob_8hz = 1 - (bg + prob_4hz + prob_6hz)

        # 根據機率決定當前手勢名稱 (需與 GUI 定義的名稱一致)
        if prob_4hz > 0.5:
            gesture = "4Hz"
        elif prob_6hz > 0.5:
            gesture = "6Hz"
        elif prob_8hz > 0.5:
            gesture = "8Hz"
        else:
            gesture = "Background"

        # 呼叫更新函式
        window.update_probabilities(bg, prob_4hz, prob_6hz, prob_8hz, gesture)

    timer = QTimer()
    timer.timeout.connect(simulate_data)
    timer.start(1000)  # 每 1000 毫秒更新一次

    sys.exit(app.exec())
