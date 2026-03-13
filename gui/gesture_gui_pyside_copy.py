"""
gesture_gui_pyside_copy.py  ─  手部震顫檢測介面（30 秒統計版）
=================================================================
公開 API（與舊版完全相容）：
    update_probabilities(prob_bg, prob_4hz, prob_6hz, prob_8hz, current_gesture)

新增：
    start_session()   ─ 由外部（InferenceUpdater）呼叫，啟動一次 30 秒統計
    show_final_result(hz_label)  ─ 統計結束後顯示最終結果（也可由外部呼叫）

使用者操作：
    點按「開始檢測」按鈕 → 30 秒倒數 → 自動顯示最終結果
    硬體 / FRM 持續在背景串流，按鈕只控制「統計視窗」
"""

import sys
import random
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QFrame, QSizePolicy, QSpacerItem
)
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, Property, QObject
from PySide6.QtGui import QColor, QPainter, QPen, QFont, QFontDatabase, QPalette

DETECTION_DURATION = 30   # 秒
RESULT_COLORS = {
    "4Hz":    "#38BDF8",   # 天藍
    "6Hz":    "#34D399",   # 翠綠
    "8Hz":    "#F472B6",   # 玫瑰
    "Normal": "#22C55E",   # 綠（正常）
    "None":   "#64748B",   # 灰（無法判斷）
}

# ── 臨床參考建議（純資訊，非診斷）──────────────────────────────────────
CLINICAL_ADVICE = {
    "4Hz": {
        "tag":  "低頻靜止性震顫（3–5 Hz）",
        "note": "常見於帕金森氏症靜止性震顫。\n建議盡早至神經內科就診評估。",
    },
    "6Hz": {
        "tag":  "中頻姿勢性震顫（5–7 Hz）",
        "note": "屬重疊判讀區，需結合姿勢或任務情境判斷。\n建議至神經內科或家醫科就診。",
    },
    "8Hz": {
        "tag":  "高頻生理性震顫（7–9 Hz）",
        "note": "較常見於高頻姿勢性或生理性震顫。\n建議先調整生活型態，若症狀持續請至家醫科就診。",
    },
    "Normal": {
        "tag":  "未偵測到明顯震顫",
        "note": "未偵測到明顯震顫，手部活動在正常範圍內。\n若仍有不適感，建議至家醫科說明症狀。",
    },
    "None": {
        "tag":  "訊號不足，無法判定",
        "note": "未收集到足夠訊號，請確認手部位置後重新檢測。",
    },
}

# ─────────────────────────── 圓形倒數元件 ──────────────────────────────
class CountdownRing(QWidget):
    """畫一個圓弧倒數環，progress 0.0→1.0"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(140, 140)
        self._progress = 1.0          # 1.0 = 滿；0.0 = 空
        self._seconds_left = 0
        self._active = False

    def set_state(self, progress: float, seconds_left: int, active: bool):
        self._progress = max(0.0, min(1.0, progress))
        self._seconds_left = seconds_left
        self._active = active
        self.update()

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter
        painter = QPainter()
        if not painter.begin(self):
            return
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        margin = 12
        rect_size = min(w, h) - margin * 2
        x = (w - rect_size) / 2
        y = (h - rect_size) / 2

        # 背景圓環
        pen = QPen(QColor("#1E293B"), 10)
        painter.setPen(pen)
        painter.drawEllipse(int(x), int(y), rect_size, rect_size)

        # 進度弧（從 12 點鐘順時針）
        arc_color = QColor("#38BDF8") if self._active else QColor("#334155")
        pen2 = QPen(arc_color, 10)
        pen2.setCapStyle(Qt.RoundCap)
        painter.setPen(pen2)
        span = int(self._progress * 360 * 16)
        painter.drawArc(int(x), int(y), rect_size, rect_size, 90 * 16, -span)

        # 中央文字
        painter.setPen(QColor("#F1F5F9"))
        if self._active:
            painter.setFont(QFont("Courier New", 28, QFont.Bold))
            painter.drawText(self.rect(), Qt.AlignCenter, str(self._seconds_left))
        else:
            painter.setFont(QFont("Courier New", 13))
            painter.drawText(self.rect(), Qt.AlignCenter, "READY")

        painter.end()


# ─────────────────────────── 機率小條 ──────────────────────────────────
class MiniBar(QWidget):
    """單一類別的垂直機率條 + 標籤"""
    BAR_COLORS = {
        "0 Hz": "#475569",
        "4Hz":  "#38BDF8",
        "6Hz":  "#34D399",
        "8Hz":  "#F472B6",
    }

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = name
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.bar = QProgressBar()
        self.bar.setOrientation(Qt.Vertical)
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedWidth(20)
        self.bar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        color = self.BAR_COLORS.get(name, "#64748B")
        self.bar.setStyleSheet(f"""
            QProgressBar {{
                background: #1E293B;
                border-radius: 4px;
                border: 1px solid #334155;
            }}
            QProgressBar::chunk {{
                background: {color};
                border-radius: 3px;
            }}
        """)

        lbl = QLabel(name)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color:#CBD5E1; font-size:13px; font-weight:bold; font-family: 'Courier New';")

        layout.addWidget(self.bar, alignment=Qt.AlignHCenter)
        layout.addWidget(lbl)

    def set_value(self, v: float):
        self.bar.setValue(int(v * 100))


# ─────────────────────────── 主 GUI ────────────────────────────────────
class GestureGUI(QWidget):
    """
    主視窗。

    公開方法（向後相容）：
        update_probabilities(prob_bg, prob_4hz, prob_6hz, prob_8hz, current_gesture)
        start_session()
        show_final_result(hz_label)
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("手部震顫檢測系統")
        self._hw_ready = False
        self._session_active = False
        self._elapsed = 0
        self._scores = {"Background": 0.0, "4Hz": 0.0, "6Hz": 0.0, "8Hz": 0.0}

        self._build_ui()
        self._build_timer()

        # 自動適應螢幕高度（保留工作列空間）
        from PySide6.QtWidgets import QApplication
        screen = QApplication.primaryScreen().availableGeometry()
        win_w = min(560, screen.width())
        win_h = min(screen.height() - 40, screen.height())
        self.resize(win_w, win_h)
        # 置中顯示
        self.move(
            screen.x() + (screen.width() - win_w) // 2,
            screen.y() + (screen.height() - win_h) // 2,
        )

    # ── UI 建構 ───────────────────────────────────────────────────────
    def _build_ui(self):
        from PySide6.QtWidgets import QScrollArea
        self.setStyleSheet("""
            QWidget {
                background-color: #0F172A;
                color: #F1F5F9;
                font-family: 'Courier New', monospace;
            }
            QScrollArea { border: none; }
            QScrollBar:vertical {
                background: #0F172A; width: 6px;
            }
            QScrollBar::handle:vertical {
                background: #334155; border-radius: 3px;
            }
        """)

        # 外層：視窗主 layout 放 ScrollArea
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 內層容器
        container = QWidget()
        root = QVBoxLayout(container)
        root.setContentsMargins(28, 28, 28, 28)
        root.setSpacing(14)

        scroll.setWidget(container)
        outer.addWidget(scroll)

        # 標題
        title = QLabel("手部震顫檢測系統")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            letter-spacing: 3px;
            color: #E2E8F0;
            padding-bottom: 4px;
        """)
        root.addWidget(title)

        # 分隔線
        root.addWidget(self._divider())

        # 倒數環 + 即時手勢標籤
        ring_row = QHBoxLayout()
        ring_row.addStretch()

        ring_col = QVBoxLayout()
        self.ring = CountdownRing()
        ring_col.addWidget(self.ring, alignment=Qt.AlignCenter)

        self.live_label = QLabel("")
        self.live_label.setAlignment(Qt.AlignCenter)
        self.live_label.setStyleSheet("color:#94A3B8; font-size:18px; letter-spacing:2px; font-weight:bold;")
        self.live_label.setVisible(False)
        ring_col.addWidget(self.live_label)

        ring_row.addLayout(ring_col)
        ring_row.addStretch()
        root.addLayout(ring_row)

        # 機率條（偵測狀態）
        root.addWidget(self._section_label("偵測狀態"))
        bars_row = QHBoxLayout()
        bars_row.setSpacing(0)
        self._mini_bars: dict[str, MiniBar] = {}
        for name in ["0 Hz", "4Hz", "6Hz", "8Hz"]:
            bars_row.addStretch()
            mb = MiniBar(name)
            mb.setFixedHeight(110)
            self._mini_bars[name] = mb
            bars_row.addWidget(mb)
        bars_row.addStretch()
        root.addLayout(bars_row)

        root.addWidget(self._divider())

        # 啟動按鈕
        self.start_btn = QPushButton("▶  開始檢測")
        self.start_btn.setFixedHeight(54)
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: #1D4ED8;
                color: #FFFFFF;
                font-size: 18px;
                font-weight: bold;
                letter-spacing: 2px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover  { background: #2563EB; }
            QPushButton:pressed{ background: #1E40AF; }
            QPushButton:disabled{
                background: #1E293B;
                color: #64748B;
            }
        """)
        self.start_btn.clicked.connect(self.start_session)
        self.start_btn.setEnabled(False)
        self.start_btn.setText("⏳  裝置初始化中…")
        root.addWidget(self.start_btn)

        # 結果框
        result_frame = QFrame()
        result_frame.setStyleSheet("""
            QFrame {
                background: #0F172A;
                border: 1px solid #1E293B;
                border-radius: 10px;
            }
        """)
        result_inner = QVBoxLayout(result_frame)
        result_inner.setContentsMargins(16, 16, 16, 16)

        result_title = QLabel("最終結果")
        result_title.setAlignment(Qt.AlignCenter)
        result_title.setStyleSheet("color:#94A3B8; font-size:14px; letter-spacing:3px;")
        result_inner.addWidget(result_title)

        self.result_label = QLabel("──")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("""
            font-size: 56px;
            font-weight: bold;
            color: #475569;
            letter-spacing: 2px;
        """)
        result_inner.addWidget(self.result_label)

        self.result_sub = QLabel("")
        self.result_sub.setAlignment(Qt.AlignCenter)
        self.result_sub.setStyleSheet("color:#94A3B8; font-size:13px;")
        result_inner.addWidget(self.result_sub)

        # 分隔線
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #1E293B; margin: 4px 0;")
        result_inner.addWidget(sep)

        # 頻率標籤
        self.advice_tag = QLabel("")
        self.advice_tag.setAlignment(Qt.AlignCenter)
        self.advice_tag.setWordWrap(True)
        self.advice_tag.setStyleSheet(
            "color:#CBD5E1; font-size:14px; font-weight:bold; letter-spacing:1px;"
        )
        result_inner.addWidget(self.advice_tag)

        # 臨床建議文字
        self.advice_label = QLabel("")
        self.advice_label.setAlignment(Qt.AlignCenter)
        self.advice_label.setWordWrap(True)
        self.advice_label.setStyleSheet(
            "color:#CBD5E1; font-size:13px; line-height:180%; padding: 6px 4px;"
        )
        result_inner.addWidget(self.advice_label)

        root.addWidget(result_frame)

    def _divider(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #1E293B;")
        return line

    def _section_label(self, text: str):
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#94A3B8; font-size:13px; letter-spacing:2px;")
        return lbl

    # ── 計時器 ────────────────────────────────────────────────────────
    def _build_timer(self):
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)

    def _tick(self):
        self._elapsed += 1
        left = DETECTION_DURATION - self._elapsed
        progress = left / DETECTION_DURATION
        self.ring.set_state(progress, left, True)

        if left <= 0:
            self._timer.stop()
            self._session_active = False
            if self._valid_frames == 0:
                final = "None"
            else:
                winner = max(self._scores, key=self._scores.get)
                final = "Normal" if winner == "Background" else winner
            self.show_final_result(final)
            self._restore_button()

    def set_hardware_ready(self):
        """硬體初始化完成後由主程式呼叫，解鎖開始按鈕"""
        self._hw_ready = True
        self.start_btn.setEnabled(True)
        self.start_btn.setText("▶  開始檢測")

    # ── 公開 API ──────────────────────────────────────────────────────
    def start_session(self):
        """啟動 30 秒統計視窗（硬體持續串流，只重設統計）"""
        if self._session_active:
            return
        self._session_active = True
        self._elapsed = 0
        self._scores = {"Background": 0.0, "4Hz": 0.0, "6Hz": 0.0, "8Hz": 0.0}
        self._valid_frames = 0

        # 重設結果顯示
        self.result_label.setText("──")
        self.result_label.setStyleSheet("font-size:48px; font-weight:bold; color:#334155;")
        self.result_sub.setText("偵測中…")
        self.advice_tag.setText("")
        self.advice_label.setText("")

        # 按鈕禁用
        self.start_btn.setEnabled(False)
        self.start_btn.setText("⏱  偵測中…")

        # 環重設
        self.ring.set_state(1.0, DETECTION_DURATION, True)
        self._timer.start()

    def update_probabilities(self, prob_bg, prob_4hz, prob_6hz, prob_8hz, current_gesture):
        """
        向後相容公開方法 — 由 InferenceUpdater 每幀呼叫。
        只有在統計視窗啟動後（按鈕按下）才更新 UI 與累積分數。
        """
        if not self._session_active:
            return   # 尚未按下開始，畫面完全靜止

        # 更新即時機率條
        self._mini_bars["0 Hz"].set_value(prob_bg)
        self._mini_bars["4Hz"].set_value(prob_4hz)
        self._mini_bars["6Hz"].set_value(prob_6hz)
        self._mini_bars["8Hz"].set_value(prob_8hz)

        # 即時手勢標籤偵測中隱藏，避免與最終結果產生混淆
        self.live_label.setVisible(False)

        # 累積加權信心分數（Background 也參與競爭）
        SCORE_TH = 0.5
        if prob_bg >= SCORE_TH:
            self._scores["Background"] += prob_bg
            self._valid_frames += 1
        if prob_4hz >= SCORE_TH:
            self._scores["4Hz"] += prob_4hz
            self._valid_frames += 1
        if prob_6hz >= SCORE_TH:
            self._scores["6Hz"] += prob_6hz
            self._valid_frames += 1
        if prob_8hz >= SCORE_TH:
            self._scores["8Hz"] += prob_8hz
            self._valid_frames += 1

    def show_final_result(self, hz_label: str):
        """顯示最終結果與臨床參考建議（也可由外部 InferenceUpdater 呼叫）"""
        color = RESULT_COLORS.get(hz_label, "#64748B")
        display_text = {"None": "訊號不足", "Normal": "正常"}.get(hz_label, hz_label)
        self.result_label.setText(display_text)
        self.result_label.setStyleSheet(
            f"font-size:48px; font-weight:bold; color:{color}; letter-spacing:2px;"
        )
        self.result_sub.setText("30 秒檢測完成")
        self.ring.set_state(0.0, 0, False)

        # 填入臨床參考建議
        advice = CLINICAL_ADVICE.get(hz_label, CLINICAL_ADVICE["None"])
        self.advice_tag.setText(advice["tag"])
        self.advice_tag.setStyleSheet(
            f"color:{color}; font-size:12px; font-weight:bold; letter-spacing:1px;"
        )
        self.advice_label.setText(advice["note"])

    def _restore_button(self):
        self.start_btn.setEnabled(True)
        self.start_btn.setText("▶  重新檢測")


# ─────────────────────────── 獨立測試 ──────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = GestureGUI()
    win.show()

    def simulate():
        bg = random.uniform(0, 0.4)
        p4 = random.uniform(0, 0.6)
        p6 = random.uniform(0, 0.4)
        p8 = max(0, 1 - bg - p4 - p6)
        gesture = max({"4Hz": p4, "6Hz": p6, "8Hz": p8}, key=lambda k: {"4Hz": p4, "6Hz": p6, "8Hz": p8}[k])
        if max(p4, p6, p8) < 0.4:
            gesture = "Background"
        win.update_probabilities(bg, p4, p6, p8, gesture)

    timer = QTimer()
    timer.timeout.connect(simulate)
    timer.start(200)

    sys.exit(app.exec())
