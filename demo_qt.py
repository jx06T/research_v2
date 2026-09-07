import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QLabel, QSlider, 
                             QComboBox, QScrollArea, QCheckBox, QLayout)
from PyQt6.QtCore import Qt, QPoint, QRect, QSize, QTimer
from PyQt6.QtGui import QImage, QPixmap
from src.models.mm_legacy import DynamicGenerator

# ==========================================
# 1. 基礎配置 (更新 EXPERIMENTS 結構)
# ==========================================
class DemoConfig:
    def __init__(self):
        self.bottleneck_size = 4
        self.kernel_size = 4
        self.padding = 1
        self.fixed_layers = 5
        self.nz = 256
        self.image_size = 64
        self.nc = 1
        self.ngf = 64
        self.ndf = 64

# 將模型路徑與對應的目標字體封裝在一起
EXPERIMENTS = {
    "ch_ta": {
        "model_path": "Yu2Ta_ch/gan_G_n4_l1_50_l_5_20260414_162105_olaa.pth",
        "target_font": "ttt/f/tegaki_zatsu.ttf"
    },
    "ch_ti": {
        "model_path": "Yu2Ti_ch/gan_G_n4_l1_1_l_5_20260303_083050_9o6z.pth", 
        "target_font": "ttt/f/kaiu.ttf"
    },
}

SOURCE_FONT_PATH = "ttt/f/NotoSansTC-Regular.ttf"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 自動換行佈局 (FlowLayout)
# ==========================================
class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, spacing=0):
        super(FlowLayout, self).__init__(parent)
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self.items = []

    def addItem(self, item): self.items.append(item)
    def count(self): return len(self.items)
    def itemAt(self, index): return self.items[index] if 0 <= index < len(self.items) else None
    def takeAt(self, index): return self.items.pop(index) if 0 <= index < len(self.items) else None
    def expandingDirections(self): return Qt.Orientations(0)
    def hasHeightForWidth(self): return True
    def heightForWidth(self, width): return self.doLayout(QRect(0, 0, width, 0), True)
    def setGeometry(self, rect): super(FlowLayout, self).setGeometry(rect); self.doLayout(rect, False)
    def sizeHint(self): return self.minimumSize()
    def minimumSize(self):
        size = QSize()
        for item in self.items: size = size.expandedTo(item.minimumSize())
        return size + QSize(2*self.contentsMargins().top(), 2*self.contentsMargins().top())

    def doLayout(self, rect, testOnly):
        x, y, lineHeight = rect.x(), rect.y(), 0
        for item in self.items:
            spaceX, spaceY = self.spacing(), self.spacing()
            nextX = x + item.sizeHint().width() + spaceX
            if nextX - spaceX > rect.right() and lineHeight > 0:
                x, y = rect.x(), y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0
            if not testOnly: item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))
            x, lineHeight = nextX, max(lineHeight, item.sizeHint().height())
        return y + lineHeight - rect.y()

# ==========================================
# 3. 單個字符方塊 Widget
# ==========================================
class CharBlock(QWidget):
    def __init__(self, char, pixel_size):
        super().__init__()
        self.char = char
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel(char)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ddd; font-size: 9px; color: #666;")
        
        self.src_img = QLabel()
        self.gen_img = QLabel()
        self.gt_img = QLabel()
        
        self.setFixedSize(pixel_size, pixel_size * 3 + 15)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.src_img)
        self.layout.addWidget(self.gen_img)
        self.layout.addWidget(self.gt_img)

    def update_visibility(self, show_label, show_src, show_gt, pixel_size):
        self.label.setVisible(show_label)
        self.src_img.setVisible(show_src)
        self.gt_img.setVisible(show_gt)
        h = pixel_size + (pixel_size if show_src else 0) + (pixel_size if show_gt else 0) + (15 if show_label else 0)
        self.setFixedSize(pixel_size, h)

# ==========================================
# 4. 主視窗
# ==========================================
class FontGenApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GAN Font Real-time Demo (Auto-GT Switch)")
        self.resize(1280, 850)

        self.cfg = DemoConfig()
        self.current_model = None
        self.current_model_name = ""
        self.current_gt_font_path = "" # 新增：追蹤當前應使用的 GT 字體
        
        self.pixel_size = 32
        self.content_scale = 0.93
        self.blocks = [] 
        self.last_text = "" 

        self.inference_cache = {}
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._do_update_generation)
        
        self.init_ui()
        # 初始化加載第一個模型
        self.load_selected_model(list(EXPERIMENTS.keys())[0])

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        ctrl_box = QVBoxLayout()
        ctrl_box.setSpacing(8)

        self.model_combo = QComboBox()
        self.model_combo.addItems(EXPERIMENTS.keys())
        self.model_combo.currentTextChanged.connect(self.load_selected_model)
        ctrl_box.addWidget(QLabel("<b>模型版本:</b>"))
        ctrl_box.addWidget(self.model_combo)

        ctrl_box.addWidget(QLabel("<b>文字比例:</b>"))
        self.scale_value_label = QLabel("0.93")
        self.scale_value_label.setStyleSheet("color: blue; font-weight: bold;")
        ctrl_box.addWidget(self.scale_value_label)
        
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(40, 100)
        self.scale_slider.setValue(93)
        self.scale_slider.valueChanged.connect(self.on_param_changed)
        ctrl_box.addWidget(self.scale_slider)

        ctrl_box.addWidget(QLabel("<b>顯示像素尺寸:</b>"))
        self.disp_slider = QSlider(Qt.Orientation.Horizontal)
        self.disp_slider.setRange(24, 512)
        self.disp_slider.setValue(32)
        self.disp_slider.valueChanged.connect(self.on_param_changed)
        ctrl_box.addWidget(self.disp_slider)

        self.chk_label = QCheckBox("顯示標籤"); self.chk_label.setChecked(True)
        self.chk_src = QCheckBox("顯示 Source"); self.chk_src.setChecked(True)
        self.chk_gt = QCheckBox("顯示 GT"); self.chk_gt.setChecked(True)
        for chk in [self.chk_label, self.chk_src, self.chk_gt]:
            chk.stateChanged.connect(self.refresh_ui_visibility)
            ctrl_box.addWidget(chk)

        ctrl_box.addWidget(QLabel("<b>測試文字:</b>"))
        # self.text_input = QTextEdit("星鼻澤叮朝")
        self.text_input = QTextEdit("""根據考古學家研究，臺灣至少在舊石器時代晚期，距今兩三萬年前已
有人類居住，而高雄地區雖未發現舊石器時代遺址，但很可能也是早期人
雅化的詩還不得不回向俗化，剛剛來自民間的詞，在當時不用說自然
是「雅俗共賞」的。別瞧黃山谷的有些詩不好懂，他的一些小詞可夠俗的
。柳耆卿更是個通俗的詞人。詞後來雖然漸漸雅化或文人化，可是始終不
能雅到詩的地位，它怎麼著也只是「詩餘」。""")
        self.text_input.textChanged.connect(self.update_generation)
        ctrl_box.addWidget(self.text_input)

        ctrl_box.addStretch()

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.container = QWidget()
        self.container.setStyleSheet("background-color: white;")
        self.flow_layout = FlowLayout(self.container, spacing=0)
        self.scroll.setWidget(self.container)

        left_w = QWidget(); left_w.setFixedWidth(240); left_w.setLayout(ctrl_box)
        layout.addWidget(left_w)
        layout.addWidget(self.scroll)

    def render_char_tensor(self, char, font_path):
        size = self.cfg.image_size
        img = Image.new('L', (size, size), color=255)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(font_path, int(size * self.content_scale))
            left, top, right, bottom = font.getbbox(char)
            # 簡單置中對齊邏輯
            draw.text(((size - (right-left)) / 2 - left, (size - (bottom-top)) / 2 - top), char, font=font, fill=0)
        except Exception as e:
            print(f"Render Error ({char}): {e}")
        return torch.from_numpy(1.0 - np.array(img).astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)

    def on_param_changed(self):
        self.content_scale = self.scale_slider.value() / 100.0
        self.pixel_size = self.disp_slider.value()
        self.scale_value_label.setText(f"{self.content_scale:.2f}")
        # 重大參數改變，需清除快取並重繪
        self.last_text = "" 
        self.inference_cache.clear()
        self.update_timer.start(100)

    def refresh_ui_visibility(self):
        for block in self.blocks:
            block.update_visibility(self.chk_label.isChecked(), self.chk_src.isChecked(), self.chk_gt.isChecked(), self.pixel_size)

    def load_selected_model(self, name):
        try:
            print(f"Loading model: {name}...")
            config = EXPERIMENTS[name]
            
            # 更新當前 GT 字體路徑
            self.current_gt_font_path = config["target_font"]
            self.current_model_name = name
            
            # 初始化並加載模型權重
            self.current_model = DynamicGenerator(self.cfg).to(DEVICE)
            self.current_model.load_state_dict(torch.load(config["model_path"], map_location=DEVICE))
            self.current_model.eval()
            
            # 模型換了，快取失效並重置文字狀態
            self.last_text = "" 
            self.inference_cache.clear()
            self.update_timer.start(100)
        except Exception as e: 
            print(f"Load Error: {e}")

    def update_generation(self):
        self.update_timer.start(100)

    def _do_update_generation(self):
        if not self.current_model: return
        
        raw_text = self.text_input.toPlainText()
        current_text = raw_text.replace("\n", "").replace(" ", "")
        
        if current_text == self.last_text:
            return

        # 增量更新判斷
        if current_text.startswith(self.last_text) and self.last_text != "":
            new_chars = current_text[len(self.last_text):]
            self._add_chars_to_ui(new_chars)
        else:
            self._clear_all_blocks()
            self._add_chars_to_ui(current_text)

        self.last_text = current_text

    def _clear_all_blocks(self):
        while self.flow_layout.count():
            item = self.flow_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.blocks = []

    def _add_chars_to_ui(self, text_segment):
        for char in text_segment:
            # 快取鍵包含模型名稱，確保切換模型時會重新推論
            cache_key = (char, self.current_model_name, self.content_scale)
            
            if cache_key in self.inference_cache:
                s_px, f_px, g_px = self.inference_cache[cache_key]
            else:
                src_t = self.render_char_tensor(char, SOURCE_FONT_PATH)
                # 關鍵修改：使用當前配置的 GT 字體路徑
                gt_t = self.render_char_tensor(char, self.current_gt_font_path)
                
                with torch.no_grad():
                    noise = torch.randn(1, self.cfg.nz, self.cfg.bottleneck_size, self.cfg.bottleneck_size).to(DEVICE)
                    fake_t = torch.clamp(self.current_model(noise, src_t), 0, 1)

                s_px = tensor_to_pixmap(src_t, self.pixel_size)
                f_px = tensor_to_pixmap(fake_t, self.pixel_size)
                g_px = tensor_to_pixmap(gt_t, self.pixel_size)
                self.inference_cache[cache_key] = (s_px, f_px, g_px)

            block = CharBlock(char, self.pixel_size)
            block.src_img.setPixmap(s_px)
            block.gen_img.setPixmap(f_px)
            block.gt_img.setPixmap(g_px)
            block.update_visibility(self.chk_label.isChecked(), self.chk_src.isChecked(), self.chk_gt.isChecked(), self.pixel_size)
            
            self.flow_layout.addWidget(block)
            self.blocks.append(block)

def tensor_to_pixmap(tensor, target_px):
    # 將 tensor 轉回 0-255 的灰階圖
    img_np = 255 - (tensor.detach().cpu().squeeze().numpy() * 255).astype(np.uint8)
    # PIL/Tensor 預設 64x64，轉換為 QImage
    qimg = QImage(img_np.data, 64, 64, 64, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(qimg).scaled(target_px, target_px, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FontGenApp()
    window.show()
    sys.exit(app.exec())