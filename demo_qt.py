import sys
import os
import csv
import glob
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QLabel, QSlider, 
                             QListWidget, QAbstractItemView, QScrollArea, QCheckBox, QLayout)
from PyQt6.QtCore import Qt, QPoint, QRect, QSize, QTimer
from PyQt6.QtGui import QImage, QPixmap
from src.models.generator.dynamic_gen import DynamicGenerator
from src.models.generator.attn_unet_gen import AttnUNetGenerator

# ==========================================
# 1. 基礎配置 (更新 EXPERIMENTS 結構)
# ==========================================
class DemoConfig:
    def __init__(self):
        # 這些將會在切換模型時被動態覆寫
        self.bottleneck_size = 4
        self.kernel_size = 4
        self.padding = 1
        self.fixed_layers = 5
        self.nz = 256
        self.image_size = 64
        self.nc = 1
        self.ngf = 64
        self.ndf = 64
        self.gen_type = "dynamic"

# 動態加載模型清單
EXPERIMENTS = {}
try:
    with open("model_registry.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            arch_folder = row.get("arch_folder", "")
            run_id = row.get("run_id", "")
            exp_name = row.get("experiment", "")
            
            # 掃描此架構資料夾下所有符合該 run_id 的權重檔
            search_path = os.path.join("runs", arch_folder, f"G_*{run_id}*.pth")
            found_models = glob.glob(search_path)
            
            for model_path in found_models:
                filename = os.path.basename(model_path)
                short_run_id = run_id.split('_')[-1] if '_' in run_id else run_id
                
                if "epoch" in filename:
                    epoch_str = filename.split("epoch_")[-1].replace(".pth", "")
                    display_name = f"[{exp_name}] {short_run_id} (Ep {epoch_str})"
                elif "latest" in filename:
                    display_name = f"[{exp_name}] {short_run_id} (Latest)"
                else:
                    display_name = f"[{exp_name}] {short_run_id}"
                
                EXPERIMENTS[display_name] = {
                    "model_path": model_path,
                    "target_font": row.get("tgt_font") if row.get("tgt_font") not in ["", "N/A", None] else "data/fonts/kaiu.ttf",
                    "source_font": row.get("src_font") if row.get("src_font") not in ["", "N/A", None] else "data/fonts/NotoSansTC-Regular.ttf",
                    "image_size": int(row.get("image_size", 64)) if str(row.get("image_size")).isdigit() else 64,
                    "bottleneck_size": int(row.get("bottleneck_size", 4)) if str(row.get("bottleneck_size")).isdigit() else 4,
                    "nz": int(row.get("nz", 256)) if str(row.get("nz")).isdigit() else 256,
                    "nc": int(row.get("nc", 1)) if str(row.get("nc")).isdigit() else 1,
                    "gen_type": row.get("gen_type", "dynamic")
                }
except Exception as e:
    print(f"無法讀取 model_registry.csv 或尚無紀錄: {e}")

# 給予至少一個預設選項避免 UI 崩潰
if not EXPERIMENTS:
    EXPERIMENTS["Fallback (尚無歸檔模型)"] = {
        "model_path": "runs/exp_01/G_latest_3.pth",
        "target_font": "data/fonts/kaiu.ttf",
        "source_font": "data/fonts/NotoSansTC-Regular.ttf",
        "image_size": 64,
        "bottleneck_size": 4,
        "nz": 256,
        "nc": 1,
        "gen_type": "dynamic"
    }

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
# 3. 單個字符方塊 Widget (支援多模型顯示)
# ==========================================
class CharBlock(QWidget):
    def __init__(self, char, pixel_size, active_models):
        super().__init__()
        self.char = char
        self.active_models = active_models
        
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(2)
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.setStyleSheet("background-color: #fafafa; border: 1px solid #ccc; border-radius: 4px;")
        
        self.label = QLabel(char)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("font-size: 10px; color: #333; font-weight: bold; border: none;")
        
        self.src_img = QLabel()
        self.src_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.src_img.setStyleSheet("border: none;")
        
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.src_img)
        
        self.model_rows = []
        for name, _ in self.active_models:
            # 截斷過長名稱
            short_name = name[:18] + ".." if len(name) > 20 else name
            name_lbl = QLabel(short_name)
            name_lbl.setStyleSheet("font-size: 8px; color: #555; margin-top: 4px; border: none;")
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            from PyQt6.QtWidgets import QGridLayout
            grid = QGridLayout()
            grid.setSpacing(2)
            grid.setContentsMargins(0,0,0,0)
            gen_lbl = QLabel()
            gen_lbl.setStyleSheet("border: none;")
            gt_lbl = QLabel()
            gt_lbl.setStyleSheet("border: none;")
            
            grid.addWidget(gen_lbl, 0, 0)
            grid.addWidget(gt_lbl, 0, 1)
            
            self.layout.addWidget(name_lbl)
            self.layout.addLayout(grid)
            
            self.model_rows.append((name_lbl, gen_lbl, gt_lbl, grid))

    def update_visibility(self, show_label, show_src, show_gt, gt_bottom, pixel_size):
        self.label.setVisible(show_label)
        self.src_img.setVisible(show_src)
        
        row_width = pixel_size
        if show_gt and not gt_bottom:
            row_width = pixel_size * 2 + 2
            
        width = max(pixel_size, row_width) + 4
        
        h = 4 # padding
        if show_label: h += 15
        if show_src: h += pixel_size
        
        for name_lbl, gen_lbl, gt_lbl, grid in self.model_rows:
            gt_lbl.setVisible(show_gt)
            grid.removeWidget(gt_lbl)
            if gt_bottom:
                grid.addWidget(gt_lbl, 1, 0)
                h += 12 + pixel_size + (pixel_size if show_gt else 0)
            else:
                grid.addWidget(gt_lbl, 0, 1)
                h += 12 + pixel_size # 名字高度 + 圖片高度
            
        self.setFixedSize(width, h)

# ==========================================
# 4. 主視窗
# ==========================================
class FontGenApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GAN Font Real-time Demo (Multi-Model Comparison)")
        self.resize(1280, 850)

        self.cfg_template = DemoConfig()
        self.active_models = [] # List of (display_name, dict: {model, config, current_src_font, current_gt_font})
        
        self.pixel_size = 32
        self.content_scale = 0.93
        self.blocks = [] 
        self.last_text = "" 
        self.last_active_names = []

        self.inference_cache = {}
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._do_update_generation)
        
        self.init_ui()
        # 預設選取第一個模型
        if EXPERIMENTS:
            self.model_list.setCurrentRow(0)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        ctrl_box = QVBoxLayout()
        ctrl_box.setSpacing(8)

        # 改為多選清單
        ctrl_box.addWidget(QLabel("<b>模型版本 (可按 Ctrl 多選):</b>"))
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.model_list.addItems(EXPERIMENTS.keys())
        self.model_list.itemSelectionChanged.connect(self.load_selected_models)
        self.model_list.setMaximumHeight(200)
        ctrl_box.addWidget(self.model_list)

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
        self.chk_gt = QCheckBox("顯示 GT (對照目標字體)"); self.chk_gt.setChecked(True)
        self.chk_gt_bottom = QCheckBox("GT 放於下方 (垂直比較)")
        for chk in [self.chk_label, self.chk_src, self.chk_gt, self.chk_gt_bottom]:
            chk.stateChanged.connect(self.refresh_ui_visibility)
            ctrl_box.addWidget(chk)

        ctrl_box.addWidget(QLabel("<b>測試文字:</b>"))
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
        self.flow_layout = FlowLayout(self.container, spacing=4)
        self.scroll.setWidget(self.container)

        left_w = QWidget(); left_w.setFixedWidth(260); left_w.setLayout(ctrl_box)
        layout.addWidget(left_w)
        layout.addWidget(self.scroll)

    def render_char_tensor(self, char, font_path, image_size):
        img = Image.new('L', (image_size, image_size), color=255)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(font_path, int(image_size * self.content_scale))
            left, top, right, bottom = font.getbbox(char)
            # 簡單置中對齊邏輯
            draw.text(((image_size - (right-left)) / 2 - left, (image_size - (bottom-top)) / 2 - top), char, font=font, fill=0)
        except Exception as e:
            pass # 略過報錯
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
            block.update_visibility(self.chk_label.isChecked(), self.chk_src.isChecked(), self.chk_gt.isChecked(), self.chk_gt_bottom.isChecked(), self.pixel_size)

    def load_selected_models(self):
        selected_items = self.model_list.selectedItems()
        self.active_models = []
        
        print("重新加載選取的模型...")
        for item in selected_items:
            name = item.text()
            try:
                config = EXPERIMENTS[name]
                
                # 建立獨立的 config 給這個模型
                cfg = DemoConfig()
                cfg.image_size = config["image_size"]
                cfg.bottleneck_size = config["bottleneck_size"]
                cfg.nz = config["nz"]
                cfg.nc = config["nc"]
                cfg.gen_type = config.get("gen_type", "dynamic")
                
                if cfg.gen_type == "unet":
                    model = AttnUNetGenerator(cfg).to(DEVICE)
                else:
                    model = DynamicGenerator(cfg).to(DEVICE)
                model.load_state_dict(torch.load(config["model_path"], map_location=DEVICE))
                model.eval()
                
                self.active_models.append((name, {
                    "model": model,
                    "cfg": cfg,
                    "src_font": config["source_font"],
                    "tgt_font": config["target_font"]
                }))
            except Exception as e:
                print(f"無法載入模型 {name}: {e}")
                
        # 檢查選取的組合是否改變
        current_names = [m[0] for m in self.active_models]
        if current_names != self.last_active_names:
            self.last_text = "" 
            self.last_active_names = current_names
            self.update_timer.start(100)

    def update_generation(self):
        self.update_timer.start(100)

    def _do_update_generation(self):
        if not self.active_models:
            self._clear_all_blocks()
            return
            
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
        active_names = tuple(m[0] for m in self.active_models)
        
        for char in text_segment:
            cache_key = (char, active_names, self.content_scale)
            
            if cache_key in self.inference_cache:
                s_px, model_pixmaps = self.inference_cache[cache_key]
            else:
                # 為了避免多個模型使用的 src_font 不同，我們以第一個模型的 src_font 作為顯示用
                primary_src_font = self.active_models[0][1]["src_font"]
                primary_size = self.active_models[0][1]["cfg"].image_size
                s_px = tensor_to_pixmap(self.render_char_tensor(char, primary_src_font, primary_size), self.pixel_size)
                
                model_pixmaps = []
                for name, info in self.active_models:
                    src_t = self.render_char_tensor(char, info["src_font"], info["cfg"].image_size)
                    gt_t = self.render_char_tensor(char, info["tgt_font"], info["cfg"].image_size)
                    
                    with torch.no_grad():
                        noise = torch.randn(1, info["cfg"].nz, info["cfg"].bottleneck_size, info["cfg"].bottleneck_size).to(DEVICE)
                        fake_t = torch.clamp(info["model"](noise, src_t), 0, 1)

                    f_px = tensor_to_pixmap(fake_t, self.pixel_size)
                    g_px = tensor_to_pixmap(gt_t, self.pixel_size)
                    model_pixmaps.append((f_px, g_px))
                    
                self.inference_cache[cache_key] = (s_px, model_pixmaps)

            block = CharBlock(char, self.pixel_size, self.active_models)
            block.src_img.setPixmap(s_px)
            
            for idx, (f_px, g_px) in enumerate(model_pixmaps):
                _, gen_lbl, gt_lbl, _ = block.model_rows[idx]
                gen_lbl.setPixmap(f_px)
                gt_lbl.setPixmap(g_px)
                
            block.update_visibility(self.chk_label.isChecked(), self.chk_src.isChecked(), self.chk_gt.isChecked(), self.chk_gt_bottom.isChecked(), self.pixel_size)
            
            self.flow_layout.addWidget(block)
            self.blocks.append(block)

def tensor_to_pixmap(tensor, target_px):
    # 動態讀取實際尺寸
    img_np = 255 - (tensor.detach().cpu().squeeze().numpy() * 255).astype(np.uint8)
    # img_np shape: (H, W)
    h, w = img_np.shape
    bytes_per_line = w
    qimg = QImage(img_np.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(qimg).scaled(target_px, target_px, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FontGenApp()
    window.show()
    sys.exit(app.exec())