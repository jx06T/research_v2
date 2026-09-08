import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import string
import random

class PairedFontDataset(Dataset):
    def __init__(self, font_path_a, font_path_b, config, num_samples=5000, invert=True):
        self.num_samples = num_samples
        self.img_size = config.image_size
        self.invert = invert
        self.cfg = config

        try:
            self.font_a = ImageFont.truetype(font_path_a, size=int(self.img_size * 1.1))
            self.font_b = ImageFont.truetype(font_path_b, size=int(self.img_size * 1.06))
        except IOError:
            raise RuntimeError("字體文件未找到，請檢查路徑。")

        digits = [str(i) for i in range(10)]
        upper = list(string.ascii_uppercase)
        lower = list(string.ascii_lowercase)
        
        # [v7 設計更新] 加入常用中文字元與部首，作為模型主要訓練依據
        char_string = list(
          "一丨丿乙亅日月木水火土田目手心人入八山石竹米艸衣言金雨風魚馬鳥女子口足艹氵忄扌亻小大中上下左右天地雲川星耳鼻頭身男父母友門戶車舟禾花草林泉光明刀力弓矢矛戈食住行家國學文白黑赤青黃綠王玉示貝糸犬牛羊虫的不我是有了來生在們他時出以可自這會成到為年然要得說過個著能動發臺麼那經去好開現就作後多方如事公看也長面起裡高用業你因而分市於道外沒無同法前民對兒之當教新意情所實全定美理本氣進樣都主間老想重體物知相回性果政只此代和活媽親化加影什己灣機部常見其正世髟血厂辶廴疒匚宇"
        )
        
        all_candidates = digits + upper + lower + char_string

        # 區分用於測試缺失字元的 missing_set 與實際參與訓練的 characters
        self.missing_set = [c for c in all_candidates if c in config.missing_chars]
        self.characters = [c for c in all_candidates if c not in self.missing_set]

    def _render_char_to_tensor(self, char_str: str, font: ImageFont) -> torch.Tensor:
        canvas_size = (int(self.img_size * 2), int(self.img_size * 2))
        img_pil = Image.new("L", canvas_size, color=255)
        draw = ImageDraw.Draw(img_pil)
        
        # fix backward compat for older pillow
        if hasattr(draw, 'textbbox'):
            bbox = draw.textbbox((0, 0), char_str, font=font)
        else:
            bbox = draw.textsize(char_str, font=font)
            bbox = (0, 0, bbox[0], bbox[1])
            
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((canvas_size[0] - text_width) / 2, (canvas_size[1] - text_height) / 2)
        draw.text(position, char_str, font=font, fill=0)

        img_tensor = transforms.ToTensor()(img_pil)
        non_white = torch.where(img_tensor < 1.0)
        if non_white[0].numel() == 0: return torch.ones(1, self.img_size, self.img_size)

        top, bottom = torch.min(non_white[1]), torch.max(non_white[1])
        left, right = torch.min(non_white[2]), torch.max(non_white[2])

        pad = 5
        img_cropped = img_tensor[:, max(0, top-pad):min(canvas_size[1], bottom+pad),
                                    max(0, left-pad):min(canvas_size[0], right+pad)]

        final_canvas = torch.ones(1, self.img_size, self.img_size)
        c_h, c_w = img_cropped.shape[1], img_cropped.shape[2]
        ratio = min((self.img_size * 0.8) / max(1, c_h), (self.img_size * 0.8) / max(1, c_w))
        new_h, new_w = int(c_h * ratio), int(c_w * ratio)

        resized_img = TF.resize(img_cropped, size=(new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR)

        y_off = (self.img_size - new_h) // 2
        x_off = (self.img_size - new_w) // 2
        final_canvas[:, y_off:y_off+new_h, x_off:x_off+new_w] = resized_img

        return final_canvas

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        char_str = random.choice(self.characters)
        label = 0 

        img_a = 1.0 - self._render_char_to_tensor(char_str, self.font_a)
        img_b = 1.0 - self._render_char_to_tensor(char_str, self.font_b)

        params = transforms.RandomAffine.get_params(
            degrees=(-self.cfg.aug_degrees, self.cfg.aug_degrees), 
            translate=self.cfg.aug_translate,
            scale_ranges=self.cfg.aug_scale, shears=None,
            img_size=[self.img_size, self.img_size]
        )

        aug_a = TF.affine(img_a, *params, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
        aug_b = TF.affine(img_b, *params, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)

        if random.random() < self.cfg.aug_mask_prob:
             mask_size = int(self.img_size * 0.4)
             mx = random.randint(0, self.img_size - mask_size)
             my = random.randint(0, self.img_size - mask_size)
             aug_a[:, my:my+mask_size, mx:mx+mask_size] = 0
             aug_b[:, my:my+mask_size, mx:mx+mask_size] = 0

        if not self.invert:
            return 1.0 - aug_a, 1.0 - aug_b, label
        return aug_a, aug_b, label

