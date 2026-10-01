"""Paired source-font and reviewed handwritten scan glyphs for training."""

import csv
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class ScannedGlyphDataset(Dataset):
    def __init__(self, manifest_path, src_font_path, config, writer_id=""):
        self.cfg = config
        self.image_size = config.image_size
        self.font = ImageFont.truetype(src_font_path, size=round(self.image_size * 1.1))
        with open(manifest_path, newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        self.rows = [r for r in rows if not writer_id or r["writer_id"] == writer_id]
        if not self.rows:
            raise ValueError("The training manifest has no glyphs for this writer")
        writers = {r["writer_id"] for r in self.rows}
        if len(writers) > 1:
            raise ValueError("Multiple writer_id values found; pass --writer_id to train one handwriting style")
        for row in self.rows:
            if len(row["label"]) != 1 or not Path(row["clean_path"]).is_file():
                raise ValueError(f"Invalid training row: {row['id']}")

    def __len__(self):
        return len(self.rows)

    def _source(self, char):
        size = self.image_size
        canvas = Image.new("L", (size * 2, size * 2), 255)
        draw = ImageDraw.Draw(canvas)
        bounds = draw.textbbox((0, 0), char, font=self.font)
        x = (canvas.width - (bounds[2] - bounds[0])) / 2 - bounds[0]
        y = (canvas.height - (bounds[3] - bounds[1])) / 2 - bounds[1]
        draw.text((x, y), char, font=self.font, fill=0)
        dark = Image.eval(canvas, lambda px: 255 - px)
        bbox = dark.getbbox()
        result = Image.new("L", (size, size), 255)
        if bbox:
            glyph = canvas.crop(bbox)
            scale = min(size * 0.8 / glyph.width, size * 0.8 / glyph.height)
            glyph = glyph.resize((max(1, round(glyph.width * scale)), max(1, round(glyph.height * scale))), Image.Resampling.LANCZOS)
            result.paste(glyph, ((size - glyph.width) // 2, (size - glyph.height) // 2))
        return result

    def get_unaugmented(self, index):
        """Return centered source and cleaned target for stable previews."""
        row = self.rows[index]
        source = self._source(row["label"])
        with Image.open(row["clean_path"]) as image:
            target = image.convert("L").resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        src = 1.0 - transforms.ToTensor()(source)
        tgt = 1.0 - transforms.ToTensor()(target)
        return src, tgt, row["label"]

    def __getitem__(self, index):
        src, tgt, label = self.get_unaugmented(index)
        params = transforms.RandomAffine.get_params(
            degrees=(-self.cfg.aug_degrees, self.cfg.aug_degrees),
            translate=self.cfg.aug_translate, scale_ranges=self.cfg.aug_scale,
            shears=None, img_size=[self.image_size, self.image_size],
        )
        src = TF.affine(src, *params, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
        tgt = TF.affine(tgt, *params, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
        return src, tgt, label
