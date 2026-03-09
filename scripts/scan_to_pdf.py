#!/usr/bin/env python3
"""Deskew image scans, OCR them, and assemble a searchable PDF."""

from __future__ import annotations

import argparse
import glob
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pypdf import PdfMerger


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def collect_images(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        expanded = sorted(glob.glob(item))
        if expanded:
            candidates = expanded
        else:
            candidates = [item]

        for candidate in candidates:
            path = Path(candidate)
            if path.is_dir():
                for child in sorted(path.iterdir()):
                    if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                        files.append(child)
            elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                files.append(path)

    unique_files = sorted(set(files))
    if not unique_files:
        raise ValueError("No input images found. Provide files, folders, or glob patterns.")
    return unique_files


def compute_skew_angle(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    return -angle


def deskew_image(image: np.ndarray) -> np.ndarray:
    angle = compute_skew_angle(image)
    if abs(angle) < 0.05:
        return image

    (height, width) = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def ocr_image_to_pdf_bytes(image: np.ndarray, lang: str) -> bytes:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return pytesseract.image_to_pdf_or_hocr(rgb, extension="pdf", lang=lang)


def write_text_output(image: np.ndarray, out_txt_path: Path, lang: str) -> None:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(rgb, lang=lang)
    out_txt_path.write_text(text, encoding="utf-8")


def run_pipeline(image_paths: list[Path], output_pdf: Path, lang: str, text_dir: Path | None) -> None:
    merger = PdfMerger()
    temp_dir = Path(tempfile.mkdtemp(prefix="scan_to_pdf_"))

    try:
        for index, image_path in enumerate(image_paths, start=1):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")

            deskewed = deskew_image(image)
            pdf_bytes = ocr_image_to_pdf_bytes(deskewed, lang=lang)

            page_pdf = temp_dir / f"page_{index:04}.pdf"
            page_pdf.write_bytes(pdf_bytes)
            merger.append(str(page_pdf))

            if text_dir:
                text_dir.mkdir(parents=True, exist_ok=True)
                txt_path = text_dir / f"{image_path.stem}.txt"
                write_text_output(deskewed, txt_path, lang=lang)

            print(f"Processed page {index}/{len(image_paths)}: {image_path}")

        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        merger.write(str(output_pdf))
        print(f"Created searchable PDF: {output_pdf}")
    finally:
        merger.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deskew scanned pages, OCR them, and combine them into a searchable PDF."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input images, folders, or glob patterns (for example: scans/*.jpg)",
    )
    parser.add_argument("-o", "--output", required=True, help="Output PDF path")
    parser.add_argument(
        "--lang",
        default="eng",
        help="Tesseract language code(s), e.g. 'eng' or 'eng+deu'",
    )
    parser.add_argument(
        "--text-dir",
        help="Optional folder to save per-page OCR text files",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    images = collect_images(args.inputs)
    output_pdf = Path(args.output)
    text_dir = Path(args.text_dir) if args.text_dir else None

    run_pipeline(images, output_pdf=output_pdf, lang=args.lang, text_dir=text_dir)


if __name__ == "__main__":
    main()
