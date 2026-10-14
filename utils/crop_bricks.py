
import argparse
from pathlib import Path
import sys

import cv2
repo_path = Path("/home/himoza/block_detect/ultralytics")
sys.path.insert(0, str(repo_path))

from ultralytics import YOLO


def crop_bricks(
    model_path: str,
    input_dir: str,
    output_dir: str,
    target_size: int = 640,
    conf: float = 0.5,
    padding: int = 5,
    save_format: str = "jpg",
    recursive: bool = False,
):
    """Детектирует кирпичи, вырезает и масштабирует их."""

    model_file = Path(model_path)
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not model_file.is_file():
        raise FileNotFoundError(f"Модель не найдена: {model_file}")

    if not input_path.is_dir():
        raise NotADirectoryError(
            f"Входная папка не найдена: {input_path}"
        )

    if target_size <= 0:
        raise ValueError("target_size должен быть больше нуля")

    if padding < 0:
        raise ValueError("padding не может быть отрицательным")

    if not 0 <= conf <= 1:
        raise ValueError("conf должен быть в диапазоне от 0 до 1")

    save_format = save_format.lower().lstrip(".")
    if save_format not in {"jpg", "png"}:
        raise ValueError("Поддерживаются только jpg и png")

    output_path.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    iterator = input_path.rglob("*") if recursive else input_path.iterdir()

    # Не обрабатываем изображения из выходной папки,
    # если она находится внутри входной.
    output_resolved = output_path.resolve()

    image_paths = sorted(
        p for p in iterator
        if p.is_file()
        and p.suffix.lower() in extensions
        and output_resolved not in p.resolve().parents
    )

    model = YOLO(str(model_file))

    processed = 0
    images_with_bricks = 0
    saved = 0

    print(f"Найдено изображений: {len(image_paths)}")
    print(f"Загрузка модели: {model_file}")

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[WARN] Не удалось прочитать: {image_path}")
            continue

        processed += 1

        result = model.predict(
            source=image,
            conf=conf,
            verbose=False
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            print(f"[INFO] Кирпичи не найдены: {image_path.name}")
            continue

        image_h, image_w = image.shape[:2]
        image_saved = 0

        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].cpu().tolist()
            )

            # Отступ вокруг объекта
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image_w, x2 + padding)
            y2 = min(image_h, y2 + padding)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2].copy()

            # Масштабируем большую сторону до target_size.
            # Пропорции сохраняются, поля не добавляются.
            h, w = crop.shape[:2]
            scale = target_size / max(h, w)

            new_w = max(1, round(w * scale))
            new_h = max(1, round(h * scale))

            interpolation = (
                cv2.INTER_AREA if scale < 1
                else cv2.INTER_CUBIC
            )

            crop = cv2.resize(
                crop,
                (new_w, new_h),
                interpolation=interpolation
            )

            confidence = float(box.conf[0].cpu())

            # Относительная структура папок сохраняется,
            # чтобы файлы с одинаковыми именами не перезаписывались.
            relative_path = image_path.relative_to(input_path)
            destination_dir = output_path / relative_path.parent
            destination_dir.mkdir(parents=True, exist_ok=True)

            filename = (
                f"{image_path.stem}_brick_{i + 1}"
                f"_conf{confidence:.2f}.{save_format}"
            )

            destination = destination_dir / filename

            if cv2.imwrite(str(destination), crop):
                saved += 1
                image_saved += 1

        if image_saved:
            images_with_bricks += 1

        print(
            f"[OK] {image_path.name}: "
            f"сохранено кирпичей — {image_saved}"
        )

    print("\n=== Результат ===")
    print(f"Обработано изображений: {processed}")
    print(f"Изображений с кирпичами: {images_with_bricks}")
    print(f"Сохранено вырезок: {saved}")
    print(f"Выходная папка: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Детекция кирпичей YOLO, вырезание и "
            "масштабирование для последующей сегментации."
        )
    )

    parser.add_argument(
        "--model", required=True,
        help="Путь к весам YOLO, например best.pt"
    )
    parser.add_argument(
        "--input", required=True,
        help="Папка с исходными изображениями"
    )
    parser.add_argument(
        "--output", required=True,
        help="Папка для вырезанных кирпичей"
    )
    parser.add_argument(
        "--size", type=int, default=640,
        help="Размер большей стороны (по умолчанию: 640)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="Порог уверенности (по умолчанию: 0.5)"
    )
    parser.add_argument(
        "--padding", type=int, default=5,
        help="Отступ вокруг кирпича в пикселях (по умолчанию: 5)"
    )
    parser.add_argument(
        "--format", choices=["jpg", "png"], default="jpg",
        help="Формат выходных изображений (по умолчанию: jpg)"
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Обрабатывать изображения во вложенных папках"
    )

    args = parser.parse_args()

    crop_bricks(
        model_path=args.model,
        input_dir=args.input,
        output_dir=args.output,
        target_size=args.size,
        conf=args.conf,
        padding=args.padding,
        save_format=args.format,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
