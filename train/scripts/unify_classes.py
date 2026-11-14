from pathlib import Path
import shutil
import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "data" / "raw" / "construction safety.v2-release.yolov8"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "construction safety.v2-release.yolov8"

SOURCE_YAML = DATASET_DIR / "data.yaml"

with open(SOURCE_YAML, "r", encoding="utf-8") as f:
    source_data = yaml.safe_load(f)

source_names = source_data["names"]

print("원본 클래스:", source_names)

CLASS_MAP_PATH = BASE_DIR / "configs" / "class_map.yaml"

with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
    class_map = yaml.safe_load(f)

class_map = {int(k): int(v) for k, v in class_map.items()}

print("클래스 매핑:", class_map)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("출력 폴더:", OUTPUT_DIR)

SPLITS = ["train", "valid", "test"]


def convert_label_line(line):
    parts = line.strip().split()

    if len(parts) != 5:
        return None

    class_id = int(parts[0])
    coordinates = parts[1:]

    for value in coordinates:
        value = float(value)
        if value < 0 or value > 1:
            return None

    if class_id not in class_map:
        return None

    new_class_id = class_map[class_id]

    return " ".join([str(new_class_id)] + coordinates) + "\n"

for split in SPLITS:
    source_labels_dir = DATASET_DIR / split / "labels"
    output_labels_dir = OUTPUT_DIR / split / "labels"

    output_labels_dir.mkdir(parents=True, exist_ok=True)

    label_files = list(source_labels_dir.glob("*.txt"))

    print(f"{split}: 라벨 파일 {len(label_files)}개")

    for label_file in label_files:
        output_file = output_labels_dir / label_file.name

        converted_lines = []

        with open(label_file, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                converted_line = convert_label_line(line)

                if converted_line is None:
                    print(
                        f"라벨 형식 오류 또는 매핑 불가: "
                        f"{label_file.name}:{line_number}"
                    )
                    continue

                converted_lines.append(converted_line)

        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(converted_lines)

            source_image_dir = DATASET_DIR / split / "images"
        output_image_dir = OUTPUT_DIR / split / "images"

        output_image_dir.mkdir(parents=True, exist_ok=True)

        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        for extension in image_extensions:
            source_image = source_image_dir / (label_file.stem + extension)

            if source_image.exists():
                shutil.copy2(source_image, output_image_dir / source_image.name)
                break
            output_data = {
    "path": str(OUTPUT_DIR.resolve()),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 5,
    "names": [
        "person",
        "helmet",
        "vest",
        "no-helmet",
        "no-vest"
    ]
}

output_data = {
    "path": str(OUTPUT_DIR.resolve()),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 5,
    "names": [
        "person",
        "helmet",
        "vest",
        "no-helmet",
        "no-vest"
    ]
}

with open(OUTPUT_DIR / "data.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(output_data, f, allow_unicode=True, sort_keys=False)

print("\n변환 완료!")
print(f"변환 데이터셋: {OUTPUT_DIR}")

print("\n변환 후 클래스별 박스 수를 계산합니다.")

class_counts = {
    0: 0,  # person
    1: 0,  # helmet
    2: 0,  # vest
    3: 0,  # no-helmet
    4: 0   # no-vest
}

for split in SPLITS:
    labels_dir = OUTPUT_DIR / split / "labels"

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) == 5:
                    class_id = int(parts[0])

                    if class_id in class_counts:
                        class_counts[class_id] += 1

print("\n변환 후 클래스별 박스 수:")
print(f"person     : {class_counts[0]}")
print(f"helmet     : {class_counts[1]}")
print(f"vest       : {class_counts[2]}")
print(f"no-helmet  : {class_counts[3]}")
print(f"no-vest    : {class_counts[4]}")

print("\n원본 클래스별 박스 수를 계산합니다.")

source_class_counts = {
    0: 0,  # helmet
    1: 0,  # no-helmet
    2: 0,  # no-vest
    3: 0,  # person
    4: 0   # vest
}

for split in SPLITS:
    labels_dir = DATASET_DIR / split / "labels"

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) == 5:
                    class_id = int(parts[0])

                    if class_id in source_class_counts:
                        source_class_counts[class_id] += 1

print("\n원본 클래스별 박스 수:")
print(f"helmet     : {source_class_counts[0]}")
print(f"no-helmet  : {source_class_counts[1]}")
print(f"no-vest    : {source_class_counts[2]}")
print(f"person     : {source_class_counts[3]}")
print(f"vest       : {source_class_counts[4]}")