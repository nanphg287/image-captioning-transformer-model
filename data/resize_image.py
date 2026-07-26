import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from typing import Any

from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from config import Config

IMAGE_SIZE = (Config.image_size, Config.image_size)
TARGET_SIZE = Config.image_size
MAPPING_FILE = Config.train_data_file


def load_images_path() -> list[Path]:
    """
    Đọc file image_to_captions.json và trả về danh sách đường dẫn ảnh.
    """

    image_to_captions = load_image_mapping()
    images_path: list[Path] = []

    for image_name, image_data in image_to_captions.items():
        if not isinstance(image_data, dict):
            print(f"Bỏ qua {image_name}: dữ liệu không phải object.")
            continue

        image_path_value = image_data.get("image_path")

        if not image_path_value:
            print(f"Bỏ qua {image_name}: không có trường image_path.")
            continue

        images_path.append(Path(image_path_value))

    return images_path


def load_image_mapping() -> dict:
    mapping_file = Path(MAPPING_FILE)

    if not mapping_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file JSON: {mapping_file}")

    with mapping_file.open("r", encoding="utf-8") as file:
        image_to_captions = json.load(file)

    if not isinstance(image_to_captions, dict):
        raise ValueError("Dữ liệu trong JSON phải có dạng dictionary/object.")

    return image_to_captions


def save_image_mapping(image_to_captions: dict) -> None:
    mapping_file = Path(MAPPING_FILE)
    temporary_file = mapping_file.with_name(f"{mapping_file.name}.tmp")
    with temporary_file.open(mode="w", encoding="utf-8") as file:
        json.dump(image_to_captions, file, ensure_ascii=False, indent=2)

    temporary_file.replace(mapping_file)


def resize_and_center_crop(
        original_image: Image.Image,
        target_size: int
) -> Image.Image:
    """
    Resize ảnh theo cạnh ngắn nhất, giữ nguyên tỉ lệ,
    sau đó center crop về kích thước vuông target_size x target_size.
    """

    resized_image = TF.resize(
        original_image,
        size=target_size,
        interpolation=InterpolationMode.BILINEAR,
        antialias=True
    )

    final_image = TF.center_crop(
        resized_image,
        output_size=[target_size, target_size]
    )

    return final_image


def process_single_image(image_name: str, image_data: dict[str, Any], output_dir: Path, target_size: int) \
        -> tuple[str, str]:
    """
    Tác vụ được chạy trong worker thread.

    Hàm thực hiện:
    1. Lấy image_path.
    2. Mở ảnh.
    3. Resize và center crop.
    4. Lưu ảnh.
    5. Trả về image_name và resize_image_path.

    Hàm không cập nhật dictionary dùng chung để tránh nhiều worker thread cùng thay đổi dữ liệu JSON.
    """

    if not isinstance(image_data, dict):
        raise ValueError(f"Dữ liệu của {image_name} không phải object.")

    original_path_value = image_data.get("image_path")

    if not original_path_value:
        raise ValueError(f"{image_name} không có trường image_path.")

    original_image_path = Path(original_path_value)

    if not original_image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {original_image_path}")

    output_path = output_dir / f"{original_image_path.stem}.jpg"

    with Image.open(original_image_path) as original_image:
        original_image = ImageOps.exif_transpose(original_image)

        # Đảm bảo ảnh có đúng 3 channel RGB
        original_image = original_image.convert("RGB")

        final_image = resize_and_center_crop(
            original_image=original_image,
            target_size=target_size
        )

        final_image.save(
            output_path,
            format="JPEG",
            quality=95
        )

    resize_image_path = output_path.resolve().as_posix()
    return image_name, resize_image_path


def resize_images_with_threads(
        image_to_captions: dict[str, dict[str, Any]],
        max_workers: int = 4
) -> tuple[int, int]:
    """
    Resize ảnh song song bằng ThreadPoolExecutor.
    Tối đa max_workers ảnh được xử lý đồng thời.
    """
    output_dir = Path(Config.resize_image_dir)
    target_size = Config.image_size
    output_dir.mkdir(parents=True, exist_ok=True)

    total_images = len(image_to_captions)
    success_count = 0
    failed_count = 0

    futures: dict[Future[tuple[str, str]], str] = {}

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="image-resize-worker") as executor:

        for image_name, image_data in image_to_captions.items():
            future = executor.submit(
                process_single_image,
                image_name,
                image_data,
                output_dir,
                target_size
            )

            futures[future] = image_name

        for completed_count, future in enumerate(as_completed(futures), start=1):
            image_name = futures[future]

            try:
                completed_image_name, resize_image_path = future.result()

                # Chỉ luồng chính cập nhật dictionary
                image_to_captions[completed_image_name]["resize_image_path"] = resize_image_path

                success_count += 1

                print(
                    f"[{completed_count}/{total_images}] "
                    f"Thành công: {completed_image_name}"
                )

            except Exception as error:
                failed_count += 1

                print(
                    f"[{completed_count}/{total_images}] "
                    f"Thất bại: {image_name}\n"
                    f"Nguyên nhân: {error}"
                )

    return success_count, failed_count


def resize_image() -> None:
    # Bước 1: Lấy danh sách đường dẫn ảnh từ JSON
    image_to_captions = load_image_mapping()

    print(f"Tổng số ảnh cần xử lý: {len(image_to_captions)}")
    print(f"Số worker thread: 4")
    print(f"Thư mục đầu ra: {Config.resize_image_dir}")
    print("-" * 60)

    success_count, failed_count = resize_images_with_threads(image_to_captions=image_to_captions)
    save_image_mapping(image_to_captions=image_to_captions)

    print("\n===== KẾT QUẢ =====")
    print(f"Tổng số ảnh: {len(image_to_captions)}")
    print(f"Resize thành công: {success_count}")
    print(f"Resize thất bại: {failed_count}")
    print(f"Đã cập nhật JSON: {MAPPING_FILE}")


def resize_image_validate() -> None:
    mapping_file = Path(Config.validation_data_file)

    if not mapping_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file JSON: {mapping_file}")

    with mapping_file.open("r", encoding="utf-8") as file:
        image_to_captions = json.load(file)

    if not isinstance(image_to_captions, dict):
        raise ValueError("Dữ liệu trong JSON phải có dạng dictionary/object.")

    print(f"Tổng số ảnh cần xử lý: {len(image_to_captions)}")
    print(f"Số worker thread: 4")
    print(f"Thư mục đầu ra: {Config.resize_image_dir}")
    print("-" * 60)

    success_count, failed_count = resize_images_with_threads(image_to_captions=image_to_captions)
    temporary_file = mapping_file.with_name(f"{mapping_file.name}.tmp")
    with temporary_file.open(mode="w", encoding="utf-8") as file:
        json.dump(image_to_captions, file, ensure_ascii=False, indent=2)

    temporary_file.replace(mapping_file)

    print("\n===== KẾT QUẢ =====")
    print(f"Tổng số ảnh: {len(image_to_captions)}")
    print(f"Resize thành công: {success_count}")
    print(f"Resize thất bại: {failed_count}")
    print(f"Đã cập nhật JSON: {Config.validation_data_file}")


def main() -> None:
    resize_image()


if __name__ == "__main__":
    main()
