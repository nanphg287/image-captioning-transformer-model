from data.prepare_flicrk8k_datasets import prepare_dataset
from data.resize_image import resize_image, resize_image_validate
from data.split_dataset import split_dataset


def prepare_training_data() -> None:
    prepare_dataset()
    split_dataset()
    resize_image()
    resize_image_validate()

if __name__ == '__main__':
    prepare_training_data()