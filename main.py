from pathlib import Path
from shutil import copyfile, rmtree, make_archive
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dd', help='dataset directory that contains images and labels',
                        default='dataset_name', type=str)
    parser.add_argument('--cl', help='classes file', default='classes.txt', type=str)
    parser.add_argument('--dn', help='dataset name', default='dataset_name', type=str)
    parser.add_argument('--sp', help='split percentage', default=0.2, type=float)
    parser.add_argument('--save', help='save directory', default='datasets', type=str)
    parser.add_argument('--zip', help='zip dataset', default=False, type=bool)
    return parser.parse_args()


def make_yaml(class_list):
    class_names = ""
    for index, class_name in enumerate(class_list):
        class_names += f'  {index}: {class_name}\n'
    yaml = f"""train: images/train
val: images/val
test: images/val

names:
{class_names}
"""
    return yaml


def get_labels(label_file, class_list):
    labels = {}
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            label = int(line.split()[0])
            label_name = class_list[label]
            if label_name not in labels:
                labels[label_name] = 1
            else:
                labels[label_name] += 1
    return labels


def read_dataset(dataset_dir, class_list):
    dataset = []

    # Get image files
    image_files = dataset_dir.glob('**/*.jpg')

    # Shuffle image files
    image_files = list(image_files)
    random.shuffle(image_files)
    print(f"Number of images: {len(image_files)}")

    for image in image_files:
        # Get label file
        label_file = image.parent / f'{image.stem}.txt'
        labels = {}
        if label_file.exists():
            # Get labels
            labels = get_labels(label_file, class_list)
        dataset.append({'image_file': str(image), 'labels': labels, 'label_file': str(label_file)})
    return dataset


def main():
    args = parse_args()
    print("Arguments:")
    print(args)

    dataset_dir = Path(args.dd)
    classes_file = Path(args.cl)
    dataset_name = args.dn
    split_percentage = args.sp
    save_dir = Path(args.save)
    zip_dataset = args.zip

    print("Making dataset...")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Classes file: {classes_file}")
    print(f"Dataset name: {dataset_name}")
    print(f"Split percentage: {split_percentage}")
    print(f"Save directory: {save_dir}")

    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    if (save_dir / dataset_name).exists():
        print(f"Dataset {dataset_name} already exists. Press y to delete existing dataset and continue: ", end="")
        command = input()
        if command != 'y' and command != 'Y':
            print("Exiting...")
            exit(0)
        else:
            print("Deleting existing dataset...")
            rmtree(save_dir / dataset_name)

    print("Making Folders...")
    # Create images directory
    images_dir = save_dir / dataset_name / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create train, val, test directories
    images_train_dir = images_dir / 'train'
    images_train_dir.mkdir(parents=True, exist_ok=True)
    images_val_dir = images_dir / 'val'
    images_val_dir.mkdir(parents=True, exist_ok=True)

    # Create labels directory
    labels_dir = save_dir / dataset_name / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Create train, val, test directories
    labels_train_dir = labels_dir / 'train'
    labels_train_dir.mkdir(parents=True, exist_ok=True)
    labels_val_dir = labels_dir / 'val'
    labels_val_dir.mkdir(parents=True, exist_ok=True)

    # Read classes file
    class_list = classes_file.read_text().splitlines()
    print(f"Number of classes: {len(class_list)} | {class_list}")

    # Make yaml file
    yaml_file = save_dir / dataset_name / f'{dataset_name}.yaml'
    with open(yaml_file, 'w') as f:
        f.write(make_yaml(class_list))

    # Read dataset
    dataset = read_dataset(dataset_dir, class_list)

    # Organize dataset
    organized_dataset = {'unlabelled': []}
    for cls in class_list:
        organized_dataset[cls] = []
    for data in dataset:
        if len(data['labels']) > 0:
            for label in data['labels']:
                organized_dataset[label].append(data)
        else:
            organized_dataset['unlabelled'].append(data)

    # Split dataset
    train_dataset = []
    val_dataset = []
    class_counts = {}
    for cls in organized_dataset:
        print(f"Number of {cls} images: {len(organized_dataset[cls])}")
        class_dataset = organized_dataset[cls]
        split_index = int(len(class_dataset) * split_percentage)
        train_dataset += class_dataset[split_index:]
        val_dataset += class_dataset[:split_index]
        class_counts[cls] = len(class_dataset)
        class_counts[f'{cls}_train'] = len(class_dataset) - split_index
        class_counts[f'{cls}_val'] = split_index

    print("")
    print(f"Number of train images: {len(train_dataset)}")
    print(f"Number of val images: {len(val_dataset)}")
    print(f"Class counts: \n{class_counts}")

    # Copy images and labels
    for data in train_dataset:
        image_file = Path(data['image_file'])
        label_file = Path(data['label_file'])
        copyfile(image_file, images_train_dir / image_file.name)
        if label_file.exists():
            copyfile(label_file, labels_train_dir / label_file.name)

    for data in val_dataset:
        image_file = Path(data['image_file'])
        label_file = Path(data['label_file'])
        copyfile(image_file, images_val_dir / image_file.name)
        if label_file.exists():
            copyfile(label_file, labels_val_dir / label_file.name)

    if zip_dataset:
        print("")
        if (save_dir / f'{dataset_name}.zip').exists():
            print(f"Zip file {dataset_name}.zip already exists. Press y to delete existing zip file and continue: ", end="")
            command = input()
            if command != 'y' and command != 'Y':
                print("Exiting...")
                exit(0)
            else:
                print("Deleting existing zip file...")
                (save_dir / f'{dataset_name}.zip').unlink()
        print("Zipping dataset...")
        make_archive(save_dir / dataset_name, 'zip', save_dir, dataset_name)

    print("Done!")


if __name__ == '__main__':
    main()
