from pathlib import Path
import shutil
import tarfile
import csv
import sys
import os
import pandas as pd

dataset_name = 'wake_vision'

#build dataset
path_to_dataset = Path(dataset_name)

folders_and_file_names = list()

#extract validation images metadata
img_names_0 = set()
img_names_1 = set()

# with open('cleaned_csvs/wv_validation_cleaned.csv', newline='') as f:
#     reader = csv.reader(f)
#     data = list(reader)
#     # print(len(data[1:]))
#     print(data.head())
#     for image_path, category, *_ in data[1:] :
#         if '0' == category :
#             img_names_0.add(Path(image_path))
#         elif '1' == category :
#             img_names_1.add(Path(image_path))
#         else :
#             print('Unknown image category')
#             exit()

labels = pd.read_csv('../cleaned_csvs/wv_validation_cleaned.csv')
cat_0 = labels[labels["clean_label"] == 0]["filename"].tolist()
cat_1 = labels[labels["clean_label"] == 1]["filename"].tolist()

img_names_0 = {Path(name) for name in cat_0}
img_names_1 = {Path(name) for name in cat_1}

folders_and_file_names.append({'folder': path_to_dataset / 'validation/0', 'file_names': img_names_0})
folders_and_file_names.append({'folder': path_to_dataset / 'validation/1', 'file_names': img_names_1})
print(f"Validation images: {len(img_names_0)} in category 0, {len(img_names_1)} in category 1")

#extract test images metadata
img_names_0 = set()
img_names_1 = set()

for element in folders_and_file_names :
    element['folder'].mkdir(parents=True)

path_to_unlabeled_images = path_to_dataset / 'unlabeled_images'
path_to_unlabeled_images.mkdir(parents=True)

#extract all compressed images and copy them in the corresponding folders
for zipped_file in Path('.').glob('*.tar.gz') :
    print(zipped_file)
    tar = tarfile.open(zipped_file, 'r:gz')
    tar.extractall()
    tar.close()
    images = set(Path('.').glob('*.jpg'))
    
    #copy extracted images to the respective folder
    for folder in folders_and_file_names :
        for image in images & folder['file_names'] :
            shutil.copy(image, folder['folder'])
   
    #gather unlabeled images
    for folder in folders_and_file_names :
        images = images - folder['file_names']
    for image in images :
        shutil.copy(image, path_to_unlabeled_images)
    
    #delete extracted images
    images = set(Path('.').glob('*.jpg'))
    for image in images : 
        image.unlink()

for zipped_file in Path('.').glob('*.tar.gz') :
    zipped_file.unlink()

for csv_file in Path('.').glob('*.csv') :
    csv_file.unlink()

# os.system(f"chmod 777 -R {dataset_name}")

print(f"Dataset saved in folder: {path_to_dataset}")