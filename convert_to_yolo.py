import os
import pandas as pd
import cv2 as cv
from tqdm import tqdm

# The path to directory containing the images
images_dir = os.path.join('dataset')

# The path to the directory containing all the annotation files for each image
test_annotations_file = 'test-annotations-bbox.csv'
train_annotations_file = 'oidv6-train-annotations-bbox.csv'

# Path to the file containing the labels for all the classes
class_description_file = 'class-descriptions-boxable.csv'
selected_class_names = ('Handgun', 'Rifle', 'Shotgun')

class_file = "guns.names"

classes_df = pd.read_csv(class_description_file, header=None, names=['class_label', 'class_name'])
selected_classes = classes_df.loc[classes_df['class_name'].isin(selected_class_names)]
# print(selected_classes)

with open(class_file, 'w') as class_file:
    class_file.writelines("\n".join(selected_classes['class_name'].to_list()))
    class_file.close()

test_df = pd.read_csv(test_annotations_file)
train_df = pd.read_csv(train_annotations_file)
train_df.drop(['XClick1X', 'XClick2X', 'XClick3X', 'XClick4X', 'XClick1Y', 'XClick2Y', 'XClick3Y', 'XClick4Y'], axis=1, inplace=True)

annonation_df = test_df.append(train_df)
# print(annonation_df)

path, dirs, _ = next(os.walk(images_dir))

for dir in dirs:
    dir_path, _, files = next(os.walk(os.path.join(path, dir)))

    for file in tqdm(files):

        if file.split(".")[-1] != 'jpg':
            break

        img = cv.imread(os.path.join(dir_path, file))

        image_id = file[0:-4]
        # print(image_id)

        img_height, img_width, channel = img.shape
        # print(f"width:{img_width}, height:{img_height}")

        anno_file = image_id + ".txt"

        image_annon_df = annonation_df.loc[((annonation_df['ImageID'] == image_id) & (annonation_df['LabelName'].isin(selected_classes['class_label'].to_list()))), ]
        # print(image_annon_df)

        # with open(os.path.join(dir_path, anno_file), 'w') as f:
        with open(os.path.join(path, "annotations_1class", dir, anno_file), 'w') as f:
            img_annotations = []
            for idx, row in image_annon_df.iterrows():

                class_label = row['LabelName']
                # print(class_label)

                box_left = row['XMin']*img_width
                box_top = row['YMin']*img_height
                box_right = row['XMax']*img_width
                box_bottom = row['YMax']*img_height

                box_width, box_height = box_right - box_left, box_bottom - box_top

                center_x_ratio, center_y_ratio = float((box_left + int(box_width / 2)) / img_width), float(
                    (box_top + int(box_height / 2)) / img_height)
                width_ratio, height_ratio = float(box_width / img_width), float(box_height / img_height)

                # print(f"class: {class_name} - x_min:{box_left}, y_min:{box_top}, x_max:{box_right}, y_max:{box_bottom}")

                class_id = selected_classes['class_label'].to_list().index(class_label)
                class_id = 0

                # print(f"{class_id} {center_x_ratio} {center_y_ratio} {width_ratio} {height_ratio}")
                img_annotations.append(f"{class_id} {center_x_ratio} {center_y_ratio} {width_ratio} {height_ratio}")

            f.writelines("\n".join(img_annotations))
            f.close()
