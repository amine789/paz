import os
from glob import glob
import json

import numpy as np
from tensorflow.keras.utils import Progbar

from ..abstract import Loader
from .utils import get_class_names


class FAT(Loader):
    """ Dataset loader for the falling things dataset (FAT).
    # Arguments
        path: String indicating full path to dataset
            e.g. /home/user/fat/
        split: String determining the data split to load.
            e.g. `train`, `val` or `test`
        class_names: `all` or list. If list it should contain as elements
            strings indicating each class name.

    # References
        - [Deep Object Pose
            Estimation (DOPE)](https://github.com/NVlabs/Deep_Object_Pose)
    """
    # TODO: Allow selection of class_names.
    def __init__(self, path, split = 'train', class_type = 'all'):
        self.class_type = class_type
        if class_type == 'all':
            self.class_names = get_class_names('FAT')
            self.class_to_arg = dict(
                                zip(self.class_names, list(range(len(self.class_names)))))
        else:
            self.class_names = class_type
            self.class_to_arg = {class_type: 0}
        self.split = split
        super(FAT, self).__init__(path, split, self.class_names, 'FAT')

    def load_data(self):
        if self.class_type == 'all':
            data = self.load_data_mixed()
        else:
            data = self.load_data_single()
        return data

    def load_data_mixed(self):
        scene_names = glob(self.path + 'mixed/*')
        image_paths, label_paths = [], []
        for scene_name in scene_names:
            scene_image_paths, scene_label_paths = [], []
            for image_side in ['left', 'right']:
                image_names = glob(scene_name + '/*%s.jpg' % image_side)
                side_image_paths = sorted(image_names, key=self._base_number)
                label_names = glob(scene_name + '/0*%s.json' % image_side)
                side_label_paths = sorted(label_names, key=self._base_number)
                scene_image_paths = scene_image_paths + side_image_paths
                scene_label_paths = scene_label_paths + side_label_paths
            image_paths = image_paths + scene_image_paths
            label_paths = label_paths + scene_label_paths

        data_dict = {}
        self.data = []
        progress_bar = Progbar(len(image_paths))
        for sample_arg, sample in enumerate(zip(image_paths, label_paths)):
            image_path, label_path = sample
            if not self._valid_name_match(image_path, label_path):
                raise ValueError('Invalid name match:', image_path, label_path)
            boxes = self._extract_boxes(label_path)
            if boxes is None:
                continue
            self.data.append({'image': image_path, 'boxes': boxes})
            progress_bar.update(sample_arg + 1)
        return self.data
        
    
    def load_data_single(self):
        object_name = self.class_names + '_16k'
        scene_names = glob(self.path + 'single/' + object_name + '/*')
        image_paths, label_paths = [], []
        for scene_name in scene_names:
            scene_image_paths, scene_label_paths = [], []
            for image_side in ['left', 'right']:
                image_names = glob(scene_name + '/*%s.jpg' % image_side)
                side_image_paths = sorted(image_names, key=self._base_number)
                label_names = glob(scene_name + '/0*%s.json' % image_side)
                side_label_paths = sorted(label_names, key=self._base_number)
                scene_image_paths = scene_image_paths + side_image_paths
                scene_label_paths = scene_label_paths + side_label_paths
            image_paths = image_paths + scene_image_paths
            label_paths = label_paths + scene_label_paths
        self.data = []
        progress_bar = Progbar(len(image_paths))
        for sample_arg, sample in enumerate(zip(image_paths, label_paths)):
            image_path, label_path = sample
            if not self._valid_name_match(image_path, label_path):
                raise ValueError('Invalid name match:', image_path, label_path)
            boxes = self._extract_boxes(label_path)
            if boxes is None:
                continue
            self.data.append({'image': image_path, 'boxes': boxes})
            progress_bar.update(sample_arg + 1)
        return self.data
        

    def split_data(self, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        total_ratio = train_ratio + val_ratio + test_ratio

        if total_ratio != 1.0:
            raise ValueError("Split ratio proportion is not correct")

        data_length = len(self.data)
        train_split = int(data_length * train_ratio)
        val_split = int(data_length * val_ratio)
        test_split = int(data_length * test_ratio)

        train_data = self.data[:train_split]
        val_data = self.data[train_split:train_split + val_split]
        test_data = self.data[-test_split:]
        return [train_data, val_data, test_data]

    def _extract_boxes(self, json_filename):
        json_data = json.load(open(json_filename, 'r'))
        num_objects = len(json_data['objects'])
        if num_objects == 0:
            return None
        box_data = np.zeros((num_objects, 5))
        for object_arg, object_data in enumerate(json_data['objects']):
            bounding_box = object_data['bounding_box']
            y_min, x_min = bounding_box['top_left']
            y_max, x_max = bounding_box['bottom_right']
            x_min, y_min = x_min / 960., y_min / 540.
            x_max, y_max = x_max / 960., y_max / 540.
            box_data[object_arg, :4] = x_min, y_min, x_max, y_max
            class_name = object_data['class'][:-4]
            box_data[object_arg, -1] = self.class_to_arg[class_name]
        return box_data

    def _base_number(self, filename):
        order = os.path.basename(filename)
        order = order.split('.')[0]
        order = float(order)
        return order

    def _valid_name_match(self, image_path, label_path):
        image_name = os.path.basename(image_path)
        label_name = os.path.basename(label_path)
        return image_name[:-3] == label_name[:-4]
