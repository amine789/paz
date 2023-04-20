import cv2
import numpy as np
import tensorflow as tf

from paz.abstract import SequentialProcessor

from mask_rcnn.model.model import MaskRCNN
from mask_rcnn.datasets.shapes import Shapes
from mask_rcnn.evaluation.inference_graph import InferenceGraph
from mask_rcnn.pipelines.detection import ResizeImages, NormalizeImages
from mask_rcnn.pipelines.detection import Detect, PostprocessInputs

from mask_rcnn.utils import norm_boxes_graph, display_instances


def test(image, weights_path):
    resize = SequentialProcessor([ResizeImages(128, 0, 128)])
    molded_images, windows = resize([image])
    image_shape = molded_images[0].shape
    window = norm_boxes_graph(windows[0], image_shape[:2])

    base_model = MaskRCNN(model_dir='../../mask_rcnn', image_shape=image_shape, backbone="resnet101",
                          batch_size=1, images_per_gpu=1, rpn_anchor_scales=(8, 16, 32, 64, 128),
                          train_rois_per_image=32, num_classes=4, window=window)

    inference_model = base_model.build_inference_model()

    base_model.keras_model = inference_model
    base_model.keras_model.load_weights(weights_path, by_name=True)
    preprocess = SequentialProcessor([ResizeImages(128, 0, 128),
                                      NormalizeImages()])
    postprocess = SequentialProcessor([PostprocessInputs()])
    detect = Detect(base_model, (8, 16, 32, 64, 128), 1, preprocess, postprocess)
    result = detect([image])
    return result


path = '/Users/poornimakaushik/Desktop/mask_rcnn/weights.20-0.43.hdf5'  # Weigths file after training

dataset_train = Shapes(1, (128, 128))
data = dataset_train.load_data()
images = data[0]['image']

class_names = ['BG', 'Square', 'Circle', 'Triangle']
results = test(images, path)
r = results[0]
print(r)
display_instances(images, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
