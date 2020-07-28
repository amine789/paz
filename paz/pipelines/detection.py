import numpy as np

from .. import processors as pr
from ..abstract import SequentialProcessor, Processor
from ..models import SSD512, SSD300, HaarCascadeDetector
from ..datasets import get_class_names

from .image import AugmentImage, PreprocessImage
from .classification import XceptionClassifierFER


class AugmentBoxes(SequentialProcessor):
    """Perform data augmentation with bounding boxes.
    # Arguments
        mean: List of three elements used to fill empty image spaces.
    """
    def __init__(self, mean=pr.BGR_IMAGENET_MEAN):
        super(AugmentBoxes, self).__init__()
        self.add(pr.ToAbsoluteBoxCoordinates())
        self.add(pr.Expand(mean=mean))
        self.add(pr.RandomSampleCrop())
        self.add(pr.RandomFlipBoxesLeftRight())
        self.add(pr.ToNormalizedBoxCoordinates())


class PreprocessBoxes(SequentialProcessor):
    """Preprocess bounding boxes

    # Arguments
        num_classes: Int.
        prior_boxes: Numpy array of shape ''[num_boxes, 4]'' containing
            prior/default bounding boxes.
        IOU: Float. Intersection over union used to match boxes.
        variances: List of two floats indicating variances to be encoded
            for encoding bounding boxes.
    """
    def __init__(self, num_classes, prior_boxes, IOU, variances):
        super(PreprocessBoxes, self).__init__()
        self.add(pr.MatchBoxes(prior_boxes, IOU),)
        self.add(pr.EncodeBoxes(prior_boxes, variances))
        self.add(pr.BoxClassToOneHotVector(num_classes))


class AugmentDetection(SequentialProcessor):
    """Augment boxes and images for object detection.
    # Arguments
        prior_boxes: Numpy array of shape ''[num_boxes, 4]'' containing
            prior/default bounding boxes.
        split: Flag from ''paz.processors.TRAIN'', ''paz.processors.VAL''
            or ''paz.processors.TEST''. Certain transformations would take
            place depending on the flag.
        num_classes: Int.
        size: Int. Image size.
        mean: List of three elements indicating the per channel mean.
        IOU: Float. Intersection over union used to match boxes.
        variances: List of two floats indicating variances to be encoded
            for encoding bounding boxes.
    """
    def __init__(self, prior_boxes, split=pr.TRAIN, num_classes=21, size=300,
                 mean=pr.BGR_IMAGENET_MEAN, IOU=.5, variances=[.1, .2]):
        super(AugmentDetection, self).__init__()

        self.augment_image = AugmentImage()
        self.augment_image.insert(0, pr.LoadImage())
        self.augment_image.add(pr.ConvertColorSpace(pr.RGB2BGR))
        self.preprocess_image = PreprocessImage((size, size), mean)

        self.augment_boxes = AugmentBoxes()
        args = (num_classes, prior_boxes, IOU, variances)
        self.preprocess_boxes = PreprocessBoxes(*args)

        self.add(pr.UnpackDictionary(['image', 'boxes']))
        if split == pr.TRAIN:
            self.add(pr.ControlMap(self.augment_image, [0], [0]))
            self.add(pr.ControlMap(self.augment_boxes, [0, 1], [0, 1]))
        self.add(pr.ControlMap(self.preprocess_image, [0], [0]))
        self.add(pr.ControlMap(self.preprocess_boxes, [1], [1]))
        self.add(pr.SequenceWrapper(
            {0: {'image': [size, size, 3]}},
            {1: {'boxes': [len(prior_boxes), 4 + num_classes]}}))


class SingleShotPrediction(Processor):
    """Single-shot object detection prediction.

    # Arguments
        model: Keras model.
        class_names: List of strings indicating the class names.
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        mean: List of three elements indicating the per channel mean.
    """
    def __init__(self, model, class_names, score_thresh, nms_thresh,
                 mean=pr.BGR_IMAGENET_MEAN):
        self.model = model
        self.class_names = class_names
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

        super(SingleShotPrediction, self).__init__()
        preprocessing = SequentialProcessor(
            [pr.ResizeImage(self.model.input_shape[1:3]),
             pr.ConvertColorSpace(pr.RGB2BGR),
             pr.SubtractMeanImage(mean),
             pr.CastImage(float),
             pr.ExpandDims(axis=0)])
        postprocessing = SequentialProcessor(
            [pr.Squeeze(axis=None),
             pr.DecodeBoxes(self.model.prior_boxes, variances=[.1, .2]),
             pr.NonMaximumSuppressionPerClass(self.nms_thresh),
             pr.FilterBoxes(self.class_names, self.score_thresh)])
        self.predict = pr.Predict(self.model, preprocessing, postprocessing)

        self.denormalize = pr.DenormalizeBoxes2D()
        self.draw = pr.DrawBoxes2D(self.class_names)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        boxes2D = self.predict(image)
        boxes2D = self.denormalize(image, boxes2D)
        image = self.draw(image, boxes2D)
        return self.wrap(image, boxes2D)


class SSD512COCO(SingleShotPrediction):
    def __init__(self, score_thresh=0.60, nms_thresh=0.45):
        """Single-shot inference pipeline with SSD512 trained on COCO.
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        """
        model = SSD512()
        names = get_class_names('COCO')
        super(SSD512COCO, self).__init__(
            model, names, score_thresh, nms_thresh)


class SSD512YCBVideo(SingleShotPrediction):
    def __init__(self, score_thresh=0.60, nms_thresh=0.45):
        """Single-shot inference pipeline with SSD512 trained on YCBVideo.
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        """
        model = SSD512(weights='YCBVideo')
        names = get_class_names('YCBVideo')
        super(SSD512YCBVideo, self).__init__(
            model, names, score_thresh, nms_thresh)


class SSD300VOC(SingleShotPrediction):
    def __init__(self, score_thresh=0.60, nms_thresh=0.45):
        """Single-shot inference pipeline with SSD300 trained on VOC.
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        """
        model = SSD300()
        names = get_class_names('VOC')
        super(SSD300VOC, self).__init__(model, names, score_thresh, nms_thresh)


class SSD300FAT(SingleShotPrediction):
    def __init__(self, score_thresh=0.60, nms_thresh=0.45):
        """Single-shot inference pipeline with SSD300 trained on FAT.
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        """
        model = SSD300(22, 'FAT', 'FAT')
        names = get_class_names('FAT')
        super(SSD300FAT, self).__init__(model, names, score_thresh, nms_thresh)


class HaarCascadePrediction(Processor):
    """HaarCascade prediction pipeline/function from RGB-image.

    # Arguments
        detector: An instantiated HaarCascadeDetector model.
        offsets: List of two elements. Each element must be between [0, 1].
        class_names: List of strings.
        draw: Boolean flag. If ``True`` the prediction will be drawn
            in the image.

    # Returns
        A function for predicting bounding box detections.
    """
    def __init__(self, detector, class_names=None, colors=None, draw=True):
        super(HaarCascadePrediction, self).__init__()
        self.detector = detector
        self.class_names = class_names
        self.colors = colors
        self.draw = draw
        RGB2GRAY = pr.ConvertColorSpace(pr.RGB2GRAY)
        postprocess = SequentialProcessor()
        postprocess.add(pr.ToBoxes2D(self.class_names))
        self.predict = pr.Predict(self.detector, RGB2GRAY, postprocess)
        self.draw_boxes2D = pr.DrawBoxes2D(self.class_names, self.colors)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        boxes2D = self.predict(image)
        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
        return self.wrap(image, boxes2D)


class HaarCascadeFrontalFace(HaarCascadePrediction):
    """HaarCascade pipeline for detecting frontal faces
    """
    def __init__(self, class_name='Face', color=[0, 255, 0], draw=True):
        self.model = HaarCascadeDetector('frontalface_default', class_arg=0)
        super(HaarCascadeFrontalFace, self).__init__(
            self.model, [class_name], [color], draw)


EMOTION_COLORS = [[255, 0, 0], [45, 90, 45], [255, 0, 255], [255, 255, 0],
                  [0, 0, 255], [0, 255, 255], [0, 255, 0]]


class XceptionDetectionFER(Processor):
    """Emotion classification and detection pipeline.

    # Returns
        Dictionary with ``image`` and ``boxes2D``.

    # References
       - [Real-time Convolutional Neural Networks for Emotion and
            Gender Classification](https://arxiv.org/abs/1710.07557)
    """
    def __init__(self, offsets=[0, 0], colors=EMOTION_COLORS):
        super(XceptionDetectionFER, self).__init__()
        self.offsets = offsets
        self.colors = colors

        # detection
        self.detect = HaarCascadeFrontalFace()
        self.square = SequentialProcessor()
        self.square.add(pr.SquareBoxes2D())
        self.square.add(pr.OffsetBoxes2D(offsets))
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()

        # classification
        self.classify = XceptionClassifierFER()

        # drawing and wrapping
        self.class_names = self.classify.class_names
        self.draw = pr.DrawBoxes2D(self.class_names, self.colors, True)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        boxes2D = self.detect(image.copy())['boxes2D']
        boxes2D = self.square(boxes2D)
        boxes2D = self.clip(image, boxes2D)
        cropped_images = self.crop(image, boxes2D)
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            predictions = self.classify(cropped_image)
            box2D.class_name = predictions['class_name']
            box2D.score = np.amax(predictions['scores'])
        image = self.draw(image, boxes2D)
        return self.wrap(image, boxes2D)
