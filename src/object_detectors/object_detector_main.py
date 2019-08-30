from utils import utils
from pydoc import locate

# all object detectors must implement 3 functions, __init__(optional config_file), detect(frame) and get_inference_time()

class ObjectDetector:
    def __init__(self):
        config = utils.get_config('object_detector_config.yaml')
        module_name = config['object_detector_module']
        object_detector_config = config['object_detector_config']

        ObjectDetectorClass = locate(module_name)
        self.object_detector = ObjectDetectorClass(object_detector_config)

    # frame is opencv mat frame, return list of DetectedObject
    def detect(self, frame):
        return self.object_detector.detect(frame)

    # returns inference time in seconds
    def get_inference_time(self):
        return self.object_detector.get_inference_time()