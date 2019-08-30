#!/usr/bin/env python3
# license: http://opencv.org/license.html


import sys
import numpy as np
import os.path
import cv2

from object_detectors.detected_object import DetectedObject
from utils import utils


class DarknetObjectDetector:

    def __init__(self, config_file=None):
        if (config_file):    
            self.load_configs(config_file)
        self.load_darknet_model()
        self.inference_time_s = -1

    def load_configs(self, config_file):
        self.darknet_config = utils.get_config(config_file)

        self.conf_threshold = self.darknet_config['conf_threshold']
        self.nms_threshold = self.darknet_config['nms_threshold']
        self.inp_width = self.darknet_config['inp_width']
        self.inp_height = self.darknet_config['inp_height']

        self.model_configuration = utils.get_abs_path(self.darknet_config['model_configuration'])
        self.model_weights = utils.get_abs_path(self.darknet_config['model_weights'])

        print("ML Configs loaded successfully")

    def load_darknet_model(self):
        self.net = cv2.dnn.readNetFromDarknet(self.model_configuration, self.model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("ML Model loaded successfully")


    # Get the names of the output layers
    def get_outputs_names(self):
        # Get the names of all the layers in the network
        layersNames = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, frame, outs):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        self.detected_objects = []

        class_ids = []
        confidences = []
        bboxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    bbox = [left, top, width, height]

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    bboxes.append(bbox)


        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(bboxes, confidences, self.conf_threshold, self.nms_threshold)
        for idx in indices:
            idx = idx[0]

            self.detected_objects.append(DetectedObject(class_ids[idx], confidences[idx], bboxes[idx]))



    # frame is opencv mat frame, return list of DetectedObject
    def detect(self, frame):

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1/255, (self.inp_width, self.inp_height), [0,0,0], 1, crop=False)

        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.get_outputs_names())

        # Remove the bounding boxes with low confidence
        self.postprocess(frame, outs)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = self.net.getPerfProfile()
        self.inference_time_s = t * 1.0 / cv2.getTickFrequency()

        return self.detected_objects

    # return time in seconds. time <= 0 implies ML not running
    def get_inference_time(self):
        return self.inference_time_s
