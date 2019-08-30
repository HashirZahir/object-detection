#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2 as pcl2
from geometry_msgs.msg import Point
import message_filters

from object_detection.msg import DetectedObjectMsg, DetectedObjectsMsg
from utils import utils
from object_detectors.object_detector_main import ObjectDetector

# main detection class that listens to image and pointcloud topics to output detections
# look at config/ros_config.yaml for topic and other ros parameters

class ObjectDetector3D: 
    def __init__(self):
        self.load_configs()
        rospy.loginfo("Waiting for ML model to load")
        self.object_detector = ObjectDetector()
        self.init_subscribers()

        self.init_publishers()
        self.bridge = CvBridge()
        self.detected_objects = []

    def load_configs(self):
        self.ros_config = utils.get_config('ros_config.yaml')
        self.subscribers = self.ros_config['subscribers']
        self.publishers = self.ros_config['publishers']

        rospy.loginfo("ROS configs loaded successfully")

    def init_subscribers(self):
        camera_2d = self.subscribers['camera_2d']
        camera_3d = self.subscribers['camera_3d']

        self.image_sub = message_filters.Subscriber(camera_2d['topic'], Image, buff_size=2**28)
        self.pointcloud_sub = message_filters.Subscriber(camera_3d['topic'], PointCloud2, buff_size=2**28)

        # try to synchronize the 2 topics as closely as possible for higher XYZ accuracy
        self.time_sync = message_filters.ApproximateTimeSynchronizer([self.image_sub, \
            self.pointcloud_sub], queue_size=self.subscribers['synchronized_queue_size'], slop=0.03)
        self.time_sync.registerCallback(self.callback)

        rospy.loginfo("Subscribers Synchronized Successfully")

    def init_publishers(self):
        self.detected_objects_3d_pos_pub = rospy.Publisher(self.publishers['detected_objects']['topic'], \
                DetectedObjectsMsg, queue_size=self.publishers['detected_objects']['queue_size'])

        if (self.ros_config['enable_debug_image']):
            self.debug_img_pub = rospy.Publisher(self.publishers['debug_image']['topic'], \
                Image, queue_size=self.publishers['debug_image']['queue_size'])

    def image_cb(self, ros_img):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(ros_img, "bgr8")
            self.cv_image_timestamp = ros_img.header.stamp
            self.process()
            self.publish_output()
        except CvBridgeError as e:
            print(e)
        

    def pointcloud_cb(self, ros_pointcloud):
        self.pointcloud = ros_pointcloud
        self.pointcloud_timestamp= ros_pointcloud.header.stamp

    def callback(self, ros_img, ros_pointcloud):
        self.image_cb(ros_img)
        self.pointcloud_cb(ros_pointcloud)

    # perform object detection and calculate xyz coordinate of object from pointcloud
    def process(self):
        self.detected_objects = []

        horz_bar_text = "-"*20
        rospy.loginfo(horz_bar_text)

        if hasattr(self,'pointcloud'):
            time_diff = abs(self.cv_image_timestamp-self.pointcloud_timestamp)
            rospy.logdebug("Synchronization Time Delay: {:.4f}".format(time_diff.nsecs/10.0**9))

        max_time_difference = rospy.Duration(self.subscribers['absoulte_synchronized_time_diff'])

        # double check time difference as ApproximateSync algo finds best match only, 
        # may exceed max synchronization delay threshold sometimes
        if (hasattr(self,'pointcloud') and abs(self.cv_image_timestamp-self.pointcloud_timestamp) < max_time_difference):
            # Note: ML detection can take significant time
            self.detected_objects = self.object_detector.detect(self.cv_image) 

            for detected_object in self.detected_objects:
                xyz_coord = list(pcl2.read_points(self.pointcloud, skip_nans=True, \
                   field_names=("x", "y", "z"), uvs=[detected_object.center]))

                detected_object.set_xyz(xyz_coord)
                rospy.loginfo(detected_object)    

        else:
            rospy.logwarn("Synchronization Issue\nImage and PointCloud not Synchronized")

    # publishes debug image (if specified in config/ros_config.yaml) and detected_objects custom message
    def publish_output(self):
        
        if (len(self.detected_objects) > 0):

            objects_msgs = DetectedObjectsMsg()
            header = Header()

            header.stamp = self.pointcloud_timestamp
            header.frame_id = "detected_object"

            objects_msgs.header = header

            for detected_object in self.detected_objects:
                point = Point()
                point.x, point.y, point.z = detected_object.xyz_coord

                msg = DetectedObjectMsg()
                msg.point = point
                msg.confidence = detected_object.confidence
                msg.class_name = detected_object.class_name
                msg.bbox = detected_object.centered_bbox
                objects_msgs.detected_objects_msgs.append(msg)

            self.detected_objects_3d_pos_pub.publish(objects_msgs)

        if (self.ros_config['enable_debug_image']):

            inference_time_s = self.object_detector.get_inference_time()
            inference_speed_text = "{0:.2f} fps".format(1/inference_time_s)
            cv2.putText(self.cv_image, inference_speed_text, (0,20), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

            for detected_object in self.detected_objects:
                cv2.rectangle(self.cv_image, *detected_object.cv2_rect, (0,255,0), 3)

                xyz_coord = str(["{:.2f}".format(pos) for pos in detected_object.xyz_coord])

                cv2.putText(self.cv_image, detected_object.class_name, \
                    (detected_object.center[0], detected_object.center[1]), \
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                cv2.putText(self.cv_image, xyz_coord, \
                    (detected_object.center[0], detected_object.center[1]+20), \
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
                
            

            self.debug_img_pub.publish(self.bridge.cv2_to_imgmsg(self.cv_image, "bgr8"))

        


# start ros node and spin up ros subscribers
def main():
    rospy.loginfo("Starting 3D Object Detector...")
    rospy.init_node('object_detection_3d', anonymous=True, log_level=rospy.INFO) # set to rospy.DEBUG for debugging

    object_detector_3d = ObjectDetector3D()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")


if __name__ == "__main__":
    main()