# object-detection

This is a 3D Object Detection and Localization ROS package that supports development of custom computer vision detectors
and fusion with 3D pointclouds to achieve object localization. A sample deep learning based detector (yolov3 with 
opencv bindings) has been included to demo this package.



## Dependencies

ROS: Melodic (Kinetic should work but untested)

python >= 3.6

opencv >= 3.4.2 (to run the demo yolov3 detector from cv2.dnn)


## Real Life Demo

<img src="/demo_images/object_detection_demo.png" alt="Object Detection Sample Output"/>

Demo was performed using an Intel RealSense D435, which has an RGB camera as well as infrared cameras for distance estimation.

As can be seen, a bench was detected a distance away and the yolov3 detector was used to detect the bench in the RGB space (2D image).
The realsense also provdes a pointcloud (that is conveniently aligned with the RGB camera frame) and hence we are able to fuse
the xyz and object detection information to localize the object as well. Given that the RealSense D435 utilizes active infrared
stereo for depth information, it is not very accurate at faraway distances. But regardless, it is still able to provide a fairly
decent estimate of the bench (to 80cm accuracy).

Moreover, the pointcloud and RGB camera data were not synchronized and hence software level synchronization is performed to 
ensure higher accuracy of xyz coordinates. 



## How to run
1. First build the project.

   `catkin install`

2. If there were errors during the build process, refer to [Build/Installation Problems](https://github.com/HashirZahir/object-detection/new/master?readme=1#buildinstallation-problems).
Otherwise, proceed to run your rosbag / connect your camera to stream the RGB and Pointcloud data.

3. Double check that the ROS topics specified in `config/ros_config.yaml` match the RBG camera and pointcloud topics.

4. [Download](config/darknet_resources/README) the relevant yolov3 config files (classes, configs and weights of the model) as they too large to host on Github. 

5. Run the object detection module.

   `source install/setup.bash && rosrun object_detection main`

6. Monitor the output and ensure the configs and subscribers load successfully. Otherwise, recheck the values in `config/ros_config.yaml`

   <img src="/demo_images/object_detection_demo_startup.png" alt="Object Detection Initialization messages"/>

7. Detections should be coming in if the camera contains people, laptops, keyboards, smartphones, etc. The demo darknet yolov3
detector can detect around 80 classes it can detect. 
   
   <img src="/demo_images/object_detection_demo_output.png" alt="Object Detection Text output"/>

   Notice that there are some detections with a x:0,y:0,z:0 output. This is because even though the camera RGB and Pointcloud data
was aligned, there was simply no pointcloud data at the place where the bench was detected and hence a 0,0,0 was returned instead.
This 0,0,0 result was not discarded as on embeded systems, the deep learning object detector could take a long time to provide 
a detection (aka low frame/sec rate) and hence skipping the frame would waste precious data that is already coming in very slowly.
If needed, this behaviour can be modified.



## Customizing your own detectors
This package was made with customizability and reusability in mind. To create a new detector, take a look at [object_detector_main.py](src/object_detectors/object_detector_main.py) 
which calls the object detector specified in the [object detector config file](config/object_detector_config.yaml).


All object detectors must implement 3 functions, 
`__init__(optional config_file)` , `detect(frame)` and `get_inference_time()`
with `detect(frame)` returning a list of `DetectedObject` type objects. 

Simply create a python file in `src/object_detectors/` and override these 3 functions. Following this, edit the `config/object_detector_config.yaml` 
to allow the package to load your new detector. You can also specify your own custom config file specific to your detector 
`object_detector_config: darknet_object_detector_config.yaml`. 

This config file should be created in `config/` folder and can be also used to load any additional resources (example the `config/darknet_object_detector_config.yaml` 
file loads the classes, configs and weights for yolov3).

## Future Work
- Support larger range of 3D data formats
- Add a tracker to prevent recurrent detections
- Allow package to run in object detection mode only OR object detection + localization mode (currently only the latter is supported)
- Add a non deep learning based example (using basic opencv functions)
- Make package python2 compatible (easier ROS support, although python2 is reaching end of life in Jan 2020)



## Build/Installation Problems
If you faced any ROS cv_bridge errors, it is likely that you will have to build the [cv_bridge module](https://github.com/ros-perception/vision_opencv) 
for python3 before this package can work.


Just follow this [stackoverflow solution](https://stackoverflow.com/questions/49221565/unable-to-use-cv-bridge-with-ros-kinetic-and-python3?rq=1)



## Credit
Credit to [Intel RealSense](https://github.com/IntelRealSense/realsense-ros) for RealSense D435 ROS wrapper.
Credit to [pjreddie's yolov3](https://pjreddie.com/darknet/yolo/) for a simple and easy to deploy deep learning detector
for a sample usage of this object detection package.
