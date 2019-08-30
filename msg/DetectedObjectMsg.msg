# DetectedObject

geometry_msgs/Point point   # 3D point with header timestamp
float32 confidence          # confidence of detection. 0-1
string class_name           # class label of object. example: person
uint16[] bbox               # 2D bbox: centre_x, centre_y, w, h