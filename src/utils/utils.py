#!/usr/bin/env python3

import os
import yaml
import rospkg

# Returns config file located in package
def get_config(name, package='object_detection', folder=None):
    rospack = rospkg.RosPack()
    if folder is not None:
        filepath = os.path.join(rospack.get_path(package), 'config', folder, name)
    else:
        filepath = os.path.join(rospack.get_path(package), 'config', name)
    with open(filepath, 'r') as stream:
        return yaml.load(stream)

# returns absolute path of rel_path joined with package path
def get_abs_path(rel_path, package='object_detection'):
    rospack = rospkg.RosPack()
    return os.path.join(rospack.get_path(package), rel_path)