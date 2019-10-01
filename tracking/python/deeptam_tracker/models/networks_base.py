from abc import ABC, abstractmethod
import tensorflow as tf

"""
networks_base.py
----------------

classes:
    1. TrackingNetworkBase
"""


class TrackingNetworkBase(ABC):
    """
    TrackingNetworkBase
    -------------------

    parents:
        1. ABC

    members (variables):
        1. _placeholders

    members (functions):
        1. constructor __init()___
        2. property placeholders()
        3. abstractmethod build_net()
    """

    def __init__(self, batch_size=1):
        """
        - note:
        - input:
            - batch_size: int
        - return: dictionary of placeholders
                - 6 entries (all tf.float32):
                    - depth_key:
                    - image_key:image_current:
                    - intrinsics:
                    - prev_rotation:
                    - prev_tanslation:
        - functionality:
            - create placeholders for the network
        """
        self._placeholders = {
            'depth_key': tf.placeholder(tf.float32, shape=(batch_size, 1, 96, 128)),
            'image_key': tf.placeholder(tf.float32, shape=(batch_size, 3, 96, 128)),
            'image_current': tf.placeholder(tf.float32, shape=(batch_size, 3, 96, 128)),
            'intrinsics': tf.placeholder(tf.float32, shape=(batch_size, 4)),
            'prev_rotation': tf.placeholder(tf.float32, shape=(batch_size, 3)),
            'prev_translation': tf.placeholder(tf.float32, shape=(batch_size, 3)),
        }

    @property
    def placeholders(self):
        """
        1. __init__():
            - note:
            - input: -
            - return: dictionary of the placeholders
                - All placeholders required for feeding this network
            - functionality:
                - to avail the placeholders of the network
        """
        return self._placeholders

    @abstractmethod
    def build_net(self, depth_key, image_key, image_current, intrinsics, prev_rotation, prev_translation):
        """
        2. build_net():
            - note:
            - input:
                - depth_key: the depth map of the key frame
                - image_key: the image of the key frame
                - image_current: the current image
                - intrinsics: the camera intrinsics
                - prev_rotation: the current guess for the camera rotation as angle axis representation
                - prev_translation: the current guess for the camera translation
            - return:
                - network outputs as a dict.
                - The following must be returned:
                    - predict_rotation
                    - predict_translation
            - functionality:
                - Build the tracking network
        """
        pass


""" End of the script """
