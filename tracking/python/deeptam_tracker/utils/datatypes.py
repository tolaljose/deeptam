from collections import namedtuple

"""
vis_utils.py
----------------
Globals:
    1. View
    2. Pose
functions:
    1. Pose_identity
"""

"""
1. View: (Tuple with named fields)
    - name: View
    - fields: (all about the view under consideration)
        - R: camera rotation
        - t: camera translation
        - K: image dimensions (width and height of image/depth)
        - image: a PIL.Image
        - depth: the absolute depth values (not inverse depth)
        - depth_metric: 'camera_z'

"""
View = namedtuple('View', ['R', 't', 'K', 'image', 'depth', 'depth_metric'])

"""
2. Pose: (Tuple with named fields)
    - name: Pose
    - fields: (camera pose)
        - R: camera rotation
        - t: camera translation
"""
Pose = namedtuple('Pose', ['R', 't'])


def Pose_identity():
    """
    1. Pose_identity():
        - note:
            - returns the identity pose
        - input:
        - return:
        - functionality:
    """
    from minieigen import Matrix3, Vector3
    return Pose(R=Matrix3.Identity, t=Vector3.Zero)


""" End of the script """
