from collections import namedtuple
import numpy

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
        - image: a PIL.Image # Lal: to be changed as scikit-image (numpy array) in the next update
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
        - input: -
        - return: Pose
        - functionality:
            - return identity pose:
                - Rotation matrix R = 3x3 identity matrix
                - translation vector t = 3x1 zero vector
    """
    # from minieigen import Matrix3, Vector3
    # return Pose(R=Matrix3.Identity, t=Vector3.Zero)
    return Pose(R=numpy.identity(3, dtype=numpy.float64), t=numpy.zeros((3, 1), dtype=numpy.float64))


# for testing the script
if __name__ == "__main__":

    # sample rotation matrix
    R = numpy.identity(3, dtype=numpy.float64)
    print(R.shape)
    print(R.dtype)
    """
    (3, 3)
    float64
    """
    # sample translation vector
    t = numpy.zeros((3, 1), dtype=numpy.float64)
    print(t.shape)
    print(t.dtype)
    """
    (3, 1)
    float64
    """
    # sample camera calibration matrix
    K = numpy.identity(3, dtype=numpy.float64)
    print(K.shape)
    print(K.dtype)
    """
    (3, 3)
    float64
    """
    # sample pose
    Pose_sample = Pose(R=R, t=t)
    # sample View
    View_sample = View(R=R, t=t, K=K, image=None, depth=None, depth_metric=None)
    print(Pose_sample)
    print(View_sample)
    """
    Pose(R=array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]]), t=array([[0.],
       [0.],
       [0.]]))
    View(R=array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]]), t=array([[0.],
       [0.],
       [0.]]), K=array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]]), image=None, depth=None, depth_metric=None)
    """
""" End of the script """
