"""Utility functions for dealing with human3.6m data."""
import numpy as np
import backend

data_mean3D = np.array(
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -2.55652587e-01,
     -7.11960574e+00, -9.81433061e-01, -5.65463051e+00, 3.19636009e+02,
     7.19329269e+01, -1.01705840e+01, 6.91147892e+02, 1.55352986e+02,
     -1.15560633e+01, 7.42149725e+02, 1.66477287e+02, -1.18447102e+01,
     7.36763064e+02, 1.65182437e+02, 2.55651314e-01, 7.11954604e+00,
     9.81423862e-01, -5.09729780e+00, 3.27040413e+02, 7.22258095e+01,
     -9.99656606e+00, 7.08277383e+02, 1.58016408e+02, -1.12642400e+01,
     7.48636864e+02, 1.66665977e+02, -1.14090840e+01, 7.36435064e+02,
     1.63713810e+02, 1.21660909e-03, -8.60110488e-02, -1.93000547e-02,
     2.90583675e+00, -2.11363307e+02, -4.74210915e+01, 5.67537804e+00,
     -4.35088906e+02, -9.76974016e+01, 5.93884964e+00, -4.91891970e+02,
     -1.10666618e+02, 7.37352083e+00, -5.83948619e+02, -1.31171400e+02,
     5.67537804e+00, -4.35088906e+02, -9.76974016e+01, 5.41920653e+00,
     -3.83931702e+02, -8.68145417e+01, 2.95964662e+00, -1.87567488e+02,
     -4.34536934e+01, 1.26585821e+00, -1.20170579e+02, -2.82526049e+01,
     1.26585821e+00, -1.20170579e+02, -2.82526049e+01, 1.57900698e+00,
     -1.51780249e+02, -3.52080548e+01, 8.84543990e-01, -1.07795356e+02,
     -2.56307189e+01, 8.84543990e-01, -1.07795356e+02, -2.56307189e+01,
     5.67537804e+00, -4.35088906e+02, -9.76974016e+01, 4.67186639e+00,
     -3.83644089e+02, -8.55125784e+01, 1.67648571e+00, -1.97007177e+02,
     -4.31368364e+01, 8.70569018e-01, -1.68664569e+02, -3.73902498e+01,
     8.70569018e-01, -1.68664569e+02, -3.73902498e+01, 1.39982513e+00,
     -2.00884252e+02, -4.47207875e+01, 5.24591118e-01, -1.65867774e+02,
     -3.68342864e+01, 5.24591118e-01, -1.65867774e+02, -3.68342864e+01])

data_stdev3D = np.array(
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.10722440e+02,
     2.23881762e+01, 7.24629442e+01, 1.58563111e+02, 1.89338322e+02,
     2.08804791e+02, 1.91799352e+02, 2.43200617e+02, 2.47561933e+02,
     2.17938049e+02, 2.63285217e+02, 2.96834071e+02, 2.38588944e+02,
     2.73101842e+02, 3.28803702e+02, 1.10721807e+02, 2.23880543e+01,
     7.24625257e+01, 1.58804541e+02, 1.99771878e+02, 2.14706298e+02,
     1.80019441e+02, 2.50527393e+02, 2.48532471e+02, 2.13542308e+02,
     2.68186067e+02, 3.01932987e+02, 2.36099536e+02, 2.76896854e+02,
     3.35285650e+02, 1.99331867e-02, 3.28409350e-02, 2.74274580e-02,
     5.21069402e+01, 5.21140553e+01, 6.90824149e+01, 9.51536655e+01,
     1.01330318e+02, 1.28997325e+02, 1.17424577e+02, 1.26484690e+02,
     1.64650907e+02, 1.23602966e+02, 1.30855389e+02, 1.64333365e+02,
     9.51536655e+01, 1.01330318e+02, 1.28997325e+02, 1.46022319e+02,
     9.70795598e+01, 1.39527313e+02, 2.43475318e+02, 1.29822486e+02,
     2.02301812e+02, 2.44687700e+02, 2.15018164e+02, 2.39382347e+02,
     2.44687700e+02, 2.15018164e+02, 2.39382347e+02, 2.29596780e+02,
     2.22589304e+02, 2.34161811e+02, 2.79422487e+02, 2.57310206e+02,
     2.80385546e+02, 2.79422487e+02, 2.57310206e+02, 2.80385546e+02,
     9.51536655e+01, 1.01330318e+02, 1.28997325e+02, 1.38760844e+02,
     1.00892600e+02, 1.42441097e+02, 2.36875290e+02, 1.44912190e+02,
     2.09808291e+02, 2.44006949e+02, 2.39750283e+02, 2.55205840e+02,
     2.44006949e+02, 2.39750283e+02, 2.55205840e+02, 2.32859559e+02,
     2.36910758e+02, 2.47343753e+02, 2.87219983e+02, 3.05741249e+02,
     3.08336881e+02, 2.87219983e+02, 3.05741249e+02, 3.08336881e+02])

dim_to_use3D = np.array(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25,
     26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 52, 53, 54,
     55, 56, 57, 58, 59, 75, 76, 77, 78, 79, 80, 81, 82, 83])

data_stdev2D = np.array(
    [107.73640057, 63.35908715, 119.00836215, 64.12154429,
     119.12412107, 50.53806214, 120.61688044, 56.3844489,
     101.95735273, 62.89636485, 106.24832898, 48.4117812,
     108.46734967, 54.58177069, 109.07369806, 68.70443671,
     111.20130351, 74.87287863, 113.22330789, 79.90670557,
     105.71458329, 73.27049435, 107.05804267, 73.93175782,
     107.97449418, 83.30391802, 121.60675105, 74.25691526,
     134.34378972, 77.48125087, 131.79990653, 89.86721123])

data_mean2D = np.array(
    [532.08351636, 419.74137558, 531.80953144, 418.26071409,
     530.68456968, 493.54259286, 529.36968722, 575.96448517,
     532.29767645, 421.28483335, 531.93946631, 494.72186794,
     529.71984447, 578.96110365, 532.93699382, 370.65225056,
     534.11018559, 317.90342311, 534.86955004, 282.31030884,
     534.11308568, 330.11296794, 533.53637526, 376.2742511,
     533.49380106, 391.72324565, 533.52579143, 330.09494671,
     532.50804963, 374.19047901, 532.72786933, 380.61615716])


def merge_into_mean(keypoints2D, args_to_mean):
    """Rearrange keypoints2D

            # Arguments
                    keypoints2D: keypoints2D (Nx17x2)
                    args_to_mean: list of joint indices

            # Returns
                    keypoints2D: keypoints2D after merging
            """
    for point, arrangment in args_to_mean.items():
        keypoints2D[:, point] = (keypoints2D[:, arrangment[0]] + keypoints2D[:,
                                 arrangment[1]]) / 2
    return keypoints2D


def filter_keypoints(keypoints, args_to_joints):
    """filter keypoints.

    # Arguments
            keypoints: points in camera coordinates
            args_to_joints: Array of joints indices

    # Returns
            filtered keypoints
    # """
    return keypoints[:, args_to_joints, :]


def filter_keypoints3D(keypoints3D, args_to_joints3D):
    """Selects 16 moving joints (Neck/Nose excluded) from 32 predicted
    joints in 3D

    # Arguments
            keypoints3D: Nx96 points in camera coordinates
            args_to_joints3D: list of indices

    # Returns
            filtered_joints_3D: Nx48 points (moving joints)
    """
    N = len(keypoints3D)
    keypoints3D = np.reshape(keypoints3D, [N, 32, 3])
    joints3D = filter_keypoints(keypoints3D, args_to_joints3D)
    return joints3D


def filter_keypoints2D(keypoints2D, args_to_mean, h36m_to_coco_joints2D):
    """Selects 16 moving joints (Neck/Nose excluded) from 17 predicted
            joints in 2D

            # Arguments
                    keypoints3D: Nx17x2 points in camera coordinates
                    args_to_mean: keypoints indices
                    36m_to_coco_joints2D: human36m dataset list of joints
                    indices

            # Returns
                    joints2D: Nx32 points (moving joints)
            """
    keypoints2D = merge_into_mean(keypoints2D, args_to_mean)
    joints2D = filter_keypoints(keypoints2D, h36m_to_coco_joints2D)
    joints2D = np.reshape(joints2D, [joints2D.shape[0], -1])
    return joints2D


def unnormalize_data(data, mean, stdev, valid):
    """Un-normalizes a matrix whose mean has been substracted and
    that has been divided by standard deviation. Some dimensions
    might also be missing

    # Arguments
            normalized_data: nxd matrix to unnormalize
            mean: array with the mean of the data
            std: array with the standard deviation of the data
            valid: list of dimensions to keep in the data

    # Returns
            unnormalized_data: the input normalized_data, but unnormalized
    """
    data = data.reshape(-1, 48)
    rearanged_data = np.zeros((len(data), len(mean)), dtype=np.float32)
    rearanged_data[:, valid] = data
    unnormalized_data = backend.destandardize(rearanged_data, mean, stdev)
    return unnormalized_data
