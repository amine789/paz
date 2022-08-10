import numpy as np
from scipy import spatial
from scipy.spatial import distance
from paz.backend.groups.quaternion import quaternion_to_rotation_matrix

def get_translations(pose6D):

    """It outputs the predicted Translation from the detected object.

      # Arguments
          pose6D: The 6d pose


      # Returns
          Return Translation vector
    """

    translation = np.array(pose6D.translation)#np.array(inferences['poses6D'][0].translation)

    return translation


def get_quaternion(pose6D):
    """It outputs the predicted quaternion from the detected object.

      # Arguments
          pose6D: The 6d pose


      # Returns
          Return Quaternion vector
      """

    quaternion = np.array(pose6D.quaternion)

    return quaternion


def transform_mesh_points(mesh_points, rotation, translation):
    """Transforms the object points

      # Arguments
          mesh_points: nx3 ndarray with 3D model points.
          rotaion: Rotation matrix
          translation: Translation vector
          

      # Returns
          Transformed model
      """
    assert (mesh_points.shape[1] == 3)
    pts_t = rotation.dot(mesh_points.T) + translation.reshape((3, 1))
    return pts_t.T


def compute_ADD(true_pose, pred_pose, mesh_points):
    """Calculate The ADD error.

      # Arguments
          true_pose: Real pose
          pred_pose: Predicted pose
          mesh_pts: nx3 ndarray with 3D model points.

      # Returns
          Return ADD error
      """
    pred_rotation = quaternion_to_rotation_matrix(pred_pose.quaternion)
    pred_translation = pred_pose.translation
    trans_pts_pred = transform_mesh_points(mesh_points, pred_rotation, pred_translation)
    true_rotation = quaternion_to_rotation_matrix(true_pose.quaternion)
    true_translation = pred_pose.translation
    trans_pts_true = transform_mesh_points(mesh_points, true_rotation, true_translation)
    error = np.linalg.norm(trans_pts_pred - trans_pts_true, axis=1).mean()
    return error


def compute_ADI(true_pose, pred_pose, mesh_points):
    """Calculate The ADI error.

      # Arguments
          true_pose: Real pose
          pred_pose: Predicted pose
          mesh_pts: nx3 ndarray with 3D model points.

      # Returns
          Return ADI error
      """
    pred_rotation = quaternion_to_rotation_matrix(pred_pose.quaternion)
    pred_translation = pred_pose.translation
    trans_pts_pred = transform_mesh_points(mesh_points, pred_rotation, pred_translation)
    true_rotation = quaternion_to_rotation_matrix(true_pose.quaternion)
    true_translation = pred_pose.translation
    trans_pts_true = transform_mesh_points(mesh_points, true_rotation, true_translation)

    # Calculate distances to the nearest neighbors from vertices in the
    # ground-truth pose to vertices in the estimated pose.
    nn_index = spatial.cKDTree(trans_pts_pred)
    nn_dists, _ = nn_index.query(trans_pts_true, k=1)

    error = nn_dists.mean()
    return error
