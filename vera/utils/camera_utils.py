"""
adapted from: https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/utils/camera_utils.py
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig
from transforms3d.quaternions import mat2quat

UP_VECTOR = np.array([0, 0, 1])


def make_pose(translation, rotation):
    """
    Makes a homogeneous pose matrix from a translation vector and a rotation matrix.
    Args:
        translation (np.array): (x,y,z) translation value
        rotation (np.array): a 3x3 matrix representing rotation
    Returns:
        pose (np.array): a 4x4 homogeneous matrix
    """
    pose = np.zeros((4, 4))
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    pose[3, 3] = 1.0
    return pose


def get_camera_intrinsic_matrix(env, camera_name):
    """
    Obtains camera intrinsic matrix.
    Args:
        env: simulator instance
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        K (np.array): 3x3 camera matrix
    """
    import mujoco

    camera_height = env.height
    camera_width = env.width

    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    fovy = env.model.cam_fovy[cam_id]
    f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
    K = np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])
    return K


def get_camera_extrinsic_matrix(env, camera_name):
    """
    Returns a 4x4 homogenous matrix corresponding to the camera pose in the
    world frame. MuJoCo has a weird convention for how it sets up the
    camera body axis, so we also apply a correction so that the x and y
    axis are along the camera view and the z axis points along the
    viewpoint.
    Normal camera convention: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    Args:
        env: simulator instance
        camera_name (str): name of camera
    Return:
        R (np.array): 4x4 camera extrinsic matrix
    """
    import mujoco

    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    camera_pos = env.data.cam_xpos[cam_id]
    camera_rot = env.data.cam_xmat[cam_id].reshape(3, 3)
    R = make_pose(camera_pos, camera_rot)

    return R


def get_camera_intrinsic_matrix_raw(camera_height, camera_width, fovy):
    f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
    K = np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])
    return K


def get_camera_extrinsic_matrix_raw(camera_pos, camera_rot):
    R = make_pose(camera_pos, camera_rot)

    return R


def camera_lookat(
    pos: np.ndarray,
    lookat: np.ndarray,
    up: Optional[np.ndarray] = None,
    return_quat: bool = True,
):
    # Compute the forward direction
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
    z = pos - lookat
    z /= np.linalg.norm(z)

    # Compute the up direction
    x = np.cross(up, z)
    x /= np.linalg.norm(x)

    # Compute the right direction
    y = np.cross(z, x)
    y /= np.linalg.norm(y)

    # Compute the rotation matrix
    R = np.stack((x, y, z), axis=-1)

    orientation = R
    if return_quat:
        orientation = mat2quat(R)

    return pos, orientation


def point_on_circle(angle, radius, center):
    x = radius * np.cos(angle) + center[0]
    y = radius * np.sin(angle) + center[1]
    return np.array([x, y])


def get_lookat_params(
    cam_pos: np.ndarray,
    lookat: np.ndarray,
    height: int,
    width: int,
    fovy: float = 45.0,
    return_quat: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cam_pos, cam_ori = camera_lookat(cam_pos, lookat, up=UP_VECTOR, return_quat=False)

    cam2world = get_camera_extrinsic_matrix_raw(cam_pos, cam_ori)
    intrinsic = get_camera_intrinsic_matrix_raw(height, width, fovy)

    if return_quat:
        cam_ori = mat2quat(cam_ori)

    return cam2world, intrinsic, cam_pos, cam_ori


def get_spherical_camera_params(
    height: int,
    width: int,
    fovy: float = 45.0,
    num_views: int = 100,
    radius: float = 2.0,
    cam_height: float = 1.5,
    lookat: Tuple[float, float, float] = None,
) -> Dict[str, List[np.ndarray]]:
    """
    :param height: image height
    :param width: image width
    :param fovy: focal length
    :param num_views: number of views to generate
    :param radius: radius of the sphere surrounding the lookat point
    :param cam_height: height of the camera
    :param lookat: the lookat position of the camera
    :return: dictionary that contains id_to_cam2world, id_to_intrinsics, id_to_campos, id_to_camori
    """

    if lookat is None:
        lookat = [0.0, 0.0, 0.5]
    if not isinstance(lookat, np.ndarray):
        lookat = np.array(lookat)

    # Set the center of the circular camera motion
    camera_degrees = np.linspace(0.0, 360, num_views + 1)[:-1]

    id_to_campos = []
    id_to_camori = []
    id_to_cam2world = []
    id_to_intrinsics = []

    for this_degree in camera_degrees:
        cam_pos = np.array(
            [
                *point_on_circle(np.deg2rad(this_degree), radius, center=lookat[:2]),
                cam_height,
            ]
        )
        cam2world, intrinsic, cam_pos, cam_ori = get_lookat_params(
            cam_pos, lookat, height, width, fovy, return_quat=True
        )

        id_to_campos.append(cam_pos)
        id_to_camori.append(cam_ori)
        id_to_cam2world.append(cam2world)
        id_to_intrinsics.append(intrinsic)

    return {
        "id_to_cam2world": id_to_cam2world,
        "id_to_intrinsics": id_to_intrinsics,
        "id_to_cam_pos": id_to_campos,
        "id_to_cam_ori": id_to_camori,
    }


def parse_view_cfg(view_cfg: DictConfig):
    spherical_cfg = view_cfg.get("spherical", None)
    base_cfg = view_cfg.get("base", None)
    image_size = view_cfg.image_size

    cam_params = {
        "id_to_cam2world": [],
        "id_to_intrinsics": [],
        "id_to_cam_pos": [],
        "id_to_cam_ori": [],
        "image_size": image_size,
    }

    if spherical_cfg is not None:
        cam_params.update(
            **get_spherical_camera_params(
                image_size[0],
                image_size[1],
                num_views=spherical_cfg.num_views,
                radius=spherical_cfg.radius,
                cam_height=spherical_cfg.cam_height,
                lookat=spherical_cfg.lookat,
                fovy=view_cfg.fovy,
            )
        )

    if base_cfg is not None:
        cam2world, intrinsic, cam_pos, cam_ori = get_lookat_params(
            np.array(base_cfg.cam_pos),
            lookat=np.array(base_cfg.lookat),
            height=image_size[0],
            width=image_size[1],
            fovy=view_cfg.fovy,
        )

        cam_params["id_to_cam2world"].append(cam2world)
        cam_params["id_to_intrinsics"].append(intrinsic)
        cam_params["id_to_cam_pos"].append(cam_pos)
        cam_params["id_to_cam_ori"].append(cam_ori)

    return cam_params


def save_cam_params(dump_dir, cam_params):
    camera_data = {
        "id_to_cam2world": np.stack(cam_params["id_to_cam2world"], axis=0).astype(
            np.float32
        ),
        "id_to_intrinsics": np.stack(cam_params["id_to_intrinsics"], axis=0).astype(
            np.float32
        ),
    }

    np.savez_compressed(os.path.join(dump_dir, "camera_config.npz"), **camera_data)
