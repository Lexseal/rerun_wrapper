import rerun as rr
from rerun.datatypes import Quaternion, TranslationRotationScale3D
import time
import numpy as np
import uuid
import open3d as o3d
import torch
from pytorch3d.ops import box3d_overlap
from transforms3d.quaternions import mat2quat


class Viz:
    def __init__(self, name: str = None, remote_url: str = None):
        """
        Initialize the rerun client

        Args:
            name (str, optional): Name of the rerun instance. Defaults to None.
            remote_url (str, optional): The ip:port to connect to.
        """
        if name is None:
            name = str(uuid.uuid4())
        rr.init(name)
        if remote_url is not None:
            rr.connect(remote_url)
        else:
            rr.spawn()
        rr.set_time_seconds("real_clock", time.time())
        self.uuid_to_oobb = {}  # {uuid, oobb} for each point cloud
        self.uuid_to_color = {}  # {uuid, color} for each point cloud

    def _pcd_to_p3d_oobb(self, pcd: np.ndarray):
        """
        Find the oriented bounding box of a point cloud using open3d and convert it to pytorch3d format
        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)
        """
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
        oobb = o3d_pcd.get_oriented_bounding_box()
        unit_box = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=torch.float32,
        )
        R = torch.tensor(oobb.R, dtype=torch.float32)
        extent = torch.tensor(oobb.extent, dtype=torch.float32)
        center = torch.tensor(oobb.center, dtype=torch.float32)
        # stretch the box
        p3d_oobb = unit_box * extent
        # rotate the box
        p3d_oobb = torch.matmul(p3d_oobb, R.T)
        # translate the box
        cur_center = torch.ones(8, 3, dtype=torch.float32) / 2
        p3d_oobb = p3d_oobb - cur_center + center
        return p3d_oobb, oobb.extent, oobb.R, oobb.center

    def _ious(self, oobbs: torch.Tensor, oobb2: torch.Tensor) -> torch.Tensor:
        """
        Calculate intersection over union of bounding boxes

        oobbs: [n, 8, 3]
        oobb2: [1, 8, 3]
        """
        _, ious = box3d_overlap(oobbs, oobb2)
        return ious.unsqueeze(1)

    def _find_most_likely_uuid(self, oobb: torch.Tensor, threshold=0.5) -> str:
        """Use intersection over union to find the most likely uuid of the pcd cloud that contains the pcd"""
        if len(self.uuid_to_oobb) == 0:
            return str(uuid.uuid4())
        oobbs = torch.stack(list(self.uuid_to_oobb.values()))
        ious = self._ious(oobbs, oobb.unsqueeze(0))
        argmax = torch.argmax(ious)
        if ious[argmax] > threshold:
            return list(self.uuid_to_oobb.keys())[argmax]
        else:
            return str(uuid.uuid4())  # generate one

    def _get_random_rgb(self) -> np.ndarray:
        return (np.random.rand(3) * 255).astype(np.uint8)

    def log_point_cloud(
        self,
        pcd: np.ndarray,
        obb: np.ndarray = None,
        colors: np.ndarray = None,
        id: str = None,
        classification: str = None,
        radius: float = 0.01,
        observation_time: float = None,
    ):
        """
        log point_cloud we need to take in pcd [n, 3] and optinally color, classification and time

        Args:
            pcd (np.ndarray): [n, 3]
            obb (np.ndarray, optional): [8, 3]. Oriented bounding box of the pcd.
            colors (np.ndarray, optional): [3, ] or [n, 3]. Defaults to None for persistent random colors.
            id (str, optional): For persistence. If None, will try to find the uuid of the closest point cloud.
            classification (str, optional): Defaults to None if you don't want any labels.
            observation_time (float, optional): Time since epoch in seconds. If None current time used.
            radius (float, optional): Defaults to 0.01 meters.
        """
        if observation_time is None:
            observation_time = time.time()
        rr.set_time_seconds("real_clock", observation_time)
        if obb is not None:  # reindex the obb since we need it in a specific order
            oobb, extent, R, center = self._pcd_to_p3d_oobb(obb)
        else:
            oobb, extent, R, center = self._pcd_to_p3d_oobb(pcd)
        if id is None:
            id = self._find_most_likely_uuid(oobb)
        if colors is None:
            colors = self.uuid_to_color.get(id, self._get_random_rgb())
        self.uuid_to_oobb[id] = oobb
        self.uuid_to_color[id] = colors
        rr.log(f"/pcd/pts/{id}", rr.Points3D(pcd, colors=colors, radii=radius))
        # draw bounding box too
        q = mat2quat(R)
        q = np.roll(q, -1)  # turn wxyz to xyzw
        q = Quaternion(xyzw=q)
        rr.log(
            f"/pcd/obb/{id}",
            rr.Boxes3D(
                half_sizes=extent / 2,
                centers=center,
                rotations=q,
                colors=colors,
                labels=classification,
            ),
        )

    def log_tf(
        self,
        tf: np.ndarray,
        scale: float = 0.3,
        id: str = None,
        observation_time: float = None,
    ):
        """
        log tf we take in 4x4 and optionally scale and time

        Args:
            tf (np.ndarray): [4, 4]
            scale (float, optional): Defaults to 0.3 meters.
            observation_time (float, optional): Time since epoch in seconds. If None current time used.
        """
        if observation_time is None:
            observation_time = time.time()
        rr.set_time_seconds("real_clock", observation_time)
        if id is None:
            id = str(uuid.uuid4())
        transform = TranslationRotationScale3D(
            translation=tf[:3, 3],
            rotation=Quaternion(xyzw=np.roll(mat2quat(tf[:3, :3]), -1)),
            scale=scale * 2,  # this is the halfsize scale
        )
        rr.log(f"/tf/{id}", rr.Transform3D(transform))

    def log_trajectory(
        self,
        trajectory: np.ndarray,
        id: str = None,
        colors: np.ndarray = None,
        observation_time: float = None,
    ):
        """
        log trajectory we take in [n, 2] or [n, 3] and optionally time

        Args:
            trajectory (np.ndarray): [n, 2] or [n, 3]
            observation_time (float, optional): Time since epoch in seconds. If None current time used.
        """
        if observation_time is None:
            observation_time = time.time()
        rr.set_time_seconds("real_clock", observation_time)
        if id is None:
            id = str(uuid.uuid4())
        if trajectory.shape[1] == 2:
            trajectory = np.hstack((trajectory, np.zeros((trajectory.shape[0], 1))))
        colors = colors or self._get_random_rgb()
        # first clear the old trajectory
        rr.log(f"/trajectory/{id}", rr.Clear(recursive=True))
        # draw arrows between points
        for i in range(trajectory.shape[0] - 1):
            rr.log(
                f"/trajectory/{id}/{i}",
                rr.Arrows3D(
                    origins=trajectory[i],
                    vectors=trajectory[i + 1] - trajectory[i],
                    colors=colors,
                ),
            )

    def log_camera(
        self,
        image: np.ndarray,
        tf: np.ndarray,
        intrinsics: np.ndarray = None,
        id: str = None,
        observation_time: float = None,
    ):
        """
        log camera we take in an image [h, w, 3] or [h, w] and then tf and optionally intrinsics and time

        Args:
            image (np.ndarray): [h, w, 3] or [h, w]
            tf (np.ndarray): [4, 4]
            intrinsics (np.ndarray, optional): [3, 3]. Defaults to None.
            observation_time (float, optional): Time since epoch in seconds. If None current time used.
        """
        if observation_time is None:
            observation_time = time.time()
        rr.set_time_seconds("real_clock", observation_time)
        if id is None:
            id = "color" if image.shape[-1] == 3 else "depth"
        if intrinsics is None:
            intrinsics = np.array(
                [
                    [image.shape[1] / 2, 0, image.shape[1] / 2],
                    [0, image.shape[1] / 2, image.shape[0] / 2],
                    [0, 0, 1],
                ]
            )
        rr.log(
            f"/camera/image/{id}",
            rr.Pinhole(
                image_from_camera=intrinsics,
                resolution=(image.shape[1], image.shape[0]),
            ),
        )
        img_to_log = (
            rr.Image(image) if image.shape[-1] == 3 else rr.DepthImage(image, meter=1.0)
        )
        rr.log(f"/camera/image/{id}", img_to_log)
        transform = TranslationRotationScale3D(
            translation=tf[:3, 3],
            rotation=Quaternion(xyzw=np.roll(mat2quat(tf[:3, :3]), -1)),
            scale=0.3,
        )
        rr.log(f"/camera/image/{id}", rr.Transform3D(transform))

    def log_image(self, image: np.ndarray, id: str = None, observation_time: float = None):
        """
        log image we take in an image [h, w, 3] or [h, w] and optionally time

        Args:
            image (np.ndarray): [h, w, 3] or [h, w]
            observation_time (float, optional): Time since epoch in seconds. If None current time used.
        """
        if observation_time is None:
            observation_time = time.time()
        rr.set_time_seconds("real_clock", observation_time)
        if id is None:
            id = "color" if image.shape[-1] == 3 else "depth"
        img_to_log = (
            rr.Image(image) if image.shape[-1] == 3 else rr.DepthImage(image, meter=1.0)
        )
        rr.log(f"/image/{id}", img_to_log)

if __name__ == "__main__":
    # test client. First log a random gaussian point cloud of 10 points
    viz = Viz()
    points = np.random.randn(100, 3)
    viz.log_point_cloud(points, classification="random")
    time.sleep(0.1)
    # then log a the same point cloud shifted by a tiny bit
    points += 0.1
    viz.log_point_cloud(points)  # this should have the same uuid
    time.sleep(0.1)
    # then shift it by a lot
    points += 1
    viz.log_point_cloud(points)  # this should have a new uuid
    time.sleep(0.1)
    # log tf
    tf = np.eye(4)
    tf[0, 3] = 1
    viz.log_tf(tf, scale=0.5)
    time.sleep(0.1)
    # log trajectory
    trajectory = np.random.randn(10, 2)
    viz.log_trajectory(trajectory)
    time.sleep(0.1)
    # log color camera
    image = np.random.rand(100, 100, 3)
    tf = np.eye(4)
    viz.log_camera(image, tf)
    time.sleep(0.1)
    # log depth camera
    image = np.ones(shape=(100, 100))
    viz.log_camera(image, tf)
