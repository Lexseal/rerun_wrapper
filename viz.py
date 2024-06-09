import rerun as rr
import time
import numpy as np
import uuid
import open3d as o3d
import torch
from pytorch3d.ops import box3d_overlap

class Viz():
    """"""
    def __init__(self) -> None:
        rr.init("rerun_example_dna_abacus", spawn=True)
        rr.set_time_seconds("real_clock", time.time())
        self.uuid_to_oobb = {}  # {uuid, oobb} for each point cloud
        self.uuid_to_color = {}  # {uuid, color} for each point cloud

    def _pcd_to_p3d_oobb(self, pcd: np.ndarray) -> torch.Tensor:
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
        unit_box = torch.tensor([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ], dtype=torch.float32)
        R = torch.tensor(oobb.R, dtype=torch.float32)
        extent = torch.tensor(oobb.extent, dtype=torch.float32)
        center = torch.tensor(oobb.center, dtype=torch.float32)
        # stretch the box
        p3d_oobb = unit_box * extent
        # rotate the box
        p3d_oobb = torch.matmul(unit_box, R.T)
        # translate the box
        cur_center = torch.ones(8, 3, dtype=torch.float32) / 2
        p3d_oobb = p3d_oobb - cur_center + center
        return p3d_oobb

    def _ious(self, oobbs: torch.Tensor, oobb2: torch.Tensor) -> torch.Tensor:
        """
        Calculate intersection over union of bounding boxes

        oobbs: [n, 8, 3]
        oobb2: [1, 8, 3]
        """
        _, ious = box3d_overlap(oobbs, oobb2)
        return ious.unsqueeze(1)

    def _find_most_likely_uuid(self, oobb: torch.Tensor, threshold=0.5) -> str:
        """ Use intersection over union to find the most likely uuid of the pcd cloud that contains the pcd """
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

    def log_point_cloud(self, pcd: np.ndarray,
                        colors: np.ndarray=None,
                        uuid: str=None,
                        classification: str=None,
                        observation_time: float=None):
        """
        log point_cloud we need to take in pcd [n, 3] and optinally color, classification and time

        Args:
            pcd (np.ndarray): [n, 3]
            colors (np.ndarray, optional): [3, ] or [n, 3]. Defaults to None for persistent random colors.
            uuid (str, optional): For persistence. If None, will try to find the uuid of the closest point cloud.
            classification (str, optional): Defaults to None if you don't want any labels.
            observation_time (float, optional): Time since epoch in seconds. If None current time used.
        """
        oobb = self._pcd_to_p3d_oobb(pcd)
        if observation_time is None:
            observation_time = time.time()
        rr.set_time_seconds("real_clock", observation_time)
        if uuid is None:
            uuid = self._find_most_likely_uuid(oobb)
        if colors is None:
            colors = self.uuid_to_color.get(uuid, self._get_random_rgb())
        self.uuid_to_oobb[uuid] = oobb
        self.uuid_to_color[uuid] = colors
        rr.log(f"/pcd/{uuid}", rr.Points3D(pcd, colors=colors, radii=0.08))


# log tf we take in 4x4 and optionally scale and time

# log camera we take in an image [h, w, 3] or [h, w] and then tf and optionally intrinsics and time

# log trajectory we take in [n, 2] or [n, 3] and optionally time

if __name__ == "__main__":
    # test client. First log a random gaussian point cloud of 10 points
    viz = Viz()
    points = np.random.randn(10, 3)
    viz.log_point_cloud(points)
    time.sleep(1)
    # then log a the same point cloud shifted by a tiny bit
    points += 0.1
    viz.log_point_cloud(points)  # this should have the same uuid
    time.sleep(1)
    # then shift it by a lot
    points += 1
    viz.log_point_cloud(points)  # this should have a new uuid
