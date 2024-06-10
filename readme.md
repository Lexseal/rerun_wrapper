# Rerun Wrapper

This is a simple wrapper for the rerun-sdk. It allows for logging of image data, camera data, point cloud data, etc. To install, clone this repo and run `pip install -e .`

## Usage

```
from rerun_wrapper.viz import Viz
import numpy as np

viz = Viz("name of the session", "optional ip:port of remote viewer")
viz.log_point_cloud(np.random.randn(100, 3), classification="random pts")
```