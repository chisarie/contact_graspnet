import pathlib
import os
import sys
import glob
import time
from uois.inference import UOISInference

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
import config_utils
from data import load_available_input_data
from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image


class ContactGraspNetInference:
    def __init__(self) -> None:
        ckpt_dir = pathlib.Path(__file__).parents[1] / "checkpoints/scene_test_2048_bs3_hor_sigma_001"

        # Build the model
        global_config = config_utils.load_config(ckpt_dir, batch_size=1)
        self.grasp_estimator = GraspEstimator(global_config)
        self.grasp_estimator.build_network()

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(save_relative_paths=True)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

        # Load weights
        self.grasp_estimator.load_weights(self.sess, saver, ckpt_dir, mode="test")
        return

    def predict(self, rgb, depth, depth_K, segmap=None):
        """
        Args:
            rgb (np.array): RGB image. Shape: (720, 1280, 3). dtype: np.uint8
            depth (np.array): Depth image. Shape: (720, 1280). dtype: np.float32
            depth_K (np.array): Camera intrinsics. Shape: (3, 3). dtype: np.float64
            segmap (np.array): Segmentation map. Shape: (720, 1280). dtype: np.float32 (but values are actually int)
        """
        # Converting depth to point cloud(s)
        pc_full, pc_segments, pc_colors = self.grasp_estimator.extract_point_clouds(
            depth,
            depth_K,
            segmap=segmap,
            rgb=rgb,
            skip_border_objects=False,
            z_range=[0.2, 1.8],
        )

        # Generating Grasps
        pred_grasps_cam, scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(
            self.sess,
            pc_full,
            pc_segments=pc_segments,
            local_regions=True if segmap is not None else False,
            filter_grasps=True if segmap is not None else False,
            forward_passes=1,
        )
        return pc_full, pc_colors, pred_grasps_cam, scores

    def visualize_results(self, rgb, segmap, pc_full, pc_colors, pred_grasps_cam, scores):
        show_image(rgb, segmap)
        visualize_grasps(
            pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors
        )
        print("Results Visualized")
        time.sleep(0.01)
        return

    def load_scene_data(self, path):
        segmap, rgb, depth, depth_K, pc_full, pc_colors = load_available_input_data(path)
        return segmap, rgb, depth, depth_K, pc_full, pc_colors


if __name__ == "__main__":
    mode = "demo"  # "demo" or "custom"
    use_segm_net = True
    net_inference = ContactGraspNetInference()

    if mode == "demo":
        np_path = "test_data/*.npy"
        path = glob.glob(np_path)[0]
        segmap, rgb, depth, depth_K, pc_full, pc_colors = net_inference.load_scene_data(path)
    elif mode == "custom":
        raise NotImplementedError

    if use_segm_net:
        segmap = None
        segm_net = UOISInference()
        segmap = segm_net.predict(rgb, depth, depth_K)

    pc_full, pc_colors, pred_grasps_cam, scores = net_inference.predict(rgb, depth, depth_K, segmap)
    net_inference.visualize_results(rgb, segmap, pc_full, pc_colors, pred_grasps_cam, scores)
