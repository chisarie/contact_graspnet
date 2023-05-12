import numpy as np
from PIL import Image
from inference_class import ContactGraspNetInference

ZED2_INTRINSICS = np.array(
    [
        [1062.88232421875, 0.0, 957.660400390625],
        [0.0, 1062.88232421875, 569.8204345703125],
        [0.0, 0.0, 1.0],
    ]
)
ZED2_INTRINSICS_HALF = np.copy(ZED2_INTRINSICS)
ZED2_INTRINSICS_HALF[0:-1, :] /= 2
ZED2_INTRINSICS_HALF[1, 2] -= 14  # Cropping
ZED2_RESOLUTION = np.array([1920, 1080], dtype=np.int32)
ZED2_RESOLUTION_HALF = ZED2_RESOLUTION // 2
ZED2_RESOLUTION_HALF[1] -= 28  # Cropping


def main():
    data_path = "inference"
    rgb_uint8 = np.array(Image.open(data_path + "/rgb.png"))
    depth = np.array(Image.open(data_path + "/depth.png"))
    depth = depth.astype(np.float32) / 1000.0
    depth_K = ZED2_INTRINSICS_HALF
    segmap = None
    contactgraspnet = ContactGraspNetInference()
    pc_full, pc_colors, pred_grasps_cam, scores = contactgraspnet.predict(
        rgb_uint8, depth, depth_K, segmap
    )
    contactgraspnet.visualize_results(
        rgb_uint8, segmap, pc_full, pc_colors, pred_grasps_cam, scores
    )
    print("done")
    return


if __name__ == "__main__":
    main()
