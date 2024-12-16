# look_ros2
## ROS2 Humble wrapper for LOOK: Keypoint-Based Gaze Detection

This repo integrates [LOOK](https://github.com/vita-epfl/looking), a keypoint-based gaze detector, with ROS2 Humble to estimate attention by detecting eye contact based on human pose.

## Prerequisites
- ROS2 Humble
- ZED SDK
- Forked ZED repositories (for ZED body detection compatibility with LOOK)
  - [zed-ros2-wrapper](https://github.com/askokostic/zed-ros2-interfaces/tree/humble-v4.1.4-skc)
  - [zed-ros2-interfaces](https://github.com/askokostic/zed-ros2-wrapper/tree/humble-v4.1.4-skc)

**Note:** The above forks have been extended to include confidence scores for each keypoint, which is required by LOOK.
  
## Setup

1. Clone repo with submodules

```
git clone --recursive https://github.com/askokostic/look_ros2.git
```

2. Clone forked ZED repositories

```
git clone https://github.com/askokostic/zed-ros2-wrapper.git -b humble-v4.1.4-skc
git clone https://github.com/askokostic/zed-ros2-interfaces.git -b humble-v4.1.4-skc
```
The `zed-ros2-wrapper` and `zed-ros2-interfaces` repositories on the `humble-v4.1.4-skc` branch have been extended to include confidence scores for each keypoint. This is necessary for LOOK to work properly.

1. Install python dependencies
```
pip install -r look_ros2/requirements.txt
```

## Configuration
Available parameters:

- `color_image_topic` (default: `/zed/zed_node/left/image_rect_color`)
- `depth_image_topic` (default: `/zed/zed_node/depth/depth_registered`)
- `color_camera_info_topic` (default: `/zed/zed_node/rgb_raw/camera_info`)
- `zed_skeletons_topic` (default: `/zed/zed_node/body_trk/skeletons`)
- `mode` (`pifpaf` or `zed`, default: `pifpaf`)
- `downscale_factor` (default: `1`)

### Note on `downscale_factor`:

- In `pifpaf` mode: it downscales the input image before processing, **improving performance** by reducing processing time.
- In `zed` mode: it **only affects visualization**. The ZED SDK always processes keypoints at full sensor resolution internally, regardless of the published image resolution.