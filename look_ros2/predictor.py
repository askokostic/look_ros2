from looking.utils.network import *
from looking.utils.utils_predict import *
import os
import argparse
import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer
from visualization_msgs.msg import MarkerArray, Marker
from zed_interfaces.msg import ObjectsStamped, Object

INPUT_SIZE = 51


class LookWrapper(Node):
    def __init__(self):
        super().__init__('look_wrapper')

        # Parameters
        self.declare_parameter('color_image_topic', '/zed/zed_node/left/image_rect_color')
        self.declare_parameter('depth_image_topic', '/zed/zed_node/depth/depth_registered')
        self.declare_parameter('color_camera_info_topic', '/zed/zed_node/rgb_raw/camera_info')
        self.declare_parameter('zed_skeletons_topic', '/zed/zed_node/body_trk/skeletons')
        self.declare_parameter('mode', 'pifpaf')
        self.declare_parameter('downscale_factor', 1)
        self.color_image_topic = self.get_parameter('color_image_topic').value
        self.depth_image_topic = self.get_parameter('depth_image_topic').value
        self.color_camera_info_topic = self.get_parameter('color_camera_info_topic').value
        self.zed_skeletons_topic = self.get_parameter('zed_skeletons_topic').value
        self.mode = self.get_parameter('mode').value
        self.downscale_factor = self.get_parameter('downscale_factor').value

        # Variables
        self.transparency = 0.4
        self.eyecontact_thresh = 0.5
        self.path_model = os.path.join(os.path.dirname(__file__), '../looking/models/predictor')
        self.intrinsics = None
        self.depth_image_cv = None
        self.keypoints = None
        self.boxes = None

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pifpaf
        self.pifpaf_predictor = load_pifpaf(self.parse_pifpaf_args())

        # Load LOOK model
        self.model = self.get_model().to(self.device)

        self.bridge = CvBridge()

        self.color_camera_info_sub = self.create_subscription(
            CameraInfo,
            self.color_camera_info_topic,
            self.color_camera_info_cb,
            10
        )

        self.color_img_sub = Subscriber(self, Image, self.color_image_topic)
        self.depth_img_sub = Subscriber(self, Image, self.depth_image_topic)
        self.zed_body_det_sub = Subscriber(self, ObjectsStamped, self.zed_skeletons_topic)

        if self.mode == 'pifpaf':
            self.depth_color_sync = ApproximateTimeSynchronizer(
                [self.color_img_sub, self.depth_img_sub], queue_size=1, slop=0.1)
            self.depth_color_sync.registerCallback(self.synced_image_cb)
            self.marker_pub = self.create_publisher(MarkerArray, 'bounding_boxes_marker_array', 1)
            self.zed_object_pub = self.create_publisher(ObjectsStamped, 'detection', 10)
        elif self.mode == 'zed':
            self.body_color_sync = ApproximateTimeSynchronizer(
                [self.color_img_sub, self.zed_body_det_sub], queue_size=1, slop=0.1)
            self.body_color_sync.registerCallback(self.synced_body_image_cb)

        self.look_debug_pub = self.create_publisher(Image, '/looking', 10)

    def parse_pifpaf_args(self):
        parser = argparse.ArgumentParser(prog='python3 predict', usage='%(prog)s [options] images', description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-o', '--image-output', default=None, nargs='?', const=True, help='Whether to output an image, with the option to specify the output path or directory')
        parser.add_argument('--json-output', default=None, nargs='?', const=True, help='Whether to output a json file, with the option to specify the output path or directory')
        parser.add_argument('--batch_size', default=1, type=int, help='processing batch size')
        parser.add_argument('--device', default='0', type=str, help='cuda device')
        parser.add_argument('--long-edge', default=None, type=int, help='rescale the long side of the image (aspect ratio maintained)')
        parser.add_argument('--loader-workers', default=None, type=int, help='number of workers for data loading')
        parser.add_argument('--precise-rescaling', dest='fast_rescaling', default=True, action='store_false', help='use more exact image rescaling (requires scipy)')
        parser.add_argument('--checkpoint_', default='shufflenetv2k30', type=str, help='backbone model to use')
        parser.add_argument('--disable-cuda', action='store_true', help='disable CUDA')

        decoder.cli(parser)
        logger.cli(parser)
        network.Factory.cli(parser)
        show.cli(parser)
        visualizer.cli(parser)

        args = parser.parse_args()
        args.device = self.device

        return args

    def synced_image_cb(self, color_img, depth_img):
        try:
            color_image_cv = self.bridge.imgmsg_to_cv2(color_img, desired_encoding='rgb8')
            depth_image_cv = self.bridge.imgmsg_to_cv2(depth_img, desired_encoding='32FC1')
            scaled_color_image = cv2.resize(color_image_cv, (0, 0), fx=1/self.downscale_factor, fy=1/self.downscale_factor, interpolation=cv2.INTER_AREA)
            self.depth_image_cv = cv2.resize(depth_image_cv, (0, 0), fx=1/self.downscale_factor, fy=1/self.downscale_factor, interpolation=cv2.INTER_AREA)
            pil_image = PILImage.fromarray(scaled_color_image)
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        predictions, _, _ = self.pifpaf_predictor.pil_image(pil_image)
        pred = [ann.json_data() for ann in predictions]

        im_size = (pil_image.size[0], pil_image.size[1])
        self.boxes, self.keypoints = preprocess_pifpaf(pred, im_size, enlarge_boxes=False)
        pred_labels = self.predict_look(im_size)

        self.render_image(pil_image, pred_labels, color_img.header)

        projected_boxes_3d = self.project_2d_box_to_3d()
        self.publish_3d_bounding_boxes(projected_boxes_3d, color_img.header)
        self.publish_zed_object(projected_boxes_3d)

    def synced_body_image_cb(self, color_img, zed_object_array):
        try:
            color_image_cv = self.bridge.imgmsg_to_cv2(color_img, desired_encoding='rgb8')
            pil_image = PILImage.fromarray(color_image_cv)
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        im_size = (pil_image.size[0], pil_image.size[1])

        bboxes = []
        keypoints = []
        for obj in zed_object_array.objects:
            x_coords = [corner.kp[0] / self.downscale_factor for corner in obj.bounding_box_2d.corners]
            y_coords = [corner.kp[1] / self.downscale_factor for corner in obj.bounding_box_2d.corners]

            x1, x2 = max(min(x_coords), 0), max(max(x_coords), 0)
            y1, y2 = max(min(y_coords), 0), max(max(y_coords), 0)
            confidence = obj.confidence / 100

            bboxes.append([x1, y1, x2, y2, confidence])

            # Process keypoints (only first 18 keypoints are relevant)
            skeleton_kps = obj.skeleton_2d.keypoints[:18]
            kp_confs = obj.skeleton_keypoint_confidence[:18]

            x_kps = []
            y_kps = []
            c_kps = []

            for kp, kp_conf in zip(skeleton_kps, kp_confs):
                x_kps.append(max(0, min((kp.kp[0] / self.downscale_factor), pil_image.size[0])))
                y_kps.append(max(0, min((kp.kp[1] / self.downscale_factor), pil_image.size[1])))
                c_kps.append(0.0 if np.isnan(kp_conf) else kp_conf)

            keypoints.append([remap_keypoint(x_kps), remap_keypoint(y_kps), remap_keypoint(c_kps)])

        self.boxes = bboxes
        self.keypoints = keypoints

        pred_labels = self.predict_look(im_size)

        self.render_image(pil_image, pred_labels, color_img.header)

    def color_camera_info_cb(self, msg):
        self.intrinsics = {
            'fx': msg.k[0],
            'fy': msg.k[4],
            'cx': msg.k[2],
            'cy': msg.k[5]
        }

    # Source: https://github.com/vita-epfl/looking
    def get_model(self):
        model = LookingModel(INPUT_SIZE)
        print("Running on: ", self.device)
        if not os.path.isfile(os.path.join(self.path_model, 'LookingModel_LOOK+PIE.p')):
            """
            DOWNLOAD(LOOKING_MODEL, os.path.join(self.path_model, 'Looking_Model.zip'), quiet=False)
            with ZipFile(os.path.join(self.path_model, 'Looking_Model.zip'), 'r') as zipObj:
                # Extract all the contents of zip file in current directory
                zipObj.extractall()
            exit(0)"""
            raise NotImplementedError
        model.load_state_dict(torch.load(os.path.join(
            self.path_model, 'LookingModel_LOOK+PIE.p'), map_location=self.device))
        model.eval()
        return model

    # Source: https://github.com/vita-epfl/looking
    def predict_look(self, im_size, batch_wise=True):
        final_keypoints = []
        if batch_wise:
            if len(self.boxes) != 0:
                for i in range(len(self.boxes)):
                    kps = self.keypoints[i]
                    kps_final = np.array(
                        [kps[0], kps[1], kps[2]]).flatten().tolist()
                    X, Y = kps_final[:17], kps_final[17:34]
                    X, Y = normalize_by_image_(X, Y, im_size)
                    # X, Y = normalize(X, Y, divide=True, height_=False)
                    kps_final_normalized = np.array(
                        [X, Y, kps_final[34:]]).flatten().tolist()
                    final_keypoints.append(kps_final_normalized)
                tensor_kps = torch.Tensor([final_keypoints]).to(self.device)
                out_labels = self.model(tensor_kps.squeeze(
                    0)).detach().cpu().numpy().reshape(-1)
            else:
                out_labels = []
        else:
            if len(self.boxes) != 0:
                for i in range(len(self.boxes)):
                    kps = self.keypoints[i]
                    kps_final = np.array(
                        [kps[0], kps[1], kps[2]]).flatten().tolist()
                    X, Y = kps_final[:17], kps_final[17:34]
                    X, Y = normalize_by_image_(X, Y, im_size)
                    # X, Y = normalize(X, Y, divide=True, height_=False)
                    kps_final_normalized = np.array(
                        [X, Y, kps_final[34:]]).flatten().tolist()
                    # final_keypoints.append(kps_final_normalized)
                    tensor_kps = torch.Tensor(
                        kps_final_normalized).to(self.device)
                    out_labels = self.model(tensor_kps.unsqueeze(
                        0)).detach().cpu().numpy().reshape(-1)
            else:
                out_labels = []
        return out_labels

    # Source: https://github.com/vita-epfl/looking
    def render_image(self, image, pred_labels, header):
        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        mask = np.zeros(open_cv_image.shape, dtype=np.uint8)
        for i, label in enumerate(pred_labels):

            if label > self.eyecontact_thresh:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            mask = draw_skeleton(mask, self.keypoints[i], color)
        mask = cv2.erode(mask, (7, 7), iterations=1)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        open_cv_image = cv2.addWeighted(open_cv_image, 1, mask, self.transparency, 1.0)

        ros_image = self.bridge.cv2_to_imgmsg(open_cv_image, encoding='bgr8')
        ros_image.header = header
        self.look_debug_pub.publish(ros_image)

    def project_2d_box_to_3d(self):
        projected_boxes_3d = []

        for keypoints, box in zip(self.keypoints, self.boxes):
            x1, y1, x2, y2 = map(round, box[:4])
            median_depth = self.get_person_depth(keypoints)
            if np.isnan(median_depth):
                continue
            corners_2d = [
                (x1, y1),   # Top-left
                (x2, y1),   # Top-right
                (x1, y2),   # Bottom-left
                (x2, y2)    # Bottom-right
            ]
            corners_3d = [
                self.project_2d_to_3d(cx, cy, median_depth) for cx, cy in corners_2d
            ]
            projected_boxes_3d.append(corners_3d)

        return projected_boxes_3d

    def project_2d_to_3d(self, x, y, depth_value):
        fx, fy, cx, cy = [self.intrinsics[k] / self.downscale_factor for k in ['fx', 'fy', 'cx', 'cy']]
        Z = depth_value
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy
        return np.array([X, Y, Z])

    def get_person_depth(self, person_keypoints):
        X, Y, _, _ = convert(person_keypoints)

        torso_indices = [5, 6, 11, 12] # Left shoulder, Right shoulder, Left hip, Right hip
        if any(np.isnan(X[idx]) or np.isnan(Y[idx]) for idx in torso_indices):
            return np.nan

        torso_polygon_points = np.array([
            [int(round(X[idx])), int(round(Y[idx]))] for idx in torso_indices
        ], dtype=np.int32)

        x, y, w, h = cv2.boundingRect(torso_polygon_points)

        img_height, img_width = self.depth_image_cv.shape[:2]

        x = np.clip(x, 0, img_width - 1)
        y = np.clip(y, 0, img_height - 1)
        w = np.clip(x+w, 0, img_width - 1) - x
        h = np.clip(y+h, 0, img_height - 1) - y

        roi_polygon_points = torso_polygon_points - [x, y]
        depth_image_roi = self.depth_image_cv[y:y+h, x:x+w]

        torso_mask_roi = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(torso_mask_roi, [roi_polygon_points], 255)

        torso_depth_values = depth_image_roi[torso_mask_roi == 255]
        valid_torso_depth_values = torso_depth_values[
            ~np.isnan(torso_depth_values) & ~np.isinf(torso_depth_values)
        ]

        if valid_torso_depth_values.size == 0:
            return np.nan

        return np.median(valid_torso_depth_values)

    def publish_3d_bounding_boxes(self, boxes_3d, header):
        marker_array = MarkerArray()

        for i, front_corners in enumerate(boxes_3d):
            marker = Marker()
            marker.header = header
            marker.ns = "bounding_boxes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            x_coords = [corner[0] for corner in front_corners]
            y_coords = [corner[1] for corner in front_corners]
            z_coords = [corner[2] for corner in front_corners]
            z_front = z_coords[0]  # All z_coords are the same

            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            depth = width  # Assuming depth equals width

            center_x = (min(x_coords) + max(x_coords)) / 2.0
            center_y = (min(y_coords) + max(y_coords)) / 2.0
            center_z = z_front  # Since all z_coords are the same

            marker.pose.position.x = center_x
            marker.pose.position.y = center_y
            marker.pose.position.z = center_z

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = width
            marker.scale.y = height
            marker.scale.z = depth

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.7

            marker_array.markers.append(marker)

        for idx, m in enumerate(marker_array.markers):
            m.id = idx

        self.marker_pub.publish(marker_array)

    def publish_zed_object(self, boxes_3d):
        zed_detections = ObjectsStamped()
        zed_detections.header.frame_id = 'zed_left_camera_optical_frame'
        zed_detections.header.stamp = self.get_clock().now().to_msg()

        for i, front_corners in enumerate(boxes_3d):

            x_coords = [corner[0] for corner in front_corners]
            y_coords = [corner[1] for corner in front_corners]
            z_coords = [corner[2] for corner in front_corners]
            z_front = z_coords[0]  # All z_coords are the same

            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            depth = width  # Assuming depth equals width

            center_x = (min(x_coords) + max(x_coords)) / 2.0
            center_y = (min(y_coords) + max(y_coords)) / 2.0
            center_z = z_front  # Since all z_coords are the same

            det_object = Object()
            det_object.label = 'person'
            det_object.label_id = i
            det_object.position[0] = center_x
            det_object.position[1] = center_y
            det_object.position[2] = center_z
            det_object.tracking_available = False

            # Top face
            det_object.bounding_box_3d.corners[0].kp[0] = center_x - (width / 2)
            det_object.bounding_box_3d.corners[0].kp[1] = center_y - (height / 2)
            det_object.bounding_box_3d.corners[0].kp[2] = center_z - (depth / 2)

            det_object.bounding_box_3d.corners[1].kp[0] = center_x - (width / 2)
            det_object.bounding_box_3d.corners[1].kp[1] = center_y - (height / 2)
            det_object.bounding_box_3d.corners[1].kp[2] = center_z + (depth / 2)

            det_object.bounding_box_3d.corners[2].kp[0] = center_x + (width / 2)
            det_object.bounding_box_3d.corners[2].kp[1] = center_y - (height / 2)
            det_object.bounding_box_3d.corners[2].kp[2] = center_z + (depth / 2)

            det_object.bounding_box_3d.corners[3].kp[0] = center_x + (width / 2)
            det_object.bounding_box_3d.corners[3].kp[1] = center_y - (height / 2)
            det_object.bounding_box_3d.corners[3].kp[2] = center_z - (depth / 2)

            # Bottom face
            det_object.bounding_box_3d.corners[4].kp[0] = center_x - (width / 2)
            det_object.bounding_box_3d.corners[4].kp[1] = center_y + (height / 2)
            det_object.bounding_box_3d.corners[4].kp[2] = center_z - (depth / 2)

            det_object.bounding_box_3d.corners[5].kp[0] = center_x - (width / 2)
            det_object.bounding_box_3d.corners[5].kp[1] = center_y + (height / 2)
            det_object.bounding_box_3d.corners[5].kp[2] = center_z + (depth / 2)

            det_object.bounding_box_3d.corners[6].kp[0] = center_x + (width / 2)
            det_object.bounding_box_3d.corners[6].kp[1] = center_y + (height / 2)
            det_object.bounding_box_3d.corners[6].kp[2] = center_z + (depth / 2)

            det_object.bounding_box_3d.corners[7].kp[0] = center_x + (width / 2)
            det_object.bounding_box_3d.corners[7].kp[1] = center_y + (height / 2)
            det_object.bounding_box_3d.corners[7].kp[2] = center_z - (depth / 2)

            det_object.dimensions_3d[0] = width
            det_object.dimensions_3d[1] = depth
            det_object.dimensions_3d[2] = height

            det_object.skeleton_available = False

            zed_detections.objects.append(det_object)

        self.zed_object_pub.publish(zed_detections)


def remap_keypoint(source_list):
    # Remaps body keypoints from ZED (BODY_18) to PifPaf (COCO 17) format.
    source_indices = [0, 14, 15, 16, 17, 2, 5, 3, 6, 4, 7, 8, 11, 9, 12, 10, 13]
    return [source_list[i] for i in source_indices]


def main(args=None):
    rclpy.init(args=args)
    node = LookWrapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
