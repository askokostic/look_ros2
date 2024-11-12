from looking.utils.network import *
from looking.utils.utils_predict import *
import os
import argparse
import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node

INPUT_SIZE = 51


class LookWrapper(Node):
    def __init__(self):
        super().__init__('look_wrapper')

        # Parameters
        self.declare_parameter('color_image_topic', '/image_topic')
        self.color_image_topic = self.get_parameter('color_image_topic').value

        # Variables
        self.transparency = 0.4
        self.eyecontact_thresh = 0.5
        self.path_model = os.path.join(os.path.dirname(__file__), '../looking/models/predictor')
        self.keypoints = None
        self.boxes = None

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pifpaf
        self.pifpaf_predictor = load_pifpaf(self.parse_pifpaf_args())

        # Load LOOK model
        self.model = self.get_model().to(self.device)

        self.bridge = CvBridge()

        self.color_img_sub = self.create_subscription(
            Image,
            self.color_image_topic,
            self.color_image_cb,
            10
        )

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

    def color_image_cb(self, color_img):
        try:
            color_image_cv = self.bridge.imgmsg_to_cv2(color_img, desired_encoding='rgb8')
            pil_image = PILImage.fromarray(color_image_cv)
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        predictions, _, _ = self.pifpaf_predictor.pil_image(pil_image)
        pred = [ann.json_data() for ann in predictions]

        im_size = (pil_image.size[0], pil_image.size[1])
        self.boxes, self.keypoints = preprocess_pifpaf(pred, im_size, enlarge_boxes=False)
        pred_labels = self.predict_look(im_size)

        self.render_image(pil_image, pred_labels, color_img.header)

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


def main(args=None):
    rclpy.init(args=args)
    node = LookWrapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
