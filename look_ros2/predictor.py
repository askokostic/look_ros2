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
        self.declare_parameter('image_topic', '/image_topic')
        self.image_topic = self.get_parameter('image_topic').value

        self.transparency = 0.4
        self.eyecontact_thresh = 0.5
        self.path_model = os.path.join(os.path.dirname(__file__), '../looking/models/predictor')

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pifpaf
        self.pifpaf_predictor = load_pifpaf(self.parse_pifpaf_args())

        # Load LOOK model
        self.model = self.get_model().to(self.device)

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_cb,
            1
        )

        self.publisher_ = self.create_publisher(Image, '/looking', 10)

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

    def image_cb(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            pil_image = PILImage.fromarray(cv_image)
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        predictions, _, _ = self.pifpaf_predictor.pil_image(pil_image)
        pred = [ann.json_data() for ann in predictions]

        im_size = (pil_image.size[0], pil_image.size[1])
        boxes, keypoints = preprocess_pifpaf(pred, im_size, enlarge_boxes=False)
        if self.mode == 'joints':
            pred_labels = self.predict_look(boxes, keypoints, im_size)
        else:
            pred_labels = self.predict_look_alexnet(boxes, pil_image)

        self.render_image(pil_image, boxes, keypoints, pred_labels)

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
    def predict_look(self, boxes, keypoints, im_size, batch_wise=True):
        final_keypoints = []
        if batch_wise:
            if len(boxes) != 0:
                for i in range(len(boxes)):
                    kps = keypoints[i]
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
            if len(boxes) != 0:
                for i in range(len(boxes)):
                    kps = keypoints[i]
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
    def predict_look_alexnet(self, boxes, image, batch_wise=True):
        out_labels = []
        data_transform = transforms.Compose([
            SquarePad(),
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        if len(boxes) != 0:
            if batch_wise:
                heads = []
                for i in range(len(boxes)):
                    bbox = boxes[i]
                    x1, y1, x2, y2, _ = bbox
                    w, h = abs(x2-x1), abs(y2-y1)
                    head_image = Image.fromarray(
                        np.array(image)[int(y1):int(y1+(h/3)), int(x1):int(x2), :])
                    head_tensor = data_transform(head_image)
                    heads.append(head_tensor.detach().cpu().numpy())
                for i in range(len(boxes)):
                    bbox = boxes[i]
                    x1, y1, x2, y2, _ = bbox
                    w, h = abs(x2-x1), abs(y2-y1)
                    head_image = Image.fromarray(
                        np.array(image)[int(y1):int(y1+(h/3)), int(x1):int(x2), :])
                    head_tensor = data_transform(head_image)
                    looking_label = self.model(torch.Tensor(head_tensor).unsqueeze(
                        0).to(self.device)).detach().cpu().numpy().reshape(-1)[0]
                    out_labels.append(looking_label)
        else:
            out_labels = []
        return out_labels

    # Source: https://github.com/vita-epfl/looking
    def render_image(self, image, bbox, keypoints, pred_labels):
        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        mask = np.zeros(open_cv_image.shape, dtype=np.uint8)
        for i, label in enumerate(pred_labels):

            if label > self.eyecontact_thresh:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            mask = draw_skeleton(mask, keypoints[i], color)
        mask = cv2.erode(mask, (7, 7), iterations=1)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        open_cv_image = cv2.addWeighted(open_cv_image, 1, mask, self.transparency, 1.0)
        
        ros_image = self.bridge.cv2_to_imgmsg(open_cv_image, encoding='bgr8')
        self.publisher_.publish(ros_image)


def main(args=None):
    rclpy.init(args=args)
    node = LookWrapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
