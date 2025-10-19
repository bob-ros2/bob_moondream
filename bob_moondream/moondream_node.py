import json
import time
import threading
import rclpy
import cv2
from PIL import Image as PILImage
from cv_bridge import CvBridge
from rclpy.node import Node
from std_msgs.msg import String   
from sensor_msgs.msg import Image   
from vision_msgs.msg import Detection2DArray, Detection2D
from rclpy.callback_groups import ReentrantCallbackGroup
from bob_moondream_msgs.srv import VisualQuery
from transformers import AutoModelForCausalLM

class ImageCache:
    def __init__(self, node):
        self.node = node
        self.latest_msg = None
        self.latest_image = None  # Store the latest PIL Image
        self.latest_cv_image = None # Store the latest cv2 image
        self.latest_image_time = 0
        self.lock = threading.Lock()

    def update(self, image_msg):
        with self.lock:
            self.latest_msg = image_msg

    def cv_image(self):
        return self.get()[0]

    def pil_image(self):
        return self.get()[1]

    def get(self):
        with self.lock:
            if self.latest_msg:
                msg_to_process = self.latest_msg
                self.latest_msg = None  # Consume message immediately
                try:
                    self.latest_cv_image = \
                        self.node.bridge.imgmsg_to_cv2(
                            msg_to_process, "bgr8")
                    # Convert BGR to RGB
                    self.latest_image = PILImage.fromarray(
                        self.latest_cv_image[:, :, ::-1])
                    self.latest_image_time = \
                        msg_to_process.header.stamp.sec \
                            + msg_to_process.header.stamp.nanosec/1000000000.0
                except Exception as e:
                    self.node.get_logger().error(
                        f"Could not convert image: {e}")
        return (self.latest_cv_image, self.latest_image)

class MoondreamNode(Node):
    def __init__(self):
        """
        Initialize MoondreamNode.
        """
        super().__init__('moondream')

        # Use ReentrantCallbackGroup to allow service calls 
        # to use the same resources as the subscriber.
        self.callback_group = ReentrantCallbackGroup()

        # ROS Parameters
        self.model_name = self.declare_parameter(
            'model_name', "vikhyatk/moondream2").value

        self.model_revision = self.declare_parameter(
            'model_revision', "2025-06-21").value

        self.cache_dir = self.declare_parameter(
            'cache_dir', "./models").value

        self.frame_id = self.declare_parameter(
            'frame_id', "moondream").value

        self.device = self.declare_parameter(
            'device', "cuda").value

        # Parameters for continuous detection or pointing.
        # These params can be changed during runtime.
        self.declare_parameter('prompt_point', "")
        self.declare_parameter('prompt_detect', "")

        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.model_revision,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            device_map={"": self.device}
        )
        self.get_logger().info(
            f"Moondream model loaded from {self.model_name}")

        # Image handler 
        self.bridge = CvBridge()
        self.image_cache = ImageCache(self)

        # Image subscriber
        self.subscription = self.create_subscription(
            Image,
            'image_input',
            self.image_callback,
            10,
            callback_group=self.callback_group)

        # Detection2DArray publisher
        self.detection_publisher = self.create_publisher(
            Detection2DArray,
            'detected_objects',
            10)

        # Point publishers
        self.point_publisher = self.create_publisher(
            Detection2DArray,
            'pointed_objects',
            10)

        # Image publisher
        self.annotated_image_publisher = self.create_publisher(
            Image,
            'annotated_image',
            10)

        # String publishers for results

        self.visual_query_result_publisher = \
            self.create_publisher(
                String, 'visual_query_result', 10)
        
        self.object_detection_result_publisher = \
            self.create_publisher(
                String, 'object_detection_result', 10)

        self.pointing_result_publisher = \
            self.create_publisher(
                String, 'pointing_result', 10)

        # String subscribers for prompts

        self.visual_query_prompt_subscriber = self.create_subscription(
            String,
            'visual_query_prompt',
            self.visual_query_prompt_callback,
            10,
            callback_group=self.callback_group)

        self.object_detection_prompt_subscriber = self.create_subscription(
            String,
            'object_detection_prompt',
            self.object_detection_prompt_callback,
            10,
            callback_group=self.callback_group)

        self.pointing_prompt_subscriber = self.create_subscription(
            String,
            'pointing_prompt',
            self.pointing_prompt_callback,
            10,
            callback_group=self.callback_group)

        # Service definitions

        self.caption_service = self.create_service(
            VisualQuery,
            'caption',
            self.caption_callback,
            callback_group=self.callback_group)

        self.visual_query_service = self.create_service(
            VisualQuery,
            'visual_query',
            self.visual_query_callback,
            callback_group=self.callback_group)

        self.object_detection_service = self.create_service(
            VisualQuery,
            'object_detection',
            self.object_detection_callback,
            callback_group=self.callback_group)

        self.pointing_service = self.create_service(
            VisualQuery,
            'pointing',
            self.pointing_callback,
            callback_group=self.callback_group)

        self.get_logger().info(
            "Moondream node initialized")

    #  Callbacks sensor_msgs Image

    def image_callback(self, msg):
        """
        Callback function for the image subscriber.

        Args:
            msg (sensor_msgs.msg.Image): The ROS Image message received from the image topic.
        """

        start_time = time.monotonic()
        self.image_cache.update(msg)
        self.annotated_image = None

        if self.get_parameter('prompt_detect').value:
            self.detect()

        if self.get_parameter('prompt_point').value:
            self.point()

        end_time = time.monotonic()
        processing_time_ms = (end_time - start_time) * 1000
        self.get_logger().debug(
            f"Image processing time: {processing_time_ms:.3f} ms")

        if self.annotated_image is not None:
            image_msg = self.bridge.cv2_to_imgmsg(
                self.annotated_image, encoding="bgr8")
            self.annotated_image_publisher.publish(image_msg)
            self.get_logger().info("Published annotated point image")

    # Callbacks std_msgs String

    def call_service(self, service, prompt):
        """Wrapper to call one of the VisualQuery services."""
        res = VisualQuery.Response()
        service(VisualQuery.Request(prompt=prompt), res)
        return res.response

    def visual_query_prompt_callback(self, msg):
        """Callback for topic-based visual query requests."""
        self.visual_query_result_publisher.publish(
            String(data=self.call_service(
                self.visual_query_callback,
                msg.data)))

    def object_detection_prompt_callback(self, msg):
        """Callback for topic-based object detection requests."""
        self.object_detection_result_publisher.publish(
            String(data=self.call_service(
                self.object_detection_callback,
                msg.data)))

    def pointing_prompt_callback(self, msg):
        """Callback for topic-based pointing requests."""
        self.pointing_result_publisher.publish(
            String(data=self.call_service(
                self.pointing_callback,
                msg.data)))

    # Realtime functions

    def detect(self):
        """
        Performs object detection and publishes results if subscribers are present.

        This method is intended to be called periodically. It checks for active subscribers
        on the detection topics. If any exist, it calls the model to find all objects
        in the provided image and then uses `publish_detections` to process and
        publish the results.
        """
        if self.detection_publisher.get_subscription_count() > 0 \
            or self.annotated_image_publisher.get_subscription_count() > 0:
            try:
                objects = self.model.detect(
                    self.image_cache.pil_image(), 
                    self.get_parameter(
                        'prompt_detect').value)["objects"]
                if len(objects):
                    self.publish_detections(
                        objects, 
                        self.image_cache.cv_image())
                else:
                    self.get_logger().info(
                        "detect: No detections in the last image")
            except Exception as e:
                self.get_logger().error(f"detect: {e}")

    def publish_detections(self, objects, image):
        """
        Processes and publishes object detection results.

        This function converts the raw detection data from the model into a
        `vision_msgs/Detection2DArray` message and publishes it. It also draws
        the bounding boxes on a copy of the input image.

        Args:
            objects (list): A list of dictionaries, where each dictionary represents a detected object.
            image (numpy.ndarray): The OpenCV image on which to draw the detections.

        Returns:
            bool: True if publishing was successful, False otherwise.
        """
        try:
            detection_array = Detection2DArray()
            detection_array.header.stamp = self.get_clock().now().to_msg()
            detection_array.header.frame_id = self.frame_id
            if self.annotated_image_publisher.get_subscription_count():
                self.annotated_image = self.annotated_image \
                    if self.annotated_image is not None else image.copy()
            h, w, _ = image.shape

            for obj in objects:
                detection = Detection2D()
                x_min = int(obj['x_min'] * w)
                y_min = int(obj['y_min'] * h)
                x_max = int(obj['x_max'] * w)
                y_max = int(obj['y_max'] * h)
                # Calculate the center of the bounding box
                x_center = (x_min + x_max) / 2.0
                y_center = (y_min + y_max) / 2.0
                detection.bbox.center.position.x = x_center
                detection.bbox.center.position.y = y_center
                # Bounding box size (must be scaled by image dimensions)
                detection.bbox.size_x = float((obj['x_max'] - obj['x_min']) * w)
                detection.bbox.size_y = float((obj['y_max'] - obj['y_min']) * h)
                detection_array.detections.append(detection)
                if self.annotated_image is not None:
                    cv2.rectangle( # Green box
                        self.annotated_image, 
                        (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Publish the Detection2DArray message
            if self.detection_publisher.get_subscription_count():
                self.detection_publisher.publish(detection_array)
                self.get_logger().info(f"Published {len(objects)} detections to detected_objects")

            return True

        except Exception as e:
            self.get_logger().error(f"Failed to publish detections: {e}")
            return False

    def point(self):
        """
        Performs pointing and publishes results if subscribers are present.

        This method is intended to be called periodically. It checks for active subscribers
        on the pointing topics. If any exist, it calls the model to find all points
        in the provided image based on the 'prompt_point' parameter and then uses
        `publish_points` to process and publish the results.
        """
        if self.point_publisher.get_subscription_count() > 0 \
            or self.annotated_image_publisher.get_subscription_count() > 0:
            try:
                points = self.model.point(
                    self.image_cache.pil_image(),
                    self.get_parameter('prompt_point').value)["points"]
                if len(points):
                    self.publish_points(
                        points,
                        self.image_cache.cv_image())
                else:
                    self.get_logger().info(
                        "point: No points found in the last image")
            except Exception as e:
                self.get_logger().error(f"point: {e}")
                
    def publish_points(self, points, image):
        """
        Processes and publishes pointing results.

        This function converts the raw point data from the model into a
        `vision_msgs/Detection2DArray` message (where each point is a zero-size bbox)
        and publishes it. It also draws markers on a copy of the input image.

        Args:
            points (list): A list of dictionaries, where each dict represents a point (e.g., {'x': 0.5, 'y': 0.5}).
            image (numpy.ndarray): The OpenCV image on which to draw the points.

        Returns:
            bool: True if publishing was successful, False otherwise.
        """
        try:
            detection_array = Detection2DArray()
            detection_array.header.stamp = self.get_clock().now().to_msg()
            detection_array.header.frame_id = self.frame_id
            if self.annotated_image_publisher.get_subscription_count() > 0:
                self.annotated_image = self.annotated_image \
                    if self.annotated_image is not None else image.copy()
            h, w, _ = image.shape

            for point in points:
                detection = Detection2D()
                x = int(point['x'] * w)
                y = int(point['y'] * h)

                detection.bbox.center.position.x = float(x)
                detection.bbox.center.position.y = float(y)
                # A point has no size, so size_x and size_y are 0
                detection.bbox.size_x = 0.0
                detection.bbox.size_y = 0.0
                detection_array.detections.append(detection)

                if self.annotated_image is not None:
                    cv2.drawMarker(
                        self.annotated_image, (x, y), (0, 0, 255),  # Red cross
                        markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            # Publish the Detection2DArray message
            if self.point_publisher.get_subscription_count():
                self.point_publisher.publish(detection_array)
                self.get_logger().info(
                    f"Published {len(points)} points to pointed_objects")

            return True

        except Exception as e:
            self.get_logger().error(f"Failed to publish points: {e}")
            return False

    # Services
            
    def check_image_availability(self):
        """
        Verifies that a recent image is available for processing.

        Checks if an image has been received and if it is not too old (older than 2 seconds).

        Returns:
            bool: True if a recent image is available, False otherwise.
        """
        if self.image_cache.pil_image() is None:
            self.get_logger().warn("No image received yet.")
            return False
        # Check if image is too old
        now = self.get_clock().now().seconds_nanoseconds()
        now_sec = now[0] + now[1] / 1000000000.0
        if (now_sec - self.image_cache.latest_image_time) > 2.0:
            self.get_logger().warn("No recent image received, using potentially old image.")
        return True

    def caption_callback(self, request, response):
        """
        ROS service callback to generate a caption for the latest image.

        Args:
            request: The service request object (prompt is ignored).
            response: The service response object to be populated.

        Returns:
            The populated service response object.
        """
        if not self.check_image_availability():
            response.response = "No image available."
            return response

        try:
            result = self.model.caption(
                self.image_cache.pil_image(), length="normal")["caption"]
            response.response = str(result)
            self.get_logger().info("Captioning successful")
        except Exception as e:
            response.response = f"Captioning failed: {e}"
            self.get_logger().error(f"Captioning failed: {e}")
        return response

    def visual_query_callback(self, request, response):
        """
        ROS service callback to answer a question about the latest image.

        Args:
            request: The service request object containing the text prompt (question).
            response: The service response object to be populated.

        Returns:
            The populated service response object.
        """
        if not self.check_image_availability():
            response.response = "No image available."
            return response
        try:
            result = self.model.query(
                self.image_cache.pil_image(), 
                request.prompt)["answer"]
            response.response = str(result)
            self.get_logger().info(
                f"Visual query successful: {response.response}")
        except Exception as e:
            response.response = f"Visual query failed: {e}"
            self.get_logger().error(f"Visual query failed: {e}")
        return response

    def object_detection_callback(self, request, response):
        """
        ROS service callback to detect objects in the latest image.

        The results are returned as a JSON string in the service response. If there are
        subscribers to the detection topics, it also calls `publish_detections`.

        Args:
            request: The service request object containing the prompt describing the object.
            response: The service response object to be populated.

        Returns:
            The populated service response object.
        """
        if not self.check_image_availability():
            response.response = "No image available."
            return response
        try:
            objects = self.model.detect(
                self.image_cache.pil_image(), request.prompt)["objects"]
            response.response = json.dumps(objects)
            self.get_logger().info(
                f"Object detection found {len(objects)} objects(s): {response.response}")
        except Exception as e:
            response.response = f"Object detection failed: {e}"
            self.get_logger().error(f"Object detection failed: {e}")
        return response

    def pointing_callback(self, request, response):
        """
        ROS service callback to find specific points in the latest image.

        Args:
            request: The service request object containing the prompt describing the point.
            response: The service response object to be populated with a JSON string of points.

        Returns:
            The populated service response object.
        """
        if not self.check_image_availability():
            response.response = "No image available."
            return response
        try:
            points = self.model.point(
                self.image_cache.pil_image(), 
                request.prompt)["points"]
            response.response = json.dumps(points)
            self.get_logger().info(
                f"Pointing successful, found {len(points)} point(s): {response.response}")
        except Exception as e:
            response.response = f"Pointing failed: {e}"
            self.get_logger().error(f"Pointing failed: {e}")
        return response

def main(args=None):
    rclpy.init(args=args)
    node = MoondreamNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()