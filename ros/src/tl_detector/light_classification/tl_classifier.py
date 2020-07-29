from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2


class TLClassifier(object):
    def __init__(self, site=False):
        self.__model_loaded = False
        self.session = None
        self.tf_graph = None
        self.load_model(not site)

    def load_model(self, simulator):
        if simulator:
            self.tf_graph = load_graph('../../models/sim.pb')
        else:
            self.tf_graph = load_graph('../../models/real.pb')
        self.config = tf.ConfigProto(log_device_placement=False)
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.8
        self.config.operation_timeout_in_ms = 50000
        with self.tf_graph.as_default():
            self.tf_session = tf.Session(graph=self.tf_graph, config=self.config)
        self.__model_loaded = True

    def get_classification(self, image, score_threshold=0.5):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if not self.__model_loaded:
            return TrafficLight.UNKNOWN

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

        image_tensor = self.tf_graph.get_tensor_by_name('prefix/image_tensor:0')
        detection_scores = self.tf_graph.get_tensor_by_name('prefix/detection_scores:0')
        num_detections = self.tf_graph.get_tensor_by_name('prefix/num_detections:0')
        detection_classes = self.tf_graph.get_tensor_by_name('prefix/detection_classes:0')
        detection_boxes = self.tf_graph.get_tensor_by_name('prefix/detection_boxes:0')
        # Get the scores, classes and number of detections
        (boxes, scores, classes, num) = self.tf_session.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        for j, box in enumerate(boxes):
            width = (box[3] - box[1]) * image.shape[1]
            height = (box[2] - box[0]) * image.shape[0]
            # only tall boxes are close enough to be interesting traffic lights
            if height > 42:
                if scores[j] > score_threshold:
                    return classes[j] - 1 # model trained with {'red': 1, 'yellow': 2, 'green':3}, but we want idices of array ["RED", "YELLOW", "GREEN"]
        # if no box of sufficient height exceeds the score threshold, we don't have a recognized traffic light
        return TrafficLight.UNKNOWN

def load_graph (graph_file):
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='prefix')
    return graph
