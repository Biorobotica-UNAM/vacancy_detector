#!/usr/bin/env python3
"""YOLO-World puntual para delimitar contenedores antes de la geometria 3D."""

import gc
import threading
import time
from typing import List, Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool
from vision_interfaces.msg import YoloDetection
from vision_interfaces.srv import GetDetections


class ContainerYoloWorldNode(Node):
    def __init__(self):
        super().__init__("container_yolo_world_node")

        self.declare_parameter("image_topic", "/hardware/camera/image_raw")
        self.declare_parameter("model_name", "yolov8s-worldv2.pt")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("conf_th", 0.12)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("load_timeout_sec", 12.0)
        self.declare_parameter("auto_load_on_detect", False)
        self.declare_parameter(
            "classes",
            [
                "bucket",
                "trash can",
                "waste basket",
                "plastic bin",
                "container",
                "red bucket",
            ],
        )

        self.bridge = CvBridge()
        self.cb_group = ReentrantCallbackGroup()
        self.model = None
        self.model_ready = False
        self.loading = False
        self.last_error = ""
        self.active_classes: List[str] = []
        self.last_image: Optional[Image] = None
        self.model_lock = threading.RLock()
        self.load_thread: Optional[threading.Thread] = None

        image_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1)
        self.create_subscription(
            Image,
            self.get_parameter("image_topic").value,
            self.image_callback,
            image_qos,
            callback_group=self.cb_group,
        )

        self.create_service(
            SetBool,
            "container_yolo_world/enable",
            self.enable_callback,
            callback_group=self.cb_group,
        )
        self.create_service(
            GetDetections,
            "container_yolo_world/detection",
            self.detect_callback,
            callback_group=self.cb_group,
        )

        self.get_logger().info(
            "container_yolo_world listo. Modelo dormido hasta enable=true. "
            f"image_topic={self.get_parameter('image_topic').value}"
        )

    def image_callback(self, msg: Image):
        self.last_image = msg

    def enable_callback(self, req: SetBool.Request, res: SetBool.Response):
        if req.data:
            self.start_loading_async()
            res.success = True
            res.message = "container_yolo_world_loading"
            return res

        self.unload_model()
        res.success = True
        res.message = "container_yolo_world_unloaded"
        return res

    def detect_callback(self, req: GetDetections.Request, res: GetDetections.Response):
        queries = list(req.queries) if req.queries else self.default_classes()
        if "*" in queries:
            queries = self.default_classes()

        if not self.model_ready:
            if self.get_parameter("auto_load_on_detect").value and not self.loading:
                self.start_loading_async()
            if not self.wait_until_ready(self.get_parameter("load_timeout_sec").value):
                res.success = False
                res.message = (
                    "YOLO-World no listo: "
                    + (self.last_error or "modelo no cargado o timeout")
                )
                return res

        if self.last_image is None:
            res.success = False
            res.message = "No image received yet"
            return res

        try:
            bgr = self.bridge.imgmsg_to_cv2(self.last_image, desired_encoding="bgr8")
            detections = self.run_inference(bgr, queries)
        except RuntimeError as exc:
            if "Expected all tensors to be on the same device" not in str(exc):
                res.success = False
                res.message = f"Inference error: {exc}"
                self.get_logger().error(res.message)
                return res

            self.get_logger().warn(
                "YOLO-World fallo en CUDA por mezcla CPU/GPU; "
                "reintentando esta inferencia en CPU"
            )
            try:
                with self.model_lock:
                    if self.model is not None:
                        self.model.to("cpu")
                detections = self.run_inference(bgr, queries, force_device="cpu")
            except Exception as retry_exc:
                res.success = False
                res.message = f"Inference retry on CPU failed: {retry_exc}"
                self.get_logger().error(res.message)
                return res
        except Exception as exc:
            res.success = False
            res.message = f"Inference error: {exc}"
            self.get_logger().error(res.message)
            return res

        res.detections = detections
        res.success = len(detections) > 0
        seen = ", ".join(
            f"{d.class_name}:{d.confidence:.2f}" for d in detections
        )
        res.message = (
            f"Detected {len(detections)} with YOLO-World: {seen}"
            if res.success
            else f"Nothing found with YOLO-World queries={queries}"
        )
        self.get_logger().info(res.message)
        return res

    def start_loading_async(self):
        with self.model_lock:
            if self.model_ready:
                self.get_logger().info("YOLO-World ya estaba cargado")
                return
            if self.loading:
                self.get_logger().info("YOLO-World ya esta cargando")
                return
            self.loading = True
            self.last_error = ""
            self.load_thread = threading.Thread(target=self.load_model, daemon=True)
            self.load_thread.start()
            self.get_logger().info("Carga YOLO-World iniciada en background")

    def load_model(self):
        model_name = self.get_parameter("model_name").value
        device = self.get_parameter("device").value
        classes = self.default_classes()
        try:
            from ultralytics import YOLOWorld

            self.get_logger().info(
                f"Cargando YOLO-World model={model_name} device={device} "
                f"classes={classes}"
            )
            model = YOLOWorld(model_name)
            model.set_classes(classes)
            if device:
                model.to(device)
            model.predict(
                np.zeros((640, 640, 3), dtype=np.uint8),
                imgsz=self.get_parameter("imgsz").value,
                conf=self.get_parameter("conf_th").value,
                verbose=False,
            )

            with self.model_lock:
                self.model = model
                self.active_classes = classes
                self.model_ready = True
                self.loading = False
            self.get_logger().info("YOLO-World cargado y listo")
        except Exception as exc:
            with self.model_lock:
                self.model = None
                self.active_classes = []
                self.model_ready = False
                self.loading = False
                self.last_error = str(exc)
            self.get_logger().error(f"No se pudo cargar YOLO-World: {exc}")

    def unload_model(self):
        with self.model_lock:
            self.model = None
            self.active_classes = []
            self.model_ready = False
            self.loading = False
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self.get_logger().info("YOLO-World descargado")

    def wait_until_ready(self, timeout_sec: float) -> bool:
        start = time.monotonic()
        while time.monotonic() - start < timeout_sec:
            if self.model_ready:
                return True
            if not self.loading:
                return False
            time.sleep(0.05)
        return self.model_ready

    def run_inference(
        self, bgr, queries: List[str], force_device: Optional[str] = None
    ) -> List[YoloDetection]:
        with self.model_lock:
            model = self.model
        if model is None:
            return []

        device = force_device or self.get_parameter("device").value
        # Las clases se fijan una sola vez al cargar el modelo. Evitamos llamar
        # set_classes() aqui porque en YOLO-World puede dejar embeddings de
        # texto en CPU mientras el modelo esta en CUDA.
        if force_device:
            model.to(device)
        results = model.predict(
            bgr,
            imgsz=self.get_parameter("imgsz").value,
            conf=self.get_parameter("conf_th").value,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []

        result = results[0]
        with self.model_lock:
            active_classes = list(self.active_classes)
        names = result.names or {idx: name for idx, name in enumerate(active_classes)}
        wanted = {query.strip().lower() for query in queries}
        detections: List[YoloDetection] = []
        for box in result.boxes:
            det = YoloDetection()
            cls_id = int(box.cls[0])
            det.class_name = str(
                names.get(
                    cls_id,
                    active_classes[cls_id] if cls_id < len(active_classes) else cls_id,
                )
            )
            if wanted and det.class_name.strip().lower() not in wanted:
                continue
            det.confidence = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            det.x_min = x1
            det.y_min = y1
            det.x_max = x2
            det.y_max = y2
            detections.append(det)

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def default_classes(self) -> List[str]:
        return [str(item) for item in self.get_parameter("classes").value]


def main():
    rclpy.init()
    node = ContainerYoloWorldNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.unload_model()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
