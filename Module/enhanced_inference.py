# Module/enhanced_inference.py
"""
增強的推論管理器，支援 YOLOv5 和 YOLOv8 ONNX 模型。
自動偵測模型格式並相應地處理輸出。
"""
import os
import time
import numpy as np
import onnxruntime as ort
import cv2
import traceback
import ast
from Module.logger import logger
from Module.config import Config

class EnhancedInferenceManager:
    """支援 YOLOv5 和 YOLOv8 格式的增強型 ONNX 推論管理器。"""
    
    def __init__(self, app_instance, yolo_version):
        """
        初始化增強型 ONNX 推論管理器。
        Args:
            app_instance: 應用程式的主類別實例 (SudaneseboyApp)。
            yolo_version: 指定的 YOLO 版本 ('v5' 或 'v8')。
        """
        if yolo_version not in ['v5', 'v8']:
            raise ValueError("yolo_version must be 'v5' or 'v8'")
            
        self.app = app_instance
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.output_name = None
        self.output_shape = None
        self.model_names = {}
        self.yolo_version = yolo_version
        self.num_classes = 80  # Default, will be updated based on model
    
    def initialize(self) -> bool:
        """載入 ONNX 模型並準備進行推論。"""
        # 直接從配置中獲取模型路徑，而不是依賴UI元件
        model_path = Config.get("model_file")
        
        if not model_path:
            logger.error("Model file path is not configured in settings.")
            return False
        
        # 處理打包後的資源路徑
        if not os.path.isabs(model_path):
            from Module.utils import resource_path
            model_path = resource_path(model_path)

        try:
            if not os.path.exists(model_path):
                logger.error(f"ONNX model file not found: {model_path}")
                return False

            logger.info(f"Loading model from: {model_path}")

            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']

            # 載入標準 ONNX 模型
            logger.info("Loading standard ONNX model.")
            self.session = ort.InferenceSession(model_path, providers=providers)
            
            provider = self.session.get_providers()[0]
            logger.info(f"ONNX Runtime is using provider: {provider}")

            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_name = self.session.get_outputs()[0].name
            self.output_shape = self.session.get_outputs()[0].shape
            
            # Set number of classes based on the specified YOLO version and model output shape
            logger.info(f"🔒 Using specified YOLO version: {self.yolo_version}")
            if len(self.output_shape) == 3:
                if self.yolo_version == 'v5':
                    # YOLOv5 format: [batch, anchors, 5 + num_classes]
                    self.num_classes = max(1, self.output_shape[2] - 5)
                else: # 'v8'
                    # YOLOv8 format: [batch, 4 + num_classes, anchors]
                    self.num_classes = max(1, self.output_shape[1] - 4)
            else:
                # Fallback for unexpected output shapes
                self.num_classes = 80
                logger.warning(f"Unexpected output shape {self.output_shape}. Defaulting to {self.num_classes} classes.")
            
            logger.info(f"Number of classes: {self.num_classes}")
            
            # Update the main app's model_size directly
            self.app.model_size = self.input_shape[2]
            
            # Try to get class names from model metadata
            meta = self.session.get_modelmeta()
            if 'names' in meta.custom_metadata_map:
                try:
                    self.model_names = ast.literal_eval(meta.custom_metadata_map['names'])
                    logger.info(f"Model class names loaded: {self.model_names}")
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Could not parse model names from metadata: {e}. Using numeric IDs.")
                    self.model_names = {}
            else:
                logger.warning("Could not find class names in model metadata. Using numeric IDs.")

            logger.info(f"Model input name: {self.input_name}, shape: {self.input_shape}")
            logger.info(f"Model output name: {self.output_name}, shape: {self.output_shape}")

            
            # Warm-up the model
            logger.info("Warming up the ONNX model...")
            dummy_input = np.zeros(self.input_shape, dtype=np.float32)
            for _ in range(5):
                self.session.run([self.output_name], {self.input_name: dummy_input})
            
            logger.info(f"Enhanced ONNX Runtime engine initialized successfully for YOLO{self.yolo_version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ONNX Runtime engine: {e}")
            logger.error(traceback.format_exc())
            return False

    def process_yolov5_output(self, output: np.ndarray) -> tuple:
        """
        處理 YOLOv5 輸出格式。
        YOLOv5 輸出形狀: [batch, anchors, 85+] 其中 85+ = x,y,w,h,conf + classes
        """
        try:
            # Remove batch dimension
            predictions = output[0]  # Shape: [anchors, 85+]
            
            logger.debug(f"YOLOv5 predictions shape: {predictions.shape}")
            logger.debug(f"YOLOv5 raw output range: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
            
            # Extract boxes, objectness, and class predictions
            boxes = predictions[:, :4]  # x, y, w, h (center format)
            objectness = predictions[:, 4]  # confidence score
            class_probs = predictions[:, 5:]  # class probabilities
            
            logger.debug(f"Boxes shape: {boxes.shape}, Objectness shape: {objectness.shape}, Class probs shape: {class_probs.shape}")
            logger.debug(f"Raw objectness range: [{np.min(objectness):.4f}, {np.max(objectness):.4f}]")
            logger.debug(f"Raw class_probs range: [{np.min(class_probs):.4f}, {np.max(class_probs):.4f}]")
            
            # Check if sigmoid activation is needed
            # If values are not in [0,1] range, apply sigmoid
            if np.min(objectness) < 0 or np.max(objectness) > 1:
                logger.debug("Applying sigmoid to objectness scores")
                objectness = 1 / (1 + np.exp(-objectness))  # sigmoid
            
            if np.min(class_probs) < 0 or np.max(class_probs) > 1:
                logger.debug("Applying sigmoid to class probabilities")
                class_probs = 1 / (1 + np.exp(-class_probs))  # sigmoid
            
            # For YOLOv5, keep boxes in their original scale (don't auto-scale here)
            # The main application will handle scaling based on YOLO version
            logger.debug(f"YOLOv5 boxes range: [{np.min(boxes):.2f}, {np.max(boxes):.2f}]")
            # Note: Removed automatic scaling to let main app handle it properly
            
            # Calculate final scores (objectness * class_prob)
            # Reshape objectness to broadcast correctly
            objectness_expanded = objectness[:, np.newaxis]  # Shape: [anchors, 1]
            scores = objectness_expanded * class_probs  # Shape: [anchors, num_classes]
            
            # Get the best class for each detection
            max_scores = np.max(scores, axis=1)  # Shape: [anchors]
            class_ids = np.argmax(scores, axis=1)  # Shape: [anchors]
            
            logger.debug(f"Final scores range: [{np.min(max_scores):.4f}, {np.max(max_scores):.4f}]")
            logger.debug(f"Valid detections (>0.01): {np.sum(max_scores > 0.01)}")
            logger.debug(f"Valid detections (>0.1): {np.sum(max_scores > 0.1)}")
            logger.debug(f"Valid detections (>0.3): {np.sum(max_scores > 0.3)}")
            
            return boxes, max_scores, class_ids
            
        except Exception as e:
            logger.error(f"Error processing YOLOv5 output: {e}")
            logger.error(f"Output shape: {output.shape if output is not None else 'None'}")
            import traceback
            logger.error(traceback.format_exc())
            return np.array([]), np.array([]), np.array([])

    def process_yolov8_output(self, output: np.ndarray) -> tuple:
        """
        處理 YOLOv8 輸出格式。
        YOLOv8 輸出形狀: [batch, 84+, anchors] 其中 84+ = 4 + classes
        """
        try:
            # Remove batch dimension and transpose
            predictions = output[0].T  # Shape: [anchors, 84+]
            
            logger.debug(f"YOLOv8 predictions shape: {predictions.shape}")
            
            # Extract boxes and class predictions
            boxes = predictions[:, :4]  # x, y, w, h (center format)
            class_probs = predictions[:, 4:]  # class probabilities (no separate objectness)
            
            logger.debug(f"Boxes shape: {boxes.shape}, Class probs shape: {class_probs.shape}")
            logger.debug(f"YOLOv8 boxes range: [{np.min(boxes):.2f}, {np.max(boxes):.2f}]")
            
            # For YOLOv8, the max class probability is the confidence
            max_scores = np.max(class_probs, axis=1)  # Shape: [anchors]
            class_ids = np.argmax(class_probs, axis=1)  # Shape: [anchors]
            
            logger.debug(f"Max scores range: [{np.min(max_scores):.4f}, {np.max(max_scores):.4f}]")
            logger.debug(f"Valid detections (>0.01): {np.sum(max_scores > 0.01)}")
            
            return boxes, max_scores, class_ids
            
        except Exception as e:
            logger.error(f"Error processing YOLOv8 output: {e}")
            logger.error(f"Output shape: {output.shape if output is not None else 'None'}")
            import traceback
            logger.error(traceback.format_exc())
            return np.array([]), np.array([]), np.array([])

    def run_inference(self, frame: np.ndarray) -> tuple[tuple, float]:
        """
        Run inference on a single frame and return processed results.
        Returns: ((boxes, scores, class_ids), inference_time)
        """
        try:
            start_time = time.perf_counter()
            
            # Preprocess input
            input_img = cv2.resize(frame, (self.input_shape[3], self.input_shape[2]))
            input_img = input_img.astype(np.float32) / 255.0
            input_img = np.transpose(input_img, (2, 0, 1))
            input_img = np.expand_dims(input_img, axis=0)
            input_img = np.ascontiguousarray(input_img)

            # Run inference
            output = self.session.run([self.output_name], {self.input_name: input_img})[0]
            
            inference_time = (time.perf_counter() - start_time) * 1000  # in ms

            # Process output based on YOLO version
            if self.yolo_version == 'v5':
                boxes, scores, class_ids = self.process_yolov5_output(output)
                logger.debug(f"YOLOv5 processing result: {len(boxes)} boxes, max score: {np.max(scores) if len(scores) > 0 else 'N/A'}")
            else:  # v8 or unknown (default to v8)
                boxes, scores, class_ids = self.process_yolov8_output(output)
                logger.debug(f"YOLOv8 processing result: {len(boxes)} boxes, max score: {np.max(scores) if len(scores) > 0 else 'N/A'}")
            
            # Additional debugging for low detection rates
            if len(scores) > 0:
                high_conf_detections = np.sum(scores > 0.5)
                medium_conf_detections = np.sum(scores > 0.25)
                low_conf_detections = np.sum(scores > 0.1)
                logger.debug(f"Detection confidence distribution: >0.5: {high_conf_detections}, >0.25: {medium_conf_detections}, >0.1: {low_conf_detections}")
            
            return (boxes, scores, class_ids), inference_time
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            logger.error(traceback.format_exc())
            return (np.array([]), np.array([]), np.array([])), 0.0

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Enhanced ONNX Runtime resources released")
        self.session = None