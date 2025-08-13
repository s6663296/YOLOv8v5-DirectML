# Module/optimized_nms.py
"""
用於提升效能的優化 NMS（非極大值抑制）演算法。
以更快的實作和預先篩選取代 OpenCV 的 NMSBoxes。
"""
import numpy as np
import time
from typing import List, Tuple, Optional
from Module.logger import logger

class OptimizedNMS:
    """
    具有預先篩選和快速演算法的優化 NMS 實作。
    """
    
    def __init__(self, max_boxes: int = 100, pre_filter_top_k: int = 200):
        """
        初始化優化的 NMS 處理器。
        
        Args:
            max_boxes: 要處理的最大方框數
            pre_filter_top_k: 在 NMS 之前要保留的最大方框數
        """
        self.max_boxes = max_boxes
        self.pre_filter_top_k = pre_filter_top_k
        self.stats = {
            'total_calls': 0,
            'total_input_boxes': 0,
            'total_output_boxes': 0,
            'total_time': 0.0,
            'pre_filter_time': 0.0,
            'nms_time': 0.0
        }
    
    def calculate_iou_vectorized(self, boxes: np.ndarray) -> np.ndarray:
        """
        用於加速處理的向量化 IoU 計算。
        
        Args:
            boxes: 格式為 [x1, y1, x2, y2] 的方框陣列
            
        Returns:
            形狀為 (N, N) 的 IoU 矩陣
        """
        # Convert to x1, y1, x2, y2 format if needed
        if boxes.shape[1] == 4 and boxes.dtype == np.float32:
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
        else:
            # Assume input is [x, y, w, h] format
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 0] + boxes[:, 2]
            y2 = boxes[:, 1] + boxes[:, 3]
        
        # Calculate areas
        areas = (x2 - x1) * (y2 - y1)
        
        # Broadcast for vectorized computation
        x1_broadcast = x1[:, None]
        y1_broadcast = y1[:, None]
        x2_broadcast = x2[:, None]
        y2_broadcast = y2[:, None]
        
        # Calculate intersection
        inter_x1 = np.maximum(x1_broadcast, x1)
        inter_y1 = np.maximum(y1_broadcast, y1)
        inter_x2 = np.minimum(x2_broadcast, x2)
        inter_y2 = np.minimum(y2_broadcast, y2)
        
        # Calculate intersection area
        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        # Calculate union area
        union_area = areas[:, None] + areas - inter_area
        
        # Calculate IoU
        iou = inter_area / (union_area + 1e-8)
        
        return iou
    
    def fast_nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
        """
        為速度優化的快速 NMS 實作。
        
        Args:
            boxes: 格式為 [x1, y1, x2, y2] 或 [x, y, w, h] 的方框陣列
            scores: 信賴度分數陣列
            iou_threshold: 用於抑制的 IoU 閾值
            
        Returns:
            要保留的索引列表
        """
        if len(boxes) == 0:
            return []
        
        # Sort by scores in descending order
        sorted_indices = np.argsort(scores)[::-1]
        
        keep = []
        suppressed = np.zeros(len(boxes), dtype=bool)
        
        for i in sorted_indices:
            if suppressed[i]:
                continue
                
            keep.append(i)
            
            if len(keep) >= self.max_boxes:
                break
            
            # Calculate IoU with remaining boxes
            current_box = boxes[i:i+1]
            remaining_indices = sorted_indices[sorted_indices > i]
            remaining_indices = remaining_indices[~suppressed[remaining_indices]]
            
            if len(remaining_indices) == 0:
                continue
            
            remaining_boxes = boxes[remaining_indices]
            
            # Fast IoU calculation for current box vs remaining boxes
            iou_scores = self._calculate_iou_fast(current_box, remaining_boxes)
            
            # Suppress boxes with high IoU
            suppress_mask = iou_scores > iou_threshold
            suppressed[remaining_indices[suppress_mask]] = True
        
        return keep
    
    def _calculate_iou_fast(self, box1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        一個方框與多個方框之間的快速 IoU 計算。
        
        Args:
            box1: 單一方框 [x1, y1, x2, y2] 或 [x, y, w, h]
            boxes2: 多個方框的陣列
            
        Returns:
            IoU 分數陣列
        """
        # Convert to x1, y1, x2, y2 format
        if box1.shape[1] == 4:
            if len(box1.shape) == 2:
                x1_1, y1_1, x2_1, y2_1 = box1[0]
            else:
                x1_1, y1_1, x2_1, y2_1 = box1
        else:
            x1_1, y1_1, w1, h1 = box1[0] if len(box1.shape) == 2 else box1
            x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        if boxes2.shape[1] == 4 and boxes2.dtype == np.float32:
            x1_2 = boxes2[:, 0]
            y1_2 = boxes2[:, 1]
            x2_2 = boxes2[:, 2]
            y2_2 = boxes2[:, 3]
        else:
            x1_2 = boxes2[:, 0]
            y1_2 = boxes2[:, 1]
            x2_2 = boxes2[:, 0] + boxes2[:, 2]
            y2_2 = boxes2[:, 1] + boxes2[:, 3]
        
        # Calculate intersection
        inter_x1 = np.maximum(x1_1, x1_2)
        inter_y1 = np.maximum(y1_1, y1_2)
        inter_x2 = np.minimum(x2_1, x2_2)
        inter_y2 = np.minimum(y2_1, y2_2)
        
        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate union and IoU
        union_area = area1 + area2 - inter_area
        iou = inter_area / (union_area + 1e-8)
        
        return iou
    
    def pre_filter_boxes(self, boxes: List, scores: List, score_threshold: float) -> Tuple[List, List, List[int]]:
        """
        預先篩選方框以減少 NMS 計算量。
        
        Args:
            boxes: 方框列表
            scores: 分數列表
            score_threshold: 最小分數閾值
            
        Returns:
            篩選後的方框、分數和原始索引
        """
        start_time = time.perf_counter()
        
        if len(boxes) == 0:
            return [], [], []
        
        # Convert to numpy arrays for faster processing
        scores_array = np.array(scores, dtype=np.float32)
        
        # Filter by score threshold
        valid_mask = scores_array >= score_threshold
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            self.stats['pre_filter_time'] += time.perf_counter() - start_time
            return [], [], []
        
        # Keep only top-k boxes by score
        if len(valid_indices) > self.pre_filter_top_k:
            top_k_scores = scores_array[valid_indices]
            top_k_indices = np.argsort(top_k_scores)[::-1][:self.pre_filter_top_k]
            valid_indices = valid_indices[top_k_indices]
        
        # Filter boxes and scores
        filtered_boxes = [boxes[i] for i in valid_indices]
        filtered_scores = [scores[i] for i in valid_indices]
        
        self.stats['pre_filter_time'] += time.perf_counter() - start_time
        return filtered_boxes, filtered_scores, valid_indices.tolist()
    
    def optimized_nms(self, boxes: List, scores: List, score_threshold: float, 
                     iou_threshold: float = 0.45) -> List[int]:
        """
        具有預先篩選的主要優化 NMS 函式。
        
        Args:
            boxes: 格式為 [x, y, w, h] 的方框列表
            scores: 信賴度分數列表
            score_threshold: 最小分數閾值
            iou_threshold: NMS 的 IoU 閾值
            
        Returns:
            要保留的索引列表（參考原始輸入）
        """
        start_time = time.perf_counter()
        
        self.stats['total_calls'] += 1
        self.stats['total_input_boxes'] += len(boxes)
        
        if len(boxes) == 0:
            return []
        
        # Pre-filter boxes
        filtered_boxes, filtered_scores, original_indices = self.pre_filter_boxes(
            boxes, scores, score_threshold
        )
        
        if len(filtered_boxes) == 0:
            self.stats['total_time'] += time.perf_counter() - start_time
            return []
        
        # Convert to numpy arrays
        boxes_array = np.array(filtered_boxes, dtype=np.float32)
        scores_array = np.array(filtered_scores, dtype=np.float32)
        
        # Apply fast NMS
        nms_start_time = time.perf_counter()
        keep_indices = self.fast_nms(boxes_array, scores_array, iou_threshold)
        self.stats['nms_time'] += time.perf_counter() - nms_start_time
        
        # Map back to original indices
        final_indices = [original_indices[i] for i in keep_indices]
        
        self.stats['total_output_boxes'] += len(final_indices)
        self.stats['total_time'] += time.perf_counter() - start_time
        
        return final_indices
    
    def distance_based_pre_filter(self, boxes: List, scores: List, center_point: Tuple[float, float], 
                                 max_distance: float) -> Tuple[List, List, List[int]]:
        """
        根據與中心點的距離預先篩選方框（例如，瞄準範圍）。
        
        Args:
            boxes: 方框列表
            scores: 分數列表
            center_point: (x, y) 中心點
            max_distance: 與中心的最大距離
            
        Returns:
            篩選後的方框、分數和原始索引
        """
        if len(boxes) == 0:
            return [], [], []
        
        center_x, center_y = center_point
        valid_indices = []
        
        for i, box in enumerate(boxes):
            # Calculate box center
            if len(box) == 4:
                box_center_x = box[0] + box[2] / 2  # x + w/2
                box_center_y = box[1] + box[3] / 2  # y + h/2
            else:
                box_center_x, box_center_y = box[0], box[1]
            
            # Calculate distance
            distance = np.sqrt((box_center_x - center_x)**2 + (box_center_y - center_y)**2)
            
            if distance <= max_distance:
                valid_indices.append(i)
        
        filtered_boxes = [boxes[i] for i in valid_indices]
        filtered_scores = [scores[i] for i in valid_indices]
        
        return filtered_boxes, filtered_scores, valid_indices
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        stats = self.stats.copy()
        if stats['total_calls'] > 0:
            stats['avg_input_boxes'] = stats['total_input_boxes'] / stats['total_calls']
            stats['avg_output_boxes'] = stats['total_output_boxes'] / stats['total_calls']
            stats['avg_time_per_call'] = stats['total_time'] / stats['total_calls'] * 1000  # ms
            stats['avg_pre_filter_time'] = stats['pre_filter_time'] / stats['total_calls'] * 1000  # ms
            stats['avg_nms_time'] = stats['nms_time'] / stats['total_calls'] * 1000  # ms
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'total_calls': 0,
            'total_input_boxes': 0,
            'total_output_boxes': 0,
            'total_time': 0.0,
            'pre_filter_time': 0.0,
            'nms_time': 0.0
        }

# Global optimized NMS instance
_global_nms_processor = None

def get_optimized_nms() -> OptimizedNMS:
    """Get the global optimized NMS processor."""
    global _global_nms_processor
    if _global_nms_processor is None:
        _global_nms_processor = OptimizedNMS()
    return _global_nms_processor

def optimized_nms_boxes(boxes: List, scores: List, score_threshold: float, 
                       iou_threshold: float = 0.45, center_point: Optional[Tuple[float, float]] = None,
                       max_distance: Optional[float] = None) -> List[int]:
    """
    Convenience function for optimized NMS with optional distance filtering.
    
    Args:
        boxes: List of boxes in format [x, y, w, h]
        scores: List of confidence scores
        score_threshold: Minimum score threshold
        iou_threshold: IoU threshold for NMS
        center_point: Optional center point for distance filtering
        max_distance: Optional maximum distance for distance filtering
        
    Returns:
        List of indices to keep
    """
    nms_processor = get_optimized_nms()
    
    # Apply distance-based pre-filtering if specified
    if center_point is not None and max_distance is not None:
        boxes, scores, original_indices = nms_processor.distance_based_pre_filter(
            boxes, scores, center_point, max_distance
        )
        if len(boxes) == 0:
            return []
        
        # Apply optimized NMS
        keep_indices = nms_processor.optimized_nms(boxes, scores, score_threshold, iou_threshold)
        
        # Map back to original indices
        return [original_indices[i] for i in keep_indices]
    else:
        return nms_processor.optimized_nms(boxes, scores, score_threshold, iou_threshold)