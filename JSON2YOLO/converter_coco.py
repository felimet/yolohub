"""
Label Studio 遮擋標註處理完整流程
專門處理 Brush labels 轉換的 COCO 格式，生成最佳化邊界框
"""

import numpy as np
import json
import cv2
from typing import List, Dict, Tuple, Optional
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
import os
from pathlib import Path


class LabelStudioOcclusionProcessor:
    """
    專門處理 Label Studio Brush labels 產生的遮擋標註
    """
    
    def __init__(self, 
                 min_segment_area: int = 100,
                 merge_distance_ratio: float = 0.02,
                 noise_filter_ratio: float = 0.005):
        """
        初始化處理器
        
        Args:
            min_segment_area: 最小分割區域面積（像素）
            merge_distance_ratio: 合併距離比例（相對於圖像對角線）
            noise_filter_ratio: 雜訊過濾比例（相對於圖像面積）
        """
        self.min_segment_area = min_segment_area
        self.merge_distance_ratio = merge_distance_ratio
        self.noise_filter_ratio = noise_filter_ratio
    
    def parse_label_studio_rle(self, rle_data: Dict) -> np.ndarray:
        """
        解析 Label Studio 的 RLE 編碼資料
        
        Args:
            rle_data: RLE 格式的遮罩資料
            
        Returns:
            二值遮罩陣列
        """
        if isinstance(rle_data, dict) and 'counts' in rle_data:
            # 標準 COCO RLE 格式
            mask = maskUtils.decode(rle_data)
        elif isinstance(rle_data, list):
            # Label Studio 可能的平坦座標格式
            # 需要根據實際格式調整
            raise NotImplementedError("需要根據實際 Label Studio 輸出格式調整")
        else:
            raise ValueError(f"不支援的 RLE 格式: {type(rle_data)}")
        
        return mask
    
    def extract_contours_from_mask(self, mask: np.ndarray, 
                                 min_area: int = None) -> List[np.ndarray]:
        """
        從二值遮罩提取輪廓
        
        Args:
            mask: 二值遮罩陣列
            min_area: 最小輪廓面積閾值
            
        Returns:
            輪廓座標列表
        """
        if min_area is None:
            min_area = self.min_segment_area
        
        # 尋找輪廓
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 過濾小面積輪廓
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                # 轉換為標準格式 (N, 2)
                contour_points = contour.reshape(-1, 2)
                valid_contours.append(contour_points)
        
        return valid_contours
    
    def analyze_occlusion_pattern(self, segments: List[np.ndarray], 
                                image_shape: Tuple[int, int]) -> Dict:
        """
        分析遮擋模式，提供標註品質評估
        
        Args:
            segments: 分割區域列表
            image_shape: 圖像尺寸
            
        Returns:
            遮擋分析結果字典
        """
        analysis = {
            'num_segments': len(segments),
            'total_area': 0,
            'largest_segment_area': 0,
            'fragmentation_ratio': 0,
            'spatial_distribution': None,
            'recommended_strategy': 'minimal_enclosing'
        }
        
        if not segments:
            return analysis
        
        # 計算面積統計
        areas = []
        for seg in segments:
            area = cv2.contourArea(seg.astype(np.float32))
            areas.append(area)
            analysis['total_area'] += area
        
        analysis['largest_segment_area'] = max(areas)
        
        # 計算碎片化比例
        if analysis['largest_segment_area'] > 0:
            analysis['fragmentation_ratio'] = (
                analysis['total_area'] - analysis['largest_segment_area']
            ) / analysis['total_area']
        
        # 空間分佈分析
        all_points = np.vstack(segments)
        centroid = np.mean(all_points, axis=0)
        
        # 計算各區域質心到整體質心的距離
        segment_centroids = []
        distances = []
        
        for seg in segments:
            seg_centroid = np.mean(seg, axis=0)
            segment_centroids.append(seg_centroid)
            distance = np.linalg.norm(seg_centroid - centroid)
            distances.append(distance)
        
        analysis['spatial_distribution'] = {
            'centroid': centroid.tolist(),
            'max_distance': max(distances) if distances else 0,
            'mean_distance': np.mean(distances) if distances else 0
        }
        
        # 推薦策略
        if analysis['fragmentation_ratio'] > 0.3:
            analysis['recommended_strategy'] = 'density_weighted'
        elif analysis['num_segments'] > 5:
            analysis['recommended_strategy'] = 'convex_hull'
        
        return analysis
    
    def process_single_annotation(self, annotation: Dict, 
                                image_info: Dict) -> Dict:
        """
        處理單一標註的多區塊分割
        
        Args:
            annotation: COCO 格式的單一標註
            image_info: 對應圖像資訊
            
        Returns:
            處理後的標註
        """
        image_shape = (image_info['height'], image_info['width'])
        segmentation = annotation.get('segmentation', [])
        
        # 處理不同格式的分割資料
        segments = []
        
        if isinstance(segmentation, dict):
            # RLE 格式
            try:
                mask = self.parse_label_studio_rle(segmentation)
                contours = self.extract_contours_from_mask(mask)
                segments.extend(contours)
            except Exception as e:
                print(f"RLE 解析失敗: {e}")
                return annotation
                
        elif isinstance(segmentation, list) and segmentation:
            # 多邊形格式
            for seg in segmentation:
                if isinstance(seg, list) and len(seg) >= 6:
                    points = np.array(seg).reshape(-1, 2)
                    segments.append(points)
        
        if not segments:
            return annotation
        
        # 遮擋模式分析
        occlusion_analysis = self.analyze_occlusion_pattern(segments, image_shape)
        
        # 根據分析結果選擇最佳策略
        strategy = occlusion_analysis['recommended_strategy']
        
        # 生成最佳化邊界框
        unified_bbox = self.calculate_optimized_bbox(
            segments, image_shape, strategy
        )
        
        # 更新標註
        processed_annotation = annotation.copy()
        processed_annotation['bbox'] = unified_bbox
        processed_annotation['area'] = occlusion_analysis['total_area']
        
        # 添加遮擋分析元資料
        processed_annotation['occlusion_analysis'] = occlusion_analysis
        
        # 重新編碼分割資料
        processed_segmentation = []
        for seg in segments:
            flattened = seg.flatten().tolist()
            processed_segmentation.append(flattened)
        processed_annotation['segmentation'] = processed_segmentation
        
        return processed_annotation
    
    def calculate_optimized_bbox(self, segments: List[np.ndarray], 
                               image_shape: Tuple[int, int],
                               strategy: str = 'adaptive') -> List[float]:
        """
        計算最佳化邊界框
        
        Args:
            segments: 分割區域列表
            image_shape: 圖像尺寸
            strategy: 計算策略
            
        Returns:
            [x, y, width, height] 格式的邊界框
        """
        if not segments:
            return [0, 0, 0, 0]
        
        # 過濾雜訊
        filtered_segments = self.filter_noise_segments(segments, image_shape)
        
        if not filtered_segments:
            filtered_segments = segments  # 如果全部被過濾，保留原始資料
        
        # 合併近鄰區域
        merged_segments = self.merge_nearby_segments(filtered_segments, image_shape)
        
        # 根據策略計算邊界框
        if strategy == 'adaptive':
            strategy = self.select_adaptive_strategy(merged_segments, image_shape)
        
        all_points = np.vstack(merged_segments)
        
        if strategy == 'minimal_enclosing':
            x_min, y_min = np.min(all_points, axis=0)
            x_max, y_max = np.max(all_points, axis=0)
            
        elif strategy == 'convex_hull':
            hull = cv2.convexHull(all_points.astype(np.float32))
            hull_points = hull.reshape(-1, 2)
            x_min, y_min = np.min(hull_points, axis=0)
            x_max, y_max = np.max(hull_points, axis=0)
            
        elif strategy == 'density_weighted':
            # 根據區域面積加權計算邊界
            weighted_points = []
            for seg in merged_segments:
                area = cv2.contourArea(seg.astype(np.float32))
                weight = int(max(1, area / 1000))  # 面積權重
                for _ in range(weight):
                    weighted_points.extend(seg)
            
            if weighted_points:
                weighted_array = np.array(weighted_points)
                x_min, y_min = np.min(weighted_array, axis=0)
                x_max, y_max = np.max(weighted_array, axis=0)
            else:
                x_min, y_min = np.min(all_points, axis=0)
                x_max, y_max = np.max(all_points, axis=0)
        
        # 確保邊界框在圖像範圍內
        x_min = max(0, float(x_min))
        y_min = max(0, float(y_min))
        x_max = min(image_shape[1], float(x_max))
        y_max = min(image_shape[0], float(y_max))
        
        width = x_max - x_min
        height = y_max - y_min
        
        return [x_min, y_min, width, height]
    
    def filter_noise_segments(self, segments: List[np.ndarray], 
                            image_shape: Tuple[int, int]) -> List[np.ndarray]:
        """過濾雜訊分割區域"""
        image_area = image_shape[0] * image_shape[1]
        min_area = image_area * self.noise_filter_ratio
        
        filtered = []
        for seg in segments:
            area = cv2.contourArea(seg.astype(np.float32))
            if area >= min_area:
                filtered.append(seg)
        
        return filtered
    
    def merge_nearby_segments(self, segments: List[np.ndarray], 
                            image_shape: Tuple[int, int]) -> List[np.ndarray]:
        """合併近鄰分割區域"""
        if len(segments) <= 1:
            return segments
        
        diagonal = np.sqrt(image_shape[0]**2 + image_shape[1]**2)
        merge_distance = diagonal * self.merge_distance_ratio
        
        merged = []
        processed = set()
        
        for i, seg1 in enumerate(segments):
            if i in processed:
                continue
            
            current_cluster = [seg1]
            processed.add(i)
            
            for j, seg2 in enumerate(segments[i+1:], i+1):
                if j not in processed:
                    min_dist = self.calculate_min_distance(seg1, seg2)
                    if min_dist < merge_distance:
                        current_cluster.append(seg2)
                        processed.add(j)
            
            if len(current_cluster) > 1:
                # 合併多個區域
                combined_points = np.vstack(current_cluster)
                hull = cv2.convexHull(combined_points.astype(np.float32))
                merged.append(hull.reshape(-1, 2))
            else:
                merged.append(seg1)
        
        return merged
    
    def calculate_min_distance(self, seg1: np.ndarray, seg2: np.ndarray) -> float:
        """計算兩個分割區域的最小距離"""
        from scipy.spatial.distance import cdist
        distances = cdist(seg1, seg2)
        return np.min(distances)
    
    def select_adaptive_strategy(self, segments: List[np.ndarray], 
                               image_shape: Tuple[int, int]) -> str:
        """自適應選擇最佳邊界框策略"""
        if len(segments) == 1:
            return 'minimal_enclosing'
        
        # 計算分散程度
        all_points = np.vstack(segments)
        centroid = np.mean(all_points, axis=0)
        
        max_distance = 0
        for seg in segments:
            seg_centroid = np.mean(seg, axis=0)
            distance = np.linalg.norm(seg_centroid - centroid)
            max_distance = max(max_distance, distance)
        
        # 根據分散程度選擇策略
        diagonal = np.sqrt(image_shape[0]**2 + image_shape[1]**2)
        dispersion_ratio = max_distance / diagonal
        
        if dispersion_ratio > 0.3:
            return 'convex_hull'
        elif len(segments) > 3:
            return 'density_weighted'
        else:
            return 'minimal_enclosing'
    
    def process_coco_file(self, input_path: str, output_path: str = None) -> str:
        """
        處理完整的 COCO 標註檔案
        
        Args:
            input_path: 輸入檔案路徑
            output_path: 輸出檔案路徑（可選）
            
        Returns:
            輸出檔案路徑
        """
        if output_path is None:
            input_file = Path(input_path)
            output_path = str(input_file.parent / f"{input_file.stem}_processed.json")
        
        # 載入 COCO 資料
        with open(input_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 建立圖像資訊映射
        images_dict = {img['id']: img for img in coco_data.get('images', [])}
        
        # 處理標註
        processed_annotations = []
        processing_stats = {
            'total_annotations': len(coco_data.get('annotations', [])),
            'processed_annotations': 0,
            'multi_segment_annotations': 0,
            'average_segments_per_annotation': 0
        }
        
        total_segments = 0
        
        for annotation in coco_data.get('annotations', []):
            image_id = annotation.get('image_id')
            
            if image_id in images_dict:
                image_info = images_dict[image_id]
                
                # 檢查是否為多區塊標註
                segmentation = annotation.get('segmentation', [])
                is_multi_segment = (
                    isinstance(segmentation, list) and 
                    len(segmentation) > 1
                ) or isinstance(segmentation, dict)
                
                if is_multi_segment:
                    processed_ann = self.process_single_annotation(annotation, image_info)
                    processing_stats['multi_segment_annotations'] += 1
                    
                    # 計算處理後的區域數量
                    if 'occlusion_analysis' in processed_ann:
                        num_segments = processed_ann['occlusion_analysis']['num_segments']
                        total_segments += num_segments
                else:
                    processed_ann = annotation
                
                processed_annotations.append(processed_ann)
                processing_stats['processed_annotations'] += 1
        
        # 計算統計資訊
        if processing_stats['multi_segment_annotations'] > 0:
            processing_stats['average_segments_per_annotation'] = (
                total_segments / processing_stats['multi_segment_annotations']
            )
        
        # 更新 COCO 資料
        processed_coco = coco_data.copy()
        processed_coco['annotations'] = processed_annotations
        
        # 添加處理元資料
        processed_coco['processing_info'] = {
            'processor': 'LabelStudioOcclusionProcessor',
            'parameters': {
                'min_segment_area': self.min_segment_area,
                'merge_distance_ratio': self.merge_distance_ratio,
                'noise_filter_ratio': self.noise_filter_ratio
            },
            'statistics': processing_stats
        }
        
        # 儲存結果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_coco, f, indent=2, ensure_ascii=False)
        
        print(f"處理完成！統計資訊：")
        print(f"  總標註數: {processing_stats['total_annotations']}")
        print(f"  多區塊標註數: {processing_stats['multi_segment_annotations']}")
        print(f"  平均每個標註的區塊數: {processing_stats['average_segments_per_annotation']:.2f}")
        print(f"  結果儲存至: {output_path}")
        
        return output_path


def main():
    """使用範例"""
    # 初始化處理器
    processor = LabelStudioOcclusionProcessor(
        min_segment_area=50,          # 最小區域 50 像素
        merge_distance_ratio=0.03,    # 3% 對角線距離內合併
        noise_filter_ratio=0.001      # 過濾小於 0.1% 圖像面積的區域
    )
    
    # 處理 Label Studio 輸出的 COCO 檔案
    input_file = "path/to/your/labelstudio_export.json"
    output_file = processor.process_coco_file(input_file)
    
    print(f"處理結果已儲存至: {output_file}")


if __name__ == "__main__":
    main()