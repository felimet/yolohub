"""

COCO格式標註的合併與個體識別設計

安裝依賴：
    pip install numpy matplotlib pandas  # 可選，用於視覺化和進階分析

主要類別：
    - ClusteringConfig: 配置參數類別
    - BoundingBoxAnalyzer: 邊界框分析器
    - CattleClusterer: 主要分群器類別
    - ClusteringVisualizer: 視覺化工具（可選）

"""

import json
import math
import warnings
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path

# 可選依賴檢查
_HAS_NUMPY = False
_HAS_MATPLOTLIB = False
_HAS_PANDAS = False

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    _HAS_MATPLOTLIB = True
except ImportError:
    pass

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    pass


# === 配置類別 ===
@dataclass
class ClusteringConfig:
    """
    分群演算法配置參數
    
    Attributes:
        base_distance_threshold (float): 基礎距離閾值（像素）
        similarity_threshold (float): 整體相似性閾值（0-1）
        aspect_ratio_weight (float): 長寬比相似性權重
        area_ratio_weight (float): 面積比例權重  
        spatial_weight (float): 空間距離權重
        max_overlap_ratio (float): 最大重疊比例容忍度
        max_aspect_ratio_diff (float): 最大長寬比差異容忍度
        max_area_ratio_diff (float): 最大面積比例差異容忍度
        
    Examples:
        >>> config = ClusteringConfig(base_distance_threshold=60)
        >>> config = ClusteringConfig(similarity_threshold=0.8, spatial_weight=0.6)
    """
    base_distance_threshold: float = 80
    similarity_threshold: float = 0.65
    aspect_ratio_weight: float = 0.3
    area_ratio_weight: float = 0.2
    spatial_weight: float = 0.5
    max_overlap_ratio: float = 0.15
    max_aspect_ratio_diff: float = 0.8
    max_area_ratio_diff: float = 0.7
    
    def __post_init__(self):
        """驗證並正規化權重參數"""
        total_weight = self.aspect_ratio_weight + self.area_ratio_weight + self.spatial_weight
        if abs(total_weight - 1.0) > 0.01:
            warnings.warn(f"權重總和應為1.0，當前為{total_weight:.3f}，將自動正規化")
            self.aspect_ratio_weight /= total_weight
            self.area_ratio_weight /= total_weight
            self.spatial_weight /= total_weight
    
    @classmethod
    def preset_strict(cls) -> 'ClusteringConfig':
        """嚴格模式預設配置 - 適用於密集牛群場景"""
        return cls(
            base_distance_threshold=50,
            similarity_threshold=0.8,
            spatial_weight=0.4,
            aspect_ratio_weight=0.4,
            area_ratio_weight=0.2,
            max_overlap_ratio=0.1
        )
    
    @classmethod
    def preset_loose(cls) -> 'ClusteringConfig':
        """寬鬆模式預設配置 - 適用於稀疏分佈場景"""
        return cls(
            base_distance_threshold=120,
            similarity_threshold=0.5,
            spatial_weight=0.6,
            aspect_ratio_weight=0.2,
            area_ratio_weight=0.2,
            max_overlap_ratio=0.2
        )
    
    @classmethod
    def preset_balanced(cls) -> 'ClusteringConfig':
        """平衡模式預設配置 - 一般用途"""
        return cls()  # 使用預設值


# === 幾何分析器 ===
class BoundingBoxAnalyzer:
    """
    邊界框幾何分析器
    提供豐富的幾何計算與空間關係分析功能
    
    Attributes:
        x, y, w, h (float): 邊界框座標與尺寸
        left, right, top, bottom (float): 邊界框邊界
        center_x, center_y (float): 中心點座標
        area (float): 面積
        aspect_ratio (float): 長寬比
    """
    
    def __init__(self, bbox: List[float]):
        """
        初始化邊界框分析器
        
        Args:
            bbox: [x, y, width, height] 格式的邊界框
        """
        self.x, self.y, self.w, self.h = bbox
        self.left = self.x
        self.right = self.x + self.w
        self.top = self.y
        self.bottom = self.y + self.h
        self.center_x = self.x + self.w / 2
        self.center_y = self.y + self.h / 2
        self.area = self.w * self.h
        self.aspect_ratio = self.w / self.h if self.h > 0 else float('inf')
    
    def distance_to(self, other: 'BoundingBoxAnalyzer') -> float:
        """
        計算到另一個邊界框的最小邊緣距離
        
        Args:
            other: 另一個BoundingBoxAnalyzer實例
            
        Returns:
            float: 最小邊緣距離，相交時返回0
        """
        if self.intersects_with(other):
            return 0.0
        
        dx = max(0, max(other.left - self.right, self.left - other.right))
        dy = max(0, max(other.top - self.bottom, self.top - other.bottom))
        return math.sqrt(dx**2 + dy**2)
    
    def intersects_with(self, other: 'BoundingBoxAnalyzer') -> bool:
        """檢查是否與另一個邊界框相交"""
        return not (self.right <= other.left or other.right <= self.left or 
                   self.bottom <= other.top or other.bottom <= self.top)
    
    def overlap_ratio_with(self, other: 'BoundingBoxAnalyzer') -> float:
        """
        計算重疊面積比例（相對於較小邊界框）
        
        Returns:
            float: 重疊比例（0-1）
        """
        if not self.intersects_with(other):
            return 0.0
        
        intersection_area = (min(self.right, other.right) - max(self.left, other.left)) * \
                          (min(self.bottom, other.bottom) - max(self.top, other.top))
        smaller_area = min(self.area, other.area)
        return intersection_area / smaller_area if smaller_area > 0 else 0.0
    
    def iou_with(self, other: 'BoundingBoxAnalyzer') -> float:
        """
        計算IoU (Intersection over Union)
        
        Returns:
            float: IoU值（0-1）
        """
        if not self.intersects_with(other):
            return 0.0
        
        intersection_area = (min(self.right, other.right) - max(self.left, other.left)) * \
                          (min(self.bottom, other.bottom) - max(self.top, other.top))
        union_area = self.area + other.area - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0


# === 相似性計算器 ===
class SimilarityCalculator:
    """相似性計算工具類別"""
    
    @staticmethod
    def aspect_ratio_similarity(bbox1: BoundingBoxAnalyzer, bbox2: BoundingBoxAnalyzer) -> float:
        """計算長寬比相似性（0-1）"""
        if bbox1.aspect_ratio == float('inf') or bbox2.aspect_ratio == float('inf'):
            return 0.0
        
        ratio_diff = abs(bbox1.aspect_ratio - bbox2.aspect_ratio) / max(bbox1.aspect_ratio, bbox2.aspect_ratio)
        return max(0, 1 - ratio_diff)
    
    @staticmethod
    def area_similarity(bbox1: BoundingBoxAnalyzer, bbox2: BoundingBoxAnalyzer) -> float:
        """計算面積相似性（0-1）"""
        if bbox1.area == 0 or bbox2.area == 0:
            return 0.0
        return min(bbox1.area, bbox2.area) / max(bbox1.area, bbox2.area)
    
    @staticmethod
    def spatial_proximity(bbox1: BoundingBoxAnalyzer, bbox2: BoundingBoxAnalyzer, max_distance: float) -> float:
        """計算空間鄰近性（0-1）"""
        distance = bbox1.distance_to(bbox2)
        if distance >= max_distance:
            return 0.0
        return 1 - (distance / max_distance)
    
    @staticmethod
    def overall_similarity(bbox1: BoundingBoxAnalyzer, bbox2: BoundingBoxAnalyzer, config: ClusteringConfig) -> float:
        """計算綜合相似性分數"""
        aspect_sim = SimilarityCalculator.aspect_ratio_similarity(bbox1, bbox2)
        area_sim = SimilarityCalculator.area_similarity(bbox1, bbox2)
        spatial_sim = SimilarityCalculator.spatial_proximity(bbox1, bbox2, config.base_distance_threshold)
        
        return (config.aspect_ratio_weight * aspect_sim +
                config.area_ratio_weight * area_sim +
                config.spatial_weight * spatial_sim)


# === 主要分群器類別 ===
class CattleClusterer:
    """
    牛隻個體分群器
    
    提供完整的COCO格式標註分群功能，包括：
    - 智慧型相似性分析
    - 圖論連通分量演算法
    - 詳細處理統計
    - 品質評估
    
    Examples:
        >>> clusterer = CattleClusterer()
        >>> result = clusterer.cluster_file('input.json', 'output.json')
        
        >>> config = ClusteringConfig.preset_strict()
        >>> clusterer = CattleClusterer(config)
        >>> data, stats = clusterer.cluster_data(coco_dict)
    """
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        """
        初始化分群器
        
        Args:
            config: 分群配置，若為None則使用預設配置
        """
        self.config = config if config is not None else ClusteringConfig()
        self.last_stats = None
    
    def should_merge(self, ann1: Dict, ann2: Dict) -> bool:
        """
        判斷兩個標註是否應該合併
        
        Args:
            ann1, ann2: COCO格式標註字典
            
        Returns:
            bool: True表示應該合併
        """
        bbox1 = BoundingBoxAnalyzer(ann1['bbox'])
        bbox2 = BoundingBoxAnalyzer(ann2['bbox'])
        
        # 基礎距離篩選
        if bbox1.distance_to(bbox2) > self.config.base_distance_threshold:
            return False
        
        # 重疊度檢查
        if bbox1.overlap_ratio_with(bbox2) > self.config.max_overlap_ratio:
            return False
        
        # 形狀相似性檢查
        if bbox1.aspect_ratio != float('inf') and bbox2.aspect_ratio != float('inf'):
            aspect_diff = abs(bbox1.aspect_ratio - bbox2.aspect_ratio) / max(bbox1.aspect_ratio, bbox2.aspect_ratio)
            if aspect_diff > self.config.max_aspect_ratio_diff:
                return False
        
        # 面積相似性檢查
        if bbox1.area > 0 and bbox2.area > 0:
            area_ratio = min(bbox1.area, bbox2.area) / max(bbox1.area, bbox2.area)
            if area_ratio < (1 - self.config.max_area_ratio_diff):
                return False
        
        # 綜合相似性評估
        similarity = SimilarityCalculator.overall_similarity(bbox1, bbox2, self.config)
        return similarity >= self.config.similarity_threshold
    
    def find_clusters(self, annotations: List[Dict]) -> List[List[Dict]]:
        """
        使用圖論演算法尋找標註群集
        
        Args:
            annotations: 標註列表
            
        Returns:
            List[List[Dict]]: 群集列表
        """
        n = len(annotations)
        if n == 0:
            return []
        
        # 建立鄰接列表
        adjacency_list = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if self.should_merge(annotations[i], annotations[j]):
                    adjacency_list[i].append(j)
                    adjacency_list[j].append(i)
        
        # DFS尋找連通分量
        visited = [False] * n
        clusters = []
        
        def dfs(node: int, cluster_indices: List[int]):
            visited[node] = True
            cluster_indices.append(node)
            for neighbor in adjacency_list[node]:
                if not visited[neighbor]:
                    dfs(neighbor, cluster_indices)
        
        for i in range(n):
            if not visited[i]:
                cluster_indices = []
                dfs(i, cluster_indices)
                cluster = [annotations[idx] for idx in cluster_indices]
                clusters.append(cluster)
        
        return clusters
    
    def merge_cluster(self, cluster: List[Dict]) -> Optional[Dict]:
        """
        合併標註群集為單一標註
        
        Args:
            cluster: 標註群集
            
        Returns:
            Optional[Dict]: 合併後的標註
        """
        if not cluster:
            return None
        
        if len(cluster) == 1:
            merged_ann = cluster[0].copy()
            merged_ann['cluster_info'] = {'size': 1, 'original_ids': [merged_ann.get('id', -1)]}
            return merged_ann
        
        # 計算合併幾何屬性
        merged_segmentation = []
        merged_area = 0.0
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        first_ann = cluster[0]
        original_ids = []
        
        for ann in cluster:
            merged_segmentation.extend(ann['segmentation'])
            merged_area += ann['area']
            x, y, w, h = ann['bbox']
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
            original_ids.append(ann.get('id', -1))
        
        merged_bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
        
        return {
            "category_id": first_ann['category_id'],
            "segmentation": merged_segmentation,
            "area": merged_area,
            "bbox": merged_bbox,
            "iscrowd": first_ann.get('iscrowd', 0),
            "annotator": first_ann.get('annotator'),
            "cluster_info": {
                "size": len(cluster),
                "original_ids": original_ids
            }
        }
    
    def cluster_data(self, coco_data: Dict, return_stats: bool = True) -> Union[Dict, Tuple[Dict, Dict]]:
        """
        對COCO格式資料進行分群處理
        
        Args:
            coco_data: COCO格式字典
            return_stats: 是否返回統計資訊
            
        Returns:
            處理後的COCO資料，可選地包含統計資訊
        """
        # 按影像分組標註
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        # 統計資訊
        stats = {
            'total_images': len(annotations_by_image),
            'original_annotations': len(coco_data['annotations']),
            'merged_annotations': 0,
            'total_clusters': 0,
            'single_clusters': 0,
            'multi_clusters': 0,
            'max_cluster_size': 0,
            'config_used': asdict(self.config)
        }
        
        new_annotations = []
        new_ann_id = 0
        
        # 逐影像處理
        for image_id, image_annotations in annotations_by_image.items():
            if not image_annotations:
                continue
            
            clusters = self.find_clusters(image_annotations)
            stats['total_clusters'] += len(clusters)
            
            for cluster in clusters:
                merged_ann = self.merge_cluster(cluster)
                if merged_ann:
                    merged_ann['id'] = new_ann_id
                    merged_ann['image_id'] = image_id
                    new_annotations.append(merged_ann)
                    new_ann_id += 1
                    
                    cluster_size = len(cluster)
                    stats['max_cluster_size'] = max(stats['max_cluster_size'], cluster_size)
                    
                    if cluster_size == 1:
                        stats['single_clusters'] += 1
                    else:
                        stats['multi_clusters'] += 1
        
        stats['merged_annotations'] = len(new_annotations)
        self.last_stats = stats
        
        result_data = {
            "info": coco_data.get('info', {}),
            "categories": coco_data.get('categories', []),
            "images": coco_data.get('images', []),
            "annotations": new_annotations
        }
        
        return (result_data, stats) if return_stats else result_data
    
    def cluster_file(self, input_path: Union[str, Path], 
                    output_path: Union[str, Path],
                    return_stats: bool = True) -> Union[Dict, Tuple[Dict, Dict]]:
        """
        從檔案讀取並處理COCO資料
        
        Args:
            input_path: 輸入檔案路徑
            output_path: 輸出檔案路徑
            return_stats: 是否返回統計資訊
            
        Returns:
            處理統計資訊（如果return_stats=True）
        """
        # 讀取輸入檔案
        with open(input_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 處理資料
        result = self.cluster_data(coco_data, return_stats=True)
        result_data, stats = result
        
        # 儲存結果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        return (result_data, stats) if return_stats else stats
    
    def get_stats(self) -> Optional[Dict]:
        """獲取最後一次處理的統計資訊"""
        return self.last_stats


# === 視覺化工具 ===
class ClusteringVisualizer:
    """分群結果視覺化工具"""
    
    def __init__(self):
        if not _HAS_MATPLOTLIB:
            raise ImportError("視覺化功能需要 matplotlib，請使用 'pip install matplotlib' 安裝")
    
    @staticmethod
    def plot_clustering_comparison(image_path: str,
                                 original_annotations: List[Dict],
                                 clustered_annotations: List[Dict],
                                 figsize: Tuple[int, int] = (15, 6),
                                 save_path: Optional[str] = None):
        """
        繪製分群結果比較圖
        
        Args:
            image_path: 影像檔案路徑
            original_annotations: 原始標註
            clustered_annotations: 分群後標註
            figsize: 圖片大小
            save_path: 儲存路徑（可選）
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 讀取影像
        img = plt.imread(image_path)
        ax1.imshow(img)
        ax2.imshow(img)
        
        # 原始標註
        ax1.set_title(f'原始標註 ({len(original_annotations)}個)', fontsize=14)
        for ann in original_annotations:
            x, y, w, h = ann['bbox']
            rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                   edgecolor='red', facecolor='none', alpha=0.7)
            ax1.add_patch(rect)
        
        # 分群後標註
        ax2.set_title(f'分群後結果 ({len(clustered_annotations)}個)', fontsize=14)
        colors = plt.cm.Set3(range(len(clustered_annotations)))
        
        for i, ann in enumerate(clustered_annotations):
            x, y, w, h = ann['bbox']
            cluster_size = ann.get('cluster_info', {}).get('size', 1)
            rect = patches.Rectangle((x, y), w, h, linewidth=3,
                                   edgecolor=colors[i], facecolor='none', alpha=0.8)
            ax2.add_patch(rect)
            
            if cluster_size > 1:
                ax2.text(x + w/2, y + h/2, str(cluster_size),
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
        
        ax1.axis('off')
        ax2.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# === 分析工具 ===
class ClusteringAnalyzer:
    """分群品質分析工具"""
    
    @staticmethod
    def analyze_quality(original_coco: Union[Dict, str], clustered_coco: Union[Dict, str]) -> Dict:
        """
        分析分群品質
        
        Args:
            original_coco: 原始COCO資料或檔案路徑
            clustered_coco: 分群後COCO資料或檔案路徑
            
        Returns:
            Dict: 品質分析報告
        """
        # 載入資料
        if isinstance(original_coco, str):
            with open(original_coco, 'r', encoding='utf-8') as f:
                original_coco = json.load(f)
        
        if isinstance(clustered_coco, str):
            with open(clustered_coco, 'r', encoding='utf-8') as f:
                clustered_coco = json.load(f)
        
        original_anns = original_coco['annotations']
        clustered_anns = clustered_coco['annotations']
        
        # 分析指標
        quality_report = {
            'compression_ratio': len(clustered_anns) / len(original_anns) if len(original_anns) > 0 else 1,
            'total_merges': len(original_anns) - len(clustered_anns),
            'cluster_size_distribution': {},
            'avg_cluster_size': 0,
            'max_cluster_size': 0
        }
        
        # 群集大小分析
        cluster_sizes = []
        for ann in clustered_anns:
            cluster_size = ann.get('cluster_info', {}).get('size', 1)
            cluster_sizes.append(cluster_size)
            
            if cluster_size not in quality_report['cluster_size_distribution']:
                quality_report['cluster_size_distribution'][cluster_size] = 0
            quality_report['cluster_size_distribution'][cluster_size] += 1
        
        if cluster_sizes:
            quality_report['avg_cluster_size'] = sum(cluster_sizes) / len(cluster_sizes)
            quality_report['max_cluster_size'] = max(cluster_sizes)
        
        return quality_report


# === 便利函數 ===
def quick_cluster(input_file: Union[str, Path],
                 output_file: Union[str, Path],
                 distance_threshold: float = 80,
                 similarity_threshold: float = 0.65,
                 print_report: bool = True) -> Dict:
    """
    快速分群函數
    
    Args:
        input_file: 輸入檔案路徑
        output_file: 輸出檔案路徑
        distance_threshold: 距離閾值
        similarity_threshold: 相似性閾值
        print_report: 是否列印報告
        
    Returns:
        Dict: 處理統計資訊
    """
    config = ClusteringConfig(
        base_distance_threshold=distance_threshold,
        similarity_threshold=similarity_threshold
    )
    
    clusterer = CattleClusterer(config)
    _, stats = clusterer.cluster_file(input_file, output_file)
    
    if print_report:
        print_stats(stats)
    
    return stats


def print_stats(stats: Dict):
    """列印處理統計報告"""
    print("=" * 60)
    print("🐄 牛隻個體分群處理報告")
    print("=" * 60)
    
    print(f"📊 基本統計：")
    print(f"   ├─ 處理影像數：{stats['total_images']:,}")
    print(f"   ├─ 原始標註數：{stats['original_annotations']:,}")
    print(f"   ├─ 合併後個體數：{stats['merged_annotations']:,}")
    
    if stats['original_annotations'] > 0:
        reduction_rate = (stats['original_annotations'] - stats['merged_annotations']) / stats['original_annotations'] * 100
        print(f"   └─ 標註減少率：{reduction_rate:.1f}%")
    
    print(f"\n🔍 分群分析：")
    print(f"   ├─ 總群集數：{stats['total_clusters']:,}")
    print(f"   ├─ 單一標註群集：{stats['single_clusters']:,}")
    print(f"   ├─ 多標註群集：{stats['multi_clusters']:,}")
    print(f"   ├─ 最大群集大小：{stats['max_cluster_size']}")
    
    if stats['total_clusters'] > 0:
        avg_size = stats['original_annotations'] / stats['total_clusters']
        print(f"   └─ 平均群集大小：{avg_size:.2f}")
    
    config = stats['config_used']
    print(f"\n⚙️  演算法參數：")
    print(f"   ├─ 距離閾值：{config['base_distance_threshold']} px")
    print(f"   ├─ 相似性閾值：{config['similarity_threshold']:.2f}")
    print(f"   └─ 權重配置：空間{config['spatial_weight']:.1f} | 形狀{config['aspect_ratio_weight']:.1f} | 面積{config['area_ratio_weight']:.1f}")


# === 模組公開介面 ===
__all__ = [
    'ClusteringConfig',
    'BoundingBoxAnalyzer', 
    'SimilarityCalculator',
    'CattleClusterer',
    'ClusteringVisualizer',
    'ClusteringAnalyzer',
    'quick_cluster',
    'print_stats'
]

