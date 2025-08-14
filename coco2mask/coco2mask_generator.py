import json
import numpy as np
import cv2
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union
import argparse


class COCOToMaskGenerator:
    """COCO格式轉二值化遮罩生成器
    
    支援處理COCO格式的JSON標註檔案，生成對應的二值化PNG遮罩。
    支援多邊形(polygon)和RLE兩種分割格式。
    """
    
    def __init__(self, coco_json_path: str, images_dir: str = None):
        """
        初始化生成器
        
        Args:
            coco_json_path: COCO格式JSON檔案路徑
            images_dir: 影像檔案目錄路徑（可選）
        """
        self.coco_json_path = coco_json_path
        self.images_dir = images_dir
        self.coco_data = self._load_coco_data()
        self.images_info = {img['id']: img for img in self.coco_data['images']}
        self.categories_info = {cat['id']: cat for cat in self.coco_data['categories']}
        
    def _load_coco_data(self) -> Dict:
        """載入COCO格式JSON資料"""
        try:
            with open(self.coco_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"無法載入COCO JSON檔案: {e}")
    
    def _polygon_to_mask(self, polygons: List[List[float]], height: int, width: int) -> np.ndarray:
        """
        將多邊形座標轉換為二值化遮罩
        
        Args:
            polygons: 多邊形座標列表，格式為 [[x1, y1, x2, y2, ...], ...]
            height: 影像高度
            width: 影像寬度
            
        Returns:
            二值化遮罩 (numpy array)
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for polygon in polygons:
            # 將平坦的座標列表轉換為點對
            if len(polygon) >= 6:  # 至少需要3個點（6個座標值）
                points = np.array(polygon).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [points], 1)
        
        return mask
    
    def _rle_to_mask(self, rle: Dict, height: int, width: int) -> np.ndarray:
        """
        將RLE格式轉換為二值化遮罩
        
        Args:
            rle: RLE格式資料
            height: 影像高度
            width: 影像寬度
            
        Returns:
            二值化遮罩 (numpy array)
        """
        try:
            # 如果使用pycocotools
            from pycocotools import mask as coco_mask
            return coco_mask.decode(rle)
        except ImportError:
            # 簡單的RLE解碼實現
            if isinstance(rle, dict) and 'counts' in rle:
                counts = rle['counts']
                if isinstance(counts, list):
                    # 未壓縮的RLE
                    mask = np.zeros(height * width, dtype=np.uint8)
                    idx = 0
                    for i, count in enumerate(counts):
                        if i % 2 == 1:  # 奇數索引表示物體像素
                            mask[idx:idx + count] = 1
                        idx += count
                    return mask.reshape(height, width)
        
        # 如果無法解碼，返回空遮罩
        print(f"警告: 無法解碼RLE格式，返回空遮罩")
        return np.zeros((height, width), dtype=np.uint8)
    
    def generate_mask_for_annotation(self, annotation: Dict, image_info: Dict) -> np.ndarray:
        """
        為單個標註生成遮罩
        
        Args:
            annotation: COCO標註物件
            image_info: 影像資訊
            
        Returns:
            二值化遮罩
        """
        height = image_info['height']
        width = image_info['width']
        segmentation = annotation['segmentation']
        
        if isinstance(segmentation, list):
            # 多邊形格式
            return self._polygon_to_mask(segmentation, height, width)
        elif isinstance(segmentation, dict):
            # RLE格式
            return self._rle_to_mask(segmentation, height, width)
        else:
            raise ValueError(f"不支援的分割格式: {type(segmentation)}")
    
    def generate_single_mask(self, image_id: int, category_id: int = None, 
                           merge_instances: bool = True) -> Tuple[np.ndarray, str]:
        """
        為指定影像生成遮罩
        
        Args:
            image_id: 影像ID
            category_id: 類別ID（可選，如果指定則只處理該類別）
            merge_instances: 是否合併同類別的多個實例
            
        Returns:
            (遮罩陣列, 影像檔名)
        """
        if image_id not in self.images_info:
            raise ValueError(f"找不到影像ID: {image_id}")
        
        image_info = self.images_info[image_id]
        height = image_info['height']
        width = image_info['width']
        
        # 獲取該影像的所有標註
        annotations = [ann for ann in self.coco_data['annotations'] 
                      if ann['image_id'] == image_id]
        
        if category_id is not None:
            annotations = [ann for ann in annotations 
                          if ann['category_id'] == category_id]
        
        if merge_instances:
            # 合併所有實例為單一遮罩
            combined_mask = np.zeros((height, width), dtype=np.uint8)
            
            for annotation in annotations:
                if 'segmentation' in annotation:
                    mask = self.generate_mask_for_annotation(annotation, image_info)
                    combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
            
            return combined_mask * 255, image_info['file_name']
        else:
            # 每個實例使用不同的值
            instance_mask = np.zeros((height, width), dtype=np.uint8)
            
            for idx, annotation in enumerate(annotations, 1):
                if 'segmentation' in annotation:
                    mask = self.generate_mask_for_annotation(annotation, image_info)
                    instance_mask[mask > 0] = idx
            
            return instance_mask, image_info['file_name']
    
    def generate_all_masks(self, output_dir: str, category_id: int = None, 
                          merge_instances: bool = True, visualize: bool = False):
        """
        為所有影像生成遮罩
        
        Args:
            output_dir: 輸出目錄
            category_id: 類別ID（可選）
            merge_instances: 是否合併實例
            visualize: 是否產生視覺化影像
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if visualize:
            vis_path = output_path / 'visualization'
            vis_path.mkdir(exist_ok=True)
        
        total_images = len(self.images_info)
        processed = 0
        
        print(f"開始處理 {total_images} 張影像...")
        
        for image_id in self.images_info.keys():
            try:
                mask, filename = self.generate_single_mask(
                    image_id, category_id, merge_instances
                )
                
                # 確保遮罩檔名與原檔名相同（僅副檔名為.png）
                original_stem = Path(filename).stem
                mask_filename = f"{original_stem}.png"
                mask_path = output_path / mask_filename
                
                # 使用PIL儲存，確保為單通道影像
                Image.fromarray(mask, mode='L').save(mask_path)
                
                # 產生視覺化影像
                if visualize:
                    vis_filename = f"{original_stem}_visualization.png"
                    self._create_visualization(image_id, mask, vis_path / vis_filename)
                
                processed += 1
                if processed % 10 == 0:
                    print(f"已處理: {processed}/{total_images}")
                    
            except Exception as e:
                print(f"處理影像 {image_id} 時發生錯誤: {e}")
                continue
        
        print(f"完成! 總共處理了 {processed} 張影像")
        print(f"遮罩檔案儲存在: {output_path}")
    
    def _create_visualization(self, image_id: int, mask: np.ndarray, output_path: Path):
        """建立視覺化影像"""
        try:
            if self.images_dir:
                image_info = self.images_info[image_id]
                image_path = Path(self.images_dir) / image_info['file_name']
                
                if image_path.exists():
                    # 載入原始影像
                    original_img = cv2.imread(str(image_path))
                    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                    
                    # 建立視覺化
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # 原始影像
                    axes[0].imshow(original_img)
                    axes[0].set_title('Original image')
                    axes[0].axis('off')
                    
                    # 遮罩
                    axes[1].imshow(mask, cmap='gray')
                    axes[1].set_title('Mask')
                    axes[1].axis('off')
                    
                    # 重疊顯示
                    overlay = original_img.copy()
                    overlay[mask > 0] = [255, 0, 0]  # 紅色標記
                    blended = cv2.addWeighted(original_img, 0.7, overlay, 0.3, 0)
                    axes[2].imshow(blended)
                    axes[2].set_title('Overlap')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(output_path, dpi=800, bbox_inches='tight')
                    plt.close()
        except Exception as e:
            print(f"建立視覺化時發生錯誤: {e}")
    
    def get_statistics(self) -> Dict:
        """獲取資料集統計資訊"""
        stats = {
            'total_images': len(self.coco_data['images']),
            'total_annotations': len(self.coco_data['annotations']),
            'categories': len(self.coco_data['categories']),
            'category_distribution': {},
            'segmentation_formats': {'polygon': 0, 'rle': 0}
        }
        
        # 統計各類別的標註數量
        for annotation in self.coco_data['annotations']:
            cat_id = annotation['category_id']
            cat_name = self.categories_info[cat_id]['name']
            stats['category_distribution'][cat_name] = stats['category_distribution'].get(cat_name, 0) + 1
            
            # 統計分割格式
            segmentation = annotation.get('segmentation', [])
            if isinstance(segmentation, list):
                stats['segmentation_formats']['polygon'] += 1
            elif isinstance(segmentation, dict):
                stats['segmentation_formats']['rle'] += 1
        
        return stats
    
    def print_statistics(self):
        """印出資料集統計資訊"""
        stats = self.get_statistics()
        
        print("=== COCO資料集統計資訊 ===")
        print(f"影像總數: {stats['total_images']}")
        print(f"標註總數: {stats['total_annotations']}")
        print(f"類別總數: {stats['categories']}")
        
        print("\n類別分佈:")
        for cat_name, count in sorted(stats['category_distribution'].items()):
            print(f"  {cat_name}: {count}")
        
        print(f"\n分割格式分佈:")
        print(f"  多邊形: {stats['segmentation_formats']['polygon']}")
        print(f"  RLE: {stats['segmentation_formats']['rle']}")


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(description='COCO格式轉二值化遮罩生成器')
    parser.add_argument('--coco_json', required=True, help='COCO格式JSON檔案路徑')
    parser.add_argument('--images_dir', help='影像檔案目錄路徑')
    parser.add_argument('--output_dir', required=True, help='輸出目錄路徑')
    parser.add_argument('--category_id', type=int, help='指定類別ID（可選）')
    parser.add_argument('--merge_instances', action='store_true', default=True,
                       help='合併同類別的多個實例')
    parser.add_argument('--visualize', action='store_true', help='產生視覺化影像')
    parser.add_argument('--stats_only', action='store_true', help='只顯示統計資訊')
    
    args = parser.parse_args()
    
    try:
        # 建立生成器
        generator = COCOToMaskGenerator(args.coco_json, args.images_dir)
        
        # 顯示統計資訊
        generator.print_statistics()
        
        if not args.stats_only:
            # 生成遮罩
            generator.generate_all_masks(
                output_dir=args.output_dir,
                category_id=args.category_id,
                merge_instances=args.merge_instances,
                visualize=args.visualize
            )
        
    except Exception as e:
        print(f"執行過程中發生錯誤: {e}")
        return 1
    
    return 0


# 使用範例
if __name__ == "__main__":
    """
    # 基本使用範例
    generator = COCOToMaskGenerator(
        coco_json_path='annotations/instances_train2017.json',
        images_dir='images/train2017'
    )
    
    # 顯示統計資訊
    generator.print_statistics()
    
    # 生成所有遮罩
    generator.generate_all_masks(
        output_dir='output/masks',
        merge_instances=True,
        visualize=True
    )
    
    # 只生成特定類別的遮罩（例如：person類別，通常ID為1）
    generator.generate_all_masks(
        output_dir='output/person_masks',
        category_id=1,
        merge_instances=True
    )
    
    # 生成單張影像的遮罩
    mask, filename = generator.generate_single_mask(image_id=1)
    Image.fromarray(mask, mode='L').save(f'{Path(filename).stem}.png')
    """
    
    # 執行命令列程式
    exit(main())