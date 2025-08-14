import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import random

def visualize_yolo_segmentation_annotations(image_path, annotation_path, label_map, save_path=None):
    """
    視覺化 YOLO 分割標註檔案
    
    參數:
        image_path (str): 影像檔案路徑
        annotation_path (str): 標註檔案路徑
        label_map (dict): 類別 ID 到類別名稱的映射
        save_path (str, optional): 儲存路徑，若為 None 則僅顯示
    
    返回:
        tuple: (原始影像, 標註影像)
    """
    
    # 讀取影像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"無法載入影像檔案: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # 檢查標註檔案是否存在
    if not Path(annotation_path).exists():
        raise FileNotFoundError(f"標註檔案不存在: {annotation_path}")
    
    # 建立視覺化圖表
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 原始影像
    axes[0].imshow(image)
    axes[0].set_title('Original image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 標註影像
    annotated_image = image.copy()
    
    # 讀取並解析標註檔案
    annotations = parse_yolo_segmentation_file(annotation_path)
    
    print(f"載入標註檔案: {annotation_path}")
    print(f"發現 {len(annotations)} 個標註物件")
    
    # 為每個類別生成固定顏色
    class_colors = {}
    for class_id in label_map.keys():
        class_colors[class_id] = [random.randint(100, 255) for _ in range(3)]
    
    # 繪製每個標註
    for i, annotation in enumerate(annotations):
        class_id = annotation['class_id']
        polygon_points = annotation['polygon']
        
        # 轉換正規化座標為像素座標
        pixel_points = []
        for j in range(0, len(polygon_points), 2):
            x = polygon_points[j] * w
            y = polygon_points[j + 1] * h
            pixel_points.append([x, y])
        
        pixel_points = np.array(pixel_points, dtype=np.int32)
        
        # 獲取類別資訊
        class_name = label_map.get(class_id, f"Class_{class_id}")
        color = class_colors.get(class_id, [255, 0, 0])
        
        # 繪製多邊形填充
        cv2.fillPoly(annotated_image, [pixel_points], color)
        
        # 繪製多邊形邊框
        cv2.polylines(annotated_image, [pixel_points], True, color, 3)
        
        # 計算邊界框以放置標籤
        x_coords = pixel_points[:, 0]
        y_coords = pixel_points[:, 1]
        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
        
        # 繪製類別標籤
        label_text = f"{class_name} (ID: {class_id})"
        font_scale = 0.8
        thickness = 2
        
        # 計算文字尺寸
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # 繪製文字背景
        cv2.rectangle(annotated_image, 
                     (x_min, y_min - text_height - 10),
                     (x_min + text_width, y_min),
                     color, -1)
        
        # 繪製文字
        cv2.putText(annotated_image, label_text, 
                   (x_min, y_min - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                   (255, 255, 255), thickness)
        
        print(f"  物件 {i+1}: {class_name} (類別ID: {class_id})")
        print(f"    多邊形頂點數: {len(pixel_points)}")
        print(f"    邊界框: ({x_min}, {y_min}, {x_max}, {y_max})")
    
    # 顯示標註影像（透明度混合）
    blended_image = cv2.addWeighted(image, 0.4, annotated_image, 0.5, 0)
    axes[1].imshow(blended_image)
    axes[1].set_title('Visual annotation', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 添加圖例
    add_legend(fig, label_map, class_colors)
    
    plt.tight_layout()
    
    # 儲存結果
    if save_path:
        plt.savefig(save_path, dpi=900, bbox_inches='tight')
        print(f"結果已儲存至: {save_path}")
    
    plt.show()
    
    return image, blended_image

def parse_yolo_segmentation_file(annotation_path):
    """
    解析 YOLO 分割標註檔案
    
    參數:
        annotation_path (str): 標註檔案路徑
    
    返回:
        list: 包含所有標註物件的列表
    """
    annotations = []
    
    with open(annotation_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:  # 跳過空行
            continue
        
        try:
            values = list(map(float, line.split()))
            
            if len(values) < 7:  # 最少需要 class_id + 3個點(6個座標)
                print(f"警告: 第 {line_num} 行資料不足，跳過")
                continue
            
            class_id = int(values[0])
            polygon_coords = values[1:]
            
            # 確保座標點數為偶數
            if len(polygon_coords) % 2 != 0:
                print(f"警告: 第 {line_num} 行座標點數為奇數，移除最後一個座標")
                polygon_coords = polygon_coords[:-1]
            
            annotations.append({
                'class_id': class_id,
                'polygon': polygon_coords,
                'line_number': line_num
            })
            
        except ValueError as e:
            print(f"錯誤: 第 {line_num} 行格式錯誤 - {e}")
            continue
    
    return annotations

def add_legend(fig, label_map, class_colors):
    """
    添加類別圖例
    
    參數:
        fig: matplotlib 圖形物件
        label_map (dict): 類別映射
        class_colors (dict): 類別顏色映射
    """
    legend_elements = []
    
    for class_id, class_name in label_map.items():
        color = np.array(class_colors[class_id]) / 255.0  # 正規化到 [0,1]
        legend_elements.append(
            patches.Patch(color=color, label=f"{class_name} (ID: {class_id})")
        )
    
    if legend_elements:
        fig.legend(handles=legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, 0.02), ncol=len(legend_elements))

def check_annotation_format(annotation_path):
    """
    檢查標註檔案格式並提供詳細資訊
    
    參數:
        annotation_path (str): 標註檔案路徑
    
    返回:
        dict: 檔案格式分析結果
    """
    analysis = {
        'total_lines': 0,
        'valid_lines': 0,
        'format_type': 'unknown',
        'classes_found': set(),
        'min_points': float('inf'),
        'max_points': 0,
        'errors': []
    }
    
    with open(annotation_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    analysis['total_lines'] = len(lines)
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        try:
            values = list(map(float, line.split()))
            
            if len(values) == 5:
                analysis['format_type'] = 'bounding_box'
            elif len(values) > 5 and len(values) % 2 == 1:
                analysis['format_type'] = 'segmentation'
                num_points = (len(values) - 1) // 2
                analysis['min_points'] = min(analysis['min_points'], num_points)
                analysis['max_points'] = max(analysis['max_points'], num_points)
            
            analysis['classes_found'].add(int(values[0]))
            analysis['valid_lines'] += 1
            
        except (ValueError, IndexError) as e:
            analysis['errors'].append(f"第 {line_num} 行: {e}")
    
    if analysis['min_points'] == float('inf'):
        analysis['min_points'] = 0
    
    return analysis

# 主要執行函數
if __name__ == "__main__":
    # 設定檔案路徑
    image_path = "../tmp_datasets/yolo_data/images/41/2db61cbc-001909.jpg"
    annotation_path = "../tmp_datasets/yolo_data/labels/41_1/2db61cbc-001909.txt"
    label_map = {
        0: "cow",
    }
    
    save_path = "./visual_ann/visual_result/seg_visual_result.png"  # 可選：儲存結果
    
    try:
        # 首先檢查標註檔案格式
        print("分析標註檔案格式...")
        analysis = check_annotation_format(annotation_path)
        
        print(f"檔案分析結果:")
        print(f"  總行數: {analysis['total_lines']}")
        print(f"  有效行數: {analysis['valid_lines']}")
        print(f"  檔案格式: {analysis['format_type']}")
        print(f"  發現的類別: {sorted(analysis['classes_found'])}")
        
        if analysis['format_type'] == 'segmentation':
            print(f"  多邊形頂點數範圍: {analysis['min_points']} - {analysis['max_points']}")
        
        if analysis['errors']:
            print(f"  錯誤: {len(analysis['errors'])} 個")
            for error in analysis['errors'][:5]:  # 只顯示前5個錯誤
                print(f"    {error}")
        
        print("\n開始視覺化...")
        
        # 執行視覺化
        original, annotated = visualize_yolo_segmentation_annotations(
            image_path=image_path,
            annotation_path=annotation_path,
            label_map=label_map,
            save_path=save_path
        )
        
        print("視覺化完成！")
        
    except Exception as e:
        print(f"執行過程中發生錯誤: {e}")
        print("\n可能的解決方案:")
        print("1. 檢查檔案路徑是否正確")
        print("2. 確認標註檔案格式是否為 YOLO 分割格式")
        print("3. 檢查影像檔案是否可正常載入")
        print("4. 驗證 label_map 中的類別 ID 是否與標註檔案一致")