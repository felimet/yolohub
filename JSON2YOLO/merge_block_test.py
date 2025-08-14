import json
import os

# --- 使用者可調整的參數 ---
# 距離閾值 (像素): 如果兩個標註的邊界框之間距小於此值，則視為同一物體。
# 您可以根據牛隻各部位（如頭與身體）之間可能的最大間隙（像素）來設定此值。
# 建議從一個較小的值（如 50）開始，檢視合併結果，然後逐步調整。
DISTANCE_THRESHOLD_PIXELS = 50

def calculate_bbox_distance(bbox1, bbox2):
    """計算兩個邊界框之間的最小距離。如果相交，則距離為0。"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # 計算邊界框的邊緣
    left1, right1, top1, bottom1 = x1, x1 + w1, y1, y1 + h1
    left2, right2, top2, bottom2 = x2, x2 + w2, y2, y2 + h2

    # 檢查是否重疊
    if not (right1 < left2 or right2 < left1 or bottom1 < top2 or bottom2 < top1):
        return 0.0

    # 計算水平距離
    dx = 0
    if right1 < left2:
        dx = left2 - right1
    elif right2 < left1:
        dx = left1 - right2

    # 計算垂直距離
    dy = 0
    if bottom1 < top2:
        dy = top2 - bottom1
    elif bottom2 < top1:
        dy = top1 - bottom2
        
    return (dx**2 + dy**2)**0.5


def merge_coco_annotations_with_clustering(input_json_path, output_json_path, distance_threshold):
    """
    讀取 COCO 格式 JSON，並使用基於空間距離的分群演算法合併屬於同一個物體的標註。
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
        
    new_annotations = []
    new_ann_id_counter = 0

    # 對每張影像內的標註進行處理
    for image_id, anns in sorted(annotations_by_image.items()):
        if not anns:
            continue
        
        unassigned_anns = list(anns)
        
        while unassigned_anns:
            # 1. 建立一個新的群集，以第一個未指派的標註為起點
            current_cluster = [unassigned_anns.pop(0)]
            
            # 2. 不斷迭代，將鄰近的標註加入群集
            while True:
                merged_something = False
                # 從剩餘的未指派標註中尋找可以合併的
                remaining_anns = []
                for ann_to_check in unassigned_anns:
                    is_close = False
                    # 檢查是否與群集中任一元素鄰近
                    for ann_in_cluster in current_cluster:
                        dist = calculate_bbox_distance(ann_in_cluster['bbox'], ann_to_check['bbox'])
                        if dist < distance_threshold:
                            is_close = True
                            break
                    
                    if is_close:
                        current_cluster.append(ann_to_check)
                        merged_something = True
                    else:
                        remaining_anns.append(ann_to_check)
                
                unassigned_anns = remaining_anns
                if not merged_something:
                    break # 如果這一輪沒有任何標註被合併，則此群集完成

            # 3. 合併群集內的所有標註
            merged_segmentation = []
            merged_area = 0.0
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')

            first_ann = current_cluster[0]

            for ann in current_cluster:
                merged_segmentation.extend(ann['segmentation'])
                merged_area += ann['area']
                x, y, w, h = ann['bbox']
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)
            
            merged_bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
            
            new_ann = {
                "id": new_ann_id_counter,
                "image_id": image_id,
                "category_id": first_ann['category_id'],
                "segmentation": merged_segmentation,
                "area": merged_area,
                "bbox": merged_bbox,
                "iscrowd": first_ann.get('iscrowd', 0),
                "annotator": first_ann.get('annotator')
            }
            new_annotations.append(new_ann)
            new_ann_id_counter += 1

    # 建立最終的 COCO 格式字典
    merged_coco_data = {
        "info": coco_data.get('info', {}),
        "categories": coco_data.get('categories', []),
        "images": coco_data.get('images', []),
        "annotations": new_annotations
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(merged_coco_data, f, indent=2, ensure_ascii=False)
        
    print(f"分群合併完成！結果已儲存至：{output_json_path}")
    print(f"距離閾值設定為：{distance_threshold} 像素")
    print(f"原始標註數量：{len(coco_data['annotations'])}")
    print(f"合併後標註（牛隻個體）數量：{len(new_annotations)}")

# --- 主程式執行區 ---
if __name__ == "__main__":
    image_path = "coco_json2yolo_txt/tmp_dataset_seg/coco_type/38.json"
    output_file = "coco_json2yolo_txt/tmp_dataset_seg/coco_type/annnnnnn.json"
    
    if not os.path.exists(image_path):
        print(f"錯誤：找不到輸入檔案 '{image_path}'。")
    else:
        merge_coco_annotations_with_clustering(image_path, output_file, DISTANCE_THRESHOLD_PIXELS)