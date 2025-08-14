from pathlib import Path
from visual_img_ann import visualize_yolo_segmentation_annotations

# 您的原始設定
label_map = {0: "person", 1: "cow"}
image_path = "/home/isspmes/yolo/coco_json2yolo_txt/tmp_datasets/yolo_data/images/41/2db61cbc-001909.jpg"
annotation_path = "/home/isspmes/yolo/coco_json2yolo_txt/tmp_datasets/yolo_data/labels/41_1/2db61cbc-001909.txt"

# 使用新的視覺化函數
visualize_yolo_segmentation_annotations(
    image_path=image_path,
    annotation_path=annotation_path,
    label_map=label_map,
    save_path="./visual_ann/visual_result/result.png"  # 可選
)