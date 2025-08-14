import cv2
import os
import numpy as np
from pathlib import Path
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

def convert_255_to_1(image):
    """
    將圖像中的像素值進行轉換，大於等於1的值統一轉為1
    
    參數:
    - image: 輸入圖像（單通道灰度圖像）
    
    返回:
    - modified_image: 修改後的圖像
    """
    # 將大於等於1的值轉換為1，保持0值不變
    modified_image = np.where(image >= 1, 1, image)
    return modified_image


def process_masks(source_directory, output_directory):
    """
    處理資料夾下的所有掩碼檔案，將255值轉換為1，並儲存修改後的圖像
    
    參數:
    - source_directory: 來源資料夾路徑
    - output_directory: 輸出資料夾路徑
    
    返回:
    - processed_count: 成功處理的檔案數量
    """
    # 使用 pathlib 處理路徑，提升跨平台相容性
    source_path = Path(source_directory)
    output_path = Path(output_directory)
    
    # 確認來源資料夾存在
    if not source_path.exists() or not source_path.is_dir():
        print(f"錯誤：來源資料夾不存在或非有效資料夾: {source_directory}")
        return 0
    
    # 建立輸出資料夾
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 支援的圖像格式
    supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    processed_count = 0
    
    # 遍歷來源資料夾中的所有檔案
    for file_path in source_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_formats:
            try:
                # 讀取灰度圖像
                image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    print(f"警告：無法讀取圖像 {file_path.name}")
                    continue
                
                # 轉換像素值
                modified_image = convert_255_to_1(image)
                
                # 構建輸出檔案路徑
                output_file_path = output_path / file_path.name
                
                # 儲存修改後的圖像
                success = cv2.imwrite(str(output_file_path), modified_image)
                
                if success:
                    print(f"✓ 處理完成: {file_path.name}")
                    processed_count += 1
                else:
                    print(f"✗ 儲存失敗: {file_path.name}")
                    
            except Exception as e:
                print(f"✗ 處理 {file_path.name} 時發生錯誤: {str(e)}")
                continue
    
    return processed_count


def validate_and_preview(source_directory, sample_count=3):
    """
    驗證處理前後的圖像差異，並預覽部分結果
    
    參數:
    - source_directory: 來源資料夾路徑
    - sample_count: 要預覽的樣本數量
    """
    source_path = Path(source_directory)
    supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    image_files = [f for f in source_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in supported_formats]
    
    if not image_files:
        print("來源資料夾中沒有找到支援的圖像檔案")
        return
    
    print(f"\n=== 圖像處理預覽 (前 {min(sample_count, len(image_files))} 個檔案) ===")
    
    for i, file_path in enumerate(image_files[:sample_count]):
        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            unique_values_before = np.unique(image)
            modified_image = convert_255_to_1(image)
            unique_values_after = np.unique(modified_image)
            
            print(f"\n檔案: {file_path.name}")
            print(f"  原始像素值範圍: {unique_values_before}")
            print(f"  處理後像素值範圍: {unique_values_after}")
            print(f"  圖像尺寸: {image.shape}")


if __name__ == "__main__":
    """主程式函數"""
    # 設定路徑參數
    mask_folder = "coco_json2yolo_txt/coco2mask/output/37"
    binary_mask_folder = mask_folder + "binary_mask"
    yolo_txt_folder = "./37"
    
    print("=== 圖像掩碼批次處理程式 ===")
    print(f"來源資料夾: {mask_folder}")
    print(f"輸出資料夾: {binary_mask_folder}")
    
    # 預覽處理效果
    validate_and_preview(mask_folder)
    
    # 執行批次處理
    print("\n開始批次處理...")
    processed_count = process_masks(mask_folder, binary_mask_folder)
    
    if processed_count > 0:
        print(f"\n✓ 處理完成！共成功處理 {processed_count} 個檔案")
        print(f"結果已儲存至: {binary_mask_folder}")
    else:
        print("\n✗ 沒有檔案被處理，請檢查來源資料夾路徑和檔案格式")

    # 執行YOLO分割格式轉換
    print(f"\n步驟3: 轉換為YOLO分割格式")
    print(f"輸入資料夾: {binary_mask_folder}")
    print(f"輸出資料夾: {yolo_txt_folder}")

        # 確保輸出資料夾存在
    Path(binary_mask_folder).mkdir(parents=True, exist_ok=True)
        
        # 執行YOLO格式轉換
    convert_segment_masks_to_yolo_seg(
            masks_dir=binary_mask_folder, 
            output_dir=yolo_txt_folder, 
            classes=1
        )
    print("✓ YOLO分割格式轉換完成")
    print("\n=== 轉換流程完成 ===")



