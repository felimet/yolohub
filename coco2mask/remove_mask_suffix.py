from pathlib import Path

def remove_mask_suffix(mask_dir):
    """移除遮罩檔案的_mask後綴"""
    mask_path = Path(mask_dir)
    
    for mask_file in mask_path.glob("*_mask.png"):
        new_name = mask_file.name.replace("_mask.png", ".png")
        new_path = mask_path / new_name
        mask_file.rename(new_path)
        print(f"重新命名: {mask_file.name} → {new_name}")

