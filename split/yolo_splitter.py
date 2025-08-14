#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強版 YOLO 資料集自動分割工具
支援子資料夾自動掃描與靈活處理模式
"""

import os
import shutil
import random
from pathlib import Path
import yaml
from typing import Tuple, List, Set, Optional, Dict
from collections import defaultdict
import sys
import argparse
from datetime import datetime

class EnhancedYOLODatasetSplitter:
    """
    增強版 YOLO 資料集自動分割工具
    
    新功能：
    - 自動掃描並處理子資料夾
    - 支援整合模式（所有子資料夾合併）
    - 支援獨立模式（每個子資料夾分別處理）
    - 智慧檔案重命名避免衝突
    - 詳細的處理報告
    """
    
    def __init__(self, source_dir: str, output_dir: str = None):
        """
        初始化增強版分割工具
        
        Args:
            source_dir: 來源資料夾路徑
            output_dir: 輸出資料夾路徑（若為 None 則使用來源資料夾）
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir) if output_dir else self.source_dir
        
        # 確保輸出目錄存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化資料夾結構資訊
        self.subfolder_info = {}
        
        # 驗證並掃描來源資料夾結構
        self._scan_source_structure()
        
        # 顯示路徑資訊
        print(f"📂 來源路徑: {self.source_dir.absolute()}")
        print(f"📁 輸出路徑: {self.output_dir.absolute()}")
        if self.source_dir != self.output_dir:
            print("🔄 使用獨立輸出資料夾模式")
        else:
            print("📍 使用原地分割模式")
    
    def _scan_source_structure(self):
        """掃描並分析來源資料夾結構"""
        print("🔍 掃描資料夾結構...")
        
        images_dir = self.source_dir / 'images'
        labels_dir = self.source_dir / 'labels'
        
        if not images_dir.exists():
            raise FileNotFoundError(f"找不到影像資料夾: {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"找不到標籤資料夾: {labels_dir}")
        
        # 掃描 images 目錄中的子資料夾
        image_subfolders = set()
        label_subfolders = set()
        
        # 檢查是否有直接在根目錄的檔案
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        direct_images = [f for f in images_dir.iterdir() 
                        if f.is_file() and f.suffix.lower() in valid_extensions]
        direct_labels = list(labels_dir.glob('*.txt'))
        
        # 掃描子資料夾
        for item in images_dir.iterdir():
            if item.is_dir():
                image_subfolders.add(item.name)
        
        for item in labels_dir.iterdir():
            if item.is_dir():
                label_subfolders.add(item.name)
        
        # 分析結構
        all_subfolders = image_subfolders.union(label_subfolders)
        
        print(f"📊 結構分析結果:")
        print(f"   直接影像檔案: {len(direct_images)} 個")
        print(f"   直接標籤檔案: {len(direct_labels)} 個")
        print(f"   發現子資料夾: {len(all_subfolders)} 個")
        
        # 詳細分析每個子資料夾
        for subfolder in sorted(all_subfolders):
            img_subfolder = images_dir / subfolder
            label_subfolder = labels_dir / subfolder
            
            img_count = 0
            label_count = 0
            
            if img_subfolder.exists():
                img_count = len([f for f in img_subfolder.iterdir() 
                               if f.is_file() and f.suffix.lower() in valid_extensions])
            
            if label_subfolder.exists():
                label_count = len(list(label_subfolder.glob('*.txt')))
            
            self.subfolder_info[subfolder] = {
                'images': img_count,
                'labels': label_count,
                'has_images_dir': img_subfolder.exists(),
                'has_labels_dir': label_subfolder.exists()
            }
            
            status = "✅" if img_count > 0 and label_count > 0 else "⚠️"
            print(f"   {status} {subfolder}: {img_count} 影像, {label_count} 標籤")
        
        # 儲存直接檔案資訊
        if direct_images or direct_labels:
            self.subfolder_info['_root_'] = {
                'images': len(direct_images),
                'labels': len(direct_labels),
                'has_images_dir': True,
                'has_labels_dir': True
            }
            print(f"   📁 根目錄: {len(direct_images)} 影像, {len(direct_labels)} 標籤")
        
        print(f"✓ 資料夾結構掃描完成")
    
    def get_all_matched_files(self, mode: str = "integrated") -> Dict[str, List[Tuple[Path, Path]]]:
        """
        取得所有配對的影像和標籤檔案
        
        Args:
            mode: 處理模式 ("integrated" 或 "separate")
            
        Returns:
            字典：{subfolder_name: [(image_path, label_path), ...]}
        """
        print(f"📋 收集配對檔案 ({mode} 模式)...")
        
        images_dir = self.source_dir / 'images'
        labels_dir = self.source_dir / 'labels'
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        all_matched_files = {}
        
        # 處理每個子資料夾
        for subfolder, info in self.subfolder_info.items():
            if info['images'] == 0:
                continue  # 跳過沒有影像的資料夾
            
            matched_pairs = []
            unmatched_images = []
            
            if subfolder == '_root_':
                # 處理根目錄的檔案
                img_source = images_dir
                label_source = labels_dir
            else:
                # 處理子資料夾的檔案
                img_source = images_dir / subfolder
                label_source = labels_dir / subfolder
            
            if not img_source.exists():
                continue
            
            # 收集影像檔案
            for img_path in img_source.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in valid_extensions:
                    # 尋找對應的標籤檔案
                    label_path = label_source / f"{img_path.stem}.txt"
                    
                    if label_path.exists():
                        matched_pairs.append((img_path, label_path))
                    else:
                        unmatched_images.append(img_path)
            
            if matched_pairs:
                all_matched_files[subfolder] = matched_pairs
                print(f"   {subfolder}: {len(matched_pairs)} 對配對檔案")
                
                if unmatched_images:
                    print(f"     ⚠️  {len(unmatched_images)} 個影像沒有對應標籤")
        
        return all_matched_files
    
    def integrate_all_files(self, all_matched_files: Dict[str, List[Tuple[Path, Path]]]) -> List[Tuple[Path, Path]]:
        """
        整合所有子資料夾的檔案（避免檔名衝突）
        
        Args:
            all_matched_files: 所有子資料夾的配對檔案
            
        Returns:
            整合後的配對檔案列表
        """
        print("🔗 整合所有子資料夾的檔案...")
        
        integrated_pairs = []
        filename_count = defaultdict(int)
        
        # 統計檔名出現次數
        for subfolder, pairs in all_matched_files.items():
            for img_path, label_path in pairs:
                base_name = img_path.stem
                filename_count[base_name] += 1
        
        # 整合檔案，處理重名衝突
        for subfolder, pairs in all_matched_files.items():
            print(f"   處理 {subfolder}: {len(pairs)} 個檔案")
            
            for img_path, label_path in pairs:
                base_name = img_path.stem
                
                # 如果檔名重複，加上子資料夾前綴
                if filename_count[base_name] > 1 and subfolder != '_root_':
                    new_base_name = f"{subfolder}_{base_name}"
                    print(f"     重命名: {base_name} -> {new_base_name}")
                else:
                    new_base_name = base_name
                
                # 建立新的檔案路徑資訊（用於後續複製）
                integrated_pairs.append((img_path, label_path, new_base_name))
        
        print(f"✅ 整合完成，總計 {len(integrated_pairs)} 對檔案")
        return integrated_pairs
    
    def split_dataset_integrated(self, 
                               train_ratio: float = 0.7, 
                               val_ratio: float = 0.15, 
                               test_ratio: float = 0.15,
                               random_seed: int = 42,
                               force_clean: bool = False):
        """
        整合模式：將所有子資料夾的檔案合併後分割
        
        Args:
            train_ratio: 訓練集比例
            val_ratio: 驗證集比例  
            test_ratio: 測試集比例
            random_seed: 隨機種子
            force_clean: 是否強制清理現有分割
        """
        print("\n🎯 執行整合模式分割")
        print("=" * 50)
        
        # 驗證比例總和
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"比例總和必須為 1.0，目前為 {total_ratio}")
        
        # 檢查是否需要清理
        if force_clean or self._check_existing_splits_for_duplicates():
            self._clean_existing_splits()
        
        # 取得所有配對檔案
        all_matched_files = self.get_all_matched_files(mode="integrated")
        
        if not all_matched_files:
            raise ValueError("沒有找到任何配對的影像和標籤檔案")
        
        # 整合所有檔案
        integrated_pairs = self.integrate_all_files(all_matched_files)
        
        # 建立輸出結構
        self._create_output_structure()
        
        # 設定隨機種子並打亂資料
        random.seed(random_seed)
        random.shuffle(integrated_pairs)
        
        total_files = len(integrated_pairs)
        train_count = int(total_files * train_ratio)
        val_count = int(total_files * val_ratio)
        
        # 分割資料
        train_pairs = integrated_pairs[:train_count]
        val_pairs = integrated_pairs[train_count:train_count + val_count]
        test_pairs = integrated_pairs[train_count + val_count:]
        
        # 複製檔案
        print("📋 開始複製檔案...")
        self._copy_files_integrated(train_pairs, 'train')
        self._copy_files_integrated(val_pairs, 'val')
        self._copy_files_integrated(test_pairs, 'test')
        
        # 統計結果
        print(f"\n📊 整合模式分割統計:")
        print(f"  訓練集: {len(train_pairs)} 對 ({len(train_pairs)/total_files*100:.1f}%)")
        print(f"  驗證集: {len(val_pairs)} 對 ({len(val_pairs)/total_files*100:.1f}%)")
        print(f"  測試集: {len(test_pairs)} 對 ({len(test_pairs)/total_files*100:.1f}%)")
        print(f"  總計: {total_files} 對")
        
        return len(train_pairs), len(val_pairs), len(test_pairs)
    
    def split_dataset_separate(self, 
                             train_ratio: float = 0.7, 
                             val_ratio: float = 0.15, 
                             test_ratio: float = 0.15,
                             random_seed: int = 42,
                             force_clean: bool = False):
        """
        獨立模式：每個子資料夾分別處理並輸出
        
        Args:
            train_ratio: 訓練集比例
            val_ratio: 驗證集比例  
            test_ratio: 測試集比例
            random_seed: 隨機種子
            force_clean: 是否強制清理現有分割
        """
        print("\n🎯 執行獨立模式分割")
        print("=" * 50)
        
        # 驗證比例總和
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"比例總和必須為 1.0，目前為 {total_ratio}")
        
        # 取得所有配對檔案
        all_matched_files = self.get_all_matched_files(mode="separate")
        
        if not all_matched_files:
            raise ValueError("沒有找到任何配對的影像和標籤檔案")
        
        results = {}
        
        # 分別處理每個子資料夾
        for subfolder, matched_pairs in all_matched_files.items():
            print(f"\n處理子資料夾: {subfolder}")
            print("-" * 30)
            
            if len(matched_pairs) < 3:
                print(f"⚠️  {subfolder} 檔案數量太少 ({len(matched_pairs)}), 跳過分割")
                continue
            
            # 建立子資料夾專用的輸出目錄
            subfolder_output = self.output_dir / f"split_{subfolder}"
            subfolder_output.mkdir(parents=True, exist_ok=True)
            
            # 建立輸出結構
            self._create_output_structure(base_dir=subfolder_output)
            
            # 設定隨機種子並打亂資料
            random.seed(random_seed + hash(subfolder) % 1000)  # 每個子資料夾使用不同種子
            random.shuffle(matched_pairs)
            
            total_files = len(matched_pairs)
            train_count = int(total_files * train_ratio)
            val_count = int(total_files * val_ratio)
            
            # 分割資料
            train_pairs = matched_pairs[:train_count]
            val_pairs = matched_pairs[train_count:train_count + val_count]
            test_pairs = matched_pairs[train_count + val_count:]
            
            # 複製檔案
            self._copy_files_simple(train_pairs, 'train', subfolder_output)
            self._copy_files_simple(val_pairs, 'val', subfolder_output)
            self._copy_files_simple(test_pairs, 'test', subfolder_output)
            
            # 統計結果
            results[subfolder] = (len(train_pairs), len(val_pairs), len(test_pairs))
            
            print(f"✅ {subfolder} 完成:")
            print(f"   訓練集: {len(train_pairs)} 對 ({len(train_pairs)/total_files*100:.1f}%)")
            print(f"   驗證集: {len(val_pairs)} 對 ({len(val_pairs)/total_files*100:.1f}%)")
            print(f"   測試集: {len(test_pairs)} 對 ({len(test_pairs)/total_files*100:.1f}%)")
        
        # 顯示總結
        print(f"\n📊 獨立模式分割總結:")
        total_train = total_val = total_test = 0
        for subfolder, (train, val, test) in results.items():
            print(f"  {subfolder}: 訓練 {train}, 驗證 {val}, 測試 {test}")
            total_train += train
            total_val += val
            total_test += test
        
        print(f"  總計: 訓練 {total_train}, 驗證 {total_val}, 測試 {total_test}")
        
        return results
    
    def _copy_files_integrated(self, file_pairs: List[Tuple[Path, Path, str]], subset: str):
        """複製整合模式的檔案"""
        copied_count = 0
        total_count = len(file_pairs)
        
        for img_path, label_path, new_base_name in file_pairs:
            # 目標路徑
            target_img_dir = self.output_dir / 'images' / subset
            target_label_dir = self.output_dir / 'labels' / subset
            
            # 新檔名
            new_img_name = f"{new_base_name}{img_path.suffix}"
            new_label_name = f"{new_base_name}.txt"
            
            # 複製檔案
            shutil.copy2(img_path, target_img_dir / new_img_name)
            shutil.copy2(label_path, target_label_dir / new_label_name)
            
            copied_count += 1
            
            # 顯示進度
            if copied_count % max(1, total_count // 4) == 0 or copied_count == total_count:
                progress = copied_count / total_count * 100
                print(f"  {subset}: {copied_count}/{total_count} ({progress:.0f}%)")
    
    def _copy_files_simple(self, file_pairs: List[Tuple[Path, Path]], subset: str, base_dir: Path):
        """複製獨立模式的檔案"""
        for img_path, label_path in file_pairs:
            # 目標路徑
            target_img_dir = base_dir / 'images' / subset
            target_label_dir = base_dir / 'labels' / subset
            
            # 複製檔案（保持原始檔名）
            shutil.copy2(img_path, target_img_dir / img_path.name)
            shutil.copy2(label_path, target_label_dir / label_path.name)
    
    def _check_existing_splits_for_duplicates(self) -> bool:
        """檢查現有分割是否存在重複檔案"""
        print("🔍 檢查現有分割中的重複檔案...")
        
        splits = ['train', 'val', 'test']
        all_files = defaultdict(list)
        
        for split in splits:
            img_dir = self.output_dir / 'images' / split
            if img_dir.exists():
                for file_path in img_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}:
                        all_files[file_path.name].append(split)
        
        duplicates = {filename: splits for filename, splits in all_files.items() if len(splits) > 1}
        
        if duplicates:
            print(f"❌ 發現 {len(duplicates)} 個重複檔案")
            return True
        else:
            print("✅ 未發現重複檔案")
            return False
    
    def _clean_existing_splits(self):
        """清空現有的分割資料夾"""
        print("🗑️ 清空現有分割資料夾...")
        
        splits = ['train', 'val', 'test']
        for split in splits:
            img_dir = self.output_dir / 'images' / split
            label_dir = self.output_dir / 'labels' / split
            
            if img_dir.exists():
                shutil.rmtree(img_dir)
            if label_dir.exists():
                shutil.rmtree(label_dir)
        
        print("✓ 清空完成")
    
    def _create_output_structure(self, base_dir: Path = None):
        """建立輸出資料夾結構"""
        if base_dir is None:
            base_dir = self.output_dir
            
        subdirs = ['train', 'val', 'test']
        
        for subdir in subdirs:
            (base_dir / 'images' / subdir).mkdir(parents=True, exist_ok=True)
            (base_dir / 'labels' / subdir).mkdir(parents=True, exist_ok=True)
    
    def create_yaml_config(self, 
                          class_names: List[str],
                          train_count: int = 0,
                          val_count: int = 0,
                          test_count: int = 0,
                          mode: str = "integrated",
                          subfolder_results: Dict = None):
        """
        建立 YAML 配置檔案
        
        Args:
            class_names: 類別名稱列表
            train_count: 訓練集數量
            val_count: 驗證集數量
            test_count: 測試集數量
            mode: 處理模式
            subfolder_results: 子資料夾結果（獨立模式用）
        """
        if mode == "integrated":
            self._create_integrated_yaml(class_names, train_count, val_count, test_count)
        else:
            self._create_separate_yaml(class_names, subfolder_results)
    
    def _create_integrated_yaml(self, class_names: List[str], train_count: int, val_count: int, test_count: int):
        """建立整合模式的 YAML 配置"""
        path_str = str(self.output_dir.absolute()).replace(os.sep, '/')
        yaml_path = self.output_dir / 'dataset.yaml'
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write("# Enhanced YOLO Dataset Configuration (Integrated Mode)\n")
            f.write(f"# Generated from: {self.source_dir.absolute()}\n")
            f.write(f"# Output to: {self.output_dir.absolute()}\n")
            f.write(f"# Mode: Integrated (all subfolders merged)\n")
            f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Subfolders processed: {list(self.subfolder_info.keys())}\n\n")
            
            f.write(f"path: {path_str}\n")
            f.write(f"train: images/train  # {train_count} images\n")
            f.write(f"val: images/val  # {val_count} images\n")
            f.write(f"test: images/test  # {test_count} images\n\n")
            
            f.write(f"nc: {len(class_names)}\n")
            f.write("names:\n")
            for i, name in enumerate(class_names):
                f.write(f"  {i}: {name}\n")
        
        print(f"✓ YAML 配置檔案已建立: {yaml_path}")
    
    def _create_separate_yaml(self, class_names: List[str], subfolder_results: Dict):
        """建立獨立模式的 YAML 配置"""
        for subfolder, (train, val, test) in subfolder_results.items():
            subfolder_dir = self.output_dir / f"split_{subfolder}"
            yaml_path = subfolder_dir / 'dataset.yaml'
            path_str = str(subfolder_dir.absolute()).replace(os.sep, '/')
            
            with open(yaml_path, 'w', encoding='utf-8') as f:
                f.write("# Enhanced YOLO Dataset Configuration (Separate Mode)\n")
                f.write(f"# Generated from: {self.source_dir.absolute()}\n")
                f.write(f"# Output to: {subfolder_dir.absolute()}\n")
                f.write(f"# Mode: Separate (subfolder: {subfolder})\n")
                f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"path: {path_str}\n")
                f.write(f"train: images/train  # {train} images\n")
                f.write(f"val: images/val  # {val} images\n")
                f.write(f"test: images/test  # {test} images\n\n")
                
                f.write(f"nc: {len(class_names)}\n")
                f.write("names:\n")
                for i, name in enumerate(class_names):
                    f.write(f"  {i}: {name}\n")
            
            print(f"✓ YAML 配置檔案已建立: {yaml_path}")


# ========================================================================
# 增強版主函數
# ========================================================================

def enhanced_auto_split(source_directory: str = "./tmp_datasets/yolo_data/",
                       output_directory: Optional[str] = None,
                       split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                       class_names: Optional[List[str]] = None,
                       mode: str = "integrated",
                       force_clean: bool = True,
                       random_seed: int = 42) -> Optional[Dict]:
    """
    增強版自動分割 YOLO 資料集
    
    Args:
        source_directory: 資料集來源資料夾路徑
        output_directory: 輸出資料夾路徑
        split_ratios: 分割比例 (訓練, 驗證, 測試)
        class_names: 類別名稱列表
        mode: 處理模式 ("integrated" 或 "separate")
        force_clean: 是否強制清理現有分割
        random_seed: 隨機種子
        
    Returns:
        處理結果字典
    """
    
    # 預設類別名稱
    if class_names is None:
        class_names = ['cow']
    
    # 處理輸出資料夾
    if output_directory is None:
        output_directory = source_directory
        operation_mode = "同來源資料位置"
    else:
        operation_mode = "自訂輸出資料位置"
    
    try:
        print("🚀 開始執行 YOLO 資料集自動分割")
        print("=" * 60)
        print(f"📂 來源目錄: {source_directory}")
        print(f"📁 輸出目錄: {output_directory}")
        print(f"🔧 操作模式: {operation_mode}")
        print(f"🎯 處理模式: {mode}")
        print(f"📊 分割比例: {split_ratios}")
        print(f"🏷️  類別名稱: {class_names}")
        print(f"🧹 強制清理: {force_clean}")
        print(f"🎲 隨機種子: {random_seed}")
        print("-" * 60)
        
        # 初始化增強版分割器
        splitter = EnhancedYOLODatasetSplitter(source_directory, output_directory)
        
        # 根據模式執行分割
        if mode == "integrated":
            # 整合模式
            train_count, val_count, test_count = splitter.split_dataset_integrated(
                train_ratio=split_ratios[0],
                val_ratio=split_ratios[1],
                test_ratio=split_ratios[2],
                random_seed=random_seed,
                force_clean=force_clean
            )
            
            # 建立 YAML 配置
            splitter.create_yaml_config(
                class_names=class_names,
                train_count=train_count,
                val_count=val_count,
                test_count=test_count,
                mode="integrated"
            )
            
            result = {
                'mode': 'integrated',
                'train_count': train_count,
                'val_count': val_count,
                'test_count': test_count,
                'total_count': train_count + val_count + test_count
            }
            
        else:
            # 獨立模式
            subfolder_results = splitter.split_dataset_separate(
                train_ratio=split_ratios[0],
                val_ratio=split_ratios[1],
                test_ratio=split_ratios[2],
                random_seed=random_seed,
                force_clean=force_clean
            )
            
            # 建立 YAML 配置
            splitter.create_yaml_config(
                class_names=class_names,
                mode="separate",
                subfolder_results=subfolder_results
            )
            
            result = {
                'mode': 'separate',
                'subfolder_results': subfolder_results,
                'total_subfolders': len(subfolder_results)
            }
        
        print("\n🎉 增強版資料集分割完成！")
        print("✅ 支援子資料夾處理，確保無重複，可安全用於訓練")
        
        return result
        
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return None


# ========================================================================
# 增強版命令列介面
# ========================================================================

def enhanced_main():
    """增強版命令列主函數"""
    parser = argparse.ArgumentParser(description='增強版 YOLO 資料集自動分割工具')
    
    # 基本參數
    parser.add_argument('--source', '-s', type=str, default="./tmp_datasets/yolo_data/",
                       help='來源資料夾路徑（預設: ./tmp_datasets/yolo_data/）')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='輸出資料夾路徑（若未指定則使用來源資料夾）')
    
    # 分割配置
    parser.add_argument('--ratios', '-r', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                       help='分割比例 [train val test]（預設: 0.7 0.15 0.15）')
    parser.add_argument('--classes', '-c', type=str, nargs='+', default=['cow', 'person'],
                       help='類別名稱列表（預設: cow）')
    
    # 增強功能
    parser.add_argument('--mode', '-m', type=str, choices=['integrated', 'separate'], 
                       default='integrated',
                       help='處理模式：integrated(整合) 或 separate(獨立)（預設: integrated）')
    
    # 操作選項
    parser.add_argument('--force-clean', '-f', action='store_true',
                       help='強制清理現有分割資料夾')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='隨機種子（預設: 42）')
    
    args = parser.parse_args()
    
    # 顯示配置摘要
    print("⚙️  執行配置:")
    print(f"   來源: {args.source}")
    print(f"   輸出: {args.output if args.output else '與來源相同'}")
    print(f"   模式: {args.mode}")
    print(f"   比例: {args.ratios}")
    print(f"   類別: {args.classes}")
    print(f"   清理: {'是' if args.force_clean else '否'}")
    print(f"   種子: {args.random_seed}")
    print()
    
    # 執行分割
    result = enhanced_auto_split(
        source_directory=args.source,
        output_directory=args.output,
        split_ratios=tuple(args.ratios),
        class_names=args.classes,
        mode=args.mode,
        force_clean=args.force_clean,
        random_seed=args.random_seed
    )
    
    if result:
        if result['mode'] == 'integrated':
            print(f"\n✅ 整合模式分割成功: 訓練集 {result['train_count']}, 驗證集 {result['val_count']}, 測試集 {result['test_count']}")
        else:
            print(f"\n✅ 獨立模式分割成功: 處理了 {result['total_subfolders']} 個子資料夾")
        sys.exit(0)
    else:
        print(f"\n❌ 分割失敗")
        sys.exit(1)


# ========================================================================
# 程式碼執行入口
# ========================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        enhanced_main()
    else:
        # 預設執行整合模式
        print("🔧 執行預設分割操作（整合模式）...")
        result = enhanced_auto_split(mode="integrated")
        if result:
            print(f"✅ 完成：{result}")
        else:
            print("❌ 分割失敗")