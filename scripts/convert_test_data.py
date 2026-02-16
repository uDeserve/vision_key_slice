#!/usr/bin/env python3
"""
将测试数据从JSON格式转换为标准JSONL格式
用于RL训练数据准备

输入格式：
- JSON文件，包含image（相对路径列表）和conversations（human/gpt对话）

输出格式：
- JSONL文件，每行一个JSON对象，包含question、ground_truth、images（绝对路径列表）
"""

import json
import os
import sys
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any


def extract_question_from_conversation(conversations: List[Dict[str, Any]]) -> str:
    """
    从conversations中提取question（human的value）
    去掉<image>标签，只保留文本部分
    """
    for conv in conversations:
        if conv.get("from") == "human":
            value = conv.get("value", "")
            # 去掉所有的<image>标签和换行符
            # 保留文本内容
            text = value.replace("<image>", "").strip()
            # 清理多余的换行符和空格
            text = re.sub(r'\n+', ' ', text)
            text = re.sub(r' +', ' ', text)
            return text.strip()
    return ""


def extract_ground_truth_from_conversation(conversations: List[Dict[str, Any]]) -> str:
    """
    从conversations中提取ground_truth（gpt的value）
    """
    for conv in conversations:
        if conv.get("from") == "gpt":
            return conv.get("value", "").strip()
    return ""


def convert_image_paths(images: List[str], base_dir: str) -> List[str]:
    """
    将相对图像路径转换为绝对路径
    
    Args:
        images: 相对路径列表
        base_dir: 数据基础目录（解压后的目录）
    
    Returns:
        绝对路径列表
    """
    absolute_paths = []
    for img_path in images:
        # 如果已经是绝对路径，直接使用
        if os.path.isabs(img_path):
            absolute_paths.append(img_path)
        else:
            # 转换为绝对路径
            abs_path = os.path.join(base_dir, img_path)
            # 检查文件是否存在
            if os.path.exists(abs_path):
                absolute_paths.append(os.path.abspath(abs_path))
            else:
                print(f"警告: 图像文件不存在: {abs_path}", file=sys.stderr)
                # 即使不存在也添加，让后续处理决定如何处理
                absolute_paths.append(os.path.abspath(abs_path))
    return absolute_paths


def convert_json_entry(entry: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    """
    转换单个JSON条目为标准格式
    
    Args:
        entry: 原始JSON条目
        base_dir: 数据基础目录
    
    Returns:
        转换后的条目
    """
    # 提取question
    conversations = entry.get("conversations", [])
    question = extract_question_from_conversation(conversations)
    
    # 提取ground_truth
    ground_truth = extract_ground_truth_from_conversation(conversations)
    
    # 转换图像路径
    images = entry.get("image", [])
    if not isinstance(images, list):
        images = [images]
    
    absolute_images = convert_image_paths(images, base_dir)
    
    # 构造标准格式
    converted_entry = {
        "question": question,
        "ground_truth": ground_truth,
        "images": absolute_images
    }
    
    return converted_entry


def convert_json_file(json_path: str, base_dir: str) -> List[Dict[str, Any]]:
    """
    转换单个JSON文件
    
    Args:
        json_path: JSON文件路径
        base_dir: 数据基础目录
    
    Returns:
        转换后的条目列表
    """
    print(f"处理文件: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果data是列表，直接处理
    # 如果data是字典，转换为列表
    if isinstance(data, dict):
        data = [data]
    
    converted_entries = []
    for entry in data:
        try:
            converted_entry = convert_json_entry(entry, base_dir)
            # 验证必需字段
            if converted_entry["question"] and converted_entry["ground_truth"]:
                converted_entries.append(converted_entry)
            else:
                print(f"警告: 跳过无效条目（缺少question或ground_truth）", file=sys.stderr)
        except Exception as e:
            print(f"错误: 处理条目时出错: {e}", file=sys.stderr)
            continue
    
    print(f"  成功转换 {len(converted_entries)} 个条目")
    return converted_entries


def convert_test_data(
    input_dir: str,
    output_jsonl: str,
    json_pattern: str = "qwen3vl_*.json"
):
    """
    转换测试数据目录中的所有JSON文件
    
    Args:
        input_dir: 输入目录（解压后的测试数据目录）
        output_jsonl: 输出JSONL文件路径
        json_pattern: JSON文件匹配模式
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 查找所有匹配的JSON文件
    json_files = list(input_path.glob(f"**/{json_pattern}"))
    
    if not json_files:
        raise FileNotFoundError(f"未找到匹配的JSON文件: {json_pattern}")
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 获取基础目录（用于转换相对路径）
    base_dir = str(input_path.absolute())
    
    # 转换所有文件
    all_entries = []
    for json_file in json_files:
        entries = convert_json_file(str(json_file), base_dir)
        all_entries.extend(entries)
    
    # 写入JSONL文件
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n写入输出文件: {output_jsonl}")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"转换完成！共 {len(all_entries)} 个条目")
    print(f"输出文件: {output_jsonl}")
    
    # 验证输出文件
    print(f"\n验证输出文件...")
    with open(output_jsonl, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            try:
                entry = json.loads(line)
                assert "question" in entry, "缺少question字段"
                assert "ground_truth" in entry, "缺少ground_truth字段"
                assert "images" in entry, "缺少images字段"
                assert isinstance(entry["images"], list), "images必须是列表"
                count += 1
            except Exception as e:
                print(f"验证失败: {e}", file=sys.stderr)
                raise
    
    print(f"验证通过！共 {count} 个有效条目")
    
    return all_entries


def main():
    parser = argparse.ArgumentParser(
        description="将测试数据从JSON格式转换为标准JSONL格式"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="输入目录（解压后的测试数据目录，如包含single_organ_test_data的目录）"
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="输出JSONL文件路径"
    )
    parser.add_argument(
        "--json_pattern",
        type=str,
        default="qwen3vl_*.json",
        help="JSON文件匹配模式（默认: qwen3vl_*.json）"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="训练集比例（默认: 0.9，即90%%训练，10%%验证）"
    )
    parser.add_argument(
        "--output_train",
        type=str,
        default=None,
        help="训练集输出路径（如果指定，会自动分割数据）"
    )
    parser.add_argument(
        "--output_val",
        type=str,
        default=None,
        help="验证集输出路径（如果指定，会自动分割数据）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（用于数据分割，默认: 42）"
    )
    
    args = parser.parse_args()
    
    try:
        # 转换数据
        all_entries = convert_test_data(
            input_dir=args.input_dir,
            output_jsonl=args.output_jsonl,
            json_pattern=args.json_pattern
        )
        
        # 如果需要分割数据
        if args.output_train and args.output_val:
            print(f"\n分割数据（训练集: {args.train_ratio*100:.1f}%, 验证集: {(1-args.train_ratio)*100:.1f}%）...")
            random.seed(args.seed)
            random.shuffle(all_entries)
            
            split_idx = int(len(all_entries) * args.train_ratio)
            train_entries = all_entries[:split_idx]
            val_entries = all_entries[split_idx:]
            
            # 写入训练集
            Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_train, 'w', encoding='utf-8') as f:
                for entry in train_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            print(f"训练集: {len(train_entries)} 条，已保存到 {args.output_train}")
            
            # 写入验证集
            Path(args.output_val).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_val, 'w', encoding='utf-8') as f:
                for entry in val_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            print(f"验证集: {len(val_entries)} 条，已保存到 {args.output_val}")
        
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()







