#!/usr/bin/env python3
"""
更新RL训练数据中的prompt，在末尾添加格式说明以帮助正则表达式提取

使用方法:
    python scripts/update_prompt_format.py --input_file data/test_data_val.jsonl --output_file data/test_data_val_updated.jsonl
    python scripts/update_prompt_format.py --input_dir data/ --output_dir data_updated/
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any


# 格式说明文本（方案2：在prompt末尾添加）
FORMAT_INSTRUCTIONS = """ 

Important: When describing lesion details in <findings>, use these exact formats for better parsing:
- Location: "hepatic segment X" or "segment X" (e.g., "hepatic segment 7")
- Size: "X x Y cm" (e.g., "3.3 x 2.6 cm")
- HU value: "HU value of X" or "mean HU value of X +/- Y" or "X +/- Y HU" (e.g., "HU value of 112.1" or "mean HU value of 112.1 +/- 17.3")
- Enhancement pattern: "hypoattenuating", "hyperattenuating", "isoattenuating", "enhancing", or "non-enhancing"
- Volume: "X cc" or "X cc in volume" (e.g., "7.1 cc" or "7.1 cc in volume")"""


def update_prompt(question: str) -> str:
    """
    更新prompt，在末尾添加格式说明
    
    Args:
        question: 原始prompt文本
    
    Returns:
        更新后的prompt文本
    """
    # 如果已经包含格式说明，不重复添加
    if "Important: When describing lesion details" in question:
        return question
    
    # 在prompt末尾添加格式说明
    return question.rstrip() + FORMAT_INSTRUCTIONS


def process_jsonl_file(input_file: Path, output_file: Path, dry_run: bool = False):
    """
    处理单个JSONL文件，更新其中的question字段
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        dry_run: 如果为True，只打印统计信息，不实际写入文件
    """
    updated_count = 0
    total_count = 0
    
    print(f"Processing: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        if not dry_run:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            f_out = open(output_file, 'w', encoding='utf-8')
        
        try:
            for line_num, line in enumerate(f_in, 1):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    total_count += 1
                    
                    if 'question' in data:
                        old_question = data['question']
                        new_question = update_prompt(old_question)
                        
                        if old_question != new_question:
                            data['question'] = new_question
                            updated_count += 1
                        
                        if not dry_run:
                            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                        elif updated_count <= 3:  # 只打印前3个示例
                            print(f"  Example {updated_count}:")
                            print(f"    Old: {old_question[:100]}...")
                            print(f"    New: {new_question[:100]}...")
                    else:
                        # 如果没有question字段，直接写入（不更新）
                        if not dry_run:
                            f_out.write(line)
                
                except json.JSONDecodeError as e:
                    print(f"  Warning: Skipping invalid JSON at line {line_num}: {e}")
                    if not dry_run:
                        f_out.write(line)  # 保留原始行
        
        finally:
            if not dry_run:
                f_out.close()
    
    print(f"  Total: {total_count} entries, Updated: {updated_count} prompts")
    return updated_count, total_count


def main():
    parser = argparse.ArgumentParser(
        description="Update prompts in RL training data files to include format instructions"
    )
    parser.add_argument(
        '--input_file',
        type=str,
        help='Input JSONL file path'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Input directory containing JSONL files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory for updated files'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Dry run mode: only show statistics without writing files'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='_updated',
        help='Suffix to add to output filenames (default: _updated)'
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE: No files will be modified\n")
    
    # 处理单个文件
    if args.input_file and args.output_file:
        input_path = Path(args.input_file)
        output_path = Path(args.output_file)
        
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return
        
        process_jsonl_file(input_path, output_path, dry_run=args.dry_run)
    
    # 处理目录中的所有JSONL文件
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir) if args.output_dir else input_dir
        
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}")
            return
        
        jsonl_files = list(input_dir.glob('*.jsonl'))
        
        if not jsonl_files:
            print(f"No JSONL files found in: {input_dir}")
            return
        
        print(f"Found {len(jsonl_files)} JSONL file(s)\n")
        
        total_updated = 0
        total_entries = 0
        
        for jsonl_file in jsonl_files:
            # 生成输出文件名
            if args.suffix:
                output_file = output_dir / f"{jsonl_file.stem}{args.suffix}{jsonl_file.suffix}"
            else:
                output_file = output_dir / jsonl_file.name
            
            updated, total = process_jsonl_file(jsonl_file, output_file, dry_run=args.dry_run)
            total_updated += updated
            total_entries += total
            print()
        
        print(f"Summary: {total_updated}/{total_entries} prompts updated across {len(jsonl_files)} file(s)")
    
    else:
        parser.print_help()
        print("\nError: Please provide either --input_file/--output_file or --input_dir")


if __name__ == '__main__':
    main()










