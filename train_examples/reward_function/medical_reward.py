"""
医学影像报告Reward计算模块 - Vision-SR1适配版本

本模块专为Vision-SR1项目设计，实现流程图要求的所有Reward计算功能。

核心功能：
1. 关键帧验证Reward (Key Frame Verification Reward)
   - 从模型输出中解析关键帧索引
   - 使用description_answers中的二次推理文本进行一致性验证

2. 细粒度准确性Reward (Fine-grained Accuracy Reward)
   - Tumor Presence (肿瘤是否存在)
   - Location (具体解剖位置)
   - Enhancement Pattern (强化密度表现/HU值)
   - Size (大小)
   - HU Value (HU值)
   
   支持两种提取方式：
   - 正则表达式提取（默认，高效）
   - LLM提取（符合流程图要求，更灵活）

接口设计：
- compute_score() 函数兼容vision-sr1的接口格式
- 返回Dict包含各项分数，便于RL训练使用
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any, Union

# ==================== 关键帧解析 ====================

def parse_key_slice_index(report_text: str) -> Optional[int]:
    """
    从报告文本中解析关键帧索引
    
    支持的格式：
    - <key_slice>image 18</key_slice>
    - <key_slice>image 21</key_slice>
    - "image 18" (无标签)
    
    Args:
        report_text: 模型生成的报告文本
    
    Returns:
        关键帧索引（从1开始），如果无法解析返回None
    """
    if not report_text:
        return None
    
    # 方法1：查找<key_slice>标签
    pattern1 = r'<key_slice>\s*image\s*(\d+)\s*</key_slice>'
    match = re.search(pattern1, report_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # 方法2：查找"image XX"模式（在findings或impression中）
    pattern2 = r'image\s*(\d+)'
    matches = re.findall(pattern2, report_text, re.IGNORECASE)
    if matches:
        # 返回最后一个匹配（通常是最重要的）
        return int(matches[-1])
    
    return None


# ==================== 关键帧验证Reward ====================

def verify_report_consistency(
    original_report: str, 
    verification_report: str,
    threshold: float = 0.5
) -> float:
    """
    验证两次推理结果的一致性
    
    简单版本：检查关键术语是否一致
    重点关注肿瘤信息的一致性
    
    Args:
        original_report: 原始生成的报告
        verification_report: 二次推理的报告
        threshold: 一致性阈值
    
    Returns:
        一致性分数 (0.0-1.0)
    """
    # 提取关键术语（位置、大小、HU值等）
    original_keywords = extract_key_medical_terms(original_report)
    verification_keywords = extract_key_medical_terms(verification_report)
    
    if not original_keywords or not verification_keywords:
        return 0.0
    
    # 计算重叠度
    overlap = len(set(original_keywords) & set(verification_keywords))
    total = len(set(original_keywords) | set(verification_keywords))
    
    if total == 0:
        return 0.0
    
    consistency_score = overlap / total
    
    return 1.0 if consistency_score >= threshold else consistency_score


def extract_key_medical_terms(text: str) -> List[str]:
    """
    从报告中提取关键医学术语
    
    提取：
    - 解剖位置（如"hepatic segment 7"）
    - 大小信息（如"3.3 x 2.6 cm"）
    - HU值（如"112.1"）
    - 病变类型（如"hypoattenuating"）
    """
    keywords = []
    
    # 提取解剖位置
    location_pattern = r'(hepatic segment|segment|liver|pancreas|kidney)\s*\d+'
    locations = re.findall(location_pattern, text, re.IGNORECASE)
    keywords.extend([loc.lower() for loc in locations])
    
    # 提取大小
    size_pattern = r'\d+\.?\d*\s*x\s*\d+\.?\d*\s*cm'
    sizes = re.findall(size_pattern, text, re.IGNORECASE)
    keywords.extend(sizes)
    
    # 提取HU值
    hu_pattern = r'HU\s*value\s*of\s*(\d+\.?\d*)'
    hu_values = re.findall(hu_pattern, text, re.IGNORECASE)
    keywords.extend([f"hu_{hu}" for hu in hu_values])
    
    # 提取病变类型
    lesion_types = ['hypoattenuating', 'hyperattenuating', 'isoattenuating', 
                   'mass', 'lesion', 'tumor']
    for lt in lesion_types:
        if re.search(lt, text, re.IGNORECASE):
            keywords.append(lt.lower())
    
    return keywords


# ==================== 细粒度准确性Reward ====================

def extract_structured_info_regex(report_text: str) -> Dict[str, Any]:
    """
    从报告文本中提取结构化信息（使用正则表达式）
    
    提取：
    - tumor_presence: bool (是否存在肿瘤)
    - location: str (具体解剖位置)
    - size: str (大小信息)
    - hu_value: float (HU值)
    - enhancement_pattern: str (强化模式)
    
    Args:
        report_text: 报告文本
    
    Returns:
        包含结构化信息的字典
    """
    info = {
        'tumor_presence': False,
        'location': None,
        'size': None,
        'hu_value': None,
        'enhancement_pattern': None,
        'volume': None
    }
    
    # 1. 检查是否存在肿瘤
    tumor_keywords = ['tumor', 'mass', 'lesion', 'nodule', 'carcinoma']
    info['tumor_presence'] = any(
        re.search(kw, report_text, re.IGNORECASE) 
        for kw in tumor_keywords
    )
    
    # 2. 提取位置
    location_pattern = r'(hepatic segment|segment|hepatic)\s*(\d+[\/\d]*)'
    location_match = re.search(location_pattern, report_text, re.IGNORECASE)
    if location_match:
        info['location'] = location_match.group(0).lower()
    
    # 3. 提取大小
    size_pattern = r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)\s*cm'
    size_match = re.search(size_pattern, report_text, re.IGNORECASE)
    if size_match:
        info['size'] = f"{size_match.group(1)} x {size_match.group(2)} cm"
    
    # 4. 提取HU值
    hu_patterns = [
        r'HU\s*value\s*of\s*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*\+\/\-\s*\d+\.?\d*\s*HU',
        r'mean\s*HU\s*value\s*of\s*(\d+\.?\d*)'
    ]
    for pattern in hu_patterns:
        hu_match = re.search(pattern, report_text, re.IGNORECASE)
        if hu_match:
            try:
                info['hu_value'] = float(hu_match.group(1))
                break
            except:
                pass
    
    # 5. 提取强化模式
    enhancement_patterns = {
        'hypoattenuating': r'hypoattenuat',
        'hyperattenuating': r'hyperattenuat',
        'isoattenuating': r'isoattenuat',
        'enhancing': r'enhancing',
        'non-enhancing': r'non-enhancing'
    }
    for pattern_name, pattern in enhancement_patterns.items():
        if re.search(pattern, report_text, re.IGNORECASE):
            info['enhancement_pattern'] = pattern_name
            break
    
    # 6. 提取体积
    volume_pattern = r'(\d+\.?\d*)\s*cc\s*(?:in volume|volume)'
    volume_match = re.search(volume_pattern, report_text, re.IGNORECASE)
    if volume_match:
        try:
            info['volume'] = float(volume_match.group(1))
        except:
            pass
    
    return info


def extract_structured_info_llm(
    report_text: str,
    llm_client=None,
    llm_model: str = None
) -> Dict[str, Any]:
    """
    使用LLM从报告文本中提取结构化信息（符合流程图要求）
    
    如果提供了llm_client，则使用LLM提取；否则回退到正则表达式
    
    Args:
        report_text: 报告文本
        llm_client: LLM客户端（如OpenAI客户端），如果为None则使用正则表达式
        llm_model: LLM模型名称
    
    Returns:
        包含结构化信息的字典
    """
    # 如果没有提供LLM客户端，回退到正则表达式
    if llm_client is None:
        return extract_structured_info_regex(report_text)
    
    # 构造LLM prompt
    prompt = f"""请从以下医学影像报告中提取结构化信息，并以JSON格式返回。

报告文本：
{report_text}

请提取以下信息：
1. tumor_presence: 是否存在肿瘤（true/false）
2. location: 具体解剖位置（如"hepatic segment 7"）
3. size: 病灶大小（如"3.3 x 2.6 cm"）
4. hu_value: HU值（浮点数，如果提到）
5. enhancement_pattern: 强化模式（如"hypoattenuating"）
6. volume: 体积（浮点数，单位cc，如果提到）

请以JSON格式返回，格式如下：
{{
    "tumor_presence": true/false,
    "location": "位置字符串或null",
    "size": "大小字符串或null",
    "hu_value": 数值或null,
    "enhancement_pattern": "模式字符串或null",
    "volume": 数值或null
}}"""

    try:
        # 调用LLM
        if hasattr(llm_client, 'chat') or hasattr(llm_client, 'completions'):
            # OpenAI格式
            if hasattr(llm_client, 'chat'):
                response = llm_client.chat.completions.create(
                    model=llm_model or "gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                result_text = response.choices[0].message.content
            else:
                response = llm_client.completions.create(
                    model=llm_model or "gpt-4",
                    prompt=prompt,
                    temperature=0
                )
                result_text = response.choices[0].text
        else:
            # 其他格式，尝试通用接口
            result_text = str(llm_client(prompt))
        
        # 解析JSON
        # 尝试提取JSON部分
        json_match = re.search(r'\{[\s\S]*\}', result_text)
        if json_match:
            result_json = json.loads(json_match.group(0))
        else:
            result_json = json.loads(result_text)
        
        # 转换为标准格式
        info = {
            'tumor_presence': result_json.get('tumor_presence', False),
            'location': result_json.get('location'),
            'size': result_json.get('size'),
            'hu_value': result_json.get('hu_value'),
            'enhancement_pattern': result_json.get('enhancement_pattern'),
            'volume': result_json.get('volume')
        }
        
        return info
        
    except Exception as e:
        # LLM提取失败，回退到正则表达式
        print(f"LLM提取失败，回退到正则表达式: {e}")
        return extract_structured_info_regex(report_text)


def tumor_presence_reward(
    generated_report: str,
    ground_truth: Union[str, Dict[str, Any]]
) -> float:
    """
    计算肿瘤存在性Reward（独立的第一个奖励维度）
    
    这是最基础、最重要的判断，具有第一性，因此作为独立的reward存在。
    
    Args:
        generated_report: 模型生成的报告
        ground_truth: Ground Truth（可以是字符串或包含结构化信息的字典）
    
    Returns:
        肿瘤存在性分数 (0.0 或 1.0)
        - 1.0: 模型判断与GT一致（都有肿瘤或都没有肿瘤）
        - 0.0: 模型判断与GT不一致
    """
    # 检查生成报告中是否存在肿瘤
    tumor_keywords = ['tumor', 'mass', 'lesion', 'nodule', 'carcinoma']
    gen_has_tumor = any(
        re.search(kw, generated_report, re.IGNORECASE) 
        for kw in tumor_keywords
    )
    
    # 检查GT中是否存在肿瘤
    if isinstance(ground_truth, dict):
        if 'tumor_presence' in ground_truth:
            gt_has_tumor = ground_truth.get('tumor_presence', False)
        else:
            gt_text = ground_truth.get('single_organ_report', '')
            gt_has_tumor = any(
                re.search(kw, gt_text, re.IGNORECASE) 
                for kw in tumor_keywords
            )
    else:
        gt_has_tumor = any(
            re.search(kw, ground_truth, re.IGNORECASE) 
            for kw in tumor_keywords
        )
    
    # 判断是否一致
    return 1.0 if (gen_has_tumor == gt_has_tumor) else 0.0


def fine_grained_accuracy_reward(
    generated_report: str,
    ground_truth: Union[str, Dict[str, Any]],
    use_llm: bool = False,
    llm_client=None,
    llm_model: str = None
) -> Dict[str, float]:
    """
    计算细粒度准确性Reward（第二个奖励维度）
    
    从报告中提取数值信息（如肿瘤大小、HU值等），与标准答案对比。
    注意：不包含tumor_presence，因为那是独立的第一个奖励维度。
    
    维度：
    1. Location: 位置匹配度 (0.0-1.0)
    2. Enhancement Pattern: 强化模式匹配度 (0.0-1.0)
    3. Size: 大小匹配度 (0.0-1.0)
    4. HU Value: HU值匹配度 (0.0-1.0)
    
    Args:
        generated_report: 模型生成的报告
        ground_truth: Ground Truth（可以是字符串或包含结构化信息的字典）
        use_llm: 是否使用LLM提取（符合流程图要求）
        llm_client: LLM客户端（如果use_llm=True）
        llm_model: LLM模型名称
    
    Returns:
        包含各项分数的字典（不包含tumor_presence）
    """
    # 提取生成报告的结构化信息
    if use_llm:
        gen_info = extract_structured_info_llm(generated_report, llm_client, llm_model)
    else:
        gen_info = extract_structured_info_regex(generated_report)
    
    # 提取GT的结构化信息
    if isinstance(ground_truth, dict):
        if use_llm:
            gt_info = extract_structured_info_llm(
                ground_truth.get('single_organ_report', ''), 
                llm_client, 
                llm_model
            )
        else:
            gt_info = extract_structured_info_regex(ground_truth.get('single_organ_report', ''))
        # 如果GT字典中已有结构化信息，优先使用
        if 'location' in ground_truth:
            gt_info['location'] = ground_truth.get('location')
    else:
        if use_llm:
            gt_info = extract_structured_info_llm(ground_truth, llm_client, llm_model)
        else:
            gt_info = extract_structured_info_regex(ground_truth)
    
    # 计算各项分数（不包含tumor_presence）
    scores = {}
    
    # 2. Location (0.0-1.0)
    if gt_info['location'] and gen_info['location']:
        # 简单匹配：检查是否包含相同的segment编号
        gen_segment = re.search(r'segment\s*(\d+)', gen_info['location'], re.IGNORECASE)
        gt_segment = re.search(r'segment\s*(\d+)', gt_info['location'], re.IGNORECASE)
        if gen_segment and gt_segment:
            scores['location'] = 1.0 if (
                gen_segment.group(1) == gt_segment.group(1)
            ) else 0.0
        else:
            # 使用字符串相似度
            scores['location'] = 1.0 if (
                gen_info['location'].lower() == gt_info['location'].lower()
            ) else 0.5
    elif not gt_info['location'] and not gen_info['location']:
        scores['location'] = 1.0
    else:
        scores['location'] = 0.0
    
    # 3. Enhancement Pattern
    if gt_info['enhancement_pattern'] and gen_info['enhancement_pattern']:
        scores['enhancement_pattern'] = 1.0 if (
            gen_info['enhancement_pattern'] == gt_info['enhancement_pattern']
        ) else 0.0
    elif not gt_info['enhancement_pattern'] and not gen_info['enhancement_pattern']:
        scores['enhancement_pattern'] = 1.0
    else:
        scores['enhancement_pattern'] = 0.0
    
    # 4. Size (允许一定误差)
    if gt_info['size'] and gen_info['size']:
        # 提取数值进行比较
        gen_size_match = re.search(r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)', gen_info['size'])
        gt_size_match = re.search(r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)', gt_info['size'])
        if gen_size_match and gt_size_match:
            gen_w, gen_h = float(gen_size_match.group(1)), float(gen_size_match.group(2))
            gt_w, gt_h = float(gt_size_match.group(1)), float(gt_size_match.group(2))
            # 计算相对误差
            error_w = abs(gen_w - gt_w) / max(gt_w, 0.1)
            error_h = abs(gen_h - gt_h) / max(gt_h, 0.1)
            avg_error = (error_w + error_h) / 2
            scores['size'] = max(0.0, 1.0 - avg_error)  # 误差越小分数越高
        else:
            scores['size'] = 0.5
    elif not gt_info['size'] and not gen_info['size']:
        scores['size'] = 1.0
    else:
        scores['size'] = 0.0
    
    # 5. HU Value (允许一定误差)
    if gt_info['hu_value'] is not None and gen_info['hu_value'] is not None:
        error = abs(gen_info['hu_value'] - gt_info['hu_value']) / max(abs(gt_info['hu_value']), 1.0)
        scores['hu_value'] = max(0.0, 1.0 - error * 0.1)  # 允许10%误差
    elif gt_info['hu_value'] is None and gen_info['hu_value'] is None:
        scores['hu_value'] = 1.0
    else:
        scores['hu_value'] = 0.0
    
    return scores


# ==================== 统一接口 ====================

def compute_score(
    predicts: List[str],
    ground_truths: List[Union[str, Dict[str, Any]]],
    questions: List[str],
    description_answers: List[str],
    format_weight: float = 0.1,
    tumor_presence_weight: float = 0.2,  # 肿瘤存在性奖励权重（第一性）
    key_frame_weight: float = 0.3,        # 关键帧验证奖励权重
    fine_grained_weight: float = 0.5,     # 细粒度准确性奖励权重
    enable_tumor_presence: bool = True,   # 是否启用肿瘤存在性奖励
    enable_key_frame_verification: bool = True,
    enable_fine_grained_accuracy: bool = True,
    use_llm_for_fine_grained: bool = False,
    llm_client=None,
    llm_model: str = None
) -> List[Dict[str, float]]:
    """
    计算Reward分数 - 兼容vision-sr1接口
    
    这是主要的接口函数，可以被vision-sr1项目直接调用。
    
    Args:
        predicts: 模型生成的预测列表
        ground_truths: Ground Truth列表（可以是字符串或字典）
        questions: 问题列表
        description_answers: description阶段的答案列表（必须，用于关键帧验证）
        format_weight: 格式权重（暂时未使用，保留兼容性）
        tumor_presence_weight: 肿瘤存在性Reward的权重（第一性，默认0.2）
        key_frame_weight: 关键帧验证Reward的权重（默认0.3）
        fine_grained_weight: 细粒度准确性Reward的权重（默认0.5）
        enable_tumor_presence: 是否启用肿瘤存在性奖励（默认True）
        enable_key_frame_verification: 是否启用关键帧验证（默认True）
        enable_fine_grained_accuracy: 是否启用细粒度准确性评估（默认True）
        use_llm_for_fine_grained: 是否使用LLM提取细粒度信息（符合流程图要求）
        llm_client: LLM客户端（如果use_llm_for_fine_grained=True）
        llm_model: LLM模型名称
    
        Returns:
        分数列表，每个元素是一个字典，包含各项分数：
        {
            'overall': float,  # 总体分数（三个独立reward的加权和）
            'tumor_presence': float,  # 肿瘤存在性奖励分数（独立的第一个reward，0/1）
            'key_frame_verification': float,  # 关键帧验证奖励分数（第二个reward）
            'fine_grained_accuracy': float,  # 细粒度准确性综合分数（第三个reward）
            'location': float,  # 位置分数（细粒度维度）
            'enhancement_pattern': float,  # 强化模式分数（细粒度维度）
            'size': float,  # 大小分数（细粒度维度）
            'hu_value': float,  # HU值分数（细粒度维度）
        }
    """
    scores = []
    
    for i, (predict, ground_truth, question) in enumerate(
        zip(predicts, ground_truths, questions)
    ):
        score_dict = {}
        
        # 1. 肿瘤存在性Reward（独立的第一个reward，具有第一性）
        tumor_presence_score = 0.0
        if enable_tumor_presence:
            tumor_presence_score = tumor_presence_reward(predict, ground_truth)
        score_dict["tumor_presence"] = float(tumor_presence_score)
        
        # 2. 关键帧验证Reward（第二个reward）
        # 使用 rollout 阶段产出的二次推理文本（Vision-SR1 会透传到 description_answers）
        key_frame_score = 0.0
        score_dict["key_frame_verification"] = 0.0
        score_dict["key_frame_index"] = None

        if enable_key_frame_verification:
            if description_answers is not None and i < len(description_answers):
                verification_text = description_answers[i]
                try:
                    key_slice_index = parse_key_slice_index(predict)
                    key_frame_score = verify_report_consistency(predict, verification_text)
                    score_dict["key_frame_verification"] = float(key_frame_score)
                    score_dict["key_frame_index"] = key_slice_index
                except Exception as e:
                    score_dict["key_frame_verification"] = 0.0
                    score_dict["key_frame_index"] = None
                    score_dict["key_frame_error"] = f"关键帧验证失败: {e}"
            else:
                # 如果没有description_answers，关键帧验证分数为0
                score_dict["key_frame_verification"] = 0.0
                score_dict["key_frame_index"] = None
        
        # 3. 细粒度准确性Reward（第三个reward）
        # 注意：不包含tumor_presence，因为那是独立的第一个reward
        if enable_fine_grained_accuracy:
            accuracy_scores = fine_grained_accuracy_reward(
                predict, 
                ground_truth,
                use_llm=use_llm_for_fine_grained,
                llm_client=llm_client,
                llm_model=llm_model
            )
            score_dict.update(accuracy_scores)
            
            # 计算细粒度准确性综合分数（各项平均，不包含tumor_presence）
            fine_grained_items = ['location', 'enhancement_pattern', 'size', 'hu_value']
            available_scores = [accuracy_scores.get(k, 0.0) for k in fine_grained_items]
            score_dict['fine_grained_accuracy'] = sum(available_scores) / len(available_scores) if available_scores else 0.0
        else:
            score_dict['fine_grained_accuracy'] = 0.0
            score_dict.update({
                'location': 0.0,
                'enhancement_pattern': 0.0,
                'size': 0.0,
                'hu_value': 0.0
            })
        
        # 4. 计算总体分数（三个独立reward的加权和）
        overall = (
            tumor_presence_weight * tumor_presence_score +
            key_frame_weight * key_frame_score +
            fine_grained_weight * score_dict['fine_grained_accuracy']
        )
        score_dict['overall'] = overall
        
        scores.append(score_dict)
    
    return scores


