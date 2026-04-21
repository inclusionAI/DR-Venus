import re
import json
import json5

def check_tags_balance(solution_str: str) -> bool:
    tags_to_check = ['code', 'tool_call', 'think', 'answer']
    for tag in tags_to_check:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start_count = solution_str.count(start_tag)
        end_count = solution_str.count(end_tag)
        if start_count != end_count:
            return False
        last_pos = -1
        while True:
            start_pos = solution_str.find(start_tag, last_pos + 1)
            if start_pos == -1:
                break
            end_pos = solution_str.find(end_tag, start_pos)
            if end_pos == -1:
                return False
            last_pos = end_pos
    return True

def compute_score(solution_str, ground_truth, data_source_e, val_type='f1') -> float:
    solution_str = solution_str.lower()
    ground_truth = ground_truth.lower()
    if not check_tags_balance(solution_str):
        if val_type == 'noformatf1':
            return 0
        else:
            return -2.0

    # 使用正则提取第一个<answer>标签中的内容
    try:
        answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            # 对答案进行预处理
            answer_json = json.loads(answer_content)
        else:
            if val_type == 'noformatf1':
                return 0
            else:
                return -2.0
    except Exception as e:
        print(f"Error extracting answer content: {e}")
        if val_type == 'noformatf1':
            return 0
        else:
            return -2.0
    
    max_score = 0.0
    try:
        ground_truth = json5.loads(ground_truth)
        if '最高点' in ground_truth:
            """
            评分任务1：量化交易师任务
            ground_truths: list of dict [{"最高点":12.34,"最低点":10.56}, ...]
            """
            pred_high = answer_json.get("最高点", -1.0)
            pred_low = answer_json.get("最低点", -1.0)
            if pred_high == -1.0:
                high_error = 0.2
                low_error = 0.2
            else:
                gt_high = ground_truth["最高点"]
                gt_low = ground_truth["最低点"]
                high_error = abs(pred_high - gt_high) / gt_high 
                low_error = abs(pred_low - gt_low) / gt_low 
                high_error = 1.0 if high_error < 0.01 else max(0, (0.05 - high_error) * 24) 
                low_error = 1.0 if low_error < 0.01 else max(0, (0.05 - low_error) * 24) 
            score = (high_error + low_error) / 2
            max_score = score
        else:

            pred_value = answer_json.get("预测值", None)
            pred_direction = answer_json.get("美元欧元汇率多空", None) # 多 或者 空 或者 影响不大 或者 无法预测
            if pred_direction == '无法预测':
                max_score = 0.2
            else:
                gt_value = ground_truth["预测值"]
                gt_direction = ground_truth["美元欧元汇率多空"]
                direction_correct = (pred_direction == gt_direction)
                max_score = direction_correct
    except:
        return -2.0
    return max_score
    