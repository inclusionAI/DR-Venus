from openai import OpenAI
import re
import difflib
import string
    
def check_tags_balance(solution_str: str) -> bool:
    """检查标签是否正确配对
    
    Args:
        solution_str: 需要检查的字符串
    
    Returns:
        bool: 标签是否都正确配对
    """
    # 需要检查的标签对
    tags_to_check = ['code', 'tool_call', 'think', 'answer']
    
    for tag in tags_to_check:
        # 计算开始和结束标签的数量
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        
        start_count = solution_str.count(start_tag)
        end_count = solution_str.count(end_tag)
        
        # 如果开始和结束标签数量不相等，返回False
        if start_count != end_count:
            return False
            
        # 检查标签的嵌套顺序（确保结束标签不会在开始标签之前出现）
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

def preprocess_text(text: str) -> str:
    """预处理文本，用于数据集的评分
    
    处理步骤:
    1. 转换为小写
    2. 移除标点符号 (.,!?;:'"()[]{}...)
    3. 去除多余空格
    """
    # 将标点符号替换为空格
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    
    # 替换多个空格为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    # 去除首尾空格
    text = text.strip()
    return text



def compute_score(solution_str, ground_truth, data_source, val_type='f1') -> float:
    solution_str = solution_str.lower()
    ground_truth = ground_truth.lower()
    ground_truths = ground_truth.split("<|answer_split|>")
    # 首先检查标签是否配对正确(格式是否正确)
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
            answer_content = preprocess_text(answer_content)
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
    
    for gt in ground_truths:
        # 对ground truth进行预处理
        gt = preprocess_text(gt)
        
        if val_type == 'em':
            if gt == answer_content:
                return 1.0
        else:
            # 将答案和参考答案分词
            if data_source in ['zhihu','xbench_deepsearch']:
                # 中文
                common_tokens = 0
                gt_tokens = 0
                pred_tokens = len(answer_content)
                for gt_token in set(gt.split()):
                    gt_tokens += len(gt_token)
                    if gt_token in answer_content:
                        common_tokens += len(gt_token)               
                # 计算精确率和召回率
                precision = (common_tokens) / (pred_tokens) if pred_tokens else 0
                recall = (common_tokens) / (gt_tokens) if gt_tokens else 0
            else:
                pred_tokens = set(answer_content.split())
                gt_tokens = set(gt.split())
                
                if not gt_tokens:  # 避免除零错误
                    continue
                if not pred_tokens:
                    continue
                
                # 计算共同的词数
                common_tokens = pred_tokens & gt_tokens
                
                # 计算精确率和召回率
                precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
                recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
            
            # 计算F1分数
            if precision + recall > 0:  # 避免除零错误
                f1 = 2 * (precision * recall) / (precision + recall)
                max_score = max(max_score, f1)
            
    return max_score