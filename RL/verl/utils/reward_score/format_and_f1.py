from openai import OpenAI
import re
import difflib
import string

def check_tags_balance(solution_str: str) -> bool:
    """Check whether all required XML-style tags are properly paired.

    Args:
        solution_str: The string to validate.

    Returns:
        bool: True iff every tag has a matching opening/closing counterpart.
    """
    tags_to_check = ['code', 'tool_call', 'think', 'answer']

    for tag in tags_to_check:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"

        start_count = solution_str.count(start_tag)
        end_count = solution_str.count(end_tag)

        if start_count != end_count:
            return False

        # Verify nesting order: a closing tag must not appear before its opening tag.
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
    """Normalize text for dataset-level scoring.

    Steps:
        1. Lowercase.
        2. Replace punctuation with spaces.
        3. Collapse repeated whitespace and strip.
    """
    for punct in string.punctuation:
        text = text.replace(punct, ' ')

    text = re.sub(r'\s+', ' ', text)

    text = text.strip()
    return text



def compute_score(solution_str, ground_truth, data_source, val_type='f1') -> float:
    solution_str = solution_str.lower()
    ground_truth = ground_truth.lower()
    ground_truths = ground_truth.split("<|answer_split|>")
    # Verify tag pairing (format correctness) first.
    if not check_tags_balance(solution_str):
        
        if val_type == 'noformatf1':
            return 0
        else:
            return -2.0

    # Extract the content of the first <answer> tag.
    try:
        answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
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
        gt = preprocess_text(gt)
        
        if val_type == 'em':
            if gt == answer_content:
                return 1.0
        else:
            if data_source in ['zhihu','xbench_deepsearch']:
                # Chinese-language datasets: character-level token matching.
                common_tokens = 0
                gt_tokens = 0
                pred_tokens = len(answer_content)
                for gt_token in set(gt.split()):
                    gt_tokens += len(gt_token)
                    if gt_token in answer_content:
                        common_tokens += len(gt_token)
                precision = (common_tokens) / (pred_tokens) if pred_tokens else 0
                recall = (common_tokens) / (gt_tokens) if gt_tokens else 0
            else:
                pred_tokens = set(answer_content.split())
                gt_tokens = set(gt.split())
                
                if not gt_tokens:  # guard against divide-by-zero
                    continue
                if not pred_tokens:
                    continue
                
                common_tokens = pred_tokens & gt_tokens
                
                precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
                recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
            
            if precision + recall > 0:  # guard against divide-by-zero
                f1 = 2 * (precision * recall) / (precision + recall)
                max_score = max(max_score, f1)
            
    return max_score
