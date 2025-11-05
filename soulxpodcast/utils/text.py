import re
from typing import List

def remove_space_between_chinese(text):
    # Remove spaces between consecutive Chinese characters
    text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)
    
    # Remove spaces between English and Chinese (English followed by Chinese)
    text = re.sub(r'([a-zA-Z])\s+([\u4e00-\u9fff])', r'\1\2', text)
    
    # Remove spaces between Chinese and English (Chinese followed by English)
    text = re.sub(r'([\u4e00-\u9fff])\s+([a-zA-Z])', r'\1\2', text)
    
    return text

# Check whether the text ends with Chinese or English and add proper punctuation
def normalize_text(current_text):
    # keep_punctuation='，。？！.,?!<| |>'
    # pattern = f'[\\p{{P}}--[{keep_punctuation}]]'
    # current_text = re.sub(pattern, '', current_text)

    # Remove spaces between consecutive Chinese characters
    current_text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', current_text)
    
    # Remove spaces between English and Chinese (English followed by Chinese)
    current_text = re.sub(r'([a-zA-Z])\s+([\u4e00-\u9fff])', r'\1\2', current_text)
    
    # Remove spaces between Chinese and English (Chinese followed by English)
    current_text = re.sub(r'([\u4e00-\u9fff])\s+([a-zA-Z])', r'\1\2', current_text)

    # Check if the text ends with a Chinese character
    if re.search(r'[\u4e00-\u9fff]$', current_text):  # 中文字符结尾
        # If the last character is not a punctuation mark, add a full stop
        if current_text[-1] not in ",.?!。，？！":
            current_text += "。"

    # Check if the text ends with an English character
    elif re.search(r'[a-zA-Z]$', current_text):  # Ends with English
        # If the last character is not a punctuation mark, add a period
        if current_text[-1] not in ".!?":
            current_text += "."
    
    return current_text


def check_monologue_text(text: str, prefix: str = None) -> bool:
    text = text.strip()
    # Check speaker tags
    if prefix is not None and (not text.startswith(prefix)):
        return False
    # Remove prefix
    if prefix is not None:
        text = text.removeprefix(prefix)
    text = text.strip()
    # If empty?
    if len(text) == 0:
        return False
    return True

def check_dialect_prompt_text(text: str, prefix: str = None) -> bool:
    text = text.strip()
    # Check COT prefix tags
    if prefix is not None and (not text.startswith(prefix)):
        return False
    text = text.strip()
    # If empty?
    if len(text) == 0:
        return False
    return True

def check_dialogue_text(text_list: List[str]) -> bool:
    if len(text_list) == 0:
        return False
    for text in text_list:
        if not (
            check_monologue_text(text, "[S1]")
            or check_monologue_text(text, "[S2]")
            or check_monologue_text(text, "[S3]")
            or check_monologue_text(text, "[S4]")
        ):
            return False
    return True