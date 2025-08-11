import re

def remove_duplicate_anchor_links(markdown_text):
    """
    移除Markdown文档中重复的锚点链接，例如：
    [#特征向量的优化](#特征向量的优化) → 特征向量的优化
    """
    # 正则匹配 [#标题](#标题) 格式的重复锚点
    pattern = re.compile(r'\[(#.+?)\]\(#.+?\)')
    
    def replace_match(match):
        # 提取标题文本（去掉开头的#）
        title = match.group(1).lstrip('#')
        return title
    
    # 替换所有匹配项
    cleaned_text = pattern.sub(replace_match, markdown_text)
    return cleaned_text

def fix_markdown_headings(markdown_text):
    """
    修复Markdown文档中的标题格式问题：
    1. 移除重复的标题
    2. 修复未闭合的标题标记
    3. 确保标题层级正确
    """
    lines = markdown_text.split('\n')
    cleaned_lines = []
    heading_stack = []  # 用于跟踪标题层级
    
    for line in lines:
        # 检测标题行（如 ## 标题）
        heading_match = re.match(r'^(#+)\s*(.*?)\s*#*\s*$', line)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            
            # 修复重复标题（如果当前标题与上一个相同）
            if heading_stack and heading_stack[-1] == (level, text):
                continue  # 跳过重复标题
            
            # 更新标题栈并保留修复后的标题
            heading_stack.append((level, text))
            cleaned_line = f"{'#' * level} {text}"
            cleaned_lines.append(cleaned_line)
        else:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

# 示例用法
if __name__ == "__main__":
    # 读取Markdown文件
    with open('test.md', 'r', encoding='utf-8') as f:
        markdown_text = f.read()
    
    # 先修复重复锚点
    markdown_text = remove_duplicate_anchor_links(markdown_text)
    
    # 再修复标题格式
    fixed_text = fix_markdown_headings(markdown_text)
    
    # 写入修复后的文件
    with open('fixed_test.md', 'w', encoding='utf-8') as f:
        f.write(fixed_text)
