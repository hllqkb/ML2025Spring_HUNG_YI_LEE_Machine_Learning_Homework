from fastapi import FastAPI
from pydantic import BaseModel # 主要用于数据验证和设置管理
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
# 取消注释下面代码来查看 Base Model 的效果。
# class User(BaseModel):
#     id:int
#     name:str='JK'
#     is_active:bool=True
# user=User(id=1,name='Tom')
# print(user)

# 定义请求体的数据模型
class PromptRequest(BaseModel):
    prompt:str
app=FastAPI()
# 精简版GPT-2模型
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# 把模型移到GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
@app.post('/generate')
def generate_text(request:PromptRequest):
    # 定义格式要求
    prompt=request.prompt
    # 切换到评估模式
    model.eval()
    # 编码输入文本
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs= {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs, # 输入
            max_length=200, # 最大长度
            num_beams=5, # 最优的 5 个候选序列,Beam Search 的数量，提高生成文本的质量
            no_repeat_ngram_size=2, # 禁止重复的n-gram大小,防止生成重复的 n-gram
            early_stopping=True # 早停,当使用 Beam Search 时，若所有候选序列都生成了结束标记（如 <eos>），则提前停止生成，这有助于生成更自然和适当长度的文本。
        )
    # 解码输出文本
    generated_text=tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("模型生成的文本：")
    print(generated_text)    
    return {"text":generated_text}
# uvicorn 6_app_fastapi:app --host 0.0.0.0 --port 8008