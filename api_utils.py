import asyncio
from vertexai.generative_models import GenerativeModel, Part
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from cv_utils import reshape_frame_to_512, encode_frame_cv

#base_url="http://60.204.212.177:3000/v1"
# base_url="http://180.184.174.65:3000/v1"
base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key="AIzaSyBX2-EBZ6fSJtZ8szMTEl3TndOmqInQtbw" # gemini
# api_key="sk-0fb97d6740c24e4fbb8b1393a6368295"
client = AsyncOpenAI(base_url=base_url, api_key=api_key)

# 设置并发限制
MAX_RETRIES = 10
BASE_DELAY = 1
MAX_DELAY = 60
MAX_CONCURRENT = 100

# model = "gemini-1.5-pro-002"
# model = "qvq-max-latest"
model = "gemini-2.5-pro"
# encode_frame_cv(reshape_frame_to_512(img, i))

# Token 使用统计
token_usage_stats = {
    'total_prompt_tokens': 0,
    'total_completion_tokens': 0,
    'total_tokens': 0,
    'request_count': 0,
    'request_details': []
}

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=60))
async def get_chat_completion(message: dict, semaphore, retry_count=0) -> str:
    try:
        async with semaphore:  # 使用传入的信号量限制并发
            resp = {'id': message['request_id']}
            content = [{"type": "text", "text": message['prompt']}]
            #print(content)
            for i, img in enumerate(message['image_base64']):
                content.append({
	                        "type": "image_url",
	                        "image_url": {
	                            "url": f"data:image/jpeg;base64,{encode_frame_cv(reshape_frame_to_512(img, i))}"
	                        }
                        })
            # print(content)
            #import json
            #json.dump([{"role": "user","content": content}], open('message.json', 'w'))
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                    	"role": "user",
                    	"content": content
                    }
                ],
                #timeout=80
            )
            print(response)
            
            # 添加token使用统计
            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
                # 更新全局统计
                token_usage_stats['total_prompt_tokens'] += prompt_tokens
                token_usage_stats['total_completion_tokens'] += completion_tokens
                token_usage_stats['total_tokens'] += total_tokens
                token_usage_stats['request_count'] += 1
                
                # 记录详细信息
                request_detail = {
                    'request_id': message['request_id'],
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens,
                    'timestamp': asyncio.get_event_loop().time()
                }
                token_usage_stats['request_details'].append(request_detail)
                
                print(f"Token使用 - 请求ID: {message['request_id']}")
                print(f"  输入tokens: {prompt_tokens}")
                print(f"  输出tokens: {completion_tokens}")
                print(f"  总计tokens: {total_tokens}")
                print(f"  累计总tokens: {token_usage_stats['total_tokens']}")
            else:
                print("警告: 响应中未找到token使用信息")
            
            response_result = response.choices[0].message.content
            resp['response'] = response_result
            
            # 添加token信息到响应中
            if hasattr(response, 'usage') and response.usage:
                resp['token_usage'] = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            
            # message[model] = response_result
            return resp
    except Exception as e:
        print(f"Error in get_chat_completion for message  {type(e).__name__} - {str(e)}")
        raise


async def request_model(prompts):

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    async def wrapped_get_chat_completion(prompt):
        try:
            return await get_chat_completion(prompt, semaphore)
        except Exception as e:
            print(f"Task failed after all retries with error: {e}")
            return None

    tasks = [wrapped_get_chat_completion(prompt) for prompt in prompts]
    
    results = []
    for future in tqdm.as_completed(tasks, total=len(tasks), desc="Processing prompts"):
        result = await future
        results.append(result)
    
    # 显示本次批处理的token统计
    print_token_usage_summary()
    
    return results

def get_token_usage_summary():
    """获取token使用统计摘要"""
    return {
        'total_requests': token_usage_stats['request_count'],
        'total_prompt_tokens': token_usage_stats['total_prompt_tokens'],
        'total_completion_tokens': token_usage_stats['total_completion_tokens'],
        'total_tokens': token_usage_stats['total_tokens'],
        'average_tokens_per_request': token_usage_stats['total_tokens'] / max(1, token_usage_stats['request_count'])
    }

def print_token_usage_summary():
    """打印token使用统计摘要"""
    summary = get_token_usage_summary()
    print("\n" + "="*50)
    print("TOKEN 使用统计摘要")
    print("="*50)
    print(f"总请求次数: {summary['total_requests']}")
    print(f"总输入tokens: {summary['total_prompt_tokens']:,}")
    print(f"总输出tokens: {summary['total_completion_tokens']:,}")
    print(f"总计tokens: {summary['total_tokens']:,}")
    print(f"平均每请求tokens: {summary['average_tokens_per_request']:.2f}")
    print("="*50)

def save_token_usage_report(filename='token_usage_report.json'):
    """保存详细的token使用报告到文件"""
    import json
    from datetime import datetime
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'summary': get_token_usage_summary(),
        'detailed_usage': token_usage_stats['request_details']
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Token使用报告已保存到: {filename}")

def reset_token_usage_stats():
    """重置token使用统计"""
    global token_usage_stats
    token_usage_stats = {
        'total_prompt_tokens': 0,
        'total_completion_tokens': 0,
        'total_tokens': 0,
        'request_count': 0,
        'request_details': []
    }
    print("Token使用统计已重置")

def get_estimated_cost(model_name=None):
    """估算token使用成本（基于Gemini定价）"""
    if model_name is None:
        model_name = model
    
    # Gemini 2.5 Pro 定价（美元/1M tokens，仅供参考）
    pricing = {
        'gemini-2.5-pro': {'input': 1.25, 'output': 5.0},
        'gemini-1.5-pro': {'input': 1.25, 'output': 5.0},
        'gemini-1.5-pro-002': {'input': 1.25, 'output': 5.0}
    }
    
    if model_name not in pricing:
        print(f"未知模型 {model_name}，无法估算成本")
        return None
    
    input_cost = (token_usage_stats['total_prompt_tokens'] / 1_000_000) * pricing[model_name]['input']
    output_cost = (token_usage_stats['total_completion_tokens'] / 1_000_000) * pricing[model_name]['output']
    total_cost = input_cost + output_cost
    
    print(f"\n成本估算 (模型: {model_name}):")
    print(f"输入成本: ${input_cost:.4f}")
    print(f"输出成本: ${output_cost:.4f}")
    print(f"总成本: ${total_cost:.4f}")
    
    return {
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': total_cost,
        'currency': 'USD'
    }
