from ollama import generate
from ollama import Client
from constant import *

ollama_client = Client(
    host="http://127.0.0.1:11434",
    trust_env=False,
)


class LocalClient:
    def text_generation(self, prompt, max_new_tokens=100, stop=None):
        options = {"num_predict": max_new_tokens}
        if stop is not None:
            options["stop"] = stop

        result = ollama_client.generate(
            model="llama3.2:3b",
            prompt=prompt,
            raw=True,  # 不自动添加聊天模板，由你自己控制提示词
            stream=False,
            options=options,
        )
        return result.response


client = LocalClient()

# 1.模型生成到Observation前
output = client.text_generation(
    prompt,
    max_new_tokens=200,
    stop=["Observation:", "Observations:"],
)
print("1.*********")
print(output)
print("*********")


# 2. Python 执行模拟工具
def get_weather(location):
    return f"{location} 天气晴朗，气温较低（模拟数据）。"


# 3. 拼接后再次生成
new_prompt = (
        prompt
        + output.rstrip()
        + "\nObservation: "
        + get_weather("London")
        + "\n"
)

print("\n第二次发送给模型的完整提示词：\n", new_prompt)

final_output = client.text_generation(new_prompt, max_new_tokens=200)
print("\n最终输出：\n", final_output)
