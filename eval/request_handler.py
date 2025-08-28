from openai import OpenAI

def init_components(model_name, api_key=None):
    if api_key:
        client = OpenAI(api_key=api_key)
        return {"client": client}
    else:
        raise Exception(f"api_key is required for model {model_name}.")


# ========== 5. 请求 模型 ==========
def model(model_name, messages,components, temperature=0):
    if components is None:
        raise Exception(f"components is required for {model_name}.")
    client = components["client"]
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature
        )
        # print(response)
    except Exception as e:
        print(e)
        return None
    raw_output = response.choices[0].message.content
    return raw_output
