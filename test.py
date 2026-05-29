from openai import OpenAI

# Use environment variable for the API key to avoid embedding secrets in source.
client = OpenAI(
    api_key='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',
    base_url='https://openrouter.ai/api/v1'
)


def ask_llm(prompt):

    response = client.chat.completions.create(
        model='deepseek/deepseek-chat',
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )

    return response.choices[0].message.content


#response = ask_llm('Say hello')
#print(response)