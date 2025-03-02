from openai import OpenAI

client = OpenAI(api_key="your-api-key")

def generate_description(species):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a wildlife expert."},
            {"role": "user", "content": f"Describe the {species} in 50 words."}
        ]
    )
    return response.choices[0].message.content