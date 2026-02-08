import openai

from .config import get_settings

settings = get_settings()
openai.api_key = settings.openai_api_key

async def generate_ai_review(pr_title, pr_body, files):
    text = f"PR Title: {pr_title}\nPR Body: {pr_body}\n"
    for f in files[:5]:
        if f.get("patch"):
            text += f"\nFile: {f['filename']}\n{f['patch'][:2000]}\n"
    response = await openai.ChatCompletion.acreate(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert code reviewer."},
            {"role": "user", "content": text},
        ]
    )
    return response.choices[0].message["content"]
