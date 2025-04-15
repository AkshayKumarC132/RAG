# rag/llm.py

# import openai
from openai import OpenAI
from project import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

def generate_answer_with_context(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    answer = response.choices[0].message.content.strip()

    # Check if the response indicates missing information
    if "The document does not provide information" in answer:
        # Create a dynamic message based on the question
        return f"Sorry, I couldn't find any information on the topic: '{question}' in the document."

    return answer