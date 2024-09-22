import os
import google.generativeai as genai


os.environ['GOOGLE_API_KEY'] = 'api key'
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

model = genai.GenerativeModel('gemini-pro')

system_prompt = "You are an expert botanist specializing in plant infections. Your role is to educate users about various plant diseases, their symptoms, causes, and treatments. When asked about a specific plant infection, provide detailed, accurate information in a clear and concise manner. If you're not sure about something, say so rather than providing incorrect information."

def get_gemini_response(prompt):
    messages = [{"role": "user", "content": system_prompt},
                {"role": "user", "content": prompt}
                ]
    response = model.generate_content(prompt)
    return response.text

# Main chat loop
print("Welcome to the Gemini Chatbot! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    
    response = get_gemini_response(user_input)
    print("Bot:", response)

print("Thank you for chatting!")