import requests

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer token"
}

data = {
  "model": "merged_model",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Extract the nutrition content from the image."},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]
    }
  ]
}

response = requests.post("http://localhost:8000/v1/chat/completions", headers=headers, json=data)
print(response.json())
