import requests

url = "http://localhost:8000/api/v3/auto-submit/"
data = {
    "candidate_id": "candidate_20260314_164333_976875",
    "job_url": "https://www.linkedin.com/jobs/view/generative-ai-engineer-internship-in-bangalore-at-aiqwip-4373874596/",
    "job_title": "Generative AI Engineer Internship",
    "credentials": {
        "email": "kshitijdalvi22@gmail.com",
        "password": "Dr@arninzola4"
    }
}

try:
    response = requests.post(url, json=data, timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
