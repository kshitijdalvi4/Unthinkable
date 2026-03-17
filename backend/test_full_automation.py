import requests
import time
import json

BASE_URL = "http://localhost:8000"
CANDIDATE_ID = "candidate_20260314_164333_976875"
JOB_URL = "https://www.linkedin.com/jobs/view/4380405207/"
CREDS = {
    "email": "kshitijdalvi22@gmail.com",
    "password": "Dr@arninzola4"
}

def run_test():
    print(f"Triggering auto-submit for {JOB_URL}...")
    payload = {
        "candidate_id": CANDIDATE_ID,
        "job_url": JOB_URL,
        "job_title": "ML Intern - Geoscience",
        "credentials": CREDS
    }
    
    try:
        resp = requests.post(f"{BASE_URL}/api/v3/auto-submit/", json=payload)
        print(f"Submission Response: {resp.status_code}")
        print(resp.json())
        
        if resp.status_code == 200:
            print("\nPolling status...")
            for _ in range(30): # Poll for 5 minutes (10s intervals)
                status_resp = requests.get(f"{BASE_URL}/api/v3/automation-status/{CANDIDATE_ID}")
                status_data = status_resp.json()
                print(f"Current Status: {status_data['status']} | {status_data['message']}")
                
                if status_data['status'] in ['completed', 'error']:
                    print("\n--- Final Result ---")
                    print(json.dumps(status_data, indent=2))
                    break
                
                time.sleep(10)
        else:
            print("Failed to start automation.")
            
    except Exception as e:
        print(f"Error calling API: {e}")

if __name__ == "__main__":
    run_test()
