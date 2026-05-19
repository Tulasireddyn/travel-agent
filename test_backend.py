import requests
import json

def test_agent(path):
    url = f"http://localhost:8000{path}"
    payload = {
        "preferences": {
            "origin": "Mumbai",
            "destination": "Paris",
            "month": "May",
            "duration": 7,
            "num_people": "2",
            "holiday_type": "Romantic",
            "budget_type": "Luxury"
        },
        "itinerary": "Day 1: Arrive in Paris..."
    }
    print(f"Testing {path}...")
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_agent("/agent/activities")
    test_agent("/agent/weather")
