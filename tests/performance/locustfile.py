"""Locust performance testing configuration for the Churn Predictor API."""

import json
import random
from locust import HttpUser, task, between


class ChurnPredictorUser(HttpUser):
    """Simulated user for performance testing the Churn Predictor API."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize test data and authentication."""
        self.api_key = "test-api-key-for-performance-testing"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Sample customer data variations for realistic testing
        self.customer_profiles = [
            {
                "tenure": 12,
                "MonthlyCharges": 65.50,
                "TotalCharges": 786.00,
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes"
            },
            {
                "tenure": 36,
                "MonthlyCharges": 89.25,
                "TotalCharges": 3213.00,
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "Yes",
                "PhoneService": "Yes",
                "Contract": "Two year",
                "PaperlessBilling": "No"
            },
            {
                "tenure": 6,
                "MonthlyCharges": 45.20,
                "TotalCharges": 271.20,
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "PhoneService": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes"
            },
            {
                "tenure": 24,
                "MonthlyCharges": 112.50,
                "TotalCharges": 2700.00,
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "PhoneService": "Yes",
                "Contract": "One year",
                "PaperlessBilling": "No"
            }
        ]
    
    @task(4)
    def predict_single_customer(self):
        """Test single customer prediction endpoint (most common use case)."""
        customer_data = random.choice(self.customer_profiles).copy()
        
        # Add some randomization to make each request unique
        customer_data["tenure"] += random.randint(-5, 5)
        customer_data["MonthlyCharges"] += random.uniform(-10, 10)
        customer_data["TotalCharges"] += random.uniform(-100, 100)
        
        # Ensure values stay within reasonable bounds
        customer_data["tenure"] = max(1, min(72, customer_data["tenure"]))
        customer_data["MonthlyCharges"] = max(18.25, min(118.75, customer_data["MonthlyCharges"]))
        customer_data["TotalCharges"] = max(18.25, customer_data["TotalCharges"])
        
        payload = {"customer_data": customer_data}
        
        with self.client.post("/predict", 
                             json=payload, 
                             headers=self.headers,
                             name="Single Prediction") as response:
            if response.status_code == 200:
                result = response.json()
                # Validate response structure
                assert "prediction" in result
                assert "probability" in result
                assert result["prediction"] in [0, 1]
                assert 0.0 <= result["probability"] <= 1.0
    
    @task(2)
    def predict_batch_customers(self):
        """Test batch prediction endpoint."""
        batch_size = random.randint(2, 10)
        customers = []
        
        for _ in range(batch_size):
            customer_data = random.choice(self.customer_profiles).copy()
            
            # Add randomization
            customer_data["tenure"] += random.randint(-10, 10)
            customer_data["MonthlyCharges"] += random.uniform(-20, 20)
            customer_data["TotalCharges"] += random.uniform(-200, 200)
            
            # Ensure bounds
            customer_data["tenure"] = max(1, min(72, customer_data["tenure"]))
            customer_data["MonthlyCharges"] = max(18.25, min(118.75, customer_data["MonthlyCharges"]))
            customer_data["TotalCharges"] = max(18.25, customer_data["TotalCharges"])
            
            customers.append(customer_data)
        
        payload = {"customers": customers}
        
        with self.client.post("/predict/batch", 
                             json=payload, 
                             headers=self.headers,
                             name=f"Batch Prediction ({batch_size} customers)") as response:
            if response.status_code == 200:
                result = response.json()
                # Validate response structure
                assert "predictions" in result
                assert "probabilities" in result
                assert len(result["predictions"]) == batch_size
                assert len(result["probabilities"]) == batch_size
                assert all(pred in [0, 1] for pred in result["predictions"])
                assert all(0.0 <= prob <= 1.0 for prob in result["probabilities"])
    
    @task(1)
    def health_check(self):
        """Test health check endpoint."""
        with self.client.get("/health", name="Health Check") as response:
            if response.status_code == 200:
                result = response.json()
                assert "status" in result
                assert result["status"] == "healthy"
    
    @task(1)
    def api_info(self):
        """Test API info endpoint."""
        with self.client.get("/", name="API Info") as response:
            if response.status_code == 200:
                result = response.json()
                assert "name" in result
                assert "version" in result
    
    def test_invalid_request(self):
        """Test API error handling with invalid requests."""
        invalid_payload = {"invalid": "data"}
        
        with self.client.post("/predict", 
                             json=invalid_payload, 
                             headers=self.headers,
                             name="Invalid Request",
                             catch_response=True) as response:
            if response.status_code == 422:  # Validation error
                response.success()
            else:
                response.failure(f"Expected 422, got {response.status_code}")
    
    def test_unauthorized_request(self):
        """Test API authentication."""
        customer_data = random.choice(self.customer_profiles)
        payload = {"customer_data": customer_data}
        
        # Request without authentication
        with self.client.post("/predict", 
                             json=payload,
                             name="Unauthorized Request",
                             catch_response=True) as response:
            if response.status_code == 401:  # Unauthorized
                response.success()
            else:
                response.failure(f"Expected 401, got {response.status_code}")


class HighVolumeUser(HttpUser):
    """Heavy load user for stress testing."""
    
    wait_time = between(0.1, 0.5)  # Much faster requests
    weight = 2  # Higher weight for load testing
    
    def on_start(self):
        """Initialize for high-volume testing."""
        self.api_key = "test-api-key-for-performance-testing"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Pre-generate test data to reduce overhead
        self.test_customers = [
            {
                "tenure": random.randint(1, 72),
                "MonthlyCharges": round(random.uniform(18.25, 118.75), 2),
                "TotalCharges": round(random.uniform(18.25, 8500.0), 2),
                "gender": random.choice(["Male", "Female"]),
                "SeniorCitizen": random.choice([0, 1]),
                "Partner": random.choice(["Yes", "No"]),
                "Dependents": random.choice(["Yes", "No"]),
                "PhoneService": random.choice(["Yes", "No"]),
                "Contract": random.choice(["Month-to-month", "One year", "Two year"]),
                "PaperlessBilling": random.choice(["Yes", "No"])
            }
            for _ in range(100)  # Pre-generate 100 customers
        ]
    
    @task
    def rapid_predictions(self):
        """Make rapid prediction requests for stress testing."""
        customer_data = random.choice(self.test_customers)
        payload = {"customer_data": customer_data}
        
        with self.client.post("/predict", 
                             json=payload, 
                             headers=self.headers,
                             name="Rapid Prediction") as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")


# Configuration for different test scenarios
class WebsiteUser(ChurnPredictorUser):
    """Normal website user simulation."""
    weight = 3
    wait_time = between(2, 10)


class APIUser(ChurnPredictorUser):
    """API integration user simulation."""
    weight = 2
    wait_time = between(0.5, 2)


class BatchProcessingUser(HttpUser):
    """Large batch processing simulation."""
    weight = 1
    wait_time = between(10, 30)
    
    def on_start(self):
        """Initialize for batch processing."""
        self.api_key = "test-api-key-for-performance-testing"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    @task
    def large_batch_prediction(self):
        """Test large batch predictions."""
        batch_size = random.randint(50, 200)
        customers = []
        
        for _ in range(batch_size):
            customer = {
                "tenure": random.randint(1, 72),
                "MonthlyCharges": round(random.uniform(18.25, 118.75), 2),
                "TotalCharges": round(random.uniform(18.25, 8500.0), 2),
                "gender": random.choice(["Male", "Female"]),
                "SeniorCitizen": random.choice([0, 1]),
                "Partner": random.choice(["Yes", "No"]),
                "Dependents": random.choice(["Yes", "No"]),
                "PhoneService": random.choice(["Yes", "No"]),
                "Contract": random.choice(["Month-to-month", "One year", "Two year"]),
                "PaperlessBilling": random.choice(["Yes", "No"])
            }
            customers.append(customer)
        
        payload = {"customers": customers}
        
        with self.client.post("/predict/batch", 
                             json=payload, 
                             headers=self.headers,
                             name=f"Large Batch ({batch_size} customers)") as response:
            if response.status_code != 200:
                print(f"Batch Error: {response.status_code} - {response.text}")