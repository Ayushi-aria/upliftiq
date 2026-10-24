import random

from locust import HttpUser, between, task

HEADERS = {"X-API-Key": "dev-key"}


def random_user():
    return {
        "tenure": round(random.uniform(0, 10), 3),
        "activity": round(random.uniform(-3, 3), 3),
        "risk_score": round(random.uniform(-3, 3), 3),
        "age": random.randint(18, 70),
        "balance": round(random.uniform(0, 20), 3),
    }


class ScoringUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def score_new_user(self):
        # har baar naya input, to cache miss hota hai (asli model latency)
        self.client.post("/score", json=random_user(), headers=HEADERS)

    @task(2)
    def score_repeat_user(self):
        # fixed input, to cache hit hota hai
        self.client.post("/score", name="/score (cached)", headers=HEADERS, json={
            "tenure": 2.0, "activity": 1.2, "risk_score": 0.3,
            "age": 35, "balance": 1.5})

    @task(1)
    def optimize(self):
        users = [{**random_user(), "user_id": i} for i in range(200)]
        self.client.post("/optimize", headers=HEADERS, json={
            "budget": 5000, "cost_per_user": 50, "users": users})