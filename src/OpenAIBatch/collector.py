import os
import json


class Responses:

    def __init__(self):
        self.cached_requests = []

    def create(self, custom_id: str, model: str, **kwargs) -> None:
        self.cached_requests.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": {"model": model, **kwargs}
        })

    def _reset_cache(self):
        self.cached_requests = []


class RequestCollector:
    def __init__(self):
        self.responses = Responses()

    def collect_requests_to_file(self, file_path: str, reset_caches: bool = False) -> None:
        os.makedirs(file_path, exist_ok=True)
        with open(file_path, 'w') as f:
            for request in self.responses.cached_requests:
                f.write(json.dumps(request) + '\n')

        if reset_caches:
            self.reset_collected_requests()

    def reset_collected_requests(self) -> None:
        self.responses = Responses()
