bind = "0.0.0.0:8000"
workers = 1  # Only 1 worker to reduce memory usage
timeout = 120  # Increased timeout
keepalive = 5
worker_class = "sync"