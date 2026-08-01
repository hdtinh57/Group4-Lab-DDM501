"""Concurrent load generator for the monitored API."""
import argparse
import concurrent.futures
import random
import statistics
import time
import requests


def make_prediction(url: str) -> tuple[bool, float]:
    started = time.perf_counter()
    try:
        response = requests.post(f"{url}/predict", json={"user_id": str(random.randint(1, 943)), "movie_id": str(random.randint(1, 1682))}, timeout=5)
        return response.ok, (time.perf_counter() - started) * 1000
    except requests.RequestException:
        return False, (time.perf_counter() - started) * 1000


def run_load_test(url: str, duration: int, workers: int) -> dict:
    if not requests.get(f"{url}/health", timeout=5).ok:
        raise RuntimeError("API health check failed")
    started, results = time.perf_counter(), []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while time.perf_counter() - started < duration:
            results.extend(f.result() for f in [executor.submit(make_prediction, url) for _ in range(workers)])
    latencies = [latency for _, latency in results]
    summary = {"requests": len(results), "success": sum(ok for ok, _ in results), "rps": len(results) / duration, "p95_ms": sorted(latencies)[max(0, int(len(latencies) * .95) - 1)] if latencies else 0, "mean_ms": statistics.mean(latencies) if latencies else 0}
    print(summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8001")
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    run_load_test(args.url, args.duration, args.workers)
