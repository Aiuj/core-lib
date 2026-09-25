"""Tests for worker heartbeats, the stall watchdog and job-system health checks."""

import fnmatch
import json
from datetime import datetime, timedelta, timezone

import pytest

from core_lib.jobs import (
    JobConfig,
    RedisJobQueue,
    WorkerHeartbeat,
    check_job_system_health,
    check_worker_health,
    ping_job_queue,
)

PREFIX = "test:jobs:"


class FakeRedis:
    """Just enough of redis-py for the liveness and stats code paths."""

    def __init__(self, fail_ping=False):
        self.store = {}
        self.lists = {}
        self.sets = {}
        self.ttls = {}
        self.fail_ping = fail_ping

    def ping(self):
        if self.fail_ping:
            raise ConnectionError("Error -2 connecting to valkey:6379")
        return True

    def set(self, key, value, ex=None):
        self.store[key] = value
        self.ttls[key] = ex

    def get(self, key):
        return self.store.get(key)

    def delete(self, key):
        self.store.pop(key, None)

    def scan_iter(self, match, count=None):
        return [k for k in list(self.store) if fnmatch.fnmatch(k, match)]

    def mget(self, keys):
        return [self.store.get(k) for k in keys]

    def llen(self, key):
        return len(self.lists.get(key, []))

    def lindex(self, key, index):
        items = self.lists.get(key, [])
        return items[index] if len(items) > index else None

    def scard(self, key):
        return len(self.sets.get(key, set()))

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def sadd(self, key, member):
        self.sets.setdefault(key, set()).add(member)

    def srem(self, key, member):
        members = self.sets.get(key, set())
        if member in members:
            members.discard(member)
            return 1
        return 0

    def ttl(self, key):
        return self.ttls.get(key) or -1

    def setex(self, key, ttl, value):
        self.set(key, value, ex=ttl)

    def lrem(self, key, count, value):
        self.lists[key] = [v for v in self.lists.get(key, []) if v != value]

    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)


def _queue(client=None):
    queue = RedisJobQueue(JobConfig(prefix=PREFIX))
    queue.client = client or FakeRedis()
    return queue


def _heartbeat(queue, state, stall_timeout=600.0):
    stalls = []
    hb = WorkerHeartbeat(
        queue,
        interval=10,
        liveness=lambda: state,
        stall_timeout=stall_timeout,
        on_stall=lambda: stalls.append(True),
    )
    return hb, stalls


def _heartbeat_key(hb):
    return f"{PREFIX}worker:heartbeat:{hb.worker_id}"


# --- WorkerHeartbeat ------------------------------------------------------


def test_heartbeat_published_with_ttl_while_loop_polls():
    queue = _queue()
    hb, stalls = _heartbeat(queue, {"last_poll_age": 0.5, "current_job_id": None})

    hb.tick()

    assert json.loads(queue.client.store[_heartbeat_key(hb)])["last_poll_age_s"] == 0.5
    assert queue.client.ttls[_heartbeat_key(hb)] == 30
    assert stalls == []


def test_heartbeat_removed_on_stop():
    queue = _queue()
    hb, _ = _heartbeat(queue, {"last_poll_age": 0.5, "current_job_id": None})
    hb.tick()

    hb.stop()

    assert queue.client.store == {}


def test_heartbeat_withheld_when_loop_stops_polling():
    queue = _queue()
    hb, stalls = _heartbeat(queue, {"last_poll_age": 120.0, "current_job_id": None})

    hb.tick()

    assert queue.client.store == {}
    assert stalls == []


def test_heartbeat_kept_while_long_job_runs():
    queue = _queue()
    hb, stalls = _heartbeat(queue, {"last_poll_age": 5000.0, "current_job_id": "job-1"})

    hb.tick()

    assert json.loads(queue.client.store[_heartbeat_key(hb)])["current_job_id"] == "job-1"
    assert stalls == []


def test_watchdog_fires_when_idle_loop_stalls():
    queue = _queue()
    hb, stalls = _heartbeat(queue, {"last_poll_age": 700.0, "current_job_id": None})

    hb.tick()

    assert stalls == [True]
    assert queue.client.store == {}


def test_watchdog_disabled_with_zero_timeout():
    queue = _queue()
    hb, stalls = _heartbeat(
        queue, {"last_poll_age": 7000.0, "current_job_id": None}, stall_timeout=0
    )
    hb.tick()
    assert stalls == []


def test_watchdog_measures_from_startup_before_first_poll():
    queue = _queue()
    hb, stalls = _heartbeat(queue, {"last_poll_age": None, "current_job_id": None})
    hb.tick()
    assert stalls == []
    assert queue.client.store  # just started: still counts as polling

    hb._started_monotonic -= 700
    hb.tick()
    assert stalls == [True]


def test_heartbeat_survives_redis_errors():
    queue = _queue(FakeRedis())
    queue.client.set = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down"))
    hb, _ = _heartbeat(queue, {"last_poll_age": 0.5, "current_job_id": None})
    hb.tick()  # must not raise


# --- Health checks --------------------------------------------------------


def test_worker_health_reports_workers_and_queue_depth():
    queue = _queue()
    _heartbeat(queue, {"last_poll_age": 0.5, "current_job_id": None})[0].tick()
    queue.client.lists[f"{PREFIX}queue:pending"] = ["job-9", "job-10"]
    queue.client.sets[f"{PREFIX}set:processing"] = {"job-8"}
    created = datetime.now(timezone.utc) - timedelta(hours=1)
    queue.client.store[f"{PREFIX}job:job-9"] = json.dumps(
        {
            "job_id": "job-9",
            "job_type": "x",
            "status": "pending",
            "created_at": created.isoformat(),
            "updated_at": created.isoformat(),
        }
    )

    result = check_worker_health(queue)

    assert result["healthy"] is True
    assert result["active_workers"] == 1
    assert result["pending_jobs"] == 2
    assert result["processing_jobs"] == 1
    assert 3590 < result["oldest_pending_age_s"] < 3610


def test_worker_health_unhealthy_without_heartbeat():
    result = check_worker_health(_queue())
    assert result["healthy"] is False
    assert result["error"] == "No background worker heartbeat found"


def test_worker_health_unhealthy_without_queue():
    assert check_worker_health(None)["healthy"] is False


def test_ping_job_queue_up_and_down():
    assert ping_job_queue(_queue())["healthy"] is True
    down = ping_job_queue(_queue(FakeRedis(fail_ping=True)))
    assert down["healthy"] is False
    assert "valkey" in down["error"]


def test_job_system_health_disabled():
    result = check_job_system_health(None, enabled=False)
    assert result["job_queue"] == {"healthy": True, "enabled": False}
    assert result["background_worker"] == {"healthy": True, "enabled": False}


def test_job_system_health_combines_both_components():
    result = check_job_system_health(_queue())
    assert result["job_queue"]["healthy"] is True
    assert result["background_worker"]["healthy"] is False


# --- 503 handler ----------------------------------------------------------


def test_queue_unavailable_handler_returns_503():
    import redis
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from core_lib.jobs import install_queue_unavailable_handler

    app = FastAPI()
    install_queue_unavailable_handler(app, detail={"code": "QUEUE_UNAVAILABLE"})

    @app.get("/job")
    def job():
        raise redis.exceptions.ConnectionError("Error -2 connecting to valkey:6379")

    response = TestClient(app).get("/job")

    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "QUEUE_UNAVAILABLE"}}
    assert response.headers["retry-after"] == "10"


# --- Lost-job recovery ----------------------------------------------------


def _processing_job(queue, job_id, idle_seconds, retry_count=0):
    updated = datetime.now(timezone.utc) - timedelta(seconds=idle_seconds)
    queue.client.store[f"{PREFIX}job:{job_id}"] = json.dumps(
        {
            "job_id": job_id,
            "job_type": "x",
            "status": "processing",
            "created_at": updated.isoformat(),
            "updated_at": updated.isoformat(),
            "metadata": {"retry_count": retry_count},
        }
    )
    queue.client.sadd(f"{PREFIX}set:processing", job_id)


def _status(queue, job_id):
    return json.loads(queue.client.store[f"{PREFIX}job:{job_id}"])["status"]


def test_stale_processing_job_is_requeued():
    queue = _queue()
    _processing_job(queue, "lost", idle_seconds=900)

    counts = queue.recover_stale_processing_jobs(stale_after=600, max_retries=3)

    assert counts == {"requeued": 1, "failed": 0}
    assert _status(queue, "lost") == "pending"
    assert queue.client.lists[f"{PREFIX}queue:pending"] == ["lost"]
    assert "lost" not in queue.client.sets[f"{PREFIX}set:processing"]


def test_stale_job_out_of_retries_is_failed():
    queue = _queue()
    _processing_job(queue, "lost", idle_seconds=900, retry_count=3)

    counts = queue.recover_stale_processing_jobs(stale_after=600, max_retries=3)

    assert counts == {"requeued": 0, "failed": 1}
    job = json.loads(queue.client.store[f"{PREFIX}job:lost"])
    assert job["status"] == "failed"
    assert "Worker lost while processing" in job["error"]


def test_recently_refreshed_job_is_left_alone():
    queue = _queue()
    _processing_job(queue, "running", idle_seconds=30)

    counts = queue.recover_stale_processing_jobs(stale_after=600, max_retries=3)

    assert counts == {"requeued": 0, "failed": 0}
    assert _status(queue, "running") == "processing"


def test_job_claimed_by_live_worker_is_left_alone():
    # e.g. per-job heartbeats could not be written during a Redis outage, but
    # the worker running it is alive and says so in its liveness heartbeat.
    queue = _queue()
    _processing_job(queue, "running", idle_seconds=900)
    _heartbeat(queue, {"last_poll_age": 5.0, "current_job_id": "running"})[0].tick()

    counts = queue.recover_stale_processing_jobs(stale_after=600, max_retries=3)

    assert counts == {"requeued": 0, "failed": 0}
    assert _status(queue, "running") == "processing"


def test_dangling_processing_entry_without_job_record_is_dropped():
    queue = _queue()
    queue.client.sadd(f"{PREFIX}set:processing", "expired")

    queue.recover_stale_processing_jobs(stale_after=600, max_retries=3)

    assert queue.client.sets[f"{PREFIX}set:processing"] == set()


def test_reclaim_hook_receives_job_and_outcome():
    queue = _queue()
    _processing_job(queue, "requeue-me", idle_seconds=900)
    _processing_job(queue, "fail-me", idle_seconds=900, retry_count=3)
    seen = {}

    queue.recover_stale_processing_jobs(
        stale_after=600,
        max_retries=3,
        on_reclaimed=lambda job, requeued, error: seen.update({job.job_id: (requeued, error)}),
    )

    assert seen["requeue-me"][0] is True
    assert seen["fail-me"][0] is False
    assert seen["fail-me"][1].endswith("(after 3 retries)")


def test_reclaim_hook_errors_do_not_stop_recovery():
    queue = _queue()
    _processing_job(queue, "a", idle_seconds=900)
    _processing_job(queue, "b", idle_seconds=900)

    def boom(job, requeued, error):
        raise RuntimeError("audit DB down")

    counts = queue.recover_stale_processing_jobs(600, 3, on_reclaimed=boom)

    assert counts == {"requeued": 2, "failed": 0}
