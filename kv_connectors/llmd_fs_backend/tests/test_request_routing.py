from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from llmd_fs_backend.request_routing import RequestPolicyRoutes

pytestmark = pytest.mark.no_cuda_required


def routes():
    return RequestPolicyRoutes([30, 90, 2147483647], 90)


def test_explicit_and_default_requests_do_not_mutate_each_other():
    r = routes()
    short = r.start("short", 30)
    long = r.start("long", 90)
    default = r.start("default")
    assert short.namespace == ".llmd-ttl-v1-30" and long.namespace == ".llmd-ttl-v1-90"
    assert r.policy(default) == 90 and r.policy(short) == 30
    for x in [long, default, short]:
        r.finish(x)
    assert r.snapshot() == dict(requests=0, jobs=0, finished_waiting=0)


@pytest.mark.parametrize("hint", [True, False, 0, -1, 31, "30", 30.0, None, [], {}])
def test_invalid_hint_declines_storage_without_request_admission_error(hint):
    r = routes()
    x = r.start("invalid", hint)
    assert r.policy(x) is None and x.namespace is None
    with pytest.raises(RuntimeError):
        r.accept(x, "store", 1)
    r.finish(x)
    r.reset()


def test_finish_keeps_route_until_both_read_and_store_complete():
    r = routes()
    x = r.start("x", 30)
    r.accept(x, "load", 1)
    r.accept(x, "store", 1)
    r.finish(x)
    assert r.policy(x) == 30 and r.snapshot()["jobs"] == 2
    r.complete(x, "store", 1)
    assert r.snapshot()["requests"] == 1
    r.complete(x, "load", 1)
    assert r.snapshot()["requests"] == 0


def test_late_completion_cannot_hit_reused_request_id():
    r = routes()
    old = r.start("id", 30)
    r.accept(old, "load", 7)
    r.finish(old)
    with pytest.raises(RuntimeError):
        r.start("id", 90)
    r.complete(old, "load", 7)
    new = r.start("id", 90)
    r.accept(new, "load", 7)
    with pytest.raises(RuntimeError):
        r.complete(old, "load", 7)
    assert r.policy(new) == 90 and r.snapshot()["jobs"] == 1
    r.complete(new, "load", 7)
    r.finish(new)


def test_forged_policy_cannot_rebind_current_generation():
    r = routes()
    x = r.start("x", 30)
    with pytest.raises(RuntimeError):
        r.policy(replace(x, seconds=90))
    assert r.policy(x) == 30
    r.finish(x)


def test_no_new_io_after_finish_even_when_old_io_still_pending():
    r = routes()
    x = r.start("x")
    r.accept(x, "load", 1)
    r.finish(x)
    with pytest.raises(RuntimeError):
        r.accept(x, "store", 2)
    r.complete(x, "load", 1)


def test_duplicate_and_unmatched_callbacks_do_not_release_other_jobs():
    r = routes()
    x = r.start("x")
    r.accept(x, "load", 1)
    r.accept(x, "store", 2)
    with pytest.raises(RuntimeError):
        r.accept(x, "load", 1)
    with pytest.raises(RuntimeError):
        r.complete(x, "load", 2)
    r.complete(x, "load", 1)
    with pytest.raises(RuntimeError):
        r.complete(x, "load", 1)
    assert r.snapshot()["jobs"] == 1
    r.complete(x, "store", 2)
    r.finish(x)


def test_reset_requires_unfinished_requests_and_pending_io_to_drain():
    r = routes()
    x = r.start("x")
    r.accept(x, "load", 1)
    with pytest.raises(RuntimeError):
        r.reset()
    r.finish(x)
    with pytest.raises(RuntimeError):
        r.reset()
    r.complete(x, "load", 1)
    r.reset()
    r.finish(x)


def test_concurrent_interleaving_retains_each_request_policy():
    r = routes()

    def run(i):
        x = r.start(str(i), 30 if i % 2 else 90)
        r.accept(x, "load", i)
        r.accept(x, "store", i)
        r.finish(x)
        assert r.policy(x) == (30 if i % 2 else 90)
        r.complete(x, "store", i)
        r.complete(x, "load", i)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(run, range(128)))
    assert r.snapshot() == dict(requests=0, jobs=0, finished_waiting=0)


@pytest.mark.parametrize(
    "allowed,default",
    [([], 30), ([True], True), ([30], 90), ([0, 30], 30), ([2147483648], 2147483648)],
)
def test_unsupported_configuration_rejected_before_requests(allowed, default):
    with pytest.raises(ValueError):
        RequestPolicyRoutes(allowed, default)
