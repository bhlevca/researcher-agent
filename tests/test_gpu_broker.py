"""The queue in front of the one GPU, and the model residency it owns.

The behaviour that matters: work is serialised, order is kept, the GPU is only
swapped when it actually has to be, and nothing — an exception, a cancellation,
a broken progress callback — can leave the queue wedged.
"""

import asyncio

import pytest

from researcher.gpu_broker import GpuBroker, Need

OLLAMA_A = Need("ollama", "gpt-oss:20b")
OLLAMA_B = Need("ollama", "qwen3.5:9b")
TORCH_IMG = Need("torch", "zimage")


@pytest.fixture
def broker(monkeypatch):
    """A broker whose Ollama teardown is recorded rather than performed."""
    b = GpuBroker()
    calls = []
    monkeypatch.setattr("researcher.gpu_broker._release_ollama",
                        lambda tag: calls.append(tag))
    b.released = calls
    return b


async def _run(b, need, label, seconds=0.01, seen=None, user=""):
    async with b.reserve(need, label, user=user) as res:
        async for notice in res.wait():
            if seen is not None:
                seen.append((label, notice["position"]))
        await asyncio.sleep(seconds)
    return label


async def _drain(res):
    async for _ in res.wait():
        pass


class TestSerialisation:
    async def test_only_one_task_at_a_time(self, broker):
        live = peak = 0

        async def work():
            nonlocal live, peak
            async with broker.reserve(OLLAMA_A, "w") as r:
                await _drain(r)
                live += 1
                peak = max(peak, live)
                await asyncio.sleep(0.03)
                live -= 1

        await asyncio.gather(*(work() for _ in range(5)))
        assert peak == 1

    async def test_first_come_first_served(self, broker):
        done = []

        async def work(name):
            async with broker.reserve(OLLAMA_A, name) as r:
                await _drain(r)
                await asyncio.sleep(0.01)
                done.append(name)

        tasks = []
        for n in "abcd":
            tasks.append(asyncio.create_task(work(n)))
            await asyncio.sleep(0.005)
        await asyncio.gather(*tasks)
        assert done == list("abcd")


class TestResidency:
    """The point of the broker: swap only when it is actually needed."""

    async def test_the_same_model_twice_is_not_reloaded(self, broker):
        await _run(broker, OLLAMA_A, "one")
        await _run(broker, OLLAMA_A, "two")
        assert broker.released == [], "reloaded a model that was already there"
        assert broker.resident == OLLAMA_A

    async def test_ollama_to_ollama_needs_no_teardown(self, broker):
        """Ollama evicts the previous model itself."""
        await _run(broker, OLLAMA_A, "a")
        await _run(broker, OLLAMA_B, "b")
        assert broker.released == []
        assert broker.resident == OLLAMA_B

    async def test_ollama_to_torch_releases_ollama(self, broker):
        """The one teardown that is genuinely required: Ollama will not yield
        VRAM to a process it does not know about."""
        await _run(broker, OLLAMA_A, "chat")
        await _run(broker, TORCH_IMG, "image")
        assert broker.released == ["image"]
        assert broker.resident == TORCH_IMG

    async def test_torch_to_ollama_needs_no_teardown(self, broker):
        """Measured: a diffusers pipeline holds ~32 MiB once it is done,
        because CPU offload keeps the weights in system RAM."""
        await _run(broker, TORCH_IMG, "image")
        broker.released.clear()
        await _run(broker, OLLAMA_A, "chat")
        assert broker.released == []

    async def test_nothing_is_restored_after_a_task(self, broker):
        """The old behaviour reloaded the evicted model whether or not anyone
        wanted it — about ten gigabytes of pointless work."""
        await _run(broker, OLLAMA_A, "chat")
        await _run(broker, TORCH_IMG, "image")
        assert broker.resident == TORCH_IMG, "something was restored"

    async def test_an_image_after_an_image_does_not_reinstate_an_llm(self, broker):
        await _run(broker, OLLAMA_A, "chat")
        await _run(broker, TORCH_IMG, "image1")
        broker.released.clear()
        await _run(broker, TORCH_IMG, "image2")
        assert broker.released == [], "released Ollama when none was loaded"

    async def test_residency_can_be_invalidated(self, broker):
        """A manual unload or a model switch happens outside the broker."""
        await _run(broker, OLLAMA_A, "chat")
        broker.forget_residency()
        assert broker.resident is None
        await _run(broker, TORCH_IMG, "image")
        assert broker.released == ["image"]


class TestPriority:
    async def test_a_waiting_task_can_jump_with_a_reason(self, broker):
        done = []

        async def work(name):
            async with broker.reserve(OLLAMA_A, name) as r:
                await _drain(r)
                await asyncio.sleep(0.02)
                done.append(name)

        holder = asyncio.create_task(work("running"))
        await asyncio.sleep(0.01)
        first = asyncio.create_task(work("ordinary"))
        await asyncio.sleep(0.005)
        urgent = asyncio.create_task(work("urgent"))
        await asyncio.sleep(0.005)

        ticket = next(e.ticket for e in broker._line if e.label == "urgent")
        assert broker.request_priority(ticket, "homework due tomorrow")
        await asyncio.gather(holder, first, urgent)
        assert done == ["running", "urgent", "ordinary"]

    async def test_the_running_task_is_never_interrupted(self, broker):
        async def work(name, seconds):
            async with broker.reserve(OLLAMA_A, name) as r:
                await _drain(r)
                await asyncio.sleep(seconds)

        holder = asyncio.create_task(work("running", 0.08))
        await asyncio.sleep(0.02)
        running_ticket = broker._line[0].ticket
        assert not broker.request_priority(running_ticket, "mine")
        await holder

    async def test_the_reason_is_visible_to_everyone_waiting(self, broker):
        async def work(name):
            async with broker.reserve(OLLAMA_A, name, user=name) as r:
                await _drain(r)
                await asyncio.sleep(0.05)

        holder = asyncio.create_task(work("ana"))
        await asyncio.sleep(0.01)
        waiter = asyncio.create_task(work("bob"))
        await asyncio.sleep(0.01)
        ticket = next(e.ticket for e in broker._line if e.user == "bob")
        broker.request_priority(ticket, "school in the morning")

        snap = broker.snapshot()
        entry = next(w for w in snap["waiting"] if w["user"] == "bob")
        assert entry["note"] == "school in the morning"
        assert entry["priority"] is True
        await asyncio.gather(holder, waiter)

    async def test_an_unknown_ticket_is_refused(self, broker):
        assert not broker.request_priority(9999, "please")


class TestRobustness:
    async def test_an_exception_frees_the_gpu(self, broker):
        async def boom():
            async with broker.reserve(OLLAMA_A, "boom") as r:
                await _drain(r)
                raise ValueError("work failed")

        with pytest.raises(ValueError):
            await boom()
        await asyncio.wait_for(_run(broker, OLLAMA_A, "after"), timeout=1)

    async def test_cancellation_frees_the_gpu(self, broker):
        async def forever():
            async with broker.reserve(OLLAMA_A, "forever") as r:
                await _drain(r)
                await asyncio.sleep(60)

        t = asyncio.create_task(forever())
        await asyncio.sleep(0.03)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        await asyncio.wait_for(_run(broker, OLLAMA_A, "next"), timeout=1)

    async def test_a_cancelled_waiter_leaves_the_queue(self, broker):
        holder = asyncio.create_task(_run(broker, OLLAMA_A, "holder", 0.15))
        await asyncio.sleep(0.02)
        waiter = asyncio.create_task(_run(broker, OLLAMA_A, "waiter"))
        await asyncio.sleep(0.02)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await holder
        assert broker.snapshot()["waiting"] == []

    async def test_a_broken_progress_callback_does_not_break_the_work(self, broker):
        def explode(pos, est, ticket):
            raise RuntimeError("callback broken")

        async def work(name):
            async with broker.reserve(OLLAMA_A, name) as r:
                await _drain(r)
                await asyncio.sleep(0.02)
                return name

        a = asyncio.create_task(work("a"))
        await asyncio.sleep(0.005)
        b = asyncio.create_task(work("b"))
        assert await asyncio.gather(a, b) == ["a", "b"]


class TestReporting:
    async def test_a_waiter_learns_its_position_and_then_its_turn(self, broker):
        seen = []
        holder = asyncio.create_task(_run(broker, OLLAMA_A, "holder", 0.12))
        await asyncio.sleep(0.02)
        waiter = asyncio.create_task(_run(broker, OLLAMA_A, "waiter", 0.01, seen))
        await asyncio.gather(holder, waiter)
        positions = [p for _, p in seen]
        assert positions and positions[0] >= 1 and positions[-1] == 0

    async def test_the_first_caller_is_told_nothing(self, broker):
        seen = []
        await _run(broker, OLLAMA_A, "only", 0.01, seen)
        assert seen == []

    async def test_no_estimate_until_there_is_history(self, broker):
        """A made-up number is worse than none."""
        holder = asyncio.create_task(_run(broker, OLLAMA_A, "work", 0.05))
        await asyncio.sleep(0.01)
        waiter_entry = None

        async def waiting():
            nonlocal waiter_entry
            async with broker.reserve(OLLAMA_A, "work") as r:
                await _drain(r)
                pass

        t = asyncio.create_task(waiting())
        await asyncio.sleep(0.01)
        ticket = broker._line[-1].ticket
        assert broker.estimate_wait(ticket) is None
        await asyncio.gather(holder, t)

    async def test_snapshot_names_the_resident_model(self, broker):
        await _run(broker, OLLAMA_A, "chat")
        assert broker.snapshot()["resident"] == "ollama:gpt-oss:20b"


class TestEndpoints:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock
        from fastapi.testclient import TestClient

        db = tmp_path / "gpu.db"
        monkeypatch.setattr("researcher.config.DB_PATH", db)
        import researcher.main as main
        monkeypatch.setattr(main, "DB_PATH", db)
        mock = MagicMock()
        # One construction point now: user_crews._crew_class imports the class
        # from its own module when a crew is actually needed.
        monkeypatch.setattr("researcher.crew.ResearchCrew", mock)
        monkeypatch.setattr("researcher.routes.models.ResearchCrew", mock)
        with TestClient(main.app) as c:
            yield c

    @pytest.fixture(autouse=True)
    def _clean_line(self):
        """These tests touch the process-wide broker, so leave it empty.

        A stale entry would make the next test see a busy GPU and fail for a
        reason that has nothing to do with what it is checking.
        """
        from researcher.gpu_broker import GPU
        yield
        GPU._line.clear()
        GPU.forget_residency()

    def test_queue_is_readable_without_logging_in(self, client):
        """On a household instance the queue is a shared fact."""
        r = client.get("/gpu/queue")
        assert r.status_code == 200
        body = r.json()
        assert body["busy"] is False
        assert body["waiting"] == []

    def test_priority_without_a_reason_is_refused(self, client):
        """The note is the only thing keeping self-declared priority honest."""
        r = client.post("/gpu/queue/priority", json={"ticket": 1, "note": "  "})
        assert r.status_code == 400
        assert "why" in r.json()["detail"].lower()

    def test_priority_for_a_task_that_is_not_waiting_is_refused(self, client):
        r = client.post("/gpu/queue/priority",
                        json={"ticket": 9999, "note": "homework"})
        assert r.status_code == 404

    def _seed_user(self, client, uid, username):
        """Insert a user through the app's own loop."""
        async def go():
            db = client.app.state.db
            await db.execute(
                "INSERT INTO users (id, username, pw_hash, model, created_at) "
                "VALUES (?, ?, ?, ?, ?)", (uid, username, "x", "", "2026-01-01"))
            await db.commit()
        client.portal.call(go)

    def _queue_entry(self, uid, label="tutor", note="", started=True):
        """Put a row in the broker's line without running anything.

        The endpoint's job here is the id-to-name join, not the queueing, so
        driving the queue for real would only add a GPU to the test.
        """
        from researcher.gpu_broker import GPU, Need, Entry
        import time as _t
        e = Entry(ticket=next(GPU._tickets), need=Need("ollama", "m"),
                  label=label, user=uid, note=note)
        if started:
            e.started_at = _t.monotonic()
        GPU._line.append(e)
        return e

    def test_the_queue_names_people_not_user_ids(self, client):
        """"30d5da42a474 is ahead of you" explains nothing.

        The broker stores ids because it has no business knowing about the user
        table; the route does the join. In a household the whole value of a
        shared queue is being able to see that it is your brother and go and
        ask him, which needs his name.
        """
        self._seed_user(client, "uid-ana", "ana")
        self._seed_user(client, "uid-bob", "bob")
        self._queue_entry("uid-ana", started=True)
        self._queue_entry("uid-bob", note="homework due", started=False)

        body = client.get("/gpu/queue").json()
        assert body["running"]["user"] == "ana"
        assert body["waiting"][0]["user"] == "bob"
        # The reason travels with the entry: that is the messaging board.
        assert body["waiting"][0]["note"] == "homework due"

    def test_an_unknown_id_degrades_to_someone(self, client):
        """A user deleted mid-queue must not blank the board or 500 it."""
        self._queue_entry("ghost", started=True)
        body = client.get("/gpu/queue").json()
        assert body["running"]["user"] == "ghost"   # no name, but still shown
        assert body["busy"] is True


class TestWiring:
    """The call sites that actually contend for the card."""

    def test_the_tutor_stream_declares_what_it_needs(self):
        from pathlib import Path

        src = Path("src/researcher/routes/tutor.py").read_text()
        assert "GPU.reserve(" in src and 'Need("ollama"' in src

    def test_media_generation_declares_torch(self):
        from pathlib import Path

        src = Path("src/researcher/routes/chat.py").read_text()
        assert "GPU.reserve(" in src and 'Need("torch"' in src

    def test_the_old_context_manager_no_longer_restores_under_the_broker(self):
        """The reload was ~10.5 GiB of a model nobody had asked for."""
        from pathlib import Path

        src = Path("src/researcher/image.py").read_text()
        assert "_brokered" in src
        assert "_reload_list = [] if _brokered" in src

    def test_unloading_a_model_invalidates_residency(self):
        from pathlib import Path

        src = Path("src/researcher/routes/models.py").read_text()
        assert "GPU.forget_residency()" in src


class TestNoticesArriveDuringTheWait:
    """The flaw that made the whole feature invisible.

    The first design blocked inside the context manager, so a streaming
    endpoint was suspended and could not yield anything until its turn came.
    The browser got silence and then a burst of stale "waiting" messages —
    exactly the hang the queue existed to explain. These tests assert the
    notices arrive *while* waiting.
    """

    async def test_a_waiter_is_told_before_its_turn_comes(self, broker):
        import time as _t

        told_at: list[float] = []
        started_at: list[float] = []

        async def holder():
            async with broker.reserve(OLLAMA_A, "holder") as r:
                await _drain(r)
                await asyncio.sleep(0.4)

        async def waiter():
            async with broker.reserve(OLLAMA_A, "waiter") as r:
                async for notice in r.wait():
                    if notice["position"]:
                        told_at.append(_t.monotonic())
                started_at.append(_t.monotonic())

        h = asyncio.create_task(holder())
        await asyncio.sleep(0.05)
        w = asyncio.create_task(waiter())
        await asyncio.gather(h, w)

        assert told_at, "the waiter was never told it was queued"
        # The notice must land well before the turn, not with it.
        assert started_at[0] - told_at[0] > 0.2, (
            "the notice arrived at the same moment as the turn — it was "
            "buffered until the wait was over")

    async def test_the_last_notice_means_the_model_is_resident(self, broker):
        async def holder():
            async with broker.reserve(OLLAMA_A, "holder") as r:
                await _drain(r)
                await asyncio.sleep(0.15)

        h = asyncio.create_task(holder())
        await asyncio.sleep(0.03)
        async with broker.reserve(TORCH_IMG, "image") as r:
            last = None
            async for notice in r.wait():
                last = notice
            assert last is not None and last["position"] == 0
            # By the final notice the swap has happened.
            assert broker.resident == TORCH_IMG
        await h

    async def test_going_straight_through_still_loads_the_model(self, broker):
        """No queue, no notices — but residency must still be arranged."""
        async with broker.reserve(TORCH_IMG, "image") as r:
            seen = [n async for n in r.wait()]
            assert seen == []
            assert broker.resident == TORCH_IMG
