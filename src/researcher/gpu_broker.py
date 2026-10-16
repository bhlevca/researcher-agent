"""One GPU, several people: a queue that owns which model is resident.

The card holds one model at a time, so the question is never "can these run
together" but "whose turn is it, and what has to be swapped in". Today each
call site answers that for itself, and answers it wrongly for more than one
user: ``_gpu_inference_ctx`` unloads Ollama to make room for an image, then
**reloads the model it unloaded** when the image is done. Measured, that costs
about ten and a half gigabytes and several seconds restoring a model the next
person may not want.

So residency belongs to the queue rather than to the call sites. A task says
what it needs — ``Need("ollama", "gpt-oss:20b")``, ``Need("torch", "zimage")``
— and when its turn comes the broker makes that true, runs it, and stands
aside. There is no restore step: the next task declares its own need.

What that buys, measured on this machine:

- A run of tasks wanting the **same** model pays the ~4-6s load once instead of
  once each. For one person working through a lesson, that is every request
  after the first.
- Nothing is reloaded speculatively. An image followed by another image no
  longer reinstates an LLM in between.

The transitions, from measurement rather than assumption:

- *Ollama to Ollama* — nothing to do. Ollama evicts the old model itself.
- *Ollama to torch* — Ollama must be told to let go (``keep_alive=0``), because
  it will not yield VRAM to a process it does not know about.
- *torch to Ollama* — nothing to do. The diffusers pipelines use CPU offload,
  so after a generation they hold 32 MiB, not gigabytes.
- *the same thing twice* — nothing to do, which is the point.

Order is first come, first served. A task that runs for minutes holds its slot
for minutes, because it is genuinely using the card, and a second request from
the same person joins the back of the queue like anyone else. Someone who needs
to jump the line can say so, with a reason the others can read.
"""

from __future__ import annotations

import time
import asyncio
import logging
import itertools
import contextlib
import contextvars
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# True inside a task that already holds the GPU, so nested reservations pass
# through instead of queueing behind themselves.
_holding: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "gpu_holding", default=False)

# How long to wait for VRAM to come free before going ahead anyway. Going ahead
# is usually right: the allocation may be another process we do not control,
# and failing to start is worse than starting under pressure.
_VRAM_WAIT_SECONDS = 20.0
_VRAM_WANTED_GIB = 4.0


@dataclass(frozen=True)
class Need:
    """What a task needs resident on the GPU.

    *kind* is "ollama", "torch" or "song".

    "torch" covers the in-process diffusers pipelines — image and video —
    which share one allocator and one set of module-level caches. "song" is
    ACE-Step, which runs in its own process because its torch and transformers
    pins are incompatible with ours, and which therefore has to be stopped
    rather than offloaded when the card is wanted back.
    """

    kind: str
    name: str

    def __str__(self) -> str:
        return f"{self.kind}:{self.name}"


@dataclass
class Entry:
    """A task holding or waiting for the GPU."""

    ticket: int
    need: Need
    label: str
    user: str = ""
    note: str = ""              # why this is urgent, if the owner said
    priority: bool = False
    queued_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None

    @property
    def waited(self) -> float:
        return (self.started_at or time.monotonic()) - self.queued_at

    @property
    def running_for(self) -> float:
        return 0.0 if self.started_at is None else time.monotonic() - self.started_at


class Reservation:
    """A place in the queue, and the GPU once it is this caller's turn."""

    def __init__(self, broker: "GpuBroker", entry: Entry):
        self._broker = broker
        self.entry = entry

    @property
    def ticket(self) -> int:
        return self.entry.ticket

    async def wait(self):
        """Async generator: yields position notices, ends when it is our turn.

        On the last notice the model is resident and the caller may proceed.
        """
        async for notice in self._broker._await_turn(self.entry):
            if notice["position"] == 0:
                await self._broker._make_resident(self.entry.need,
                                                  self.entry.label)
                self.entry.started_at = time.monotonic()
            yield notice
        if self.entry.started_at is None:
            # Went straight through without ever queueing.
            await self._broker._make_resident(self.entry.need,
                                              self.entry.label)
            self.entry.started_at = time.monotonic()


class _Passthrough:
    """Stands in for a Reservation when this task already holds the GPU.

    Honours the same interface so call sites need no special case: wait()
    yields nothing, because there is nothing to wait for.
    """

    def __init__(self, broker: "GpuBroker", need: Need):
        self._broker = broker
        self._need = need

    @property
    def ticket(self) -> int:
        return 0

    async def wait(self):
        return
        yield       # pragma: no cover - makes this an async generator


class GpuBroker:
    """Serialises GPU work and owns which model is resident."""

    def __init__(self) -> None:
        self._tickets = itertools.count(1)
        self._line: list[Entry] = []        # [0] holds the GPU once started
        self._turn = asyncio.Condition()
        self._resident: Need | None = None
        self._durations: dict[str, list[float]] = {}

    # -- what is loaded -----------------------------------------------------

    @property
    def resident(self) -> Need | None:
        return self._resident

    def forget_residency(self) -> None:
        """Declare the GPU's contents unknown.

        Called when something outside the broker changes what is loaded — a
        manual unload, or a model switch from the settings panel — so the next
        task reloads rather than trusting a stale assumption.
        """
        self._resident = None

    # -- introspection ------------------------------------------------------

    def snapshot(self) -> dict:
        running = self._line[0] if self._line and self._line[0].started_at else None
        return {
            "resident": str(self._resident) if self._resident else None,
            "busy": running is not None,
            "running": self._describe(running) if running else None,
            "waiting": [self._describe(e) for e in self._line
                        if e is not running],
            "typical": {k: round(self._median(v), 1)
                        for k, v in self._durations.items() if len(v) >= 3},
        }

    def _describe(self, e: Entry) -> dict:
        return {"ticket": e.ticket, "label": e.label, "user": e.user,
                "need": str(e.need), "note": e.note, "priority": e.priority,
                "waiting_seconds": round(e.waited, 1),
                "running_seconds": round(e.running_for, 1)}

    @staticmethod
    def _median(xs: list[float]) -> float:
        s = sorted(xs)
        return s[len(s) // 2]

    def estimate_wait(self, ticket: int) -> float | None:
        """Roughly how long until this ticket starts, from past durations."""
        total = 0.0
        for e in self._line:
            if e.ticket == ticket:
                return round(total, 1) if total else 0.0
            typical = self._durations.get(e.label, [])
            if len(typical) < 3:
                return None             # not enough history to be honest
            remaining = self._median(typical) - e.running_for
            total += max(remaining, 0.0)
        return None

    def position_of(self, ticket: int) -> int:
        for i, e in enumerate(self._line):
            if e.ticket == ticket:
                return i
        return 0

    # -- jumping the queue --------------------------------------------------

    def request_priority(self, ticket: int, note: str) -> bool:
        """Move a waiting task ahead of the others, with a stated reason.

        Self-declared on purpose. This is a household, not a datacentre: the
        note is visible to everyone waiting, and that is the check on it.
        Whoever is already running is never interrupted.
        """
        entry = next((e for e in self._line if e.ticket == ticket), None)
        if entry is None or entry.started_at is not None:
            return False
        entry.priority = True
        entry.note = (note or "").strip()[:200]
        running = [e for e in self._line if e.started_at is not None]
        waiting = [e for e in self._line if e.started_at is None]
        # Stable: priority first, each group still first-come-first-served.
        waiting.sort(key=lambda e: (not e.priority, e.queued_at))
        self._line = running + waiting
        logger.info("[gpu] %s jumped the queue: %s", entry.user, entry.note)
        return True

    # -- the queue itself ---------------------------------------------------

    @contextlib.asynccontextmanager
    async def exclusive(self, need: Need, label: str, user: str = ""):
        """Queue, wait, and hold the GPU — for callers that cannot stream.

        Most endpoints are not SSE and have no way to say "two ahead of you",
        so they simply block. They still take their turn in the same line, and
        /gpu/queue still shows them, which is what makes the shared board
        complete rather than a view of the tutor only.
        """
        async with self.reserve(need, label, user=user) as res:
            async for _ in res.wait():
                pass
            yield res

    @contextlib.asynccontextmanager
    async def reserve(self, need: Need, label: str, user: str = ""):
        """Join the queue and hold the GPU for the duration of the block.

        Yields a :class:`Reservation`. The caller **must** consume
        ``reservation.wait()`` before using the GPU — that is what actually
        queues, and doing it as an async generator is the point: a streaming
        endpoint can emit "2 ahead of you" while it waits.

        The earlier design blocked inside this context manager, which meant an
        SSE generator was suspended and could not yield anything at all until
        its turn came. The browser saw silence followed by a burst of stale
        "waiting" messages — which is indistinguishable from the hang the queue
        was meant to explain.
        """
        if _holding.get():
            # Already ours. A crew run calls back into the model for
            # summarisation and tool steps; queueing those behind the very
            # task that is running them is a deadlock, not an ordering.
            yield _Passthrough(self, need)
            return
        entry = Entry(ticket=next(self._tickets), need=need, label=label,
                      user=user)
        self._line.append(entry)
        token = _holding.set(True)
        try:
            yield Reservation(self, entry)
        finally:
            _holding.reset(token)
            if entry.started_at is not None:
                self._durations.setdefault(label, []).append(entry.running_for)
                self._durations[label] = self._durations[label][-12:]
            with contextlib.suppress(ValueError):
                self._line.remove(entry)
            async with self._turn:
                self._turn.notify_all()

    async def _await_turn(self, entry: Entry):
        """Yield a notice while queued; return once the GPU is ours."""
        if self.position_of(entry.ticket) == 0:
            return                      # straight through, nothing to report
        last = None
        while True:
            position = self.position_of(entry.ticket)
            if position == 0 and (not self._line
                                  or self._line[0].ticket == entry.ticket):
                break
            if position != last:
                last = position
                yield {"position": position,
                       "estimate": self.estimate_wait(entry.ticket),
                       "ticket": entry.ticket}
            await asyncio.sleep(0.5)
        yield {"position": 0, "estimate": 0, "ticket": entry.ticket}

    # -- residency ----------------------------------------------------------

    async def _make_resident(self, need: Need, label: str) -> None:
        """Swap the GPU's contents to *need*, doing only what is necessary."""
        current = self._resident
        if current == need:
            logger.debug("[gpu] %s already resident", need)
            return

        loop = asyncio.get_running_loop()

        if current is not None and current.kind == "song" and need.kind != "song":
            # ACE-Step runs in its own process and has no unload endpoint —
            # `offload` is an init-time option only — so the only way to get
            # its VRAM back is to stop it. Every other transition here is a
            # no-op by measurement; this one is not, and leaving it out would
            # push the next person's model onto the CPU.
            # gpu-queue-exempt: the broker making room, not queued work.
            from researcher import song as _song

            await loop.run_in_executor(None, _song.stop)

        if (need.kind in ("torch", "song")
                and (current is None or current.kind == "ollama")):
            # Ollama will not give VRAM back to a process it does not know
            # about, so it has to be asked. For a diffusers pipeline this is
            # the only teardown required — it holds ~32 MiB once its
            # generation finishes, because the weights live in CPU RAM — and
            # ACE-Step needs the same room made for it.
            # gpu-queue-exempt: this is the broker making room; queueing
            # it behind itself is the deadlock it exists to prevent.
            await loop.run_in_executor(None, _release_ollama, label)

        # Ollama -> Ollama needs nothing: loading a model evicts the previous
        # one. torch -> Ollama needs nothing either, as measured.
        self._resident = need
        logger.info("[gpu] %s now resident for %s", need, label)


def _release_ollama(tag: str) -> None:
    """Tell Ollama to drop everything, then wait for the VRAM to appear."""
    import requests

    base = "http://localhost:11434"
    try:
        running = requests.get(f"{base}/api/ps", timeout=10).json()
        names = [m["name"] for m in running.get("models", [])]
        for name in names:
            requests.post(f"{base}/api/generate",
                          json={"model": name, "keep_alive": 0, "prompt": ""},
                          timeout=30)
        if names:
            logger.info("[gpu] %s released Ollama models: %s", tag, names)
    except Exception as e:
        logger.warning("[gpu] could not release Ollama: %s", e)

    deadline = time.monotonic() + _VRAM_WAIT_SECONDS
    while time.monotonic() < deadline:
        free = _free_vram_gib()
        if free is None or free >= _VRAM_WANTED_GIB:
            return
        time.sleep(1)
    logger.warning("[gpu] %s proceeding with only %.1f GiB free", tag,
                   _free_vram_gib() or -1)


def _free_vram_gib() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return free / 2**30
    except Exception:
        return None


async def _tell(on_wait, position: int, estimate, ticket: int) -> None:
    """Report progress. A failure here must never fail the work."""
    if on_wait is None:
        return
    try:
        got = on_wait(position, estimate, ticket)
        if asyncio.iscoroutine(got):
            await got
    except Exception as e:
        logger.debug("[gpu] progress callback failed: %s", e)


def queue_message(position: int, estimate, ticket: int) -> dict:
    """The payload a streaming endpoint sends while its caller waits.

    Shared so every tab says the same thing, and so the browser can rely on
    one shape: a position of 0 means "your turn", which is the client's cue to
    take the waiting note down.
    """
    if not position:
        return {"position": 0, "text": "Starting", "ticket": ticket}
    text = f"Waiting for the GPU \u2014 {position} ahead of you"
    if estimate:
        text += f", about {estimate:.0f}s"
    return {"position": position, "text": text, "ticket": ticket}


# One card, one broker.
GPU = GpuBroker()
