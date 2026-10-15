"""Who has the GPU, who is waiting, and asking to go first.

There is one card and several people, so contention is ordinary rather than
exceptional. The whole point of these endpoints is that a wait should be
visible: a request that is queued looks exactly like a request that has hung,
and the difference matters when the person waiting is a child who wants their
homework sheet.

Priority is self-declared and carries a reason that everyone waiting can read.
That is deliberate for a household: the note is the check on it, not a
permission system. Whoever is already running is never interrupted — their
work would be lost, and losing a four-minute lesson to jump a queue is a poor
trade.
"""

import logging

from fastapi import HTTPException, Request
from pydantic import BaseModel

from researcher.auth import get_optional_user
from researcher.gpu_broker import GPU

logger = logging.getLogger(__name__)


class PriorityRequest(BaseModel):
    """Ask to go next, and say why."""

    ticket: int
    note: str = ""


def register_gpu_routes(app):
    """Attach ``/gpu/*`` endpoints to *app*."""

    async def _name_the_users(request: Request, snap: dict) -> dict:
        """Replace user ids with usernames throughout *snap*.

        The broker holds ids because it has no business knowing about the user
        table. But "30d5da42a474 is ahead of you" explains nothing, and the
        whole point of a shared queue in a household is that you can see it is
        your brother and go and ask him.
        """
        entries = ([snap["running"]] if snap.get("running") else []) + \
            snap.get("waiting", [])
        ids = {e["user"] for e in entries if e.get("user")}
        if not ids:
            return snap
        rows = await request.app.state.db.execute(
            "SELECT id, username FROM users WHERE id IN "
            f"({','.join('?' * len(ids))})", tuple(ids))
        names = {r[0]: r[1] for r in await rows.fetchall()}
        for e in entries:
            e["user"] = names.get(e.get("user"), e.get("user") or "someone")
        return snap

    @app.get("/gpu/queue")
    async def gpu_queue(request: Request):
        """What the card is doing and who is in line.

        Readable without logging in: on a household instance the queue is a
        shared fact, and hiding it would defeat the purpose.
        """
        snap = await _name_the_users(request, GPU.snapshot())
        user = await get_optional_user(request)
        snap["you"] = user["username"] if user else None
        return snap

    @app.post("/gpu/queue/priority")
    async def gpu_priority(req: PriorityRequest, request: Request):
        """Move a waiting task to the front, with a stated reason."""
        note = (req.note or "").strip()
        if not note:
            # Without a reason this is just queue-jumping, and the reason is
            # the only thing keeping it honest.
            raise HTTPException(
                status_code=400,
                detail="Say why you need to go first — everyone waiting sees it.")
        if not GPU.request_priority(req.ticket, note):
            raise HTTPException(
                status_code=404,
                detail="That task is not waiting any more — it has either "
                       "started or finished.")
        user = await get_optional_user(request)
        logger.info("[gpu] priority granted to ticket %s (%s): %s",
                    req.ticket, (user or {}).get("username", "anon"), note)
        return await _name_the_users(request, GPU.snapshot())
