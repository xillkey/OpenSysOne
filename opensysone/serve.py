"""HTTP server exposing ``POST /v1/systemone`` with request micro-batching.

Concurrent requests that arrive within ``max_wait_ms`` of each other are run
through the model as one batch (up to ``max_batch`` requests), which raises
throughput on GPUs without hurting single-request latency much.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .schema import SchemaError

log = logging.getLogger(__name__)


class MicroBatcher:
    def __init__(self, predictor: Any, max_batch: int = 32, max_wait_ms: float = 5.0):
        self.predictor = predictor
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000
        self.queue: asyncio.Queue[tuple[dict[str, Any], asyncio.Future]] | None = None
        self.task: asyncio.Task | None = None

    async def start(self) -> None:
        self.queue = asyncio.Queue()
        self.task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()

    async def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.queue is None:  # not started (e.g. tests without lifespan): run inline
            return self.predictor.decide(request)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self.queue.put((request, fut))
        return await fut

    async def _loop(self) -> None:
        assert self.queue is not None
        loop = asyncio.get_running_loop()
        while True:
            first = await self.queue.get()
            batch = [first]
            deadline = loop.time() + self.max_wait
            while len(batch) < self.max_batch:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), timeout))
                except asyncio.TimeoutError:
                    break
            await self._run(batch)

    async def _run(self, batch: list[tuple[dict[str, Any], asyncio.Future]]) -> None:
        loop = asyncio.get_running_loop()
        try:
            outs = await loop.run_in_executor(None, self.predictor.decide_many, [r for r, _ in batch])
            for (_, fut), out in zip(batch, outs):
                fut.set_result(out)
        except Exception:
            # one bad request must not fail its neighbours: retry one by one
            for req, fut in batch:
                try:
                    fut.set_result(await loop.run_in_executor(None, self.predictor.decide, req))
                except Exception as e:  # noqa: BLE001
                    fut.set_exception(e)


def create_app(predictor: Any, max_batch: int = 32, max_wait_ms: float = 5.0) -> Any:
    try:
        from contextlib import asynccontextmanager

        from fastapi import FastAPI, HTTPException
    except ImportError as e:  # pragma: no cover
        raise ImportError("serving needs `pip install opensysone[serve]`") from e

    batcher = MicroBatcher(predictor, max_batch, max_wait_ms)

    @asynccontextmanager
    async def lifespan(_: Any):  # type: ignore[no-untyped-def]
        await batcher.start()
        yield
        await batcher.stop()

    app = FastAPI(title="OpenSysOne", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/model")
    async def model_info() -> dict[str, Any]:
        return {"model": predictor.name, "serializer": predictor.serializer.cfg.to_dict()}

    @app.post("/v1/systemone")
    async def systemone(request: dict[str, Any]) -> dict[str, Any]:
        try:
            return await batcher.submit(request)
        except (SchemaError, KeyError, TypeError, ValueError) as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return app
