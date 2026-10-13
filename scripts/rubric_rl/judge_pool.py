from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
import random
import time
import urllib.request
import urllib.error


class JudgeUnavailable(RuntimeError):
    pass


def read_config(path, raw=None):
    data = json.loads(Path(path).read_bytes() if raw is None else raw)
    endpoints = data["endpoints"]
    if not isinstance(endpoints, list) or not endpoints:
        raise ValueError("endpoints must be a nonempty list")
    seen = set()
    for e in endpoints:
        for key in ("id", "base_url", "model", "api_key_env"):
            if not isinstance(e.get(key), str) or not e[key]:
                raise ValueError("missing endpoint field: " + key)
        if e["id"] in seen:
            raise ValueError("duplicate endpoint id")
        seen.add(e["id"])
        if not e["base_url"].startswith(("https://", "http://")):
            raise ValueError("invalid base_url")
        if type(e.get("enabled", True)) is not bool:
            raise ValueError("enabled must be boolean")
        for key, default in [("max_concurrency", 8), ("weight", 1), ("timeout_s", 120)]:
            e.setdefault(key, default)
            if type(e[key]) not in (int, float) or not 0 < e[key] <= 3600:
                raise ValueError("invalid " + key)
        if type(e["max_concurrency"]) is not int:
            raise ValueError("max_concurrency must be integer")
        if "api_key" in e:
            raise ValueError("use api_key_env, never literal credentials")
    return endpoints


def http_completion(endpoint, messages, max_tokens, temperature):
    key = os.environ.get(endpoint["api_key_env"])
    if not key:
        raise JudgeUnavailable("credential environment variable unavailable")
    body = json.dumps(
        dict(
            model=endpoint["model"],
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **endpoint.get("extra_body", {}),
        )
    ).encode()
    request = urllib.request.Request(
        endpoint["base_url"].rstrip("/") + "/chat/completions",
        data=body,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=endpoint["timeout_s"]) as response:
        result = json.load(response)
    content = result["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise JudgeUnavailable("empty judge response")
    return content


class JudgePool:
    def __init__(self, config_path, transport=http_completion):
        self.path = Path(config_path)
        self.transport = transport
        self.fingerprint = self.path.read_bytes()
        self.endpoints = read_config(self.path, self.fingerprint)
        self.active = {}
        self.cooldown = {}
        self.config_error = False
        self.lock = asyncio.Lock()

    def _reload(self):
        try:
            raw = self.path.read_bytes()
            if raw != self.fingerprint:
                endpoints = read_config(self.path, raw)
                self.endpoints, self.fingerprint = (endpoints, raw)
            self.config_error = False
        except (OSError, ValueError, KeyError, TypeError):
            self.config_error = True

    def status(self):
        return dict(
            config_error=self.config_error,
            endpoints=[
                dict(
                    id=e["id"],
                    model=e["model"],
                    enabled=e.get("enabled", True),
                    active=self.active.get(e["id"], 0),
                    max_concurrency=e["max_concurrency"],
                )
                for e in self.endpoints
            ],
        )

    async def _acquire(self, deadline):
        while time.monotonic() < deadline:
            async with self.lock:
                self._reload()
                candidates = [
                    e
                    for e in self.endpoints
                    if e.get("enabled", True)
                    and self.active.get(e["id"], 0) < e["max_concurrency"]
                    and (self.cooldown.get(e["id"], 0) <= time.monotonic())
                ]
                if candidates:
                    e = random.choices(
                        candidates, weights=[x["weight"] for x in candidates]
                    )[0].copy()
                    self.active[e["id"]] = self.active.get(e["id"], 0) + 1
                    return e
            await asyncio.sleep(0.05)
        raise JudgeUnavailable("no available judge within queue timeout")

    async def complete(
        self,
        messages,
        *,
        max_tokens=1024,
        temperature=0,
        max_attempts=3,
        queue_timeout_s=180,
    ):
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        for attempt in range(max_attempts):
            e = await self._acquire(time.monotonic() + queue_timeout_s)
            task = asyncio.create_task(
                asyncio.to_thread(self.transport, e, messages, max_tokens, temperature)
            )
            try:
                content = await asyncio.shield(task)
                return dict(
                    content=content,
                    endpoint_id=e["id"],
                    model=e["model"],
                    attempts=attempt + 1,
                )
            except asyncio.CancelledError:
                try:
                    await task
                except Exception:
                    pass
                raise
            except Exception as exc:
                status = exc.code if isinstance(exc, urllib.error.HTTPError) else None
                detail = "http_" + str(status) if status else type(exc).__name__
                print(
                    json.dumps(
                        dict(
                            event="judge_request_failed", error=detail, endpoint=e["id"]
                        ),
                        sort_keys=True,
                    ),
                    flush=True,
                )
                self.cooldown[e["id"]] = time.monotonic() + min(2**attempt, 10)
                if attempt + 1 == max_attempts:
                    raise JudgeUnavailable(detail) from None
            finally:
                self.active[e["id"]] -= 1
