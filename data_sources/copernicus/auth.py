"""Copernicus Data Space OAuth2 认证。

OData 产品下载推荐使用 Copernicus Data Space 账号密码，通过 cdse-public
客户端获取 access token，并缓存到本地文件。旧的 client_id/client_secret
client credentials flow 仍作为兼容兜底保留。

推荐凭证：
    1. 环境变量 COPERNICUS_USERNAME / COPERNICUS_PASSWORD
    2. 配置文件中的 copernicus.username / copernicus.password
    3. ~/.copernicus/credentials.json 中的 username/password
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from data_sources.copernicus.config import COPERNICUS_TOKEN_URL


def _default_credentials_path() -> Path:
    return Path.home() / ".copernicus" / "credentials.json"


def _default_token_cache_path() -> Path:
    return Path.home() / ".copernicus" / "token_cache.json"


def _load_credentials_from_file(path: Path) -> tuple[str, str] | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        client_id = data.get("client_id") or data.get("username")
        client_secret = data.get("client_secret") or data.get("password")
        if client_id and client_secret:
            return str(client_id), str(client_secret)
    except Exception:
        pass
    return None


def _load_credentials_from_env() -> tuple[str, str] | None:
    client_id = os.environ.get("COPERNICUS_CLIENT_ID")
    client_secret = os.environ.get("COPERNICUS_CLIENT_SECRET")
    if client_id and client_secret:
        return client_id, client_secret
    return None


def _load_credentials_from_config(config: dict[str, Any] | None) -> tuple[str, str] | None:
    if not config:
        return None
    copernicus = config.get("copernicus", {})
    client_id = copernicus.get("client_id")
    client_secret = copernicus.get("client_secret")
    if client_id and client_secret:
        return client_id, client_secret
    return None


def _resolve_credentials(config: dict[str, Any] | None = None) -> tuple[str, str]:
    """按优先级获取凭证。"""
    for source in [
        _load_credentials_from_env(),
        _load_credentials_from_config(config),
        _load_credentials_from_file(_default_credentials_path()),
    ]:
        if source is not None:
            return source

    raise RuntimeError(
        "未找到 Copernicus Data Space 凭证。请通过以下任一方式提供：\n"
        "  1. 环境变量: COPERNICUS_CLIENT_ID / COPERNICUS_CLIENT_SECRET\n"
        "  2. 配置文件 copernicus.client_id / copernicus.client_secret\n"
        f"  3. {_default_credentials_path()}"
    )


def _request_token(client_id: str, client_secret: str) -> dict[str, Any]:
    """向哥白尼身份服务请求 access token。"""
    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }
    response = requests.post(
        COPERNICUS_TOKEN_URL,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _load_cached_token(cache_path: Path) -> str | None:
    """从缓存文件加载未过期的 token。"""
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        access_token = data.get("access_token")
        expires_at = data.get("expires_at", 0)
        # 提前 5 分钟刷新
        if access_token and time.time() < expires_at - 300:
            return str(access_token)
    except Exception:
        pass
    return None


def _save_token_cache(cache_path: Path, token_response: dict[str, Any]) -> None:
    """将 token 响应缓存到本地文件。"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    expires_in = int(token_response.get("expires_in", 3600))
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "access_token": token_response["access_token"],
                "expires_at": time.time() + expires_in,
            },
            f,
        )


# Newer Copernicus Data Space OData downloads use an OAuth2 token from the
# public CDSE client with username/password credentials.  Keep the old
# client_credentials path as a fallback for deployments that already use it.
def _load_user_password_from_env() -> tuple[str, str] | None:
    username = os.environ.get("COPERNICUS_USERNAME") or os.environ.get("COPERNICUS_USER")
    password = os.environ.get("COPERNICUS_PASSWORD")
    if username and password:
        return username, password
    return None


def _load_user_password_from_config(config: dict[str, Any] | None) -> tuple[str, str] | None:
    if not config:
        return None
    copernicus = config.get("copernicus", {})
    username = copernicus.get("username") or copernicus.get("user")
    password = copernicus.get("password")
    if username and password:
        return str(username), str(password)
    return None


def _load_user_password_from_file(path: Path) -> tuple[str, str] | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        username = data.get("username") or data.get("user")
        password = data.get("password")
        if username and password:
            return str(username), str(password)
    except Exception:
        pass
    return None


def _resolve_user_password(config: dict[str, Any] | None = None) -> tuple[str, str] | None:
    for source in [
        _load_user_password_from_env(),
        _load_user_password_from_config(config),
        _load_user_password_from_file(_default_credentials_path()),
    ]:
        if source is not None:
            return source
    return None


def _request_password_token(username: str, password: str) -> dict[str, Any]:
    payload = {
        "grant_type": "password",
        "client_id": os.environ.get("COPERNICUS_CLIENT_ID", "cdse-public"),
        "username": username,
        "password": password,
    }
    client_secret = os.environ.get("COPERNICUS_CLIENT_SECRET")
    if client_secret:
        payload["client_secret"] = client_secret

    response = requests.post(
        COPERNICUS_TOKEN_URL,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def get_access_token(config: dict[str, Any] | None = None) -> str:
    """获取有效的 Copernicus access token（优先使用缓存）。

    推荐配置：
      COPERNICUS_USERNAME / COPERNICUS_PASSWORD
    或 ~/.copernicus/credentials.json:
      {"username": "...", "password": "..."}

    如果没有 username/password，则回退到旧的 client_id/client_secret flow。
    """
    cache_path = _default_token_cache_path()

    cached = _load_cached_token(cache_path)
    if cached:
        return cached

    user_password = _resolve_user_password(config)
    if user_password is not None:
        token_response = _request_password_token(*user_password)
    else:
        client_id, client_secret = _resolve_credentials(config)
        token_response = _request_token(client_id, client_secret)

    _save_token_cache(cache_path, token_response)
    return str(token_response["access_token"])
