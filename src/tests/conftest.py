# tests/conftest.py

import socket

import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """テストが外部サービスへ接続しないことを保証する。"""

    def refuse_connection(*args, **kwargs):
        raise AssertionError("テスト中のネットワーク接続は禁止されています")

    monkeypatch.setattr(socket.socket, "connect", refuse_connection)
    monkeypatch.setattr(socket, "create_connection", refuse_connection)
