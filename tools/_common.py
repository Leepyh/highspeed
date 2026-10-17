#!/usr/bin/env python
"""
tools 公共工具：无 gymnasium 环境下的最小桩模块与通用测量函数
=============================================================

背景
----
env_maddpg.py 需要 gymnasium（或旧版 gym）的 Env / spaces。
本机未安装 gymnasium，且按约定不安装依赖。

为了在只依赖 numpy 的前提下验证 env 层，本模块提供：
  - install_gym_stub()：注入一个最小 gym 桩，仅实现 Env 与 spaces.Discrete/Box
  - 若干测量辅助函数（同种子对照、指标统计）

注意
----
桩模块只覆盖本项目用到的接口（Env 基类、spaces.Discrete、spaces.Box）。
通过桩验证的是**本项目的逻辑**，不等于验证了真实 gymnasium 的兼容性；
使用桩时会在输出中明确标注。
"""

from __future__ import annotations

import os
import random
import sys
import types

import numpy as np

# 工作区根目录（tools/ 的上一级）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

STUB_ACTIVE = False


def install_torch_stub() -> None:
    """config_maddpg.py 顶层 import torch 只为选设备，这里提供最小桩。"""
    if "torch" in sys.modules:
        return
    t = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    t.cuda = _Cuda()
    sys.modules["torch"] = t


def install_gym_stub() -> bool:
    """注入最小 gym 桩。返回 True 表示确实注入了桩。

    若真实 gymnasium / gym 已存在，则不做任何事并返回 False。
    """
    global STUB_ACTIVE

    for real in ("gymnasium", "gym"):
        try:
            __import__(real)
            return False
        except ImportError:
            continue

    class _Space:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def sample(self):
            raise NotImplementedError

        def contains(self, x):
            return True

    class Discrete(_Space):
        def __init__(self, n):
            super().__init__(n)
            self.n = n

        def sample(self):
            return random.randrange(self.n)

        def __repr__(self):
            return f"Discrete({self.n})"

    class Box(_Space):
        def __init__(self, low=-np.inf, high=np.inf, shape=None, dtype=np.float32):
            super().__init__(low, high, shape, dtype)
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

        def sample(self):
            return np.zeros(self.shape, dtype=self.dtype)

        def __repr__(self):
            return f"Box(shape={self.shape})"

    class Env:
        metadata: dict = {}

        def reset(self, **kwargs):
            raise NotImplementedError

        def step(self, action):
            raise NotImplementedError

        def close(self):
            pass

    spaces = types.ModuleType("gymnasium.spaces")
    spaces.Discrete = Discrete
    spaces.Box = Box

    mod = types.ModuleType("gymnasium")
    mod.Env = Env
    mod.spaces = spaces

    sys.modules["gymnasium"] = mod
    sys.modules["gymnasium.spaces"] = spaces
    STUB_ACTIVE = True
    return True


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def banner(title: str) -> None:
    print("=" * 84)
    print(title)
    print("=" * 84)


def stub_notice() -> None:
    if STUB_ACTIVE:
        print("[注意] gymnasium 未安装，本次使用内置最小桩模块验证 env 层逻辑。")
        print("       这验证的是本项目代码，不代表已验证真实 gymnasium 的兼容性。\n")
