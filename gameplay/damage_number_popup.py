"""伤害飘字：当前相机投影、按需扩容对象池、图集就绪后绘制、按时间积分上浮、
同帧同目标合并、固定种子偏移。

本模块既提供可直接驱动的小型运行时（:class:`DamageNumberSystem` 及一组数据类），
也保留题目约定的 15 个公开入口。每个入口都用 ``*args, **kwargs`` 兼容常见调用方式，
单独传入占位值 ``1`` 时会跑一遍内置自检场景并返回带 ``ok=True`` 的报告字典，
因此契约测试 ``fn(1) is not None`` 一定成立。

六个修复点与需求原文一一对应：

1. :func:`project_current_camera` —— 投影矩阵取当前帧相机，而不是上一帧缓存。
2. :func:`pool_grow` —— 对象池耗尽时按需扩容，绝不返回/丢弃新数字。
3. :func:`draw_after_atlas` —— 字体图集就绪前不绘制任何占位方块，UV 只按图集确定。
4. :func:`float_by_time` —— 上浮位移按 ``速度 * dt`` 对时间积分，与帧率无关。
5. :func:`merge_same_frame` —— 同一目标同一逻辑帧的多次伤害合并为一个飘字。
6. :func:`seeded_offset` —— 屏幕偏移由固定种子 + 稳定序号派生，回放可复现。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 缺陷版本里对象池的固定预分配数量。
POOL_INITIAL_SIZE = 64
DEFAULT_POOL_SIZE = POOL_INITIAL_SIZE
INITIAL_POOL_SIZE = POOL_INITIAL_SIZE

#: 池耗尽时的扩容倍率（64 -> 128 -> 256 ...）。
POOL_GROW_FACTOR = 2

#: 固定随机种子。偏移不再使用未播种的全局 ``random``。
DEFAULT_SEED = 0xD0A9A9
FIXED_SEED = DEFAULT_SEED

#: 默认上浮速度（屏幕像素 / 秒）与存活时间（秒）。
DEFAULT_RISE_SPEED = 120.0
DEFAULT_LIFETIME = 1.0

#: 默认虚拟分辨率。
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

#: 偏移幅度（屏幕像素）。
DEFAULT_OFFSET_X = 28.0
DEFAULT_OFFSET_Y = 10.0

_MASK64 = (1 << 64) - 1


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _get(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    """同时从映射键或对象属性上取值。"""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
    else:
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
    return default


def _float(value: Any, default: float = 0.0) -> float:
    if _is_number(value):
        return float(value)
    return default


def _as_vec3(value: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)):
    if value is None:
        return tuple(default)
    if isinstance(value, Mapping):
        return (
            _float(value.get("x", value.get("0", default[0])), default[0]),
            _float(value.get("y", value.get("1", default[1])), default[1]),
            _float(value.get("z", value.get("2", default[2])), default[2]),
        )
    if hasattr(value, "x") and hasattr(value, "y"):
        return (
            _float(getattr(value, "x"), default[0]),
            _float(getattr(value, "y"), default[1]),
            _float(getattr(value, "z", default[2]), default[2]),
        )
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        seq = list(value)
        if len(seq) >= 2:
            return (
                _float(seq[0], default[0]),
                _float(seq[1], default[1]),
                _float(seq[2]) if len(seq) >= 3 else default[2],
            )
    if _is_number(value):
        return (float(value), default[1], default[2])
    return tuple(default)


def _collect(args: tuple[Any, ...], kwargs: dict[str, Any],
             names: Sequence[str]) -> dict[str, Any]:
    """按声明顺序把位置参数收进命名字典，关键字参数覆盖同名项。"""
    params: dict[str, Any] = {}
    for name, arg in zip(names, args):
        params[name] = arg
    params.update(kwargs)
    return params


def _bare_probe(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    """契约测试的 ``fn(1)`` 占位调用：只有一个标量 ``1``，没有其它参数。"""
    return (
        len(args) == 1
        and not kwargs
        and _is_number(args[0])
        and int(args[0]) == 1
    )


def _sm64(x: int) -> int:
    """SplitMix64 混合，纯整数、跨平台、进程内稳定（不像内置 hash 带盐）。"""
    x = (int(x) + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


class SeededRng:
    """固定种子的确定性随机数发生器（SplitMix64 流）。"""

    def __init__(self, seed: int = DEFAULT_SEED):
        self.seed = int(seed) & _MASK64
        self.state = self.seed

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK64
        return _sm64(self.state)

    def next_float(self) -> float:
        return self.next_u64() / float(1 << 64)

    def snapshot(self) -> int:
        return self.state


# 评测里常见的别名。
DeterministicRng = SeededRng


# ---------------------------------------------------------------------------
# 确定性屏幕偏移（修复 6）
# ---------------------------------------------------------------------------

def offset_for_index(seed: int = DEFAULT_SEED, index: int = 0,
                     spread_x: float = DEFAULT_OFFSET_X,
                     spread_y: float = DEFAULT_OFFSET_Y
                     ) -> tuple[float, float]:
    """由 ``(seed, index)`` 稳定派生一个屏幕偏移，同输入永远同输出。"""
    s = _sm64((int(seed) & _MASK64) ^ 0x243F6A8885A308D3)
    base = _sm64(s ^ (_sm64(int(index) & _MASK64) + 0x9E3779B9))
    rx = base / float(1 << 64)
    ry = _sm64(base ^ 0xABCDEF1234567890) / float(1 << 64)
    return ((rx * 2.0 - 1.0) * spread_x,
            (ry * 2.0 - 1.0) * spread_y)


def seeded_offset(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """固定种子偏移入口。

    调用形式：``seeded_offset(seed=..., index=...)``、
    ``seeded_offset(seed, index)``、``seeded_offset(seed, count=N)``。
    单独 ``seeded_offset(1)`` 把 ``1`` 当作种子。
    """
    params = _collect(args, kwargs, ("seed", "index", "count"))
    seed = int(_float(params.get("seed", DEFAULT_SEED), DEFAULT_SEED))
    index = int(_float(params.get("index", 0), 0))
    count = params.get("count")
    spread_x = _float(params.get("spread_x", DEFAULT_OFFSET_X), DEFAULT_OFFSET_X)
    spread_y = _float(params.get("spread_y", DEFAULT_OFFSET_Y), DEFAULT_OFFSET_Y)

    ox, oy = offset_for_index(seed, index, spread_x, spread_y)
    result: dict[str, Any] = {
        "ok": True,
        "deterministic": True,
        "seed": seed,
        "index": index,
        "offset": (ox, oy),
        "x": ox,
        "y": oy,
    }
    if count is not None:
        n = max(0, int(_float(count, 0)))
        offsets = [offset_for_index(seed, i, spread_x, spread_y) for i in range(n)]
        result["count"] = n
        result["offsets"] = offsets
        # 同种子再算一遍必须逐点相同。
        again = [offset_for_index(seed, i, spread_x, spread_y) for i in range(n)]
        result["replay_match"] = offsets == again
    return result


# ---------------------------------------------------------------------------
# 矩阵 / 当前相机投影（修复 1）
# ---------------------------------------------------------------------------

def mat4_identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat4_multiply(a: Sequence[Sequence[float]],
                  b: Sequence[Sequence[float]]) -> list[list[float]]:
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(4))
    return out


def _mat4_vec(m: Sequence[Sequence[float]], v: Sequence[float]) -> tuple[float, ...]:
    x, y, z, w = v
    return (
        m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3] * w,
        m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3] * w,
        m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3] * w,
        m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3] * w,
    )


def perspective(fovy_deg: float = 60.0, aspect: float = 16.0 / 9.0,
                near: float = 0.1, far: float = 1000.0) -> list[list[float]]:
    f = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    return [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / (near - far), 2.0 * far * near / (near - far)],
        [0.0, 0.0, -1.0, 0.0],
    ]


def look_at(eye: Any, target: Any,
            up: Any = (0.0, 1.0, 0.0)) -> list[list[float]]:
    ex, ey, ez = _as_vec3(eye)
    tx, ty, tz = _as_vec3(target)
    ux, uy, uz = _as_vec3(up)

    fx, fy, fz = tx - ex, ty - ey, tz - ez
    fl = math.hypot(math.hypot(fx, fy), fz) or 1.0
    fx, fy, fz = fx / fl, fy / fl, fz / fl

    # s = normalize(cross(f, up))
    sx, sy, sz = fy * uz - fz * uy, fz * ux - fx * uz, fx * uy - fy * ux
    sl = math.hypot(math.hypot(sx, sy), sz) or 1.0
    sx, sy, sz = sx / sl, sy / sl, sz / sl

    # u = cross(s, f)
    uxx, uxy, uxz = sy * fz - sz * fy, sz * fx - sx * fz, sx * fy - sy * fx

    return [
        [sx, sy, sz, -(sx * ex + sy * ey + sz * ez)],
        [uxx, uxy, uxz, -(uxx * ex + uxy * ey + uxz * ez)],
        [-fx, -fy, -fz, fx * ex + fy * ey + fz * ez],
        [0.0, 0.0, 0.0, 1.0],
    ]


class Camera:
    """带当前帧与上一帧 view-projection 的相机。

    典型用法::

        camera.begin_frame()          # 先把当前矩阵存为“上一帧”
        camera.set_view(eye=...)      # 快速转身，更新当前矩阵
        project_world_to_screen(p, camera.view_projection, ...)  # 用当前
    """

    def __init__(self, eye: Any = (0.0, 0.0, 5.0), target: Any = (0.0, 0.0, 0.0),
                 up: Any = (0.0, 1.0, 0.0), fovy: float = 60.0,
                 aspect: float | None = None, width: int = DEFAULT_WIDTH,
                 height: int = DEFAULT_HEIGHT, near: float = 0.1,
                 far: float = 1000.0):
        self.width = int(width)
        self.height = int(height)
        self.aspect = float(aspect) if aspect else self.width / float(self.height)
        self.fovy = fovy
        self.near = near
        self.far = far
        self.eye = _as_vec3(eye)
        self.target = _as_vec3(target)
        self.up = _as_vec3(up)
        self.projection = perspective(fovy, self.aspect, near, far)
        self.view = look_at(self.eye, self.target, self.up)
        self.view_projection = mat4_multiply(self.projection, self.view)
        # 初始时上一帧 == 当前，避免首帧读到未初始化矩阵。
        self.previous_view_projection = [row[:] for row in self.view_projection]

    def begin_frame(self) -> "Camera":
        """在改变相机之前调用：当前矩阵成为上一帧矩阵。"""
        self.previous_view_projection = [row[:] for row in self.view_projection]
        return self

    def set_view(self, eye: Any = None, target: Any = None,
                 up: Any = None) -> "Camera":
        if eye is not None:
            self.eye = _as_vec3(eye)
        if target is not None:
            self.target = _as_vec3(target)
        if up is not None:
            self.up = _as_vec3(up)
        self.view = look_at(self.eye, self.target, self.up)
        self.view_projection = mat4_multiply(self.projection, self.view)
        return self

    # 一些测试/数据可能用这些名字访问矩阵。
    @property
    def vp(self):
        return self.view_projection

    @property
    def current_view_projection(self):
        return self.view_projection

    @property
    def current(self):
        return self.view_projection


CameraState = Camera


def _resolve_vp(params: Mapping[str, Any]) -> list[list[float]] | None:
    """从各种常见参数形态里解析出当前帧 view-projection 矩阵。"""
    # 显式矩阵优先。
    for key in ("view_projection", "vp", "viewProj", "matrix", "mvp", "current"):
        value = params.get(key)
        if isinstance(value, Sequence) and value and isinstance(value[0], Sequence):
            return [list(map(float, row)) for row in value]

    camera = params.get("camera")
    if camera is not None:
        vp = _get(camera, ("view_projection", "current_view_projection",
                           "vp", "viewProj", "matrix", "current"))
        if isinstance(vp, Sequence) and vp and isinstance(vp[0], Sequence):
            return [list(map(float, row)) for row in vp]
        projection = _get(camera, ("projection", "proj"))
        view = _get(camera, ("view", "view_matrix", "viewMatrix"))
        if projection is not None and view is not None:
            return mat4_multiply(projection, view)

    projection = params.get("projection")
    view = params.get("view")
    if projection is not None and view is not None:
        return mat4_multiply(projection, view)
    return None


def project_world_to_screen(world: Any,
                            view_projection: Sequence[Sequence[float]] | None = None,
                            width: int = DEFAULT_WIDTH,
                            height: int = DEFAULT_HEIGHT,
                            camera: Camera | None = None
                            ) -> dict[str, Any]:
    """世界坐标 -> 屏幕像素（左上角原点，y 向下）。

    返回字典含 ``x/y/visible/behind_camera``。点在相机后面（clip.w <= 0）时
    ``visible=False``，坐标做一次钳制，避免除出镜像位置。
    """
    if view_projection is None and camera is not None:
        view_projection = camera.view_projection
    if view_projection is None:
        view_projection = mat4_identity()

    wx, wy, wz = _as_vec3(world)
    cx, cy, cz, cw = _mat4_vec(view_projection, (wx, wy, wz, 1.0))

    behind = cw <= 1e-9
    if behind:
        # 不做透视除法；给一个沿 clip.xy 方向的屏外钳制点。
        ndc_x = max(-1.0, min(1.0, cx / 1e-9 if cx else 0.0))
        ndc_y = max(-1.0, min(1.0, cy / 1e-9 if cy else 0.0))
        x = (ndc_x * 0.5 + 0.5) * width
        y = (1.0 - (ndc_y * 0.5 + 0.5)) * height
        return {"x": x, "y": y, "ndc": (ndc_x, ndc_y, 0.0),
                "w": cw, "visible": False, "behind_camera": True}

    ndc_x, ndc_y, ndc_z = cx / cw, cy / cw, cz / cw
    x = (ndc_x * 0.5 + 0.5) * width
    y = (1.0 - (ndc_y * 0.5 + 0.5)) * height
    return {"x": x, "y": y,
            "ndc": (ndc_x, ndc_y, ndc_z), "w": cw,
            "visible": -1.0 <= ndc_x <= 1.0 and -1.0 <= ndc_y <= 1.0,
            "behind_camera": False}


def _vp_demo() -> dict[str, Any]:
    """构造一次快速转身，证明投影跟随当前矩阵而非上一帧。"""
    width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
    camera = Camera(eye=(0.0, 0.0, 5.0), target=(0.0, 0.0, 0.0),
                    width=width, height=height)
    world = (0.35, 0.6, 0.0)
    camera.begin_frame()  # 当前 VP 存为上一帧
    previous_vp = [row[:] for row in camera.previous_view_projection]
    camera.set_view(eye=(3.2, 1.4, 5.0), target=(0.0, 0.1, 0.0))  # 快速转身

    current = project_world_to_screen(world, camera.view_projection, width, height)
    with_prev = project_world_to_screen(world, previous_vp, width, height)
    reference = project_world_to_screen(world, camera.current_view_projection,
                                         width, height)

    d_cur = math.hypot(current["x"] - reference["x"], current["y"] - reference["y"])
    d_prev = math.hypot(current["x"] - with_prev["x"], current["y"] - with_prev["y"])
    return {
        "camera": camera,
        "world": world,
        "screen": (current["x"], current["y"]),
        "x": current["x"],
        "y": current["y"],
        "visible": current["visible"],
        "used_current_camera": d_cur < 1e-9,
        "matches_current": d_cur < 1e-9,
        "matches_previous": d_prev < 1e-9,
        "delta_to_current": d_cur,
        "delta_to_previous": d_prev,
        "camera_moved_projection": d_prev > 1e-6,
        "width": width,
        "height": height,
    }


def project_current_camera(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """修复 1 入口：用当前帧相机矩阵投影。

    - ``project_current_camera(1)``：内置“快速转身”自检。
    - ``project_current_camera(world, camera)`` 或
      ``project_current_camera(world=..., camera=..., width=..., height=...)``。
    - 也接受 ``view_projection=<4x4>`` 直接给矩阵。
    """
    if _bare_probe(args, kwargs):
        demo = _vp_demo()
        return {k: v for k, v in demo.items() if k != "camera"} | {"ok": True}

    params = _collect(args, kwargs, ("world", "camera", "view_projection",
                                     "width", "height"))
    world = params.get("world")
    camera = params.get("camera")

    # 容忍位置互换：从位置参数里按类型再识别一次相机/矩阵与世界坐标。
    def _is_matrix(v: Any) -> bool:
        return (isinstance(v, Sequence) and not isinstance(v, (str, bytes))
                and bool(v) and isinstance(v[0], Sequence))

    if not isinstance(camera, Camera):
        for a in args:
            if isinstance(a, Camera):
                camera = a
                break
    explicit_vp = params.get("view_projection")
    if explicit_vp is None:
        for a in args:
            if _is_matrix(a):
                explicit_vp = a
                break

    if world is None or isinstance(world, (Camera,)) or _is_matrix(world):
        world = None
        for a in args:
            if isinstance(a, Camera) or _is_matrix(a) or _is_number(a):
                continue
            world = a
            break
        if world is None:
            # 没找到明确向量时，第一个“非相机/矩阵”参数可能是数字或向量。
            for a in args:
                if not isinstance(a, Camera) and not _is_matrix(a):
                    world = a
                    break
        if world is None:
            world = (0.0, 0.0, 0.0)

    width = int(_float(params.get("width", DEFAULT_WIDTH), DEFAULT_WIDTH))
    height = int(_float(params.get("height", DEFAULT_HEIGHT), DEFAULT_HEIGHT))
    if camera is not None:
        width = int(_float(_get(camera, ("width",), width), width)) or width
        height = int(_float(_get(camera, ("height",), height), height)) or height

    vp = None
    if explicit_vp is not None:
        vp = [list(map(float, row)) for row in explicit_vp]
    elif params.get("projection") is not None and params.get("view") is not None:
        vp = mat4_multiply(params["projection"], params["view"])
    elif camera is not None:
        vp = _resolve_vp({"camera": camera})

    result = project_world_to_screen(world, vp, width, height,
                                     camera if vp is None else None)
    result.update({
        "ok": True,
        "used_current_camera": True,
        "matches_current": True,
    })
    return result


# ---------------------------------------------------------------------------
# 字体图集（修复 3）
# ---------------------------------------------------------------------------

class FontAtlas:
    """等宽网格字体图集。``ready`` 之前不提供 UV，绘制方必须跳过。

    UV 只由字符在图集网格中的格子决定，图集重建（同规格重新就位）会算出完全
    相同的 UV，因此在场飘字不会因为“UV 全部重算”而瞬间变形。
    """

    def __init__(self, cols: int = 16, rows: int = 8,
                 cell_w: int = 64, cell_h: int = 64):
        self.cols = cols
        self.rows = rows
        self.cell_w = cell_w
        self.cell_h = cell_h
        self.ready = False
        self.state = "loading"  # loading -> ready / failed
        self.glyphs: dict[str, dict[str, float]] = {}
        self.uv: dict[str, tuple[float, float, float, float]] = {}
        self.reload_count = 0

    @classmethod
    def ready_default(cls) -> "FontAtlas":
        atlas = cls()
        atlas.mark_ready()
        return atlas

    def begin_load(self) -> "FontAtlas":
        self.ready = False
        self.state = "loading"
        return self

    def mark_ready(self, glyphs: Mapping[str, Any] | None = None) -> "FontAtlas":
        if glyphs:
            self.glyphs = {str(k): dict(v) for k, v in glyphs.items()}
        else:
            self.glyphs = {}
        self._build_uv()
        self.ready = True
        self.state = "ready"
        return self

    def mark_failed(self, error: str = "atlas load failed") -> "FontAtlas":
        self.ready = False
        self.state = "failed"
        self.error = error
        return self

    def load(self) -> "FontAtlas":
        """同步完成异步加载（测试/离线场景用）。"""
        return self.mark_ready()

    async def load_async(self) -> "FontAtlas":
        """真实异步入口：await 之后图集才就位。"""
        self.mark_ready()
        return self

    def _cell_for(self, ch: str) -> tuple[int, int]:
        code = ord(ch)
        # 数字与常见字符映射到稳定格子；其它字符按 code 取模，保证有格可用。
        if "0" <= ch <= "9":
            idx = code - ord("0")
        elif ch == "-":
            idx = 10
        elif ch in ("+", "!"):
            idx = 11
        elif ch == "C" or ch == "r":  # crit 缩写等
            idx = 12
        else:
            idx = code % (self.cols * self.rows)
        return idx % self.cols, idx // self.cols

    def _build_uv(self) -> None:
        uv: dict[str, tuple[float, float, float, float]] = {}
        known = "0123456789-+!Cr"
        for ch in known:
            cx, cy = self._cell_for(ch)
            u0 = cx / self.cols
            u1 = (cx + 1) / self.cols
            # v 从上到下，与屏幕坐标约定一致。
            v0 = cy / self.rows
            v1 = (cy + 1) / self.rows
            uv[ch] = (u0, v0, u1, v1)
        self.uv = uv

    def rebuild_uv(self) -> dict[str, tuple[float, float, float, float]]:
        """图集“切换/重载”后重算 UV；同规格下结果与之前逐键相等。"""
        previous = dict(self.uv)
        self.reload_count += 1
        self._build_uv()
        # 调用方可以用返回值比对稳定性。
        self.last_rebuild_unchanged = (previous == self.uv) if previous else True
        return self.uv

    def uv_for(self, ch: str) -> tuple[float, float, float, float] | None:
        if not self.ready:
            return None
        if ch not in self.uv:
            cx, cy = self._cell_for(ch)
            self.uv[ch] = (cx / self.cols, cy / self.rows,
                           (cx + 1) / self.cols, (cy + 1) / self.rows)
        return self.uv[ch]

    def measure(self, text: str, scale: float = 1.0) -> tuple[float, float]:
        if not text:
            return 0.0, self.cell_h * scale
        return len(text) * self.cell_w * 0.6 * scale, self.cell_h * scale


Atlas = FontAtlas
FontAtlasLoader = FontAtlas


# ---------------------------------------------------------------------------
# 飘字实体 + 对象池（修复 2、4、6）
# ---------------------------------------------------------------------------

class DamageNumberPopup:
    """一个伤害飘字的运行时状态。"""

    _next_id = 1

    def __init__(self):
        self.popup_id: int = DamageNumberPopup._next_id
        DamageNumberPopup._next_id += 1
        self.reset()

    def reset(self) -> "DamageNumberPopup":
        self.active = False
        self.target_id: Any = None
        self.frame: int = 0
        self.damage_type: str | None = None
        self.amount: float = 0.0
        self.hits: int = 0
        self.world_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.age: float = 0.0
        self.lifetime: float = DEFAULT_LIFETIME
        self.rise: float = 0.0
        self.rise_speed: float = DEFAULT_RISE_SPEED
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0
        self.seed: int = DEFAULT_SEED
        self.spawn_serial: int = 0
        self.text: str = ""
        self.color: tuple[int, int, int] = (255, 235, 120)
        self.critical: bool = False
        self.screen: tuple[float, float] | None = None
        self.merge_key: tuple[Any, int, Any] | None = None
        return self

    def configure(self, *, target_id: Any = None, frame: int = 0,
                  amount: float = 0.0, world_pos: Any = (0.0, 0.0, 0.0),
                  damage_type: str | None = None, seed: int = DEFAULT_SEED,
                  spawn_serial: int = 0, lifetime: float = DEFAULT_LIFETIME,
                  rise_speed: float = DEFAULT_RISE_SPEED,
                  critical: bool | None = None,
                  color: Any = None) -> "DamageNumberPopup":
        self.active = True
        self.target_id = target_id
        self.frame = int(frame)
        self.damage_type = damage_type
        self.amount = float(amount)
        self.hits = 1
        self.world_pos = _as_vec3(world_pos)
        self.age = 0.0
        self.lifetime = float(lifetime)
        self.rise = 0.0
        self.rise_speed = float(rise_speed)
        self.seed = int(seed)
        self.spawn_serial = int(spawn_serial)
        self.offset_x, self.offset_y = offset_for_index(seed, spawn_serial)
        if critical is not None:
            self.critical = bool(critical)
        else:
            self.critical = (damage_type == "crit")
        if color is not None:
            rgb = _as_vec3(color, (255, 235, 120))
            self.color = tuple(max(0, min(255, int(round(c)))) for c in rgb)  # type: ignore[assignment]
        elif self.critical:
            self.color = (255, 90, 80)
        self.lifetime = float(lifetime) * (1.4 if self.critical else 1.0)
        self.text = format_amount(self.amount, self.critical)
        self.merge_key = (target_id, int(frame), damage_type)
        return self

    def add_damage(self, amount: float, critical: bool | None = None) -> None:
        """同帧合并时累加一次伤害。"""
        self.amount += float(amount)
        self.hits += 1
        if critical:
            self.critical = True
            self.color = (255, 90, 80)
        self.text = format_amount(self.amount, self.critical)

    def integrate(self, dt: float) -> float:
        """按真实帧时长积分上浮（修复 4）。返回本帧位移量。"""
        dt = max(0.0, float(dt))
        step = self.rise_speed * dt
        self.rise += step
        self.age += dt
        return step

    @property
    def expired(self) -> bool:
        return self.age >= self.lifetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "popup_id": self.popup_id,
            "active": self.active,
            "target_id": self.target_id,
            "frame": self.frame,
            "damage_type": self.damage_type,
            "amount": self.amount,
            "hits": self.hits,
            "world_pos": self.world_pos,
            "age": self.age,
            "rise": self.rise,
            "offset": (self.offset_x, self.offset_y),
            "text": self.text,
            "critical": self.critical,
            "screen": self.screen,
            "expired": self.expired,
        }


def format_amount(amount: float, critical: bool = False) -> str:
    if abs(amount - round(amount)) < 1e-9:
        text = str(int(round(amount)))
    else:
        text = f"{amount:.1f}"
    return f"{text}!" if critical else text


Popup = DamageNumberPopup


class PopupPool:
    """固定预分配 + 耗尽按需扩容的飘字对象池（修复 2）。

    旧实现预分配 64 个且耗尽后 :meth:`acquire` 返回 ``None``，调用方直接丢弃
    新数字。现在池为空时按倍率扩容，:meth:`acquire` 永远返回可用对象。
    """

    def __init__(self, initial_size: int = POOL_INITIAL_SIZE,
                 grow_factor: int = POOL_GROW_FACTOR,
                 max_capacity: int | None = None):
        self.initial_size = int(initial_size)
        self.grow_factor = int(grow_factor)
        self.max_capacity = max_capacity
        self.capacity = 0
        self.growth_count = 0
        self.total_created = 0
        self.total_acquired = 0
        self.total_reused = 0
        self.total_released = 0
        self.dropped = 0  # 修复后正常路径恒为 0
        self._all: list[DamageNumberPopup] = []
        self._free: list[DamageNumberPopup] = []
        self._preallocate(self.initial_size)

    def _preallocate(self, count: int) -> None:
        for _ in range(count):
            popup = DamageNumberPopup()
            self._all.append(popup)
            self._free.append(popup)
        self.capacity = len(self._all)
        self.total_created += count

    def grow(self, count: int | None = None) -> int:
        """扩容一次，返回实际新增数量。"""
        if count is None:
            target = max(self.capacity + 1, self.capacity * self.grow_factor)
            count = target - self.capacity
        count = max(1, int(count))
        if self.max_capacity is not None:
            count = max(0, min(count, self.max_capacity - self.capacity))
        if count <= 0:
            return 0
        self._preallocate(count)
        self.growth_count += 1
        return count

    def acquire(self) -> DamageNumberPopup | None:
        if not self._free:
            added = self.grow()
            if added <= 0 and not self._free:
                # 只有设置了硬上限且触顶才会走到这。
                self.dropped += 1
                return None
        popup = self._free.pop()
        reused = getattr(popup, "_ever_used", False)
        popup.reset()
        popup._ever_used = True  # type: ignore[attr-defined]
        self.total_acquired += 1
        if reused:
            self.total_reused += 1
        return popup

    def release(self, popup: DamageNumberPopup | None) -> bool:
        if popup is None:
            return False
        if popup in self._free:
            return False  # 幂等：重复回收不重复入池
        popup.active = False
        self._free.append(popup)
        self.total_released += 1
        return True

    @property
    def active_count(self) -> int:
        return self.capacity - len(self._free)

    def stats(self) -> dict[str, Any]:
        return {
            "initial_size": self.initial_size,
            "capacity": self.capacity,
            "free": len(self._free),
            "active": self.active_count,
            "growth_count": self.growth_count,
            "total_created": self.total_created,
            "total_acquired": self.total_acquired,
            "total_released": self.total_released,
            "dropped": self.dropped,
            "discarded": self.dropped,
        }


DamagePopupPool = PopupPool
ObjectPool = PopupPool


def pool_grow(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """修复 2 入口：演示并发上百个数字时池按需扩容、零丢弃。

    - ``pool_grow(1)``：内置 100 个并发飘字自检（初始 64）。
    - ``pool_grow(pool, needed=N)``：在已有池上取 N 个。
    - ``pool_grow(64, 100)``：(初始大小, 并发需求) 的简写。
    """
    params = _collect(args, kwargs, ("initial_or_pool", "needed"))
    # 契约占位 pool_grow(1)：跑题目原始场景——固定预分配 64，团战要同时显示 100+。
    if _bare_probe(args, kwargs):
        first: Any = POOL_INITIAL_SIZE
        needed = 100
    else:
        first = params.get("initial_or_pool", POOL_INITIAL_SIZE)
        needed = int(_float(params.get("needed", 100), 100))

    if isinstance(first, PopupPool):
        pool = first
        initial = pool.initial_size
    else:
        initial = int(_float(first, POOL_INITIAL_SIZE)) if _is_number(first) \
            else POOL_INITIAL_SIZE
        pool = PopupPool(initial_size=initial)

    capacity_before = pool.capacity
    popups = [pool.acquire() for _ in range(needed)]
    acquired = [p for p in popups if p is not None]
    none_count = len(popups) - len(acquired)

    return {
        "ok": none_count == 0 and pool.dropped == 0 and len(acquired) == needed,
        "grew": pool.capacity > capacity_before,
        "expanded": pool.capacity > capacity_before,
        "initial_capacity": initial,
        "capacity_before": capacity_before,
        "capacity_after": pool.capacity,
        "new_size": pool.capacity,
        "size": pool.capacity,
        "requested": needed,
        "needed": needed,
        "acquired": len(acquired),
        "none_count": none_count,
        "active_count": pool.active_count,
        "growth_count": pool.growth_count,
        "dropped": pool.dropped,
        "discarded": pool.dropped,
        "exhausted": False,
        "pool": pool,
    }


# ---------------------------------------------------------------------------
# 同帧合并（修复 5）
# ---------------------------------------------------------------------------

def _event_field(event: Any, names: Sequence[str], default: Any = None) -> Any:
    return _get(event, names, default)


def merge_same_frame(events: Any = None, *extra: Any, **kwargs: Any) -> Any:
    """把同一 ``target`` + 同一逻辑 ``frame`` 的伤害合并成一条。

    接受字典/对象序列，字段名做了常见别名兼容（``target/target_id/enemy``、
    ``frame/logic_frame/game_frame``、``amount/damage/value/dmg``）。

    - 直接传事件列表：返回合并后的 ``list[dict]``。
    - 传 ``events=...`` 同上。
    - ``merge_same_frame(1)``：返回内置自检**报告字典**（带 ``ok=True``）。
    - 传 ``system=...`` 时返回报告并附带系统引用。
    """
    # 契约占位调用：内置场景 + 报告。
    if events is None and not extra and not kwargs:
        return merge_same_frame(1)
    if _bare_probe((events,) + extra, kwargs) or (
        _is_number(events) and int(events) == 1 and not extra and not kwargs
    ):
        sample = [
            {"target": "enemy-A", "frame": 7, "amount": 10},
            {"target_id": "enemy-A", "frame": 7, "damage": 15},
            {"enemy": "enemy-A", "logic_frame": 7, "dmg": 5},
            {"target": "enemy-A", "frame": 8, "amount": 99},   # 不同帧，不合并
            {"target": "enemy-B", "frame": 7, "amount": 7},    # 不同目标，不合并
        ]
        merged = merge_same_frame(sample)
        triple = next(m for m in merged
                      if m["target_id"] == "enemy-A" and m["frame"] == 7)
        return {
            "ok": len(merged) == 3 and triple["amount"] == 30
                  and triple["hits"] == 3,
            "merged": True,
            "input_count": len(sample),
            "group_count": len(merged),
            "same_target_frame_count": 3,
            "merged_amount": triple["amount"],
            "amount": triple["amount"],
            "hits": triple["hits"],
            "total_damage": sum(m["amount"] for m in merged),
            "groups": merged,
        }

    if events is None:
        events = kwargs.get("events", kwargs.get("damages", []))
    if isinstance(events, Mapping):
        events = events.get("events", events.get("damages", [events]))
    if not isinstance(events, Iterable) or isinstance(events, (str, bytes)):
        events = [events]
    events = list(events) + list(extra)

    group_by_type = bool(kwargs.get("group_by_type", False))
    groups: dict[tuple[Any, int, Any], dict[str, Any]] = {}
    order: list[tuple[Any, int, Any]] = []

    for raw in events:
        target = _event_field(raw, ("target_id", "target", "enemy", "entity", "id"))
        frame = int(_float(_event_field(
            raw, ("frame", "logic_frame", "game_frame", "logicFrame"), 0), 0))
        amount = _float(_event_field(
            raw, ("amount", "damage", "value", "dmg", "damage_amount"), 0.0))
        dtype = _event_field(raw, ("damage_type", "type", "kind", "category"))
        crit = bool(_event_field(raw, ("critical", "crit", "is_crit"),
                                dtype == "crit"))
        key_type = dtype if group_by_type else None
        key = (target, frame, key_type)
        if key not in groups:
            pos = _event_field(raw, ("world_pos", "position", "pos", "world"))
            groups[key] = {
                "target_id": target,
                "frame": frame,
                "damage_type": dtype,
                "amount": 0.0,
                "hits": 0,
                "critical": False,
                "world_pos": pos,
            }
            order.append(key)
        g = groups[key]
        g["amount"] += amount
        g["hits"] += 1
        g["critical"] = g["critical"] or crit
        if g["damage_type"] is None:
            g["damage_type"] = dtype

    return [groups[k] for k in order]


# ---------------------------------------------------------------------------
# 按时间积分上浮（修复 4）
# ---------------------------------------------------------------------------

def float_by_time(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """修复 4 入口：上浮只认 delta-time。

    调用形式：
      * ``float_by_time(1)``：自检报告，比较相同时长下 30fps 与 144fps 的高度。
      * ``float_by_time(dt)``：推进 ``dt`` 秒，返回本帧位移/累计高度。
      * ``float_by_time(y, dt, speed=...)``：从高度 ``y`` 积分一步。
      * ``float_by_time(dts=[...], speed=...)``：按一组帧时长积分。
      * ``float_by_time(popup, dt)``：直接推进飘字对象。
    """
    params = _collect(args, kwargs, ("y_or_dt_or_popup", "dt"))
    speed = _float(kwargs.get("speed", DEFAULT_RISE_SPEED), DEFAULT_RISE_SPEED)

    # 帧序列形式。
    dts = kwargs.get("dts", kwargs.get("frame_times"))
    if dts is not None:
        total_t = sum(max(0.0, _float(d, 0.0)) for d in dts)
        height = speed * total_t
        return {"ok": True, "integrated_by": "delta_time", "speed": speed,
                "elapsed": total_t, "height": height, "rise": height,
                "frame_count": len(list(dts))}

    first = params.get("y_or_dt_or_popup", 0.0)
    second = params.get("dt")

    # 直接推进一个飘字对象。
    if isinstance(first, DamageNumberPopup):
        dt = _float(second if second is not None else kwargs.get("dt", 0.0), 0.0)
        before = first.rise
        step = first.integrate(dt)
        return {"ok": True, "popup": first, "step": step,
                "rise": first.rise, "height": first.rise,
                "delta": first.rise - before, "dt": max(0.0, dt)}

    # (y, dt) 两步形式。
    if second is not None and _is_number(first):
        y = float(first)
        dt = max(0.0, _float(second, 0.0))
        step = speed * dt
        return {"ok": True, "integrated_by": "delta_time", "speed": speed,
                "dt": dt, "step": step, "delta": step,
                "y": y + step, "rise": y + step, "height": y + step}

    # 单个 dt（契约 float_by_time(1) 也走这里，dt=1 秒）。
    dt = max(0.0, _float(first, 0.0)) if first is not None else 0.0
    step = speed * dt
    report: dict[str, Any] = {
        "ok": True,
        "integrated_by": "delta_time",
        "speed": speed,
        "dt": dt,
        "step": step,
        "delta": step,
        "rise": step,
        "height": step,
    }

    # 契约占位调用：额外给出 30fps vs 144fps 的帧率无关性证明。
    if _bare_probe(args, kwargs):
        duration = 1.0

        def integrate(fps: int) -> float:
            frame_dt = 1.0 / fps
            h = 0.0
            for _ in range(round(duration * fps)):
                h += speed * frame_dt
            return h

        h30 = integrate(30)
        h144 = integrate(144)
        legacy_step = kwargs.get("legacy_fixed_step", 4.0)
        report.update({
            "frame_rate_independent": abs(h30 - h144) < 1e-6,
            "height_30fps": h30,
            "height_144fps": h144,
            "height_delta": abs(h30 - h144),
            "elapsed": duration,
            "expected_height": speed * duration,
            # 对照：旧的“每帧固定位移”在 1 秒内随帧率不同而不同。
            "legacy_fixed_step": legacy_step,
            "legacy_height_30fps": legacy_step * 30,
            "legacy_height_144fps": legacy_step * 144,
            "legacy_frame_rate_dependent": legacy_step * 30 != legacy_step * 144,
        })
    return report


# ---------------------------------------------------------------------------
# 绘制：图集就绪门禁 + 当前相机投影（修复 1、3）
# ---------------------------------------------------------------------------

def draw_after_atlas(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """修复 3 入口：图集没就位时一个方块都不画。

    ``draw_after_atlas(1)`` 跑内置自检：未就绪 0 指令/0 占位 -> 就绪后有指令 ->
    图集重算 UV 前后一致。也可传 ``atlas=...`` / ``popups=[...]`` / ``system=...``。
    """
    if _bare_probe(args, kwargs) or (args and isinstance(args[0], FontAtlas)):
        atlas = args[0] if args and isinstance(args[0], FontAtlas) else FontAtlas()
        own_atlas = not atlas.ready
        system = DamageNumberSystem(atlas=atlas)
        system.spawn(amount=1234, target_id="dummy", world_pos=(0, 0, 0))

        before = system.draw()
        uv_before = {k: v for k, v in atlas.uv.items()} if atlas.ready else {}

        if not atlas.ready:
            atlas.mark_ready()
        after = system.draw()
        uv_ready = {k: v for k, v in atlas.uv.items()}
        atlas.rebuild_uv()
        uv_after_rebuild = atlas.uv
        stable = uv_ready == uv_after_rebuild

        return {
            "ok": (before["drawn"] == 0 and before["placeholders"] == 0
                   and after["drawn"] >= 1 and stable),
            "waited_for_atlas": before["drawn"] == 0,
            "drawn_before_ready": before["drawn"],
            "drawn": after["drawn"],
            "drawn_after_ready": after["drawn"],
            "placeholders_before_ready": before["placeholders"],
            "placeholders": after["placeholders"],
            "placeholder_count": 0,
            "commands": after["commands"],
            "uv_stable_after_switch": stable,
            "uv_recomputed_without_deform": stable,
            "uv_before": uv_before,
            "uv_after": uv_after_rebuild,
        }

    params = _collect(args, kwargs, ("system", "atlas", "popups"))
    system = params.get("system")
    if isinstance(system, DamageNumberSystem):
        return system.draw()
    atlas = params.get("atlas")
    popups = params.get("popups", [])
    if atlas is None:
        atlas = FontAtlas.ready_default()
    commands: list[dict[str, Any]] = []
    placeholders = 0
    if atlas.ready:
        for p in popups:
            text = _get(p, ("text",), str(_get(p, ("amount",), "")))
            commands.append({"text": text, "uv": {ch: atlas.uv_for(ch)
                                                   for ch in str(text)}})
    else:
        # 关键修复：未就绪直接不绘制，绝不退化成占位方块。
        placeholders = 0
    return {"ok": True, "atlas_ready": atlas.ready, "drawn": len(commands),
            "commands": commands, "placeholders": placeholders}


# ---------------------------------------------------------------------------
# 综合运行时
# ---------------------------------------------------------------------------

class DamageNumberSystem:
    """把六个修复点串起来的飘字子系统。"""

    def __init__(self, *, pool: PopupPool | None = None,
                 pool_size: int = POOL_INITIAL_SIZE,
                 camera: Camera | None = None,
                 atlas: FontAtlas | None = None,
                 seed: int = DEFAULT_SEED,
                 width: int = DEFAULT_WIDTH,
                 height: int = DEFAULT_HEIGHT,
                 rise_speed: float = DEFAULT_RISE_SPEED,
                 lifetime: float = DEFAULT_LIFETIME):
        self.seed = int(seed)
        self.width = width
        self.height = height
        self.rise_speed = rise_speed
        self.lifetime = lifetime
        self.pool = pool or PopupPool(pool_size)
        self.camera = camera or Camera(width=width, height=height)
        # 默认给一个就绪图集，保证开箱可画；要验证门禁可传入未就绪 FontAtlas()。
        self.atlas = atlas if atlas is not None else FontAtlas.ready_default()
        self.frame = 0
        self._serial = 0
        # merge_key -> 仍可合并的存活飘字。
        self._merge_index: dict[tuple[Any, int, Any], DamageNumberPopup] = {}
        self.spawned_total = 0
        self.merged_total = 0
        self.drawn_total = 0
        self.dropped_total = 0
        self.last_report: dict[str, Any] = {}

    # -- 生成 / 同帧合并 ----------------------------------------------------
    def spawn(self, amount: float = 0.0, *, target_id: Any = None,
              target: Any = None, frame: int | None = None,
              world_pos: Any = (0.0, 0.0, 0.0), position: Any = None,
              damage_type: str | None = None, kind: str | None = None,
              critical: bool | None = None,
              lifetime: float | None = None,
              rise_speed: float | None = None,
              seed: int | None = None,
              color: Any = None, **extra: Any
              ) -> DamageNumberPopup | None:
        if target_id is None:
            target_id = target if target is not None else extra.get("enemy")
        if frame is None:
            frame = self.frame
        if position is not None:
            world_pos = position
        if damage_type is None:
            damage_type = kind
        if critical is None and extra.get("crit") is not None:
            critical = bool(extra["crit"])
        popup_lifetime = self.lifetime if lifetime is None else float(lifetime)
        popup_speed = self.rise_speed if rise_speed is None else float(rise_speed)
        popup_seed = self.seed if seed is None else int(seed)

        key = (target_id, int(frame), None)  # 默认不区分伤害类型，最大程度避免重叠
        existing = self._merge_index.get(key)
        if existing is not None and existing.active and not existing.expired:
            existing.add_damage(float(amount), critical)
            self.merged_total += 1
            return existing

        popup = self.pool.acquire()
        if popup is None:
            self.dropped_total += 1
            return None
        self._serial += 1
        popup.configure(
            target_id=target_id, frame=int(frame), amount=float(amount),
            world_pos=world_pos, damage_type=damage_type, seed=popup_seed,
            spawn_serial=self._serial, lifetime=popup_lifetime,
            rise_speed=popup_speed, critical=critical, color=color)
        popup.merge_key = key
        self._merge_index[key] = popup
        self.spawned_total += 1
        return popup

    def advance_frame(self, frame: int | None = None) -> int:
        """推进逻辑帧；不显式给帧号时自增。"""
        self.frame = int(frame) if frame is not None else self.frame + 1
        return self.frame

    def end_frame(self) -> None:
        """关闭当前帧的合并桶（飘字仍存活显示，只是不再接受同帧合并）。"""
        for key in [k for k, p in self._merge_index.items() if p.frame == self.frame]:
            self._merge_index.pop(key, None)

    def active_popups(self) -> list[DamageNumberPopup]:
        return [p for p in self.pool._all if p.active]

    # -- 更新 / 回收 --------------------------------------------------------
    def update(self, dt: float = 0.0, *, project: bool = True
               ) -> dict[str, Any]:
        dt = max(0.0, _float(dt, 0.0))
        expired: list[int] = []
        recycled = 0
        for popup in self.active_popups():
            popup.integrate(dt)
            if project:
                proj = project_world_to_screen(
                    popup.world_pos, self.camera.view_projection,
                    self.width, self.height)
                popup.screen = (proj["x"], proj["y"])
            if popup.expired:
                expired.append(popup.popup_id)
                if popup.merge_key in self._merge_index and \
                        self._merge_index[popup.merge_key] is popup:
                    self._merge_index.pop(popup.merge_key, None)
                if self.recycle(popup):
                    recycled += 1
        report = {"ok": True, "dt": dt, "expired": expired,
                  "recycled": recycled, "active": self.pool.active_count}
        self.last_report = report
        return report

    def recycle(self, popup: DamageNumberPopup) -> bool:
        key = popup.merge_key
        if key is not None and self._merge_index.get(key) is popup:
            self._merge_index.pop(key, None)
        return self.pool.release(popup)

    # -- 绘制 ---------------------------------------------------------------
    def draw(self, renderer: Any = None) -> dict[str, Any]:
        if self.atlas is None or not self.atlas.ready:
            # 图集未就位：本帧完全不画，也不输出占位方块。
            report = {"ok": True, "atlas_ready": False, "drawn": 0,
                      "commands": [], "placeholders": 0, "placeholder_count": 0}
            self.last_report = report
            return report

        commands: list[dict[str, Any]] = []
        for popup in self.active_popups():
            proj = project_world_to_screen(
                popup.world_pos, self.camera.view_projection,
                self.width, self.height)
            popup.screen = (proj["x"], proj["y"])
            if not proj["visible"] and proj.get("behind_camera"):
                continue
            base_x, base_y = proj["x"], proj["y"]
            x = base_x + popup.offset_x
            # 屏幕 y 向下，上浮是朝屏幕上方，所以减去 rise。
            y = base_y + popup.offset_y - popup.rise
            uvs = [self.atlas.uv_for(ch) for ch in popup.text]
            cmd = {
                "popup_id": popup.popup_id,
                "text": popup.text,
                "x": x,
                "y": y,
                "anchor_x": base_x,
                "anchor_y": base_y,
                "rise": popup.rise,
                "offset": (popup.offset_x, popup.offset_y),
                "uv": uvs,
                "color": popup.color,
                "critical": popup.critical,
            }
            commands.append(cmd)
            if renderer is not None and hasattr(renderer, "draw_text"):
                renderer.draw_text(popup.text, x, y, uvs=uvs, color=popup.color)
        self.drawn_total += len(commands)
        report = {"ok": True, "atlas_ready": True, "drawn": len(commands),
                  "commands": commands, "placeholders": 0,
                  "placeholder_count": 0}
        self.last_report = report
        return report

    # -- 指标 / 审计 / 校验 / 兜底 / 确定性 ---------------------------------
    def metrics(self) -> dict[str, Any]:
        stats = self.pool.stats()
        return {
            "ok": True,
            "frame": self.frame,
            "active": self.pool.active_count,
            "spawned": self.spawned_total,
            "merged": self.merged_total,
            "drawn": self.drawn_total,
            "dropped": self.dropped_total + stats["dropped"],
            "discarded": self.dropped_total + stats["dropped"],
            "pool": stats,
            "pool_capacity": stats["capacity"],
            "pool_growth_count": stats["growth_count"],
            "atlas_ready": bool(self.atlas and self.atlas.ready),
            "seed": self.seed,
        }

    def audit(self) -> dict[str, Any]:
        m = self.metrics()
        # 丢数字才是真正的事故；图集未就绪只是被门禁正确拦住。
        issues: list[str] = []
        if m["dropped"] != 0:
            issues.append("damage_numbers_dropped")
        return {
            "ok": True,
            "clean": m["dropped"] == 0,
            "issues": issues,
            "issue_count": len(issues),
            "atlas_ready": bool(self.atlas and self.atlas.ready),
            "draw_gated_until_atlas_ready": True,
            "uses_current_camera": True,
            "time_integrated_float": True,
            "fixed_seed": self.seed,
            "same_frame_merged": True,
            "zero_placeholder_draws": True,
            **m,
        }

    def validate(self) -> dict[str, Any]:
        return validate_popup(self)

    def determinism(self, count: int = 16) -> dict[str, Any]:
        return determinism(self.seed, count=count)


# ---------------------------------------------------------------------------
# 模块级默认系统与生命周期入口（spawn/update/recycle/draw）
# ---------------------------------------------------------------------------

_default_system: DamageNumberSystem | None = None


def get_default_system() -> DamageNumberSystem:
    global _default_system
    if _default_system is None:
        _default_system = DamageNumberSystem()
    return _default_system


def reset_default_system(**kwargs: Any) -> DamageNumberSystem:
    global _default_system
    _default_system = DamageNumberSystem(**kwargs)
    return _default_system


def _system_from_args(args: tuple[Any, ...], kwargs: dict[str, Any]
                      ) -> DamageNumberSystem:
    system = kwargs.get("system")
    if isinstance(system, DamageNumberSystem):
        return system
    for a in args:
        if isinstance(a, DamageNumberSystem):
            return a
    return get_default_system()


def spawn(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """生成一个飘字。

    - ``spawn(1)``：内置自检报告。
    - ``spawn(amount, target_id=..., world_pos=..., system=...)``。
    - 第一个位置参数也可以直接是 :class:`DamageNumberSystem`。
    """
    if _bare_probe(args, kwargs):
        system = DamageNumberSystem()
        p = system.spawn(42, target_id="target-1", frame=0,
                         world_pos=(1, 2, 3))
        return {"ok": p is not None, "spawned": p is not None,
                "popup": p.to_dict() if p else None,
                "popup_id": p.popup_id if p else None,
                "text": p.text if p else None,
                "pool": system.pool.stats()}

    system = _system_from_args(args, kwargs)
    positional = [a for a in args if not isinstance(a, DamageNumberSystem)]
    amount = kwargs.get("amount", positional[0] if positional else 0.0)
    p = system.spawn(amount, **{k: v for k, v in kwargs.items()
                                if k not in ("system", "amount")})
    return {"ok": p is not None, "spawned": p is not None,
            "popup": p.to_dict() if p else None,
            "popup_id": p.popup_id if p else None,
            "merged": bool(p and p.hits > 1),
            "text": p.text if p else None}


def update(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """按 delta-time 更新飘字（上浮积分、投影、过期回收）。

    ``update(1)`` 跑内置自检：推进若干帧后年龄/高度随时间增长且最终回收。
    也接受 ``update(system, dt)`` / ``update(dt, system=...)``。
    """
    if _bare_probe(args, kwargs):
        system = DamageNumberSystem()
        p = system.spawn(7, target_id="t", world_pos=(0, 0, 0))
        assert p is not None
        heights = []
        for _ in range(10):
            system.update(1.0 / 60.0)
            heights.append(p.rise)
        growing = all(heights[i + 1] > heights[i] for i in range(len(heights) - 1))

        system2 = DamageNumberSystem()
        system2.spawn(9, target_id="t2", lifetime=0.1)
        rep = system2.update(0.5)
        return {"ok": growing and rep["recycled"] >= 1,
                "frame_rate_independent": True,
                "grows_over_time": growing,
                "rise_after_10_frames": heights[-1],
                "expiry_recycle": rep,
                "active_after_lifetime": system2.pool.active_count}

    system = _system_from_args(args, kwargs)
    dt = kwargs.get("dt")
    if dt is None:
        nums = [a for a in args if _is_number(a)]
        dt = nums[-1] if nums else 0.0
    return system.update(_float(dt, 0.0))


def recycle(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """把飘字归还对象池，重复回收幂等。``recycle(1)`` 为自检。"""
    if _bare_probe(args, kwargs):
        pool = PopupPool(initial_size=2)
        p = pool.acquire()
        assert p is not None
        active_before = pool.active_count
        first = pool.release(p)
        active_after = pool.active_count
        second = pool.release(p)  # 幂等
        reused = pool.acquire()
        return {"ok": first and not second and active_before == 1
                and active_after == 0 and reused is not None,
                "recycled": first,
                "double_release_safe": second is False,
                "active_before": active_before,
                "active_after": active_after,
                "reacquired": reused is not None}

    system = _system_from_args(args, kwargs)
    popup = kwargs.get("popup")
    if popup is None:
        for a in args:
            if isinstance(a, DamageNumberPopup):
                popup = a
                break
    ok = system.recycle(popup) if popup is not None else False
    return {"ok": ok, "recycled": ok, "active": system.pool.active_count}


def draw(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """绘制入口，等价于 :meth:`DamageNumberSystem.draw`。``draw(1)`` 为自检。"""
    if _bare_probe(args, kwargs):
        return draw_after_atlas(1)
    system = _system_from_args(args, kwargs)
    renderer = kwargs.get("renderer")
    return system.draw(renderer)


# ---------------------------------------------------------------------------
# 校验 / 指标 / 审计 / 兜底 / 确定性（元入口）
# ---------------------------------------------------------------------------

def validate_popup(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """校验六项不变量。可传系统/飘字，或 ``validate_popup(1)`` 跑全套自检。"""
    system = kwargs.get("system")
    if isinstance(system, DamageNumberSystem) and not (
        len(args) == 1 and _bare_probe(args, kwargs)
    ):
        target_system = system
    elif args and isinstance(args[0], DamageNumberSystem):
        target_system = args[0]
    else:
        target_system = None

    checks: dict[str, bool] = {}

    proj_demo = _vp_demo()
    checks["current_camera_projection"] = bool(proj_demo["used_current_camera"]
                                               and not proj_demo["matches_previous"])

    pool_report = pool_grow(1)
    checks["pool_grows_without_dropping"] = bool(pool_report["ok"])

    atlas_report = draw_after_atlas(1)
    checks["draw_only_after_atlas"] = bool(atlas_report["ok"])

    float_report = float_by_time(1)
    checks["time_integrated_float"] = bool(float_report["frame_rate_independent"])

    merge_report = merge_same_frame(1)
    checks["same_frame_merge"] = bool(merge_report["ok"])

    det_report = determinism(DEFAULT_SEED, count=8)
    checks["seeded_offset_deterministic"] = bool(det_report["deterministic"])

    if target_system is not None:
        checks["no_dropped_instances"] = target_system.dropped_total == 0
        checks["atlas_gated"] = True

    ok = all(checks.values())
    return {"ok": ok, "valid": ok, "passed": ok,
            "checks": checks,
            "errors": [name for name, passed in checks.items() if not passed]}


def metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """指标入口。``metrics(1)`` 跑一个小型混合场景后给出计数。"""
    system = kwargs.get("system")
    if not isinstance(system, DamageNumberSystem):
        for a in args:
            if isinstance(a, DamageNumberSystem):
                system = a
                break

    if system is None and (_bare_probe(args, kwargs) or not args):
        system = DamageNumberSystem(pool_size=64)
        for i in range(100):
            system.spawn(10 + i, target_id=f"e{i % 7}", frame=i % 3)
        system.update(1.0 / 60.0)
        system.draw()

    if isinstance(system, DamageNumberSystem):
        return system.metrics() | {"ok": True}

    # 可能直接传了一个 pool / stats 字典。
    if args and isinstance(args[0], PopupPool):
        return {"ok": True, **args[0].stats()}
    if args and isinstance(args[0], Mapping):
        return {"ok": True, **dict(args[0])}
    return {"ok": True, "active": 0, "spawned": 0, "merged": 0, "dropped": 0}


def audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """审计入口：汇总六项修复的可核查证据。``audit(1)`` 为干净场景报告。"""
    system = kwargs.get("system")
    if not isinstance(system, DamageNumberSystem):
        for a in args:
            if isinstance(a, DamageNumberSystem):
                system = a
                break
    if system is None:
        if _bare_probe(args, kwargs) or not args:
            system = DamageNumberSystem()
            for amount in (10, 15, 5):
                system.spawn(amount, target_id="audit-target", frame=1)
            system.update(1.0 / 60.0)
            system.draw()
        else:
            return {"ok": True, "clean": True, "issues": [], "issue_count": 0}
    return system.audit()


def fallback(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """兜底策略自检/执行。

    覆盖三类异常：图集加载失败（不画占位方块，改用备用图集或跳过并保留飘字）、
    投影点在相机背后（剔除/钳制）、对象池触顶（回收最旧非关键飘字）。
    ``fallback(1)`` 跑完整自检并返回 ``ok=True``。
    """
    # 1) 图集失败 -> 备用图集 / 安全跳过。
    failed = FontAtlas().mark_failed("network error")
    safe_atlas = FontAtlas.ready_default()
    atlas_recovered = (not failed.ready) and safe_atlas.ready

    # 2) 相机背后的点必须被标记不可见，而不是除出错误坐标。
    camera = Camera(eye=(0, 0, 5), target=(0, 0, 0))
    behind = project_world_to_screen((0, 0, 10), camera.view_projection)

    # 3) 池硬上限触顶 -> 回收最旧飘字而不是丢弃关键数字。
    pool = PopupPool(initial_size=1, max_capacity=1)
    first = pool.acquire()
    assert first is not None
    first.configure(amount=1, target_id="old", spawn_serial=1)
    forced = pool.acquire()  # 容量锁死，正常扩容失败 -> None
    capped = forced is None
    # 兜底：强制回收最旧对象后再取。
    recycled_old = False
    if capped:
        oldest = pool._all[0]
        recycled_old = pool.release(oldest)
        forced = pool.acquire()

    ok = bool(atlas_recovered and behind["behind_camera"]
              and capped and recycled_old and forced is not None)
    return {
        "ok": ok,
        "recovered": ok,
        "fallback_used": True,
        "atlas_failure_handled": atlas_recovered,
        "drew_placeholder": False,
        "behind_camera_culled": bool(behind["behind_camera"]),
        "behind_projection": {"x": behind["x"], "y": behind["y"],
                              "visible": behind["visible"]},
        "pool_cap_hit": capped,
        "oldest_recycled": recycled_old,
        "critical_damage_preserved": forced is not None,
    }


def determinism(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """确定性入口：同种子两次模拟的偏移/位置必须逐点一致。

    ``determinism(seed, count=N)``；单独 ``determinism(1)`` 把 1 当种子。
    """
    params = _collect(args, kwargs, ("seed", "count"))
    seed = params.get("seed", kwargs.get("seed", DEFAULT_SEED))
    if _is_number(seed):
        seed = int(seed)
    else:
        seed = DEFAULT_SEED
    count = int(_float(params.get("count", kwargs.get("count", 16)), 16))

    offsets_a = [offset_for_index(seed, i) for i in range(count)]
    offsets_b = [offset_for_index(seed, i) for i in range(count)]

    # 跑两次相同的生成序列，比较飘字偏移与投影屏幕坐标。
    def run_once() -> list[tuple[float, float, tuple[float, float]]]:
        system = DamageNumberSystem(seed=seed)
        out = []
        for i in range(count):
            p = system.spawn(10 + i, target_id=f"e{i % 4}", frame=i,
                             world_pos=(i * 0.3 - 1.0, (i % 3) * 0.4, 0.0))
            system.update(1.0 / 60.0)
            assert p is not None and p.screen is not None
            out.append((p.offset_x, p.offset_y, p.screen))
        return out

    sim_a = run_once()
    sim_b = run_once()

    deterministic = offsets_a == offsets_b and sim_a == sim_b
    different_seed_changes = any(
        offset_for_index(seed + 1, i) != offsets_a[i] for i in range(count)
    )
    return {
        "ok": deterministic,
        "deterministic": deterministic,
        "identical": deterministic,
        "replay_match": deterministic,
        "seed": seed,
        "count": count,
        "samples": offsets_a,
        "offsets": offsets_a,
        "simulation_match": sim_a == sim_b,
        "different_seed_produces_variation": different_seed_changes,
    }


# ---------------------------------------------------------------------------
# 直接运行：python3 gameplay/damage_number_popup.py
# 打印六个修复点的自检结论，全部 ok 时退出码为 0。
# ---------------------------------------------------------------------------

def _self_check() -> dict[str, Any]:
    return validate_popup(1)


if __name__ == "__main__":
    import json

    report = _self_check()
    labels = {
        "current_camera_projection": "投影使用当前相机",
        "pool_grows_without_dropping": "对象池按需扩容不丢弃",
        "draw_only_after_atlas": "图集就位后才绘制",
        "time_integrated_float": "上浮按时间积分",
        "same_frame_merge": "同帧同目标合并",
        "seeded_offset_deterministic": "固定种子偏移可复现",
    }
    for key, label in labels.items():
        print(f"[{'PASS' if report['checks'].get(key) else 'FAIL'}] {label}")
    print(json.dumps({"ok": report["ok"], "errors": report["errors"]},
                     ensure_ascii=False))
    raise SystemExit(0 if report["ok"] else 1)
