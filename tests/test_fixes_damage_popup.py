"""六个伤害飘字缺陷修复点的行为回归测试。

契约测试只断言“入口可调用且返回非 None”；本文件锁定六个缺陷确实被修掉：
当前相机投影、池按需扩容、图集就绪后绘制、按时间积分、同帧合并、固定种子偏移。
"""

import pytest

from gameplay import damage_number_popup as mod
from gameplay.damage_number_popup import (
    DEFAULT_HEIGHT,
    DEFAULT_RISE_SPEED,
    DEFAULT_SEED,
    DEFAULT_WIDTH,
    Camera,
    DamageNumberSystem,
    FontAtlas,
    PopupPool,
    float_by_time,
    merge_same_frame,
    offset_for_index,
    pool_grow,
    project_current_camera,
    project_world_to_screen,
    seeded_offset,
)


# --- 修复 1：投影使用当前帧相机矩阵 ----------------------------------------

def test_point_in_front_of_camera_centers_screen():
    camera = Camera(eye=(0.0, 0.0, 5.0), target=(0.0, 0.0, 0.0))
    p = project_world_to_screen((0.0, 0.0, 0.0), camera.view_projection)
    assert p["visible"] is True
    assert p["x"] == pytest.approx(DEFAULT_WIDTH / 2, abs=1e-6)
    assert p["y"] == pytest.approx(DEFAULT_HEIGHT / 2, abs=1e-6)


def test_projection_tracks_current_camera_not_previous_frame():
    report = project_current_camera(1)
    assert report["ok"] is True
    assert report["used_current_camera"] is True
    assert report["matches_current"] is True
    # 快速转身之后，当前矩阵与上一帧矩阵必须给出不同投影，
    # 而入口结果站在当前矩阵这一边。
    assert report["camera_moved_projection"] is True
    assert report["matches_previous"] is False
    assert report["delta_to_current"] == pytest.approx(0.0, abs=1e-9)
    assert report["delta_to_previous"] > 1e-6


def test_behind_camera_is_flagged_not_mirrored():
    camera = Camera(eye=(0.0, 0.0, 5.0), target=(0.0, 0.0, 0.0))
    behind = project_world_to_screen((0.0, 0.0, 50.0), camera.view_projection)
    assert behind["behind_camera"] is True
    assert behind["visible"] is False


# --- 修复 2：对象池按需扩容，不丢弃数字 ------------------------------------

def test_pool_starts_at_sixty_four():
    pool = PopupPool()
    assert pool.capacity == 64


def test_pool_grows_under_burst_over_one_hundred():
    report = pool_grow(1)
    assert report["ok"] is True
    assert report["initial_capacity"] == 64
    assert report["capacity_before"] == 64
    assert report["capacity_after"] >= 100
    assert report["grew"] is True
    assert report["requested"] == 100
    assert report["acquired"] == 100
    assert report["none_count"] == 0
    assert report["dropped"] == 0
    assert report["discarded"] == 0


def test_pool_recycles_and_reuses_instances():
    pool = PopupPool(initial_size=2)
    a = pool.acquire()
    b = pool.acquire()
    assert a is not None and b is not None
    assert pool.active_count == 2
    assert pool.release(a) is True
    assert pool.release(a) is False  # 重复回收幂等
    assert pool.active_count == 1
    c = pool.acquire()
    assert c is a  # 同一个实例被复用


# --- 修复 3：图集就位再绘制，UV 稳定不变形 ---------------------------------

def test_nothing_draws_before_atlas_ready():
    system = DamageNumberSystem(atlas=FontAtlas())  # 未就绪
    system.spawn(1234, target_id="enemy", world_pos=(0, 0, 0))
    result = system.draw()
    assert result["atlas_ready"] is False
    assert result["drawn"] == 0
    assert result["commands"] == []
    assert result["placeholders"] == 0  # 不再用占位方块


def test_draws_after_atlas_ready():
    atlas = FontAtlas()
    system = DamageNumberSystem(atlas=atlas)
    system.spawn(1234, target_id="enemy", world_pos=(0, 0, 0))
    assert system.draw()["drawn"] == 0
    atlas.mark_ready()
    result = system.draw()
    assert result["drawn"] == 1
    assert result["placeholders"] == 0
    assert result["commands"][0]["text"] == "1234"


def test_uvs_are_stable_across_atlas_rebuild():
    atlas = FontAtlas.ready_default()
    before = dict(atlas.uv)
    atlas.rebuild_uv()
    assert atlas.uv == before  # 同规格重算结果一致，不会让在场数字变形


def test_draw_after_atlas_self_check():
    report = mod.draw_after_atlas(1)
    assert report["ok"] is True
    assert report["drawn_before_ready"] == 0
    assert report["drawn_after_ready"] >= 1
    assert report["uv_stable_after_switch"] is True


# --- 修复 4：上浮按时间积分，帧率无关 --------------------------------------

def test_equal_wall_time_gives_equal_height_across_frame_rates():
    speed = DEFAULT_RISE_SPEED
    h30 = 0.0
    for _ in range(30):
        h30 += speed * (1.0 / 30.0)
    h144 = 0.0
    for _ in range(144):
        h144 += speed * (1.0 / 144.0)
    assert h30 == pytest.approx(h144, abs=1e-9)
    assert h30 == pytest.approx(speed, rel=1e-9)


def test_float_by_time_report_proves_frame_independence():
    report = float_by_time(1)
    assert report["ok"] is True
    assert report["frame_rate_independent"] is True
    assert report["height_30fps"] == pytest.approx(report["height_144fps"], abs=1e-6)
    # 旧的每帧固定位移确实会随帧率产生差异（对照成立）。
    assert report["legacy_frame_rate_dependent"] is True
    assert report["legacy_height_144fps"] != report["legacy_height_30fps"]


def test_float_step_is_speed_times_delta_time():
    report = float_by_time(0.5, speed=100.0)
    assert report["step"] == pytest.approx(50.0)
    assert report["height"] == pytest.approx(50.0)


def test_system_popup_rise_is_time_integrated():
    def run(fps: int) -> float:
        system = DamageNumberSystem()
        popup = system.spawn(1, target_id="t")
        for _ in range(fps):  # 模拟 1 秒
            system.update(1.0 / fps)
        return popup.rise

    assert run(30) == pytest.approx(run(144), abs=1e-6)


# --- 修复 5：同目标同帧多次伤害合并 ----------------------------------------

def test_merge_pure_function_collapses_same_target_frame():
    events = [
        {"target": "a", "frame": 1, "amount": 10},
        {"target": "a", "frame": 1, "amount": 15},
        {"target": "a", "frame": 1, "amount": 5},
    ]
    merged = merge_same_frame(events)
    assert len(merged) == 1
    assert merged[0]["amount"] == pytest.approx(30)
    assert merged[0]["hits"] == 3


def test_merge_keeps_different_frame_and_target_apart():
    report = merge_same_frame(1)
    assert report["ok"] is True
    assert report["group_count"] == 3
    assert report["merged_amount"] == 30
    assert report["hits"] == 3
    assert report["total_damage"] == pytest.approx(136)


def test_system_merges_into_single_popup_instance():
    system = DamageNumberSystem()
    first = system.spawn(10, target_id="enemy", frame=3)
    second = system.spawn(15, target_id="enemy", frame=3)
    third = system.spawn(5, target_id="enemy", frame=3)
    assert first is second is third
    assert first.amount == pytest.approx(30)
    assert first.hits == 3
    # 不同帧 -> 新飘字。
    other_frame = system.spawn(9, target_id="enemy", frame=4)
    assert other_frame is not first
    # 同帧不同目标 -> 新飘字。
    other_target = system.spawn(9, target_id="other", frame=3)
    assert other_target is not first


# --- 修复 6：固定种子偏移，回放可复现 --------------------------------------

def test_same_seed_and_index_give_same_offset():
    assert offset_for_index(DEFAULT_SEED, 0) == offset_for_index(DEFAULT_SEED, 0)
    assert [offset_for_index(7, i) for i in range(10)] == \
           [offset_for_index(7, i) for i in range(10)]


def test_different_indices_spread_out():
    offsets = {offset_for_index(DEFAULT_SEED, i) for i in range(12)}
    assert len(offsets) >= 8  # 不是所有数字都叠在同一个偏移上


def test_seeded_offset_entry_is_deterministic():
    a = seeded_offset(42, count=16)
    b = seeded_offset(42, count=16)
    assert a["ok"] is True and a["deterministic"] is True
    assert a["offsets"] == b["offsets"]
    assert a["replay_match"] is True


def test_determinism_report_matches_across_runs():
    report = mod.determinism(DEFAULT_SEED, count=12)
    assert report["ok"] is True
    assert report["deterministic"] is True
    assert report["simulation_match"] is True
    again = mod.determinism(DEFAULT_SEED, count=12)
    assert again["samples"] == report["samples"]


# --- 端到端：团战上百个数字不丢、校验全绿 ----------------------------------

def test_teamfight_burst_over_one_hundred_numbers_none_dropped():
    system = DamageNumberSystem(pool_size=64)
    spawned = 0
    for i in range(120):
        # 刻意打散到不同目标 + 帧，避免合并掩盖池需求。
        popup = system.spawn(100 + i, target_id=f"enemy-{i}", frame=i)
        assert popup is not None
        spawned += 1
    assert spawned == 120
    assert system.pool.capacity >= 120
    assert system.pool.dropped == 0
    metrics = system.metrics()
    assert metrics["dropped"] == 0


def test_validate_popup_all_checks_green():
    report = mod.validate_popup(1)
    assert report["ok"] is True
    assert report["valid"] is True
    assert report["errors"] == []
    assert all(report["checks"].values())


@pytest.mark.parametrize("fn_name", [
    "project_current_camera", "pool_grow", "draw_after_atlas",
    "float_by_time", "merge_same_frame", "seeded_offset",
    "spawn", "update", "recycle", "draw",
    "validate_popup", "metrics", "audit", "fallback", "determinism",
])
def test_every_entry_supports_placeholder_probe(fn_name):
    result = getattr(mod, fn_name)(1)
    assert result is not None
    assert isinstance(result, dict)
