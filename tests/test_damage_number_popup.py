import pytest
import gameplay.damage_number_popup as mod

CASES = [
    ("project_current_camera", (1,)),
    ("pool_grow", (1,)),
    ("draw_after_atlas", (1,)),
    ("float_by_time", (1,)),
    ("merge_same_frame", (1,)),
    ("seeded_offset", (1,)),
    ("spawn", (1,)),
    ("update", (1,)),
    ("recycle", (1,)),
    ("draw", (1,)),
    ("validate_popup", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
