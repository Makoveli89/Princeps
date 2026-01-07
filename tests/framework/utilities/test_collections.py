from framework.utilities.collections import dict_merge

def test_dict_merge_basic():
    d1 = {"a": 1, "b": 2}
    d2 = {"b": 3, "c": 4}
    dict_merge(d1, d2)
    assert d1 == {"a": 1, "b": 3, "c": 4}

def test_dict_merge_recursive():
    d1 = {"a": {"x": 1, "y": 2}, "b": 3}
    d2 = {"a": {"y": 3, "z": 4}, "c": 5}
    dict_merge(d1, d2)
    assert d1 == {
        "a": {"x": 1, "y": 3, "z": 4},
        "b": 3,
        "c": 5
    }

def test_dict_merge_mixed_types():
    d1 = {"a": {"x": 1}}
    d2 = {"a": 2}  # overwriting dict with int
    dict_merge(d1, d2)
    assert d1 == {"a": 2}

    d3 = {"a": 1}
    d4 = {"a": {"x": 2}} # overwriting int with dict
    dict_merge(d3, d4)
    assert d3 == {"a": {"x": 2}}

def test_dict_merge_nested_deep():
    d1 = {"a": {"b": {"c": 1}}}
    d2 = {"a": {"b": {"d": 2}}}
    dict_merge(d1, d2)
    assert d1 == {"a": {"b": {"c": 1, "d": 2}}}
