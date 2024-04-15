from pytest import mark

from lib.preproc import RelativeDirectionWrapper


@mark.parametrize("abs_act, dir_name, exp_rel_act", [
    ("go north", "north", "go forward"),
    ("go south", "south", "go forward"),
    ("go east",  "east",  "go forward"),
    ("go east",  "north", "go right"),
    ("go west",  "north", "go left"),
    ("go south", "north", "go back"),
    ("go west",  "east",  "go back"),
    ("go west",  "south", "go right"),
    ("go east",  "south", "go left"),
    ("go north", "south", "go back"),
    ("go south", "east",  "go right"),
    ("go south", "west",  "go left"),
])
def test_abs_to_rel(abs_act, dir_name, exp_rel_act):
    dir_idx = RelativeDirectionWrapper.ABSOLUTE_DIRS.index(dir_name)
    rel_act = RelativeDirectionWrapper.abs_to_rel(abs_act, dir_idx)
    assert isinstance(rel_act, str)
    assert rel_act == exp_rel_act


@mark.parametrize("rel_act, dir_name, exp_abs_act", [
    ("go forward", "north", "go north"),
    ("go right",   "north", "go east"),
    ("go back",    "north", "go south"),
    ("go left",    "north", "go west"),

    ("go forward", "east", "go east"),
    ("go right",   "east", "go south"),
    ("go back",    "east", "go west"),
    ("go left",    "east", "go north"),
])
def test_rel_to_abs(rel_act, dir_name, exp_abs_act):
    dir_idx = RelativeDirectionWrapper.ABSOLUTE_DIRS.index(dir_name)
    abs_act = RelativeDirectionWrapper.rel_to_abs(rel_act, dir_idx)
    assert isinstance(abs_act, str)
    assert abs_act == exp_abs_act


@mark.parametrize("rel_act, dir_name, exp_new_dir", [
    ("go forward", "north", "north"),
    ("go right",   "north", "east"),
    ("go left",    "north", "west"),
    ("go back",    "north", "south"),

    ("go forward", "west", "west"),
    ("go right",   "west", "north"),
    ("go left",    "west", "south"),
    ("go back",    "west", "east"),
])
def test_rel_execute(rel_act, dir_name, exp_new_dir):
    dir_idx = RelativeDirectionWrapper.ABSOLUTE_DIRS.index(dir_name)
    new_dir = RelativeDirectionWrapper.rel_execute(rel_act, dir_idx)
    assert isinstance(new_dir, int)
    new_dir_name = RelativeDirectionWrapper.ABSOLUTE_DIRS[new_dir]
    assert new_dir_name == exp_new_dir


def test_update_vocabs():
    v, v_r = {}, {}
    RelativeDirectionWrapper.update_vocabs(v, v_r)
    assert len(v) == 4
    assert len(v_r) == 4
    assert v == {0: "right", 1: "forward", 2: "left", 3: "back"}
    assert v_r == {"right": 0, "forward": 1, "left": 2, "back": 3}

    v, v_r = {0: "word", 1: "left"}, {"word": 0, "left": 1}
    RelativeDirectionWrapper.update_vocabs(v, v_r)
    assert len(v) == 5
    assert len(v_r) == 5
    assert v == {0: "word", 1: "left", 2: "right", 3: "forward", 4: "back"}
    assert v_r == {"word": 0, "left": 1, "right": 2, "forward": 3, "back": 4}
