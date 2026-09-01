from features.labels import (
    CENSORED,
    LabelConfig,
    binary_target,
    ordinal_target,
    survival_tuple,
)

CFG = LabelConfig(rpl_minutes_threshold=200, settled_age=26)


def test_binary_breakthrough():
    assert binary_target(rpl_minutes_ever=250, current_age=21, cfg=CFG) == 1
    assert binary_target(rpl_minutes_ever=200, current_age=19, cfg=CFG) == 1


def test_binary_settled_negative():
    assert binary_target(rpl_minutes_ever=0, current_age=27, cfg=CFG) == 0
    assert binary_target(rpl_minutes_ever=199, current_age=26, cfg=CFG) == 0


def test_binary_censored():
    assert binary_target(rpl_minutes_ever=50, current_age=22, cfg=CFG) == CENSORED


def test_ordinal():
    assert ordinal_target(300, reached_pro_level=True, cfg=CFG) == "rpl"
    assert ordinal_target(0, reached_pro_level=True, cfg=CFG) == "lower_leagues"
    assert ordinal_target(0, reached_pro_level=False, cfg=CFG) == "none"


def test_survival_tuple():
    assert survival_tuple(rpl_debut_age=18.4, current_age=25) == (18.4, 1)
    assert survival_tuple(rpl_debut_age=None, current_age=20) == (20.0, 0)
