import numpy as np
from football.env import Football2v2, PASS, SHOOT_TOP, SHOOT_BOT, FWD, STAY, N_MOVE


def test_shapes_and_random_rollout():
    env = Football2v2(n_envs=32, seed=1)
    obs, st = env.reset()
    assert obs.shape == (32, 4, env.obs_dim) and st.shape == (32, 2, env.state_dim)
    rng = np.random.default_rng(0)
    goals = 0
    for _ in range(300):
        obs, st, r, d, info = env.step(rng.integers(0, N_MOVE, (32, 4)), rng.integers(0, 8, (32, 4)))
        assert np.allclose(r.sum(1), 0)  # zero-sum
        goals += (info["goal"] >= 0).sum()
        assert env.pos[..., 0].min() >= 0 and env.pos[..., 0].max() < env.W
    assert goals > 0


def test_keeper_side_matters():
    rates = []
    for act in (SHOOT_TOP, SHOOT_BOT):
        env = Football2v2(n_envs=4000, seed=2, p_keeper_switch=0.0)
        env.reset()
        env.pos[:, 0] = [env.W - 1, 2]   # แดงยืนหน้าประตู
        env.pos[:, 1] = [0, 0]
        env.pos[:, 2] = [0, 5]; env.pos[:, 3] = [1, 5]  # ฝ่ายรับอยู่ไกล
        env.holder[:] = 0
        env.keeper[:, 1] = 1  # ผู้รักษาประตูน้ำเงินยืนครึ่งล่าง
        mv = np.zeros((4000, 4), int); mv[:, 0] = act
        _, _, r, d, info = env.step(mv, np.zeros((4000, 4), int))
        rates.append((info["goal"] == 0).mean())
    assert 0.85 < rates[0] <= 0.93 and rates[1] < 0.12


def test_blue_frame_mirrored():
    env = Football2v2(n_envs=1, seed=3)
    env.reset()
    env.pos[0] = [[0, 0], [0, 1], [5, 3], [9, 3]]
    env.holder[:] = 2
    mv = np.full((1, 4), STAY); mv[0, 2] = FWD  # น้ำเงินเดินหน้า = x ลดลง
    env.step(mv, np.zeros((1, 4), int))
    assert env.pos[0, 2, 0] == 4


def test_messages_private_vs_eavesdrop():
    for hear, expect in [((False, False), 0), ((True, True), 1)]:
        env = Football2v2(n_envs=1, vocab=4, hear_opp=hear, seed=4)
        env.reset()
        obs, *_ = env.step(np.zeros((1, 4), int), np.array([[1, 2, 3, 0]]))
        base = 6 + 4 * env.grid_cells
        assert obs[0, 0, base + 2] == 1  # แดง 0 ได้ยินเพื่อน (token 2)
        assert obs[0, 0, base + 4:base + 12].sum() == 2 * expect


def test_pass_intercept():
    env = Football2v2(n_envs=1000, p_intercept=1.0, seed=5)
    env.reset()
    env.pos[:, 0] = [1, 2]; env.pos[:, 1] = [7, 2]
    env.pos[:, 2] = [4, 2]; env.pos[:, 3] = [9, 5]
    env.holder[:] = 0
    mv = np.zeros((1000, 4), int); mv[:, 0] = PASS
    env.step(mv, np.zeros((1000, 4), int))
    assert (env.holder == 2).all()


def test_keeper_visible_only_when_close():
    env = Football2v2(n_envs=1, seed=6)
    env.reset()
    env.keeper[:] = 0
    g = env.grid_cells
    env.pos[0] = [[env.W - 1, 2], [0, 0], [5, 5], [6, 5]]
    near = env.observe()[0, 0, 6 + 3 * g: 6 + 4 * g].sum()
    env.pos[0, 0] = [env.W - 4, 2]
    far = env.observe()[0, 0, 6 + 3 * g: 6 + 4 * g].sum()
    assert near == 2 and far == 0


def test_carrier_slower():
    env = Football2v2(n_envs=4000, seed=7, p_carry_move=0.6, p_tackle_adj=0.0)
    env.reset()
    env.pos[:, 0] = [2, 2]; env.pos[:, 1] = [2, 4]
    env.pos[:, 2] = [9, 0]; env.pos[:, 3] = [9, 5]
    env.holder[:] = 0
    mv = np.zeros((4000, 4), int); mv[:, 0] = FWD; mv[:, 1] = FWD
    env.step(mv, np.zeros((4000, 4), int))
    assert 0.55 < (env.pos[:, 0, 0] == 3).mean() < 0.65 and (env.pos[:, 1, 0] == 3).all()
