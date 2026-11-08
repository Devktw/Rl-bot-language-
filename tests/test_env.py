import numpy as np
from football.env import Football2v2, PASS, SHOOT, FWD, STAY


def test_shapes_and_random_rollout():
    env = Football2v2(n_envs=32, seed=1)
    obs, st = env.reset()
    assert obs.shape == (32, 4, env.obs_dim) and st.shape == (32, 2, env.state_dim)
    rng = np.random.default_rng(0)
    goals = 0
    for _ in range(300):
        obs, st, r, d, info = env.step(rng.integers(0, 7, (32, 4)), rng.integers(0, 8, (32, 4)))
        assert np.allclose(r.sum(1), 0)  # zero-sum
        goals += (info["goal"] >= 0).sum()
        assert env.pos[..., 0].min() >= 0 and env.pos[..., 0].max() < env.W
    assert goals > 0


def test_shot_from_close_scores_often():
    env = Football2v2(n_envs=2000, seed=2)
    env.reset()
    env.pos[:, 0] = [env.W - 1, 2]   # แดงยืนหน้าประตู
    env.pos[:, 1] = [0, 0]
    env.pos[:, 2] = [0, 5]; env.pos[:, 3] = [1, 5]  # ฝ่ายรับอยู่ไกล
    env.holder[:] = 0
    mv = np.zeros((2000, 4), int); mv[:, 0] = SHOOT
    _, _, r, d, info = env.step(mv, np.zeros((2000, 4), int))
    assert 0.8 < (info["goal"] == 0).mean() <= 0.95 and r[:, 0].max() == 1


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
        base = 6 + 3 * env.grid_cells
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
