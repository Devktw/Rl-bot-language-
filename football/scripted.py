"""คู่แข่งแบบเขียนกฎเอง (เห็นทั้งสนาม) ใช้เป็นคู่ซ้อมที่นิ่ง ไม่เปลี่ยนตามการเทรน

กฎ (team frame, บุกไปทางขวา):
  - ถือบอล: ใกล้ประตูพอ (d <= 3) ยิง; มีคู่แข่งขวางหน้า หลบขึ้น/ลง; ไม่งั้นเดินหน้า
  - เพื่อนถือบอล: วิ่งขึ้นหน้าไปรอ
  - คู่แข่งถือบอล: คนที่ใกล้บอลกว่าไล่บอล อีกคนถอยมาคุมหน้าประตู
"""
import numpy as np

from .env import BACK, DOWN, FWD, SHOOT, STAY, UP


def scripted_actions(env, team, eps=0.1, rng=None):
    rng = rng or env.rng
    N = env.N
    ar = np.arange(N)
    players = (0, 1) if team == 0 else (2, 3)
    fx = env.frame_x(env.pos[..., 0], team)  # (N,4) ใน frame ของทีมนี้
    y = env.pos[..., 1]
    hteam = env.holder // 2
    hx, hy = fx[ar, env.holder], y[ar, env.holder]
    out = np.zeros((N, 2), dtype=np.int64)
    # ใครใกล้บอลกว่าในทีม
    d = [np.abs(fx[:, p] - hx) + np.abs(y[:, p] - hy) for p in players]
    chaser = np.where(d[0] <= d[1], players[0], players[1])
    for k, p in enumerate(players):
        x0, y0 = fx[:, p], y[:, p]
        has = env.holder == p
        a = np.full(N, STAY)
        # ถือบอล
        gy = np.clip(y0, *env.mouth)
        dist = (env.W - x0) + np.abs(y0 - gy)
        blocked = np.zeros(N, dtype=bool)
        for o in ((2, 3) if team == 0 else (0, 1)):
            blocked |= (fx[:, o] == x0 + 1) & (y[:, o] == y0)
        side = np.where(y0 < env.H / 2, DOWN, UP)
        a_hold = np.where(dist <= 3, SHOOT, np.where(blocked, side, FWD))
        # เพื่อนถือบอล
        a_support = np.where(x0 < env.W - 2, FWD, np.where(y0 < gy, DOWN, np.where(y0 > gy, UP, STAY)))
        # คู่แข่งถือบอล
        dx, dy = hx - x0, hy - y0
        chase = np.where(np.abs(dx) >= np.abs(dy), np.where(dx > 0, FWD, BACK), np.where(dy > 0, DOWN, UP))
        chase = np.where((dx == 0) & (dy == 0), STAY, chase)
        # คนที่ไม่ไล่บอลถอยไปยืนหน้าประตูตัวเอง (x'=1) ในแถวเดียวกับบอล
        gx, gy2 = 1, np.clip(hy, *env.mouth)
        ddx, ddy = gx - x0, gy2 - y0
        guard = np.where(ddx != 0, np.where(ddx > 0, FWD, BACK), np.where(ddy > 0, DOWN, np.where(ddy < 0, UP, STAY)))
        a_def = np.where(chaser == p, chase, guard)
        a = np.where(has, a_hold, np.where(hteam == team, a_support, a_def))
        rnd = rng.random(N) < eps
        a = np.where(rnd, rng.integers(0, 5, N), a)
        out[:, k] = a
    return out
