"""ฟุตบอล 2v2 บน grid 2 มิติ แบบ vectorized (numpy) สำหรับศึกษาภาษาที่เกิดขึ้นเอง

ผู้เล่น index 0,1 = ทีมแดง (team 0, บุกไปทางขวา ประตูอยู่ที่ x=W)
ผู้เล่น index 2,3 = ทีมน้ำเงิน (team 1, บุกไปทางซ้าย ประตูอยู่ที่ x=-1)

ทุกอย่างที่ agent เห็นอยู่ใน "team frame" คือพลิกแกน x ให้ทุกทีมบุกไปทางขวาเสมอ

แต่ละ step ผู้เล่นทุกคนเลือก
  - move action: 0 อยู่นิ่ง, 1 ขึ้น, 2 ลง, 3 ถอยหลัง, 4 เดินหน้า, 5 ส่งบอล,
                 6 ยิงมุมบน, 7 ยิงมุมล่าง
  - message token: 0..K-1 (เพื่อนร่วมทีมได้ยินใน step ถัดไป;
    ถ้าเปิด hear_opp คู่แข่งก็ได้ยินด้วย)

กลไกที่ทำให้ข้อมูลที่มองไม่เห็นมีค่า (v2)
  - ผู้รักษาประตู (ควบคุมโดย env) ยืนครึ่งบนหรือครึ่งล่างของปากประตู และสลับฝั่งแบบสุ่ม
    ยิงไปฝั่งที่เขายืน โอกาสเข้าเหลือแค่ 10% แต่คนยิงจากไกลมองไม่เห็นเขา
  - คนถือบอลเดินได้ช้ากว่า (สำเร็จ p_carry_move) และคู่แข่งที่ยืนติดกันแย่งบอลได้
    จึงต้องส่งบอลกันมากขึ้น

การมองเห็นจำกัด: เห็นเฉพาะสิ่งที่อยู่ในรัศมี R (Chebyshev) รอบตัว
รู้เสมอแค่ตำแหน่งตัวเอง, ตัวเองถือบอลไหม, และทีมไหนครองบอล
"""
from __future__ import annotations

import numpy as np

STAY, UP, DOWN, BACK, FWD, PASS, SHOOT_TOP, SHOOT_BOT = range(8)
SHOOT = SHOOT_TOP  # ชื่อเดิม เผื่อโค้ดเก่า
N_MOVE = 8
_DY = np.array([0, -1, 1, 0, 0, 0, 0, 0])
_DX = np.array([0, 0, 0, -1, 1, 0, 0, 0])  # ใน team frame
N_CH = 4  # ช่องของ ego grid: เพื่อน, คู่แข่ง, บอล, ผู้รักษาประตู


class Football2v2:
    def __init__(
        self,
        n_envs: int = 256,
        width: int = 10,
        height: int = 6,
        vision: int = 2,
        vocab: int = 8,
        max_steps: int = 50,
        speak=(True, True),
        hear_opp=(False, False),
        p_tackle: float = 0.5,
        p_tackle_adj: float = 0.2,
        p_intercept: float = 0.8,
        p_carry_move: float = 0.6,
        p_keeper_switch: float = 0.1,
        keeper_save: float = 0.9,
        seed: int = 0,
    ):
        self.N, self.W, self.H, self.R, self.K = n_envs, width, height, vision, vocab
        self.max_steps = max_steps
        self.speak = np.array(speak, dtype=bool)
        self.hear_opp = np.array(hear_opp, dtype=bool)
        self.p_tackle, self.p_tackle_adj, self.p_intercept = p_tackle, p_tackle_adj, p_intercept
        self.p_carry_move, self.p_keeper_switch, self.keeper_save = p_carry_move, p_keeper_switch, keeper_save
        self.rng = np.random.default_rng(seed)
        g = 2 * vision + 1
        self.grid_cells = g * g
        self.obs_dim = 6 + N_CH * self.grid_cells + 3 * vocab + 1
        self.state_dim = 8 + 4 + 2 + 1
        # ปากประตูกว้าง H-2 แถว แบ่งเป็นครึ่งบน/ล่าง
        self.mouth = (1, height - 2)
        mid = height // 2
        self.halves = np.array([[1, mid - 1], [mid, height - 2]])  # (half, [lo, hi])
        self.pos = np.zeros((n_envs, 4, 2), dtype=np.int64)  # (x, y) พิกัดจริง
        self.holder = np.zeros(n_envs, dtype=np.int64)
        self.keeper = np.zeros((n_envs, 2), dtype=np.int64)  # keeper[:, d] = ฝั่งที่ผู้รักษาประตูของทีม d ยืน (0 บน, 1 ล่าง)
        self.t = np.zeros(n_envs, dtype=np.int64)
        self.msg = np.zeros((n_envs, 4), dtype=np.int64)  # token ที่ส่งเมื่อ step ก่อน
        self.msg_opp = np.zeros((n_envs, 4), dtype=np.int64)  # token ที่คู่แข่งได้ยิน (ปกติ = msg)
        self.reset_envs(np.arange(n_envs))

    # ------------------------------------------------------------------ helpers
    def frame_x(self, x, team):
        """แปลง x จริงเป็น team frame (team 1 พลิกแกน)"""
        return np.where(team == 1, self.W - 1 - x, x)

    def reset_envs(self, idx):
        n = len(idx)
        if n == 0:
            return
        half = self.W // 2
        xs = np.empty((n, 4), dtype=np.int64)
        xs[:, :2] = self.rng.integers(0, half, size=(n, 2))
        xs[:, 2:] = self.rng.integers(half, self.W, size=(n, 2))
        ys = self.rng.integers(0, self.H, size=(n, 4))
        self.pos[idx, :, 0] = xs
        self.pos[idx, :, 1] = ys
        self.holder[idx] = self.rng.integers(0, 4, size=n)
        self.keeper[idx] = self.rng.integers(0, 2, size=(n, 2))
        self.t[idx] = 0
        self.msg[idx] = 0
        self.msg_opp[idx] = 0

    def reset(self):
        self.reset_envs(np.arange(self.N))
        return self.observe(), self.state()

    def keeper_cells(self, d):
        """ตำแหน่งจริงสองช่องที่ผู้รักษาประตูของทีม d ยืน: (N,2,2)"""
        x = -1 if d == 0 else self.W
        lo = self.halves[self.keeper[:, d], 0]
        hi = self.halves[self.keeper[:, d], 1]
        cells = np.zeros((self.N, 2, 2), dtype=np.int64)
        cells[:, :, 0] = x
        cells[:, 0, 1] = lo
        cells[:, 1, 1] = hi
        return cells

    # ------------------------------------------------------------------ observe
    def observe(self):
        N, R, K = self.N, self.R, self.K
        g = 2 * R + 1
        ar = np.arange(N)
        obs = np.zeros((N, 4, self.obs_dim), dtype=np.float32)
        ball = self.pos[ar, self.holder]  # (N,2)
        kc = [self.keeper_cells(d) for d in range(2)]
        for i in range(4):
            team = i // 2
            mate = i ^ 1
            opps = (2, 3) if team == 0 else (0, 1)
            me = self.pos[:, i]
            fx = self.frame_x(me[:, 0], team)
            o = obs[:, i]
            o[:, 0] = fx / (self.W - 1)
            o[:, 1] = me[:, 1] / (self.H - 1)
            o[:, 2] = self.holder == i
            o[:, 3] = (self.holder // 2) == team
            o[:, 4 + (i % 2)] = 1.0
            base = 6
            ents = (
                [self.pos[:, mate]],
                [self.pos[:, e] for e in opps],
                [ball],
                [kc[d][:, c] for d in range(2) for c in range(2)],
            )
            for ch, plist in enumerate(ents):
                for p in plist:
                    dx = p[:, 0] - me[:, 0]
                    if team == 1:
                        dx = -dx
                    dy = p[:, 1] - me[:, 1]
                    vis = (np.abs(dx) <= R) & (np.abs(dy) <= R)
                    cell = (dy + R) * g + (dx + R)
                    rows = np.nonzero(vis)[0]
                    o[rows, base + ch * self.grid_cells + cell[rows]] = 1.0
            base += N_CH * self.grid_cells
            # token ของเพื่อนร่วมทีม
            if self.speak[team]:
                o[ar, base + self.msg[:, mate]] = 1.0
            base += K
            # token ของคู่แข่ง (ดักฟัง)
            if self.hear_opp[team] and self.speak[1 - team]:
                for k, e in enumerate(opps):
                    o[ar, base + k * K + self.msg_opp[:, e]] = 1.0
            base += 2 * K
            o[:, base] = self.t / self.max_steps
        return obs

    def state(self):
        """state เต็มสำหรับ centralized critic ของแต่ละทีม (team frame)"""
        N = self.N
        st = np.zeros((N, 2, self.state_dim), dtype=np.float32)
        for team in range(2):
            order = (0, 1, 2, 3) if team == 0 else (2, 3, 0, 1)
            for k, p in enumerate(order):
                st[:, team, 2 * k] = self.frame_x(self.pos[:, p, 0], team) / (self.W - 1)
                st[:, team, 2 * k + 1] = self.pos[:, p, 1] / (self.H - 1)
                st[:, team, 8 + k] = self.holder == p
            st[:, team, 12] = self.keeper[:, 1 - team]  # ผู้รักษาประตูฝั่งที่เราบุก
            st[:, team, 13] = self.keeper[:, team]
            st[:, team, 14] = self.t / self.max_steps
        return st

    def shot_prob(self, shooter_pos, team, half):
        """โอกาสยิงเข้า (ไม่รวมการถูกบล็อก) ของผู้เล่นที่ยืนอยู่ shooter_pos ยิงไปครึ่ง half"""
        fx = self.frame_x(shooter_pos[:, 0], team)
        lo, hi = self.halves[half, 0], self.halves[half, 1]
        gy = np.clip(shooter_pos[:, 1], lo, hi)
        d = (self.W - fx) + np.abs(shooter_pos[:, 1] - gy)
        p = np.clip(1.0 - 0.12 * (d - 1), 0.0, 0.9)
        saved = self.keeper[np.arange(self.N), 1 - team] == half
        return np.where(saved, p * (1 - self.keeper_save), p), gy

    # ------------------------------------------------------------------ dynamics
    def _on_path(self, src, dst, cand, include_end):
        """cand อยู่บนเส้นทางจาก src ไป dst ไหม (N,) bool"""
        fr = np.linspace(0, 1, 11)[1:] if include_end else np.linspace(0, 1, 11)[1:-1]
        pts = np.rint(src[:, None, :] + fr[None, :, None] * (dst - src)[:, None, :])
        return np.any(np.all(pts == cand[:, None, :], axis=-1), axis=1)

    def step(self, move, token, token_opp=None):
        """move, token: (N,4) int. คืน obs, state, reward (N,2), done (N,), info

        token_opp (ไม่บังคับ): token ที่คู่แข่งจะได้ยินแทน token จริง ใช้ทำ intervention"""
        N, ar = self.N, np.arange(self.N)
        move = np.asarray(move)
        h = self.holder.copy()
        h_act = move[ar, h]
        # 1) เคลื่อนที่ (คนถือบอลเดินสำเร็จแค่บางครั้ง)
        carry_ok = self.rng.random(N) < self.p_carry_move
        for i in range(4):
            a = move[:, i]
            dx = _DX[a] * (1 if i < 2 else -1)
            dy = _DY[a]
            ok = (h != i) | carry_ok
            self.pos[:, i, 0] = np.clip(self.pos[:, i, 0] + dx * ok, 0, self.W - 1)
            self.pos[:, i, 1] = np.clip(self.pos[:, i, 1] + dy * ok, 0, self.H - 1)
        hteam = h // 2
        opp0 = (1 - hteam) * 2
        mate = h ^ 1
        hp = self.pos[ar, h]
        reward = np.zeros((N, 2), dtype=np.float32)
        new_h = h.copy()
        u = self.rng.random((N, 3))

        # 2) ส่งบอล
        is_pass = h_act == PASS
        mp = self.pos[ar, mate]
        intercepted = np.zeros(N, dtype=bool)
        for k in range(2):
            o = opp0 + k
            on = self._on_path(hp, mp, self.pos[ar, o], include_end=True)
            hit = is_pass & ~intercepted & on & (u[:, k] < self.p_intercept)
            new_h = np.where(hit, o, new_h)
            intercepted |= hit
        new_h = np.where(is_pass & ~intercepted, mate, new_h)

        # 3) ยิง
        is_shot = (h_act == SHOOT_TOP) | (h_act == SHOOT_BOT)
        half = np.where(h_act == SHOOT_BOT, 1, 0)
        p_goal = np.zeros(N)
        gy = np.zeros(N, dtype=np.int64)
        for T in range(2):
            m = hteam == T
            pg, g = self.shot_prob(hp, T, half)
            p_goal = np.where(m, pg, p_goal)
            gy = np.where(m, g, gy)
        goal_real_x = np.where(hteam == 0, self.W, -1)
        gpt = np.stack([goal_real_x, gy], axis=1)
        for k in range(2):
            blocked = self._on_path(hp, gpt, self.pos[ar, opp0 + k], include_end=False)
            p_goal = np.where(blocked, p_goal * 0.3, p_goal)
        goal = is_shot & (u[:, 2] < p_goal)
        shot_at_keeper = self.keeper[ar, 1 - hteam] == half
        scored = np.where(goal, hteam, -1)
        # ยิงพลาด: ผู้รักษาประตูส่งบอลให้ผู้เล่นฝ่ายรับที่ใกล้ประตูตัวเองที่สุด
        miss = is_shot & ~goal
        ox = self.pos[ar[:, None], opp0[:, None] + np.arange(2)[None, :], 0]  # (N,2)
        ofx = self.frame_x(ox, hteam[:, None])  # ฝ่ายรับป้องกันที่ x' = W
        keeper_to = opp0 + np.argmax(ofx, axis=1)
        new_h = np.where(miss, keeper_to, new_h)

        # 4) แย่งบอล (คนถือบอลที่ไม่ได้ส่ง/ยิง): ช่องเดียวกัน หรือยืนติดกัน
        ut = self.rng.random((N, 2))
        tackled = np.zeros(N, dtype=bool)
        for k in range(2):
            o = opp0 + k
            cheb = np.abs(self.pos[ar, o] - hp).max(-1)
            p = np.where(cheb == 0, self.p_tackle, np.where(cheb == 1, self.p_tackle_adj, 0.0))
            hit = ~is_pass & ~is_shot & ~tackled & (ut[:, k] < p)
            new_h = np.where(hit, o, new_h)
            tackled |= hit
        self.holder = new_h

        # 5) ผู้รักษาประตูสลับฝั่ง
        sw = self.rng.random((N, 2)) < self.p_keeper_switch
        self.keeper = np.where(sw, 1 - self.keeper, self.keeper)

        # 6) reward / done
        g = scored >= 0
        reward[g, scored[g]] = 1.0
        reward[g, 1 - scored[g]] = -1.0
        self.t += 1
        done = g | (self.t >= self.max_steps)
        self.msg = np.asarray(token, dtype=np.int64).copy()
        self.msg_opp = self.msg.copy() if token_opp is None else np.asarray(token_opp, dtype=np.int64).copy()
        info = {
            "goal": scored,
            "pass": is_pass,
            "pass_ok": is_pass & ~intercepted,
            "shot": is_shot,
            "shot_open_side": is_shot & ~shot_at_keeper,
            "tackle": tackled,
            "holder_team": hteam,
        }
        self.reset_envs(np.nonzero(done)[0])
        return self.observe(), self.state(), reward, done, info
