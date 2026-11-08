"""วิเคราะห์ว่า token ที่ agent ส่งหากันเป็น "ภาษา" หรือไม่

สิ่งที่วัด
1. Positive signaling  : MI(token ; สถานการณ์ของผู้พูด) เทียบกับ baseline ที่สับข้อมูล
2. Positive listening  : MI(token ที่ได้ยิน ; action ของผู้ฟัง)
3. Causal intervention : สลับ token เป็นค่าสุ่ม แล้ววัดว่า win rate ตกไหม
     - *_mate : เพื่อนร่วมทีมได้ยิน token สุ่ม (ภาษานี้ช่วยทีมจริงไหม)
     - *_opp  : คู่แข่งได้ยิน token สุ่ม (คู่แข่งได้ประโยชน์จากการดักฟังไหม)
4. พจนานุกรม (lexicon) : heatmap P(token | สถานการณ์)
5. Drift               : ภาษาเปลี่ยนไปเร็วแค่ไหนระหว่าง checkpoint

ตัวอย่าง:
    python analyze.py runs/private_s0 runs/eavesdrop_s0
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from football.agent import TeamPolicy
from football.env import Football2v2
from football.scripted import scripted_actions
from train import CONDITIONS

ROLE_NAMES = ["ฉันถือบอล", "เพื่อนถือบอล", "คู่แข่งถือบอล"]
ROLE_NAMES_EN = ["I have ball", "mate has ball", "opp has ball"]
ZONE_NAMES_EN = ["own third", "middle", "attack third"]


def load(ckpt_path, n_envs=512, seed=123):
    ck = torch.load(ckpt_path, weights_only=False)
    a = ck["args"]
    env = Football2v2(
        n_envs=n_envs, width=a["width"], height=a["height"], vision=a["vision"],
        vocab=a["vocab"], seed=seed, **CONDITIONS[a["cond"]],
    )
    teams = []
    for name in ("red", "blue"):
        t = TeamPolicy(env.obs_dim, env.state_dim, a["vocab"])
        t.load_state_dict(ck[name])
        t.eval()
        teams.append(t)
    a.setdefault("opponent", "selfplay")
    return env, teams, a


def bots(a):
    return (1,) if a.get("opponent") == "scripted" else ()


def features(env):
    """คุณลักษณะสถานการณ์ของผู้เล่นแต่ละคน (N,4) ในมุมมองทีมตัวเอง"""
    N = env.N
    f = {}
    team = np.array([0, 0, 1, 1])
    fx = env.frame_x(env.pos[..., 0], team[None, :])
    f["has_ball"] = (env.holder[:, None] == np.arange(4)[None]).astype(int)
    hteam = env.holder // 2
    mate_has = (env.holder[:, None] == (np.arange(4) ^ 1)[None]).astype(int)
    opp_has = (hteam[:, None] != team[None]).astype(int)
    f["role"] = np.where(f["has_ball"] == 1, 0, np.where(mate_has == 1, 1, 2))
    f["xzone"] = np.minimum(fx * 3 // env.W, 2)
    f["yzone"] = np.minimum(env.pos[..., 1] * 3 // env.H, 2)
    open_ = np.ones((N, 4), dtype=int)
    for i in range(4):
        for o in ((2, 3) if i < 2 else (0, 1)):
            d = np.abs(env.pos[:, i] - env.pos[:, o]).max(-1)
            open_[:, i] &= (d > 1).astype(int)
    f["open"] = open_
    mate = np.arange(4) ^ 1
    f["mate_visible"] = (np.abs(env.pos - env.pos[:, mate]).max(-1) <= env.R).astype(int)
    ball = env.pos[np.arange(N), env.holder]
    f["ball_visible"] = (np.abs(env.pos - ball[:, None]).max(-1) <= env.R).astype(int)
    f["situation"] = f["role"] * 3 + f["xzone"]
    # ผู้รักษาประตูฝั่งที่ผู้เล่นคนนี้บุก: ยืนครึ่งไหน และมองเห็นไหม
    f["keeper_side"] = np.stack([env.keeper[:, 1 - (i // 2)] for i in range(4)], 1)
    f["keeper_visible"] = env.keeper_visible().astype(int)
    return f


def scramble(tok, cols, rng, K):
    out = tok.copy()
    out[:, cols] = rng.integers(0, K, size=(tok.shape[0], len(cols)))
    return out


@torch.no_grad()
def rollout(env, teams, steps=300, interventions=(), seed=0, bot_teams=()):
    """interventions: ชุดของ 'red_mate', 'blue_mate', 'red_opp', 'blue_opp'"""
    rng = np.random.default_rng(seed)
    obs, _ = env.reset()
    N, K = env.N, env.K
    rec = {k: [] for k in ("tok", "mv", "heard", "shot", "shot_open", "shooter")}
    feats = []
    goals = np.zeros(2, dtype=int)
    eps = 0
    for _ in range(steps):
        feats.append(features(env))
        rec["heard"].append(env.msg[:, [1, 0, 3, 2]].copy())  # token ที่แต่ละคนได้ยินจากเพื่อน
        o = torch.from_numpy(obs)
        mv = np.zeros((N, 4), dtype=np.int64)
        tk = np.zeros((N, 4), dtype=np.int64)
        for T in range(2):
            if T in bot_teams:
                mv[:, 2 * T : 2 * T + 2] = scripted_actions(env, T)
                continue
            m, t, _ = teams[T].act(o[:, 2 * T : 2 * T + 2].reshape(N * 2, -1))
            mv[:, 2 * T : 2 * T + 2] = m.view(N, 2).numpy()
            tk[:, 2 * T : 2 * T + 2] = t.view(N, 2).numpy()
        tk_mate, tk_opp = tk.copy(), tk.copy()
        if "red_mate" in interventions:
            tk_mate = scramble(tk_mate, [0, 1], rng, K)
        if "blue_mate" in interventions:
            tk_mate = scramble(tk_mate, [2, 3], rng, K)
        if "red_opp" in interventions:
            tk_opp = scramble(tk_opp, [0, 1], rng, K)
        if "blue_opp" in interventions:
            tk_opp = scramble(tk_opp, [2, 3], rng, K)
        env_holder_before = env.holder.copy()
        obs, _, _, done, info = env.step(mv, tk_mate, tk_opp)
        rec["tok"].append(tk)
        rec["shot"].append(info["shot"])
        rec["shot_open"].append(info["shot_open_side"])
        rec["shooter"].append(env_holder_before)
        rec["mv"].append(mv)
        g = info["goal"]
        goals[0] += (g == 0).sum()
        goals[1] += (g == 1).sum()
        eps += done.sum()
    out = {k: np.stack(v) for k, v in rec.items()}
    out["feat"] = {k: np.stack([f[k] for f in feats]) for k in feats[0]}
    # ยิงไปฝั่งที่ผู้รักษาประตูไม่อยู่ได้บ่อยแค่ไหน แยกตามว่าคนยิงมองเห็นผู้รักษาประตูหรือไม่
    kv = out["feat"]["keeper_visible"]
    S_, N_ = out["shooter"].shape
    shooter_vis = kv[np.arange(S_)[:, None], np.arange(N_)[None, :], out["shooter"]]
    for T, team in enumerate(("red", "blue")):
        m = out["shot"] & (out["shooter"] // 2 == T)
        for name, cond in (("blind", shooter_vis == 0), ("seeing", shooter_vis == 1)):
            mm = m & cond
            out[f"{team}_open_rate_{name}"] = float(out["shot_open"][mm].mean()) if mm.sum() else float("nan")
            out[f"{team}_n_shots_{name}"] = int(mm.sum())
    out["red_win"] = goals[0] / eps
    out["blue_win"] = goals[1] / eps
    out["episodes"] = int(eps)
    return out


def mutual_info(x, y):
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    nx, ny = x.max() + 1, y.max() + 1
    joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny).astype(float)
    joint /= joint.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (px @ py)[nz])).sum())


def mi_excess(x, y, rng, n_shuffle=3):
    """MI ลบ baseline ที่สับ y (ตัด bias ของการประมาณ)"""
    base = np.mean([mutual_info(x, rng.permutation(np.asarray(y).ravel())) for _ in range(n_shuffle)])
    return mutual_info(x, y) - base


def lexicon(tok, situation, K, n_sit=9):
    """P(token | situation) (n_sit, K) และความถี่ของแต่ละ situation"""
    M = np.zeros((n_sit, K))
    np.add.at(M, (situation.ravel(), tok.ravel()), 1)
    freq = M.sum(1)
    return M / np.maximum(freq[:, None], 1), freq / freq.sum()


def jsd(p, q):
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return (a[nz] * np.log2(a[nz] / b[nz])).sum()

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def signaling_listening(r, K, rng):
    res = {}
    for T, name in enumerate(("red", "blue")):
        cols = [2 * T, 2 * T + 1]
        tok = r["tok"][:, :, cols]
        d = {}
        for feat in ("has_ball", "role", "xzone", "yzone", "open", "situation", "keeper_side"):
            d[f"MI_tok_{feat}"] = mi_excess(tok, r["feat"][feat][:, :, cols], rng)
        seen = r["feat"]["keeper_visible"][:, :, cols] == 1
        if seen.sum() > 500:
            d["MI_tok_keeper_side_when_seen"] = mi_excess(tok[seen], r["feat"]["keeper_side"][:, :, cols][seen], rng)
        for k in ("open_rate_blind", "open_rate_seeing", "n_shots_blind", "n_shots_seeing"):
            d[k] = r[f"{name}_{k}"]
        # listening: token ที่ได้ยินจากเพื่อน vs action ของผู้ฟัง (ตัดช่วง step แรกที่ยังไม่มีข้อความ)
        heard = r["heard"][1:, :, cols]
        mv = r["mv"][1:, :, cols]
        d["MI_heard_action"] = mi_excess(heard, mv, rng)
        # listening แบบควบคุมสถานการณ์ของผู้ฟังเอง: เฉลี่ย MI ภายในแต่ละ situation
        sit = r["feat"]["situation"][1:, :, cols]
        tot, acc = 0, 0.0
        for s in range(9):
            m = sit == s
            if m.sum() > 500:
                acc += m.sum() * mi_excess(heard[m], mv[m], rng)
                tot += m.sum()
        d["MI_heard_action_given_sit"] = acc / max(tot, 1)
        p = np.bincount(tok.ravel(), minlength=K) / tok.size
        d["token_entropy"] = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
        res[name] = d
    return res


def plot_lexicon(lex_by_team, freq_by_team, title, path, labels=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if labels is None:
        labels = [f"{ROLE_NAMES_EN[s // 3]} / {ZONE_NAMES_EN[s % 3]}" for s in range(9)]
    for ax, (name, lex) in zip(axes, lex_by_team.items()):
        im = ax.imshow(lex, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([f"{l} ({f:.0%})" for l, f in zip(labels, freq_by_team[name])], fontsize=7)
        ax.set_xticks(range(lex.shape[1]))
        ax.set_xlabel("token")
        ax.set_title(f"{name}: P(token | situation)")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle(title)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def analyze_run(run_dir, steps, n_envs, out_root):
    ckpts = sorted(glob.glob(os.path.join(run_dir, "ckpt_*.pt")))
    name = os.path.basename(run_dir.rstrip("/"))
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    env, teams, a = load(ckpts[-1], n_envs=n_envs)
    K = a["vocab"]
    cond = CONDITIONS[a["cond"]]
    summary = {"run": name, "cond": a["cond"], "ckpt": os.path.basename(ckpts[-1])}

    bt = bots(a)
    base = rollout(env, teams, steps, bot_teams=bt)
    summary["win"] = {"red": base["red_win"], "blue": base["blue_win"], "episodes": base["episodes"]}
    summary["language"] = signaling_listening(base, K, rng)

    # interventions
    iv = {}
    for team_i, team in enumerate(("red", "blue")):
        if cond["speak"][team_i]:
            r = rollout(env, teams, steps, interventions=(f"{team}_mate",), seed=1, bot_teams=bt)
            iv[f"{team}_mate_scrambled"] = {
                "red": r["red_win"], "blue": r["blue_win"],
                f"{team}_open_rate_blind": r[f"{team}_open_rate_blind"],
            }
            if cond["hear_opp"][1 - team_i]:
                r = rollout(env, teams, steps, interventions=(f"{team}_opp",), seed=2, bot_teams=bt)
                iv[f"{team}_opp_scrambled"] = {"red": r["red_win"], "blue": r["blue_win"]}
    summary["interventions"] = iv

    # lexicon ของ checkpoint สุดท้าย
    lex, freq = {}, {}
    for T, team in enumerate(("red", "blue")):
        cols = [2 * T, 2 * T + 1]
        lex[team], freq[team] = lexicon(base["tok"][:, :, cols], base["feat"]["situation"][:, :, cols], K)
    plot_lexicon(lex, freq, f"{name} ({a['cond']})", os.path.join(out_dir, "lexicon.png"))
    summary["lexicon"] = {t: np.round(lex[t], 3).tolist() for t in lex}
    # lexicon เทียบกับผู้รักษาประตู: (มองเห็นไหม) x (ยืนฝั่งไหน)
    klabels = ["keeper unseen / top", "keeper unseen / bottom", "keeper SEEN / top", "keeper SEEN / bottom"]
    lex2, freq2 = {}, {}
    for T, team in enumerate(("red", "blue")):
        cols = [2 * T, 2 * T + 1]
        sit2 = base["feat"]["keeper_visible"][:, :, cols] * 2 + base["feat"]["keeper_side"][:, :, cols]
        lex2[team], freq2[team] = lexicon(base["tok"][:, :, cols], sit2, K, n_sit=4)
    plot_lexicon(lex2, freq2, f"{name}: token vs goalkeeper", os.path.join(out_dir, "lexicon_keeper.png"), klabels)
    summary["lexicon_keeper"] = {t: np.round(lex2[t], 3).tolist() for t in lex2}

    # drift ระหว่าง checkpoint
    drift, prev = [], None
    for c in ckpts:
        e, tm, _ = load(c, n_envs=256)
        r = rollout(e, tm, 150, bot_teams=bt)
        cur = {}
        row = {"ckpt": os.path.basename(c)}
        for T, team in enumerate(("red", "blue")):
            cols = [2 * T, 2 * T + 1]
            cur[team] = lexicon(r["tok"][:, :, cols], r["feat"]["situation"][:, :, cols], K)
            row[f"{team}_MI_situation"] = mi_excess(r["tok"][:, :, cols], r["feat"]["situation"][:, :, cols], rng)
            if prev is not None:
                (L1, f1), (L0, _) = cur[team], prev[team]
                row[f"{team}_jsd_from_prev"] = float(sum(f1[s] * jsd(L1[s], L0[s]) for s in range(9)))
        drift.append(row)
        prev = cur
    summary["drift"] = drift
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--out", default="results")
    args = p.parse_args()
    torch.set_num_threads(1)
    for run in args.runs:
        s = analyze_run(run, args.steps, args.envs, args.out)
        print(json.dumps({k: s[k] for k in ("run", "win", "language", "interventions")}, indent=1, default=float))


if __name__ == "__main__":
    main()
