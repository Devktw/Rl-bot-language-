"""เทรนฟุตบอล 2v2 ด้วย PPO (self-play สองทีม แต่ละทีมมีเครือข่ายของตัวเอง)

ตัวอย่าง:
    python train.py --cond private --updates 300 --seed 0
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from football.agent import TeamPolicy
from football.env import Football2v2
from football.scripted import scripted_actions

# speak: ทีมนั้นส่ง token ได้ไหม, hear_opp: ทีมนั้นได้ยิน token ของคู่แข่งไหม
CONDITIONS = {
    "nocomm": dict(speak=(False, False), hear_opp=(False, False)),
    "asym": dict(speak=(True, False), hear_opp=(False, False)),  # แดงคุยได้ น้ำเงินใบ้
    "private": dict(speak=(True, True), hear_opp=(False, False)),
    "eavesdrop": dict(speak=(True, True), hear_opp=(True, True)),  # ทุกคนได้ยินทุกคน
    "spy": dict(speak=(True, False), hear_opp=(False, True)),  # แดงคุย น้ำเงินแอบฟัง
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--cond", default="private", choices=CONDITIONS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=64, help="rollout length")
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--vocab", type=int, default=8)
    p.add_argument("--width", type=int, default=10)
    p.add_argument("--height", type=int, default=6)
    p.add_argument("--vision", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--ent_move", type=float, default=0.01)
    p.add_argument("--ent_msg", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--out", default="runs")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--opponent", default="selfplay", choices=["selfplay", "scripted"],
                   help="scripted = ทีมน้ำเงินเป็นบอทเขียนกฎ ไม่ถูกเทรน")
    p.add_argument("--tag", default="", help="ต่อท้ายชื่อโฟลเดอร์ run")
    return p.parse_args()


def make_env(args, n_envs=None, seed=None):
    return Football2v2(
        n_envs=n_envs or args.envs,
        width=args.width,
        height=args.height,
        vision=args.vision,
        vocab=args.vocab,
        seed=args.seed if seed is None else seed,
        **CONDITIONS[args.cond],
    )


def gae(rew, val, done, last_val, gamma, lam):
    S = rew.shape[0]
    adv = torch.zeros_like(rew)
    last = torch.zeros_like(last_val)
    for t in reversed(range(S)):
        nv = last_val if t == S - 1 else val[t + 1]
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nv * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + val


def main():
    args = parse()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    name = f"{args.cond}{'_vs_bot' if args.opponent == 'scripted' else ''}{args.tag}_s{args.seed}"
    run_dir = os.path.join(args.out, name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    env = make_env(args)
    teams = [TeamPolicy(env.obs_dim, env.state_dim, args.vocab) for _ in range(2)]
    opts = [torch.optim.Adam(t.parameters(), lr=args.lr, eps=1e-5) for t in teams]
    S, N = args.steps, args.envs
    bot_teams = (1,) if args.opponent == "scripted" else ()

    obs_np, st_np = env.reset()
    log_f = open(os.path.join(run_dir, "log.csv"), "w", newline="")
    log = None
    t0 = time.time()
    for upd in range(1, args.updates + 1):
        buf = {
            k: [] for k in ("obs", "st", "mv", "tk", "lp", "val", "rew", "done")
        }
        ep_stats = dict(goal_red=0, goal_blue=0, draw=0, passes=0, pass_ok=0, shots=0, tackles=0)
        tok_hist = np.zeros((2, args.vocab))
        for _ in range(S):
            obs = torch.from_numpy(obs_np)
            st = torch.from_numpy(st_np)
            mv = np.zeros((N, 4), dtype=np.int64)
            tk = np.zeros((N, 4), dtype=np.int64)
            lps, vals = [], []
            for T in range(2):
                if T in bot_teams:
                    mv[:, 2 * T : 2 * T + 2] = scripted_actions(env, T)
                    lps.append(torch.zeros(N, 2))
                    vals.append(torch.zeros(N))
                    continue
                o = obs[:, 2 * T : 2 * T + 2].reshape(N * 2, -1)
                m, t, lp = teams[T].act(o)
                mv[:, 2 * T : 2 * T + 2] = m.view(N, 2).numpy()
                tk[:, 2 * T : 2 * T + 2] = t.view(N, 2).numpy()
                lps.append(lp.view(N, 2))
                with torch.no_grad():
                    vals.append(teams[T].value(st[:, T]))
                tok_hist[T] += np.bincount(tk[:, 2 * T : 2 * T + 2].ravel(), minlength=args.vocab)
            obs_np, st_np, rew, done, info = env.step(mv, tk)
            buf["obs"].append(obs)
            buf["st"].append(st)
            buf["mv"].append(torch.from_numpy(mv))
            buf["tk"].append(torch.from_numpy(tk))
            buf["lp"].append(torch.stack(lps, 1))  # (N,2team,2player)
            buf["val"].append(torch.stack(vals, 1))  # (N,2)
            buf["rew"].append(torch.from_numpy(rew))
            buf["done"].append(torch.from_numpy(done.astype(np.float32)))
            g = info["goal"]
            ep_stats["goal_red"] += int((g == 0).sum())
            ep_stats["goal_blue"] += int((g == 1).sum())
            ep_stats["draw"] += int((done & (g < 0)).sum())
            ep_stats["passes"] += int(info["pass"].sum())
            ep_stats["pass_ok"] += int(info["pass_ok"].sum())
            ep_stats["shots"] += int(info["shot"].sum())
            ep_stats["tackles"] += int(info["tackle"].sum())

        B = {k: torch.stack(v) for k, v in buf.items()}
        st_last = torch.from_numpy(st_np)
        metrics = {}
        for T in range(2):
            if T in bot_teams:
                continue
            pol, opt = teams[T], opts[T]
            with torch.no_grad():
                last_v = pol.value(st_last[:, T])
            adv, ret = gae(B["rew"][:, :, T], B["val"][:, :, T], B["done"], last_v, args.gamma, args.lam)
            # flatten: ตัวอย่างละผู้เล่น (S*N*2) ใช้ advantage ของทีม
            o = B["obs"][:, :, 2 * T : 2 * T + 2].reshape(S * N * 2, -1)
            m = B["mv"][:, :, 2 * T : 2 * T + 2].reshape(-1)
            tk = B["tk"][:, :, 2 * T : 2 * T + 2].reshape(-1)
            lp_old = B["lp"][:, :, T].reshape(-1)
            a = adv.unsqueeze(-1).expand(S, N, 2).reshape(-1)
            a = (a - a.mean()) / (a.std() + 1e-8)
            st_flat = B["st"][:, :, T].reshape(S * N, -1)
            ret_flat = ret.reshape(-1)
            n_pl, n_st = o.shape[0], st_flat.shape[0]
            mb_pl = args.minibatch
            n_mb = max(1, n_pl // mb_pl)
            mb_st = n_st // n_mb
            for _ in range(args.epochs):
                perm_p = torch.randperm(n_pl)
                perm_s = torch.randperm(n_st)
                for k in range(n_mb):
                    ip = perm_p[k * mb_pl : (k + 1) * mb_pl]
                    is_ = perm_s[k * mb_st : (k + 1) * mb_st]
                    dm, dt = pol.dists(o[ip])
                    lp = dm.log_prob(m[ip]) + dt.log_prob(tk[ip])
                    ratio = (lp - lp_old[ip]).exp()
                    pg = -torch.min(
                        ratio * a[ip], ratio.clamp(1 - args.clip, 1 + args.clip) * a[ip]
                    ).mean()
                    v = pol.value(st_flat[is_])
                    vl = 0.5 * ((v - ret_flat[is_]) ** 2).mean()
                    ent_m, ent_t = dm.entropy().mean(), dt.entropy().mean()
                    loss = pg + args.vf_coef * vl - args.ent_move * ent_m - args.ent_msg * ent_t
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
                    opt.step()
            team = "red" if T == 0 else "blue"
            metrics[f"{team}_ent_move"] = ent_m.item()
            metrics[f"{team}_ent_msg"] = ent_t.item()
            metrics[f"{team}_vloss"] = vl.item()
            p = tok_hist[T] / tok_hist[T].sum()
            metrics[f"{team}_tok_entropy"] = float(-(p[p > 0] * np.log(p[p > 0])).sum())

        n_ep = ep_stats["goal_red"] + ep_stats["goal_blue"] + ep_stats["draw"]
        row = dict(
            update=upd,
            env_steps=upd * S * N,
            episodes=n_ep,
            red_win=ep_stats["goal_red"] / max(n_ep, 1),
            blue_win=ep_stats["goal_blue"] / max(n_ep, 1),
            draw=ep_stats["draw"] / max(n_ep, 1),
            pass_per_ep=ep_stats["passes"] / max(n_ep, 1),
            pass_success=ep_stats["pass_ok"] / max(ep_stats["passes"], 1),
            shots_per_ep=ep_stats["shots"] / max(n_ep, 1),
            tackles_per_ep=ep_stats["tackles"] / max(n_ep, 1),
            ep_len=S * N / max(n_ep, 1),
            sps=upd * S * N / (time.time() - t0),
            **metrics,
        )
        if log is None:
            log = csv.DictWriter(log_f, fieldnames=list(row))
            log.writeheader()
        log.writerow(row)
        log_f.flush()
        if upd % 10 == 0 or upd == 1:
            print(
                f"[{args.cond} s{args.seed}] upd {upd} steps {row['env_steps']:,} "
                f"red {row['red_win']:.2f} blue {row['blue_win']:.2f} draw {row['draw']:.2f} "
                f"len {row['ep_len']:.1f} pass {row['pass_per_ep']:.2f}({row['pass_success']:.2f}) "
                f"tokH {metrics['red_tok_entropy']:.2f} "
                f"sps {row['sps']:.0f}",
                flush=True,
            )
        if upd % args.save_every == 0 or upd == args.updates:
            torch.save(
                {"red": teams[0].state_dict(), "blue": teams[1].state_dict(), "args": vars(args), "update": upd},
                os.path.join(run_dir, f"ckpt_{upd:05d}.pt"),
            )
    log_f.close()


if __name__ == "__main__":
    main()
