"""Path Y: one consolidated expert-guided SAC training and evaluation run.

Starts with the user's SAC 53k, not PPO or Z. No teacher, PID, or cap is
called at inference. Only one OT-2 PyBullet simulator exists at any time.
Creates a timestamped run directory. Never overwrites original artifacts.

Approach:
  1. Collect real full-range P-expert rollouts, including disturbed starts.
  2. Label old SAC replay observations using the known world-coordinate teacher.
  3. Supervised actor refinement with balanced fine/coarse distance bins.
  4. Add genuine expert transitions to SAC replay and run a brief SAC follow-up.
  5. Pick checkpoint on one fixed selection set; evaluate once on new holdout.

Use from repository root:
  python .\\task11_rl_controller\\Path_Y\\train_path_y_final_push.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for directory in (REPO / 'src', HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from path_y_env import PathYEnv
from evaluate_sac import fixed_cases, evaluate, summarize, save_csv
from ot2.envelope import PRACTICAL_ENVELOPE

SOURCE_RUN = HERE / 'runs' / 'sac_continue43k_20261011_201840_630818'
SOURCE_MODEL = SOURCE_RUN / 'checkpoints' / 'sac_0053000_steps.zip'
SOURCE_REPLAY = SOURCE_RUN / 'checkpoints' / 'sac_0053000_steps.replay.pkl'
ENV_PATH = HERE / 'path_y_env.py'
GAIN = 80.0
TRAIN_SEED = 820161
SELECTION_SEED = 32671841
HOLDOUT_SEED = 32671842
BINS = np.array([0., .001, .003, .010, .060, np.inf])
DIST_NAMES = ['under_1mm', '1to3mm', '3to10mm', '10to60mm', 'over_60mm']


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for piece in iter(lambda: handle.read(1 << 20), b''):
            h.update(piece)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def expert_from_info(info):
    error = np.asarray(info['error_xyz_m'], dtype=np.float64)
    return np.clip(GAIN * error, -1, 1).astype(np.float32)


def expert_from_obs(obs):
    # coarse observation channels are physical target error / envelope extent.
    arr = np.asarray(obs, dtype=np.float32)
    err_m = arr[..., 3:6] * PRACTICAL_ENVELOPE.extent
    return np.clip(GAIN * err_m, -1, 1).astype(np.float32)


def distance_from_obs(obs):
    arr = np.asarray(obs, dtype=np.float32)
    return np.linalg.norm(arr[..., 3:6] * PRACTICAL_ENVELOPE.extent, axis=-1)


def make_case(rng, kind):
    low = PRACTICAL_ENVELOPE.low
    high = PRACTICAL_ENVELOPE.high
    ranges = {
        'micro': (.00015, .00095),
        'small': (.0010, .0030),
        'local': (.0030, .0100),
        'near': (.0100, .0600),
        'mid': (.0600, .1500),
    }
    for _ in range(5000):
        target = rng.uniform(low + .015, high - .015)
        if kind == 'wide':
            start = rng.uniform(low + .010, high - .010)
            valid = np.linalg.norm(start - target) >= .10
        else:
            direction = rng.normal(size=3)
            direction /= max(np.linalg.norm(direction), 1e-8)
            a, b = ranges[kind]
            start = target + rng.uniform(a, b) * direction
            valid = True
        if valid and np.all(start >= low + .010) and np.all(start <= high - .010):
            return start, target
    raise RuntimeError(f'Unable to sample {kind} inside envelope')


def collect_data(env, model, rng, out, demo_counts):
    rows = []
    transitions = []
    teacher_trials = []
    sequence = []
    for kind, count in demo_counts.items():
        sequence.extend([kind] * count)
    rng.shuffle(sequence)
    episode_id = 0
    for kind in sequence:
        start, target = make_case(rng, kind)
        obs, info = env.reset(options={'start': start, 'target': target})
        disturbed = (episode_id % 4 == 0)
        # Disturb 1-2 steps before corrective supervision: teacher sees moving states.
        for step in range(600):
            expert = expert_from_info(info)
            # Store labels for ALL distances, never only the last 10 mm.
            rows.append((episode_id, obs.copy(), expert.copy(), float(info['distance_m'])))
            if disturbed and step < 2:
                # Deliberate bounded disturbance to expose recovery states.
                action = rng.uniform(-.5, .5, size=3).astype(np.float32)
            else:
                action = expert
            next_obs, reward, term, trunc, next_info = env.step(action)
            # Only true expert-controlled actions join offline SAC replay.
            if not (disturbed and step < 2):
                transitions.append((obs.copy(), next_obs.copy(), action.copy(),
                                    float(reward), bool(term or trunc), bool(trunc)))
            obs, info = next_obs, next_info
            if term or trunc:
                break
        teacher_trials.append({
            'id':episode_id, 'kind':kind, 'disturbed':disturbed,
            'settled':bool(info['settled']), 'unsafe':bool(info['unsafe']),
            'steps':int(info['steps']), 'error_mm':float(info['distance_mm'])
        })
        episode_id += 1
        if episode_id % 25 == 0:
            print(f'EXPERT {episode_id}/{len(sequence)} | samples={len(rows):,} '
                  f'| settled={sum(v["settled"] for v in teacher_trials)}/{episode_id}', flush=True)
    save_csv(out/'expert_trials.csv', teacher_trials)
    check(sum(v['settled'] for v in teacher_trials) >= .9*len(teacher_trials),
          'Expert teacher failed on >10% of cases; do not train with failed demonstrations')
    return rows, transitions, teacher_trials


def build_training_arrays(rows, replay, out, rng):
    # Split by *whole expert trajectory* before training. Replay observations
    # were collected on earlier SAC trials and may contain correlated states.
    ids = np.unique([x[0] for x in rows])
    rng.shuffle(ids)
    validation_ids = set(ids[:max(20, int(.20 * len(ids)))].tolist())
    expert_train = [r for r in rows if r[0] not in validation_ids]
    expert_val = [r for r in rows if r[0] in validation_ids]
    X = np.stack([r[1] for r in expert_train]).astype(np.float32)
    Y = np.stack([r[2] for r in expert_train]).astype(np.float32)
    Xval = np.stack([r[1] for r in expert_val]).astype(np.float32)
    Yval = np.stack([r[2] for r in expert_val]).astype(np.float32)
    # Add real state distribution from 53k SAC replay, corrected by the expert.
    n = replay.size()
    sac_obs = np.asarray(replay.observations[:n], dtype=np.float32).reshape(-1, 12)
    check(np.all(np.isfinite(sac_obs)), 'Replay observations contain NaN/Inf')
    # Up to 40k historic learner states, balanced together with expert rollouts.
    indexes = rng.choice(len(sac_obs), size=min(40000, len(sac_obs)), replace=False)
    sac_sample = sac_obs[indexes]
    X = np.concatenate([X, sac_sample], axis=0)
    Y = np.concatenate([Y, expert_from_obs(sac_sample)], axis=0)
    distances = distance_from_obs(X)
    per_bin = [np.flatnonzero((distances >= BINS[i]) & (distances < BINS[i+1]))
               for i in range(5)]
    counts = {DIST_NAMES[i]:len(v) for i,v in enumerate(per_bin)}
    check(all(len(v) >= 50 for v in per_bin), f'Too few examples in some bins: {counts}')
    check(np.all(np.isfinite(Y)), 'Expert action contains NaN/Inf')
    np.savez_compressed(out/'expert_training_data.npz', X=X, Y=Y, Xval=Xval, Yval=Yval)
    print('Training examples by range:', counts, '| independent expert validation:', len(Xval), flush=True)
    return X, Y, Xval, Yval, per_bin, counts


def supervise(model, train, args, out):
    import torch
    X, Y, Xval, Yval, bins, _ = train
    torch.manual_seed(TRAIN_SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rng = np.random.default_rng(TRAIN_SEED + 2)
    actor = model.policy.actor
    actor.train()
    # Independent optimizer; do not mutate existing checkpoint or critic.
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.bc_lr)
    device = model.device
    xt = torch.as_tensor(X, device=device)
    yt = torch.as_tensor(Y, device=device)
    xv = torch.as_tensor(Xval, device=device)
    yv = torch.as_tensor(Yval, device=device)
    checkpoints = {}
    metrics = []
    for update in range(1, args.bc_updates + 1):
        idxs = np.concatenate([rng.choice(b, size=96, replace=True) for b in bins])
        # Mixed examples with equally represented distance scales, including <1mm.
        index_t = torch.as_tensor(idxs, device=device)
        pred = actor(xt[index_t], deterministic=True)
        loss = ((pred - yt[index_t])**2).mean()
        check(bool(torch.isfinite(loss).item()), 'Nonfinite behavior-cloning loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()
        if update % 250 == 0 or update == args.bc_updates:
            actor.eval()
            with torch.no_grad():
                total = 0.
                for i in range(0, len(xv), 2048):
                    total += float(((actor(xv[i:i+2048], deterministic=True)
                                      - yv[i:i+2048])**2).sum().item())
                val_mse = total / (len(xv)*3)
            actor.train()
            metrics.append({'update':update, 'train_loss':float(loss.item()),
                            'expert_val_mse':val_mse})
            print(f'BC {update:4d}/{args.bc_updates} | loss={loss.item():.6f} '
                  f'| independent-expert MSE={val_mse:.6f}', flush=True)
        if update in set([750, 1500, 2500, args.bc_updates]):
            path = out/'checkpoints'/f'bc_{update:04d}.zip'
            model.save(path)
            checkpoints[f'bc_{update}'] = path
    save_csv(out/'bc_training.csv', metrics)
    actor.eval()
    return checkpoints


def add_teacher_replay(model, transitions, rng):
    replay = model.replay_buffer
    check(replay is not None, 'Missing SAC replay')
    # Retain old 53k transitions, append up to 20k supervised trajectories.
    chosen = rng.choice(len(transitions), size=min(20000, len(transitions)), replace=False)
    for i in chosen:
        obs, next_obs, action, reward, done, timeout = transitions[int(i)]
        replay.add(obs.reshape(1, -1), next_obs.reshape(1, -1),
                   action.reshape(1, -1), np.array([reward], dtype=np.float32),
                   np.array([done], dtype=np.float32),
                   [{'TimeLimit.truncated': bool(timeout)}])
    print('SAC replay augmented with', len(chosen), 'real expert transitions;',
          'total replay size=', replay.size(), flush=True)
    return len(chosen)


def eval_one(model, env, cases, label, out):
    rows = evaluate(model, env, cases)
    save_csv(out/f'select_{label}.csv', rows)
    st = summarize(rows)['all']
    print(f'SELECTION {label:13s} | settled={st["settled_count"]}/{st["count"]} '
          f'mean={st["mean_final_error_mm"]:.3f}mm max={st["max_final_error_mm"]:.3f}mm '
          f'unsafe={st["unsafe_count"]}', flush=True)
    return st


def main():
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    import torch
    parser = argparse.ArgumentParser(description='One consolidated Path Y training and eval pipeline')
    parser.add_argument('--bc-updates', type=int, default=3500)
    parser.add_argument('--bc-lr', type=float, default=0.0001)
    parser.add_argument('--rl-steps', type=int, default=3000)
    parser.add_argument('--selection-per-group', type=int, default=20)
    parser.add_argument('--holdout-per-group', type=int, default=50)
    args = parser.parse_args()
    check(1500 <= args.bc_updates <= 10000, 'bc-updates must be >=1500 and <=10000')
    check(args.rl_steps in (0, 3000), 'rl-steps must be 0 or 3000')
    for f in (SOURCE_MODEL, SOURCE_REPLAY, ENV_PATH, HERE/'evaluate_sac.py'):
        check(f.is_file(), f'Missing required source: {f}')
    config_path = SOURCE_RUN/'config.json'
    if config_path.is_file():
        old_config = json.loads(config_path.read_text(encoding='utf-8'))
        check(old_config.get('environment_sha256') == hash_file(ENV_PATH),
              'Path Y environment does not match original 53k replay. Abort.')
    check(abs(PathYEnv.distance_reward(.05) - (-.75*.05/.10-.25*.05/.052-.005))<1e-9,
          'Path Y reward is not V2. Stop.')
    source = SAC.load(SOURCE_MODEL, device='cpu')
    check(source.num_timesteps == 53000, 'Source model is not 53k SAC')
    check(source.observation_space.shape == (12,) and source.action_space.shape == (3,),
          'Wrong observation/action shapes')
    check(np.isclose(source.gamma, .99), 'Unexpected gamma')
    source.load_replay_buffer(SOURCE_REPLAY)
    check(source.replay_buffer.size() in (52999,53000), '53k replay size unexpected')
    run = HERE/'runs'/('final_push_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    (run/'checkpoints').mkdir(parents=True, exist_ok=False)
    print('PATH Y — FULL-RANGE EXPERT-GUIDED SAC FINAL PUSH',flush=True)
    print('Output:',run,flush=True)
    print('Source vanilla 53k protected and untouched:', SOURCE_MODEL,flush=True)
    config = {
        'source_model':str(SOURCE_MODEL), 'source_sha256':hash_file(SOURCE_MODEL),
        'source_replay':str(SOURCE_REPLAY), 'source_replay_sha256':hash_file(SOURCE_REPLAY),
        'environment_sha256':hash_file(ENV_PATH),'seed':TRAIN_SEED,
        'selection_seed':SELECTION_SEED,'holdout_seed':HOLDOUT_SEED,
        'P_teacher_gain':GAIN, 'teacher_only_training':True,
        'teacher_at_inference':False, 'inference_action_caps':False,
        'alg':'SAC + full-workspace expert data + BC actor fine-tune + SAC follow-up',
        'parameters':vars(args),
        'limitation':'Performance must be measured; imitation and SAC can regress.'
    }
    write_json(run/'config.json',config)
    rng = np.random.default_rng(TRAIN_SEED)
    training_env = None
    trained = False
    bc_checkpoints = {}
    followup_checkpoints = {}
    try:
        training_env = PathYEnv(start_strategy='uniform', max_steps=600)
        # New training data seed, unrelated to test cases.
        expert_counts = {'micro':60,'small':45,'local':45,'near':40,'mid':25,'wide':35}
        expert_rows, expert_transitions, teacher_info = collect_data(
            training_env, source, rng, run, expert_counts)
        train = build_training_arrays(expert_rows, source.replay_buffer, run, rng)
        write_json(run/'data_summary.json', {
            'expert_cases':len(teacher_info), 'expert_settled':sum(x['settled'] for x in teacher_info),
            'examples':len(train[0]), 'counts':train[-1], 'expert_transitions':len(expert_transitions)})
    finally:
        if training_env is not None:
            training_env.close()
    # No PyBullet used during supervised training.
    bc_checkpoints = supervise(source, train, args, run)
    # Supervised training changes actor; clear stale vanilla optimizer moments.
    source.policy.actor.optimizer.state.clear()
    final_bc = run/'checkpoints'/'bc_final.zip'
    source.save(final_bc)
    bc_checkpoints['bc_final'] = final_bc
    followup = None
    if args.rl_steps:
        from stable_baselines3.common.utils import set_random_seed
        set_random_seed(TRAIN_SEED + 3)
        followup = SAC.load(final_bc, device='cpu')
        followup.load_replay_buffer(SOURCE_REPLAY)
        followup.policy.actor.optimizer.state.clear()
        used = add_teacher_replay(followup, expert_transitions, rng)
        write_json(run/'replay_augmentation.json', {'injected_expert_steps':used,
          'starting_replay_steps':52999, 'final_replay_size':followup.replay_buffer.size()})
        # One simulator only; no evaluation while training.
        class SaveRL(BaseCallback):
            def _on_step(self):
                r = np.asarray(self.locals['rewards'], dtype=np.float64)
                ob = np.asarray(self.locals['new_obs'], dtype=np.float64)
                if not np.isfinite(r).all() or not np.isfinite(ob).all():
                    raise FloatingPointError('Nonfinite SAC update')
                if self.num_timesteps in (54000,56000):
                    f = run/'checkpoints'/f'guided_rl_{self.num_timesteps}.zip'
                    self.model.save(f)
                    followup_checkpoints[f'guided_rl_{self.num_timesteps}'] = f
                    print('SAC checkpoint',self.num_timesteps,flush=True)
                return True
        env = None
        try:
            env = PathYEnv(start_strategy='uniform', max_steps=600)
            followup.set_env(env)
            followup.set_logger(configure(str(run/'sb3_logs'), ['csv']))
            followup.learn(total_timesteps=args.rl_steps,reset_num_timesteps=False,
                           callback=SaveRL(), progress_bar=False)
            trained = True
        finally:
            followup.save(run/'checkpoints'/'guided_rl_final_or_partial.zip')
            followup.save_replay_buffer(run/'guided_rl_final_or_partial_replay.pkl')
            if env is not None:
                env.close()
        if trained:
            followup_checkpoints['guided_rl_final'] = run/'checkpoints'/'guided_rl_final_or_partial.zip'
    # Serial evaluation. No other simulator exists.
    cases = fixed_cases(seed=SELECTION_SEED, per_group=args.selection_per_group)
    write_json(run/'selection_cases.json', cases)
    candidate_paths = {'vanilla53':SOURCE_MODEL, **bc_checkpoints, **followup_checkpoints}
    scores = {}
    evaluate_env = None
    try:
        evaluate_env = PathYEnv(start_strategy='uniform', max_steps=600)
        for name,path in candidate_paths.items():
            candidate = SAC.load(path,device='cpu')
            scores[name] = eval_one(candidate,evaluate_env,cases,name,run)
    finally:
        if evaluate_env is not None:
            evaluate_env.close()
    # Select by number of settled targets; then mean, maximum error and safety.
    best_name = min(scores, key=lambda label:(scores[label]['unsafe_count'],
                  -scores[label]['settled_count'],scores[label]['mean_final_error_mm'],
                  scores[label]['max_final_error_mm']))
    best_path = candidate_paths[best_name]
    print('BEST SELECTION:', best_name, 'on', len(cases), 'cases',flush=True)
    # The holdout is generated ONCE after selecting. Never use to select/retune.
    holdout = fixed_cases(seed=HOLDOUT_SEED,per_group=args.holdout_per_group)
    write_json(run/'holdout_cases_DO_NOT_TUNE_ON.json',holdout)
    test_env = None
    try:
        test_env = PathYEnv(start_strategy='uniform',max_steps=600)
        selected_model = SAC.load(best_path,device='cpu')
        hold_rows = evaluate(selected_model,test_env,holdout)
        hold_stats = summarize(hold_rows)
        save_csv(run/'final_holdout.csv',hold_rows)
    finally:
        if test_env is not None:
            test_env.close()
    shutil.copy2(best_path,run/'best_learned_sac_model.zip')
    held = hold_stats['all']
    print(f'FRESH HOLDOUT best={best_name} | settled={held["settled_count"]}/{held["count"]} '
          f'| mean={held["mean_final_error_mm"]:.4f}mm '
          f'| max={held["max_final_error_mm"]:.4f}mm unsafe={held["unsafe_count"]}',flush=True)
    top_tier_evidence = bool(held['settled_count']==held['count'] and held['unsafe_count']==0
                             and held['max_final_error_mm'] < 1.0)
    write_json(run/'final_summary.json',{'best_selection':best_name,
      'best_path':str(best_path),'model_for_task12':str(run/'best_learned_sac_model.zip'),
      'selection_scores':scores,'holdout_stats':hold_stats,
      'strict_submm_all_targets_test':top_tier_evidence,
      'status':'completed',
      'notes':'Selected using validation only. No PID, caps or teacher at inference; '
              'not an official grading decision. Task 12 integration not tested.'})
    print('STRICT ALL-HOLDOUT UNDER-1MM + SETTLED:',top_tier_evidence,flush=True)
    print('DONE: results and candidate model at',run,flush=True)

if __name__=='__main__':
    main()
