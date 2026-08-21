from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv
from historical_version_guard import require_exact_version

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation'/'rl_benchmark_v0.6.json'

ACTIONS=[]
def a(o2=0.0, vent=0.0):
    x=np.zeros(8,dtype=np.float32); x[4]=o2; x[5]=vent; return x
ACTIONS=[a(),a(0,.08),a(0,.20),a(.10,.12),a(.20,0)]

CO2_BINS=np.array([42,48,54,62,75],dtype=float)
PH_BINS=np.array([7.20,7.30,7.36,7.42,7.50],dtype=float)
SPO2_BINS=np.array([90,94,97,99],dtype=float)


def state_key(info):
    s=info['state']
    return (int(np.digitize(s['paco2_mmHg'],CO2_BINS)),int(np.digitize(s['ph_arterial'],PH_BINS)),int(np.digitize(s['spo2_pct'],SPO2_BINS)))


def run_policy(policy, seed, episode_minutes=30):
    env=HumanHomeostasisEnv(config=HumanConfig(episode_minutes=episode_minutes),scenario='respiratory_acidosis')
    _,info=env.reset(seed=seed); total=0.0; steps=0
    while True:
        ai=int(policy(info))
        _,r,term,trunc,info=env.step(ACTIONS[ai]); total+=float(r); steps+=1
        if term or trunc: break
    return total,info,steps


def train_q(episodes=40, seed=20260817):
    rng=np.random.default_rng(seed); Q={}; alpha=.25; gamma=.92
    eps0=.45; eps1=.05
    for ep in range(episodes):
        env=HumanHomeostasisEnv(config=HumanConfig(episode_minutes=30),scenario='respiratory_acidosis')
        _,info=env.reset(seed=1000+ep)
        eps=eps0+(eps1-eps0)*(ep/max(1,episodes-1))
        while True:
            k=state_key(info); q=Q.setdefault(k,np.zeros(len(ACTIONS),dtype=float))
            if rng.random()<eps:
                ai=int(rng.integers(len(ACTIONS)))
            else:
                best=np.flatnonzero(q==q.max()); ai=int(rng.choice(best))
            _,r,term,trunc,ninfo=env.step(ACTIONS[ai])
            nk=state_key(ninfo); nq=Q.setdefault(nk,np.zeros(len(ACTIONS),dtype=float))
            target=float(r) if (term or trunc) else float(r)+gamma*float(nq.max())
            q[ai]+=alpha*(target-q[ai]); info=ninfo
            if term or trunc: break
    return Q


def evaluate(name,policy,seeds):
    vals=[]; finals=[]
    for seed in seeds:
        ret,info,steps=run_policy(policy,seed)
        vals.append(ret); s=info['state']; finals.append({'paco2':s['paco2_mmHg'],'pH':s['ph_arterial'],'spo2':s['spo2_pct'],'steps':steps})
    return {'name':name,'returns':vals,'mean_return':float(np.mean(vals)),'sd_return':float(np.std(vals)),'finals':finals}


def main():
    require_exact_version("0.6.0")
    rng=np.random.default_rng(99)
    Q=train_q()
    learned=lambda info: int(np.argmax(Q.get(state_key(info),np.zeros(len(ACTIONS)))))
    noop=lambda info:0
    # deterministic per-state random baseline is not wanted; use a closure RNG.
    random_policy=lambda info:int(rng.integers(len(ACTIONS)))
    heuristic=lambda info: (3 if info['state']['spo2_pct']<94 else (2 if info['state']['paco2_mmHg']>48 else (1 if info['state']['paco2_mmHg']>43 else 0)))
    seeds=list(range(200,210))
    results=[evaluate('no_op',noop,seeds),evaluate('random_discrete',random_policy,seeds),evaluate('heuristic',heuristic,seeds),evaluate('tabular_q_learning',learned,seeds)]
    qstates=len(Q)
    result={
      'task':'30-min respiratory-acidosis control through discrete wrapper over native continuous actions',
      'training':{'algorithm':'tabular Q-learning','episodes':40,'q_states_visited':qstates,'actions':{'0':'none','1':'low ventilation','2':'medium ventilation','3':'low O2 + ventilation','4':'O2 only'}},
      'evaluation_seeds':seeds,
      'results':results,
      'interpretation':'This is an algorithmic learnability smoke benchmark, not evidence of clinical-policy validity. The default OpenHumSim observation remains partially observable; recurrent/function-approximation policies should be benchmarked when Gymnasium/SB3 are available.'
    }
    OUT.write_text(json.dumps(result,indent=2))
    for x in results: print(x['name'],round(x['mean_return'],6),'+/-',round(x['sd_return'],6))
    print('Q states',qstates)

if __name__=='__main__': main()
