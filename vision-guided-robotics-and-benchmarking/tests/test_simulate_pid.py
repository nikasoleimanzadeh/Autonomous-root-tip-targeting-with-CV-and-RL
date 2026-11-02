"""Offline tests of 10-step settlement logic using an explicit fake simulator."""
from pathlib import Path
import sys
import types
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from simulate_pid import THRESHOLD_M, DWELL_STEPS


def test_pid_step_interface_and_dwell(monkeypatch):
    ot2=types.ModuleType('ot2'); sim_mod=types.ModuleType('ot2.sim')
    sim_mod.precise_pipette_position=lambda sim: sim.pos.copy()
    monkeypatch.setitem(sys.modules,'ot2',ot2)
    monkeypatch.setitem(sys.modules,'ot2.sim',sim_mod)
    from simulate_pid import move_pid
    class FakePid:
        def __init__(self):self.resets=0
        def reset(self):self.resets+=1
        def step(self,goal,current,dt): return np.clip((goal-current)*100, -1, 1)
    class FakeSim:
        def __init__(self):self.pos=np.zeros(3)
        def run(self,actions,num_steps=1):self.pos += np.array(actions[0][:3])*0.01
    sim=FakeSim(); pid=FakePid()
    record=move_pid(sim,pid,np.array([0.03,0.04,0.05]),max_steps=300)
    assert record['settled'] and record['final_error_mm']<1.0
    assert pid.resets==1
    assert record['steps']>=DWELL_STEPS
