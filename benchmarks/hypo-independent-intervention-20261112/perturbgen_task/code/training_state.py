"""Portable random-state capture for checkpoint continuation."""
import random
import numpy as np
import torch


def capture_rng(rng):
    n=np.random.get_state()
    return dict(sampler=rng.bit_generator.state,python=random.getstate(),numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]],torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state,rng):
    rng.bit_generator.state=state['sampler'];random.setstate(state['python'])
    n=state['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


def replay_legacy_rng(rng,epochs,train_ids,rows,controls,draw_mask):
    """Replay draws only, for the original fixed 8 x 4 mask-training loop.

    Validation used torch.random.fork_rng and a separate sampling generator.
    Model dropout was zero and the frozen encoder remained in eval mode.
    Native mask draws do not depend on token values or padding lengths.
    """
    for epoch in range(epochs):
        for sid in rng.permutation(train_ids):
            for micro in range(4):
                rng.choice(rows[controls[sid]],8,replace=True);rng.choice(rows[sid],8,replace=True)
                np.random.choice([1]);draw_mask()
        for _ in range(21*8):np.random.choice([1])
