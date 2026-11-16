import copy,random,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from training_state import capture_rng,restore_rng,replay_legacy_rng


class TrainingContinuation(unittest.TestCase):
    def test_checkpoint_preserves_next_optimizer_step(self):
        torch.manual_seed(43);np.random.seed(43);random.seed(43);rng=np.random.default_rng(43)
        model=torch.nn.Linear(3,1);optim=torch.optim.AdamW(model.parameters(),lr=.01)
        scheduler=torch.optim.lr_scheduler.StepLR(optim,step_size=1,gamma=.9)
        def step():
            x=torch.randn(5,3)+float(rng.normal())+float(np.random.random())+random.random()
            loss=model(x).square().mean();optim.zero_grad();loss.backward();optim.step();scheduler.step()
            return loss.detach().clone(),copy.deepcopy(model.state_dict())
        step()
        checkpoint=copy.deepcopy(dict(model=model.state_dict(),optimizer=optim.state_dict(),scheduler=scheduler.state_dict(),rng=capture_rng(rng)))
        expected=step()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'state.pt';torch.save(checkpoint,path);state=torch.load(path,weights_only=True)
        model.load_state_dict(state['model']);optim.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);restore_rng(state['rng'],rng)
        actual=step();self.assertTrue(torch.equal(expected[0],actual[0]))
        for k in expected[1]:self.assertTrue(torch.equal(expected[1][k],actual[1][k]))

    def test_legacy_replay_preserves_sampling_and_mask_streams(self):
        ids=['a','b'];rows={'a':np.arange(21),'b':np.arange(31),'c':np.arange(17)};controls={'a':'c','b':'c'}
        def draw():torch.rand(8);torch.rand(8,2048)
        def reset():
            torch.manual_seed(45);np.random.seed(45);random.seed(45)
            return np.random.default_rng(45)
        rng=reset()
        for epoch in range(2):
            for sid in rng.permutation(ids):
                for _ in range(4):
                    rng.choice(rows[controls[sid]],8,replace=True);rng.choice(rows[sid],8,replace=True)
                    np.random.choice([1]);draw()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(4242)
                for _ in range(21*8):np.random.choice([1]);draw()
        expected=(rng.integers(100000,size=20),torch.rand(20),np.random.random(20))
        rng=reset();replay_legacy_rng(rng,2,ids,rows,controls,draw)
        self.assertTrue(np.array_equal(expected[0],rng.integers(100000,size=20)))
        self.assertTrue(torch.equal(expected[1],torch.rand(20)))
        self.assertTrue(np.array_equal(expected[2],np.random.random(20)))


if __name__=='__main__':unittest.main()
