"""Segment recomputation with boundary adjoints; no truncated backpropagation.

Only checkpoint primals and one differentiable window stay resident. Targets
remain resident for the complete trial. Rust owns the independent reference.
"""
import numpy as np
import taichi as ti
from level0 import Level0, replay_safe_tape


@ti.data_oriented
class CheckpointLevel0(Level0):
    def __init__(self, fixture, checkpoint_steps, batch=1, dtype=ti.f32):
        if checkpoint_steps < 1:
            raise ValueError('checkpoint_steps must be positive')
        window = min(checkpoint_steps, int(fixture['steps']))
        super().__init__(fixture, batch=batch, dtype=dtype, state_steps=window)
        self.blocks = (self.steps + window - 1) // window
        shape = (self.blocks, batch, self.n)
        self.saved_v = ti.field(dtype, shape=shape)
        self.saved_c = ti.field(dtype, shape=shape)
        self.saved_s = ti.field(dtype, shape=shape)
        self.seed_v = ti.field(dtype, shape=(batch, self.n))
        self.seed_c = ti.field(dtype, shape=(batch, self.n))
        self.seed_s = ti.field(dtype, shape=(batch, self.n))
        self.total_gradient = ti.field(dtype, shape=self.p)

    @ti.kernel
    def save_boundary(self, block: ti.i32):
        for b, i in ti.ndrange(self.batch, self.n):
            self.saved_v[block,b,i] = self.voltage[0,b,i]
            self.saved_c[block,b,i] = self.calcium[0,b,i]
            self.saved_s[block,b,i] = self.gate[0,b,i]

    @ti.kernel
    def restore_boundary(self, block: ti.i32):
        for b, i in ti.ndrange(self.batch, self.n):
            self.voltage[0,b,i] = self.saved_v[block,b,i]
            self.calcium[0,b,i] = self.saved_c[block,b,i]
            self.gate[0,b,i] = self.saved_s[block,b,i]

    @ti.kernel
    def move_boundary(self, end: ti.i32):
        for b, i in ti.ndrange(self.batch, self.n):
            self.voltage[0,b,i] = self.voltage[end,b,i]
            self.calcium[0,b,i] = self.calcium[end,b,i]
            self.gate[0,b,i] = self.gate[end,b,i]

    @ti.kernel
    def observe_block(self, start: ti.i32, end: ti.i32):
        for t,b,i in ti.ndrange(end+1,self.batch,self.n):
            # Shared boundary belongs to the preceding block. Initial t=0 is
            # included exactly once, so every observed sample has one owner.
            if t > 0 or start == 0:
                error = self.calcium[t,b,i]*self.prepared[5*self.n+i]-self.target[start+t,i]
                self.loss[None] += self.weight[start+t,i]*error*error/self.normalizer

    @ti.kernel
    def terminal_objective(self, end: ti.i32):
        for b,i in ti.ndrange(self.batch,self.n):
            self.loss[None] += (self.voltage[end,b,i]*self.seed_v[b,i]
                               + self.calcium[end,b,i]*self.seed_c[b,i]
                               + self.gate[end,b,i]*self.seed_s[b,i])

    @ti.kernel
    def capture_boundary_adjoint(self):
        for b,i in ti.ndrange(self.batch,self.n):
            self.seed_v[b,i] = self.voltage.grad[0,b,i]
            self.seed_c[b,i] = self.calcium.grad[0,b,i]
            self.seed_s[b,i] = self.gate.grad[0,b,i]

    @ti.kernel
    def accumulate_gradient(self):
        for i in range(self.p):
            self.total_gradient[i] += self.raw.grad[i]

    def segments(self):
        for block in range(self.blocks):
            start = block*self.state_steps
            yield block, start, min(self.state_steps,self.steps-start)

    def forward(self):
        self.loss[None] = 0.0
        self.prepare(); self.initialize()
        for block,start,length in self.segments():
            self.save_boundary(block)
            for t in range(length):self.advance(t)
            self.observe_block(start,length)
            if block+1 < self.blocks:self.move_boundary(length)
        ti.sync()
        return float(self.loss[None])

    def value_and_grad(self, validation=False):
        # The first pass stores primals only; reverse replays one window at a
        # time. Tape clears window/parameter adjoints but not our untracked
        # checkpoint, carry, or gradient-accumulator fields.
        value = self.forward()
        self.seed_v.fill(0); self.seed_c.fill(0); self.seed_s.fill(0)
        self.total_gradient.fill(0)
        for block,start,length in reversed(list(self.segments())):
            self.restore_boundary(block)
            with replay_safe_tape(self.loss,validation=validation):
                self.prepare()
                if block == 0:self.initialize()
                for t in range(length):self.advance(t)
                self.observe_block(start,length)
                self.terminal_objective(length)
            self.capture_boundary_adjoint()
            self.accumulate_gradient()
        ti.sync()
        return value,self.total_gradient.to_numpy()

    def audit_states(self, reference, atol, rtol):
        """Replay and compare all frames without assembling a full device tape.

        Only used outside measured timing. The host reference is audit overhead.
        """
        passed = True
        max_v = max_f = 0.0
        self.prepare(); self.initialize()
        scale = self.prepared.to_numpy()[5*self.n:6*self.n][None,None,:]
        for block,start,length in self.segments():
            for t in range(length):self.advance(t)
            voltage = self.voltage.to_numpy()[:length+1]
            fluorescence = self.calcium.to_numpy()[:length+1]*scale
            expected_v = np.asarray(reference['voltage'][start:start+length+1])[:,None,:]
            expected_f = np.asarray(reference['fluorescence'][start:start+length+1])[:,None,:]
            passed &= bool(np.allclose(voltage,expected_v,atol=atol,rtol=rtol)
                           and np.allclose(fluorescence,expected_f,atol=atol,rtol=rtol))
            max_v = max(max_v,float(np.max(np.abs(voltage-expected_v))))
            max_f = max(max_f,float(np.max(np.abs(fluorescence-expected_f))))
            if block+1 < self.blocks:self.move_boundary(length)
        return passed,max_v,max_f

    def memory_accounting(self):
        scalar_bytes = np.dtype(self.npdtype).itemsize
        frame_bytes = 3*self.batch*self.n*scalar_bytes
        return {
            'window_state_and_adjoint_bytes':2*(self.state_steps+1)*frame_bytes,
            'checkpoint_primal_bytes':self.blocks*frame_bytes,
            'boundary_adjoint_bytes':frame_bytes,
            'gradient_accumulator_bytes':self.p*scalar_bytes,
            'full_tape_state_and_adjoint_bytes':2*(self.steps+1)*frame_bytes,
            'resident_target_and_weight_bytes':2*(self.steps+1)*self.n*scalar_bytes,
        }
