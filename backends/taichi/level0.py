"""Experimental DiffTaichi-style reverse-mode Level 0 Euler backend.
Rust owns graph compilation, codecs, parameters, and the numerical reference.
States are time-indexed to satisfy Taichi's global data access rules.
"""
import numpy as np
import taichi as ti

@ti.data_oriented
class Level0:
    def __init__(self,fixture,batch=1,dtype=ti.f32):
        if fixture.get('schema_version')!=1 or fixture.get('method')!='euler':
            raise ValueError('Expected schema 1 Euler reference')
        self.n=int(fixture['neurons']);self.steps=int(fixture['steps'])
        self.m=len(fixture['pre']);self.g=len(fixture['gap_a']);self.batch=batch
        self.p=len(fixture['parameters_raw']);self.dt=float(fixture['dt'])
        self.dtype=dtype;self.npdtype=np.float64 if dtype==ti.f64 else np.float32
        if self.p!=6*self.n+2*self.m+self.g+1 or batch<1 or self.steps<1 or self.dt<=0:
            raise ValueError('Invalid Level 0 fixture')
        self.raw=ti.field(dtype,shape=self.p,needs_grad=True)
        self.prepared=ti.field(dtype,shape=self.p,needs_grad=True)
        shape=(self.steps+1,batch,self.n)
        self.voltage=ti.field(dtype,shape=shape,needs_grad=True)
        self.calcium=ti.field(dtype,shape=shape,needs_grad=True)
        self.gate=ti.field(dtype,shape=shape,needs_grad=True)
        self.loss=ti.field(dtype,shape=(),needs_grad=True)
        self.target=ti.field(dtype,shape=(self.steps+1,self.n))
        self.weight=ti.field(dtype,shape=(self.steps+1,self.n))
        self.current=ti.field(dtype,shape=self.n)
        self.pre=ti.field(ti.i32,shape=max(1,self.m))
        self.edge_parameter=ti.field(ti.i32,shape=max(1,self.m))
        self.count=ti.field(dtype,shape=max(1,self.m))
        self.offset=ti.field(ti.i32,shape=self.n+1)
        gaps=[[] for _ in range(self.n)]
        for edge,(a,b,size) in enumerate(zip(fixture['gap_a'],fixture['gap_b'],fixture['gap_sizes'])):
            gaps[a].append((b,edge,size));gaps[b].append((a,edge,size))
        flat=[entry for group in gaps for entry in group]
        self.gap_offset=ti.field(ti.i32,shape=self.n+1)
        self.gap_peer=ti.field(ti.i32,shape=max(1,len(flat)))
        self.gap_parameter=ti.field(ti.i32,shape=max(1,len(flat)))
        self.gap_size=ti.field(dtype,shape=max(1,len(flat)))
        self.raw.from_numpy(np.asarray(fixture['parameters_raw'],dtype=self.npdtype))
        self.current.from_numpy(np.asarray(fixture['current'],dtype=self.npdtype))
        if self.m:
            self.pre.from_numpy(np.asarray(fixture['pre'],dtype=np.int32))
            self.edge_parameter.from_numpy(np.asarray(fixture['parameter_edge'],dtype=np.int32))
            self.count.from_numpy(np.asarray(fixture['counts'],dtype=self.npdtype))
        self.max_chemical_degree=max(np.diff(fixture['incoming_offsets']), default=0)
        self.max_gap_degree=max((len(group) for group in gaps),default=0)
        self.offset.from_numpy(np.asarray(fixture['incoming_offsets'],dtype=np.int32))
        self.gap_offset.from_numpy(np.asarray([0]+list(np.cumsum([len(g) for g in gaps])),dtype=np.int32))
        if flat:
            self.gap_peer.from_numpy(np.asarray([x[0] for x in flat],dtype=np.int32))
            self.gap_parameter.from_numpy(np.asarray([x[1] for x in flat],dtype=np.int32))
            self.gap_size.from_numpy(np.asarray([x[2] for x in flat],dtype=self.npdtype))
        target=np.asarray([[0.0 if v is None else v for v in row] for row in fixture['targets']],dtype=self.npdtype)
        mask=np.asarray([[v is not None for v in row] for row in fixture['targets']])
        weight=mask*np.asarray(fixture['confidence'],dtype=self.npdtype)[None,:]
        self.normalizer=float(weight.sum()*batch)
        if self.normalizer<=0:raise ValueError('No observed targets')
        self.target.from_numpy(target);self.weight.from_numpy(weight.astype(self.npdtype))

    @ti.func
    def sigmoid(self,x):
        value=0.0
        if x>=0:value=1.0/(1.0+ti.exp(-x))
        else:
            e=ti.exp(x);value=e/(1.0+e)
        return value

    @ti.func
    def positive(self,x):
        value=0.0
        if x>0:value=x+ti.log(1.0+ti.exp(-x))
        else:value=ti.log(1.0+ti.exp(x))
        return value+1e-9

    @ti.kernel
    def prepare(self):
        for i in range(self.p):
            value=self.raw[i]
            if self.n<=i<3*self.n:self.prepared[i]=value
            elif 6*self.n+self.m<=i<6*self.n+2*self.m:self.prepared[i]=2.0*self.sigmoid(value)-1.0
            else:self.prepared[i]=self.positive(value)

    @ti.kernel
    def initialize(self):
        for b,i in ti.ndrange(self.batch,self.n):
            v=self.prepared[self.n+i]
            r=self.sigmoid((v-self.prepared[2*self.n+i])*self.prepared[3*self.n+i])
            self.voltage[0,b,i]=v
            self.calcium[0,b,i]=r
            self.gate[0,b,i]=r/(1.0+r)

    @ti.kernel
    def advance(self,t:ti.i32):
        for b,i in ti.ndrange(self.batch,self.n):
            v=self.voltage[t,b,i]
            r=self.sigmoid((v-self.prepared[2*self.n+i])*self.prepared[3*self.n+i])
            current=-(v-self.prepared[self.n+i])+self.current[i]
            # Static slots avoid incorrect Metal reverse gradients observed with
            # dynamic nested CSR loops in Taichi 1.7.4. Keep the regression audit.
            for slot in ti.static(range(self.max_chemical_degree)):
                if self.offset[i]+slot<self.offset[i+1]:
                    edge=self.offset[i]+slot
                    a=self.pre[edge];p=self.edge_parameter[edge]
                    strength=self.prepared[6*self.n+p]*self.count[edge]
                    reversal=self.prepared[6*self.n+self.m+p]
                    current+=strength*self.gate[t,b,a]*(reversal-v)
            for slot in ti.static(range(self.max_gap_degree)):
                if self.gap_offset[i]+slot<self.gap_offset[i+1]:
                    edge=self.gap_offset[i]+slot
                    a=self.gap_peer[edge];p=self.gap_parameter[edge]
                    conductance=self.prepared[6*self.n+2*self.m+p]*self.gap_size[edge]
                    current+=conductance*(self.voltage[t,b,a]-v)
            self.voltage[t+1,b,i]=v+self.dt*current/self.prepared[i]
            c=self.calcium[t,b,i]
            self.calcium[t+1,b,i]=c+self.dt*(r-c)/self.prepared[4*self.n+i]
            s=self.gate[t,b,i]
            self.gate[t+1,b,i]=s+self.dt*(r*(1.0-s)-s)/self.prepared[self.p-1]

    @ti.kernel
    def observe(self):
        for t,b,i in ti.ndrange(self.steps+1,self.batch,self.n):
            predicted=self.calcium[t,b,i]*self.prepared[5*self.n+i]
            error=predicted-self.target[t,i]
            self.loss[None]+=self.weight[t,i]*error*error/self.normalizer

    def rollout(self):
        self.prepare();self.initialize()
        for t in range(self.steps):self.advance(t)
        self.observe()

    def forward(self):
        self.loss[None]=0.0
        self.rollout();ti.sync()
        return float(self.loss[None])

    def value_and_grad(self,validation=False):
        with ti.ad.Tape(self.loss,validation=validation):self.rollout()
        ti.sync()
        return float(self.loss[None]),self.raw.grad.to_numpy()
