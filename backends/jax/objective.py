"""Pure JAX atlas objective; data selection and sufficient statistics come from Rust."""
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from extensions import initialize


def build(model, graph, training, configuration=None):
    if training.get('schema_version')!=2:
        raise ValueError('training export requires schema version 2')
    for key in ['graph_hash','dataset_hash','split_hash','training_trials']:
        if model[key]!=training[key]:
            raise ValueError(f'training export lineage mismatch: {key}')
    names=training['names']
    if names!=sorted(n['id'] for n in graph['neurons']):
        raise ValueError('training neuron order mismatch')
    n=len(names)
    index={name:i for i,name in enumerate(names)}
    chemical=sorted(graph['chemical'],key=lambda e:(e['pre'],e['post']))
    gaps=sorted(graph['gaps'],key=lambda e:(e['a'],e['b']))
    if ([[index[e['pre']],index[e['post']],e['synapse_count']] for e in chemical]!=training['chemical_topology']
            or [[index[e['a']],index[e['b']],e['size']] for e in gaps]!=training['gap_topology']):
        raise ValueError('JAX graph differs from authoritative Rust topology')
    times=training['groups'][0]['recording']['times']
    engine,theta,active=initialize(model,graph,times,configuration)
    classifier=model.get('classifier')
    total=sum(g['sample_weight'] for g in training['groups'])
    pairs=training['classification_pairs']
    groups=[]
    correlation_pairs=0
    for g in training['groups']:
        rec=g['recording']
        if rec['times']!=times:
            raise ValueError('JAX atlas fitting requires a common grid')
        mean=np.zeros((len(times),n)); weights=np.zeros(n); eligible=np.zeros(n)
        for trace in rec['traces']:
            i=index[trace['neuron']]
            mean[:,i]=trace['values']
            weights[i]=trace['provenance']['id_confidence']
            eligible[i]=weights[i]>0 and np.any(mean[:,i]!=mean[0,i])
        labels=np.zeros(n); label_mask=np.zeros(n)
        for i,y in g['labels']:
            labels[i]=y; label_mask[i]=1
        scale=g['sample_weight']/total
        weights=weights/(weights.sum()*len(times))*scale
        correlation_pairs+=int(eligible.sum())
        groups.append(tuple(jnp.asarray(v) for v in (g['target'],mean,weights,scale*g['irreducible_mse'],labels,label_mask,eligible)))
    if (classifier is not None)!=(pairs>0) or sum(float(g[5].sum()) for g in groups)!=pairs:
        raise ValueError('training pair count mismatch')
    config=model['config']
    correlation=config.get('correlation')
    if correlation and correlation_pairs==0:
        raise ValueError('no eligible training correlation pairs')

    def freeze(p):
        return jax.tree.map(lambda v,a:jnp.where(a,v,jax.lax.stop_gradient(v)),p,active)

    def data_loss(p, group):
        p=freeze(p)
        target,mean,weights,floor,labels,label_mask,eligible=group
        prediction=engine.response(p,target)
        mse=jnp.sum((prediction-mean)**2*weights)+floor
        bce=jnp.asarray(0.); shape=jnp.asarray(0.)
        loss=mse
        if classifier:
            eps=classifier['epsilon']
            area=model['sample_dt']*jnp.sum(prediction*(prediction/(jnp.hypot(prediction,eps)+eps)),axis=0)
            logits=p['classifier'][0]+jax.nn.softplus(p['classifier'][1])*jnp.log1p(area/classifier['area_scale'])
            bce=jnp.sum(label_mask*jax.nn.softplus(jnp.where(labels>0,-logits,logits)))/pairs
            loss+=config['classification']['weight']*bce
        if correlation:
            pc=prediction-jnp.mean(prediction,axis=0)
            yc=mean-jnp.mean(mean,axis=0)
            floor_var=correlation['epsilon']**2
            corr=jnp.mean(pc*yc,axis=0)/jnp.sqrt((jnp.mean(pc**2,axis=0)+floor_var)*(jnp.mean(yc**2,axis=0)+floor_var))
            shape=jnp.sum(eligible*(1-corr))/correlation_pairs
            loss+=correlation['weight']*shape
        return loss,jnp.stack([mse,bce,shape])

    centers=jnp.asarray([g['prior_mean'] for g in model['parameters']['groups']])
    probabilities=jnp.asarray(training['sign_probabilities'])
    if len(probabilities)!=engine.m:
        raise ValueError('sign prior count mismatch')
    kernel_prior=jnp.asarray(model['kernel_prior'])
    def prior_loss(p):
        p=freeze(p)
        penalty=config['prior_strength']*jnp.sum(jnp.where(active['groups'],(p['groups']-centers)**2,0.))/max(sum(g['trainable'] for g in model['parameters']['groups']),1)
        if engine.m:
            raw=p['groups'][engine.mapping]
            signs=raw[6*n+engine.m:6*n+2*engine.m]
            penalty+=config['sign_prior_strength']*jnp.mean(jax.nn.softplus(signs)-probabilities*signs)
        penalty+=config['kernel_prior_strength']*jnp.mean((p['kernel']-kernel_prior)**2)
        if config.get('observation_gain'):
            c=config['observation_gain']
            penalty+=c['prior_strength']*(p['log_gain']-np.log(c['initial_gain']))**2
        return penalty+engine.extension_penalty(p)
    return theta,active,groups,eqx.filter_jit(eqx.filter_value_and_grad(data_loss,has_aux=True)),eqx.filter_jit(eqx.filter_value_and_grad(prior_loss))


def evaluate(theta, groups, data_gradient, prior_gradient, progress=None):
    """Stream target gradients, then add priors exactly once. No parameter update."""
    penalty,gradient=prior_gradient(theta)
    metrics=np.zeros(3)
    value=penalty
    for i,group in enumerate(groups):
        (loss,components),g=data_gradient(theta,group)
        value=value+loss
        gradient=jax.tree.map(lambda a,b:a+b,gradient,g)
        metrics+=np.asarray(components)
        if progress is not None:
            progress(i+1,len(groups))
    return value,gradient,{'mse':float(metrics[0]),'bce':float(metrics[1]),'correlation':float(metrics[2]),'prior':float(penalty)}
