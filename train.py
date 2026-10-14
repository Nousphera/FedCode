"""FedCode local sequential execution with stock Flower message serialization."""
import argparse
import functools
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile

ROOT=Path(__file__).resolve().parent


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['synthetic','cifar10','cifar100','spcm'],default='cifar10')
    p.add_argument('--model',choices=['resnet20','mobilenet','yamnet'],default='resnet20')
    p.add_argument('--data-root',type=Path,default=Path('datasets'))
    p.add_argument('--output',type=Path,default=Path('runs/fedcode'))
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--clients',type=int,default=10)
    p.add_argument('--rounds',type=int,default=100)
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--participation',type=float,default=1.)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--clusters',type=int,default=64)
    p.add_argument('--start-round',type=int,default=2)
    p.add_argument('--broadcast-rate',type=int,default=5,help='Downlink calibration interval')
    p.add_argument('--request-rate',type=int,default=2,help='Uplink calibration interval')
    p.add_argument('--split',choices=['iid','noniid','imbalanced'],default='iid')
    p.add_argument('--alpha',type=float,default=.1,help='Dirichlet concentration for non-IID/imbalanced split')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--init-state',type=Path,help='Trusted state-dict checkpoint (weights-only loading)')
    args=p.parse_args(argv)
    if min(args.clients,args.rounds,args.epochs,args.batch_size,args.clusters,args.threads,args.broadcast_rate,args.request_rate)<1:
        p.error('Counts and intervals must be positive')
    if not 0<args.participation<=1: p.error('participation must be in (0,1]')
    if (args.dataset=='spcm')!=(args.model=='yamnet'): p.error('Speech Commands requires YAMNet; vision data requires a vision model')
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()): p.error('Use a new output directory')
    os.environ['TORCH_DATA_DIR']=str(args.data_root.resolve())
    sys.path.insert(0,str(ROOT/'src'))
    import numpy as np
    import torch
    import flwr as fl
    from flwr.common import serde
    from utils.fed_utils import Client,Server
    from network import get_model_fn
    from data import get_dataset_fn
    from utils.shard import get_split_fn
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    if args.device.startswith('cuda') and not torch.cuda.is_available(): p.error('CUDA unavailable; use --device cpu')
    model=get_model_fn(args.model)
    loader=get_dataset_fn(args.dataset)
    split=get_split_fn(args.split)
    if args.split!='iid': split=functools.partial(split,skewness=args.alpha)
    output.mkdir(parents=True,exist_ok=True)
    server=Server(model,loader,args.rounds,num_clients=args.clients,participation=args.participation,
        init_model=str(args.init_state.resolve()) if args.init_state else None,batch_size=args.batch_size,
        device=args.device,num_epochs=args.epochs,lr=args.lr,num_clusters=args.clusters,
        start_rnd=args.start_round,broadcast_freq=args.broadcast_rate,request_freq=args.request_rate,
        file_logger=str(output/'training.log'))
    parameters=server.strategy.initial_parameters
    report={'backend':'sequential-stock-flower','paper_reproduction':False,
        'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'runtime':{'python':platform.python_version(),'torch':torch.__version__,'flower':fl.__version__,'numpy':np.__version__},'rounds':[]}
    class Manager:
        def num_available(self): return args.clients
        def sample(self,num_clients,min_num_clients=None):
            from types import SimpleNamespace
            return [SimpleNamespace(cid=str(i)) for i in sorted(random.sample(range(args.clients),num_clients))]
    with tempfile.TemporaryDirectory(prefix='clients-',dir=output) as cache:
        for rnd in range(1,args.rounds+1):
            before=fl.common.parameters_to_ndarrays(parameters)
            responses=[]; transfers=[]
            for proxy,instruction in server.strategy.configure_fit(rnd,parameters,Manager()):
                torch.manual_seed(args.seed+rnd*args.clients+int(proxy.cid))
                client=Client(int(proxy.cid),args.clients,model,loader,split,args.batch_size,args.seed,
                    temp_dir=cache,device=args.device)
                request=serde.fit_ins_to_proto(instruction)
                response=client.to_client().fit(serde.fit_ins_from_proto(request))
                proto=serde.fit_res_to_proto(response)
                responses.append((proxy,serde.fit_res_from_proto(proto)))
                transfers.append({'client':proxy.cid,'downlink':'codebook' if instruction.config['broadcast_clusters'] else 'calibration',
                    'uplink':'calibration' if response.metrics['send_compressed_weights'] else 'codebook',
                    'cold_start':instruction.config['cold_start_calibration'],'request_bytes':request.ByteSize(),'response_bytes':proto.ByteSize()})
            parameters,_=server.strategy.aggregate_fit(rnd,responses,[])
            loss,metrics=server.strategy.evaluate(rnd,parameters)
            after=fl.common.parameters_to_ndarrays(parameters)
            row={'round':rnd,'loss':float(loss),'accuracy':float(metrics['accuracy']),
                'parameters_changed':any(not np.array_equal(a,b) for a,b in zip(before,after)),
                'examples':sum(r.num_examples for _,r in responses),'transfers':transfers}
            report['rounds'].append(row)
            (output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(json.dumps(row),flush=True)
    torch.save(server.model.state_dict(),output/'model-state.pt')
    np.savez_compressed(output/'final-parameters.npz',*fl.common.parameters_to_ndarrays(parameters))
    return report


if __name__=='__main__': main()
