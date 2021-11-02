import torch

from build_datasets import *
from training_constractor import Trainer
import argparse

datasets_dict = {'breast_cancer':breast_cancer_dataset, 'concrete':concrete_dataset, 'yacht':yacht_dataset,
                 'boston_housing':boson_housing_dataset,'protein':protein_dataset}


def main(args):
    if args.seeds is None:
        seeds = np.random.choice(range(100), args.amount_of_seeds, replace=False).tolist()
    else:
        seeds = [int(item) for item in args.seeds.split(',')]
    for seed in seeds:
        print(seed)
        torch_seed = seed
        numpy_seed = seed
        torch.manual_seed(torch_seed)
        np.random.seed(numpy_seed)
        dataset = args.dataset
        label_mask_prob = args.label_mask
        features_mask_prob = args.features_mask
        cv = args.cv
        embedding_dim = args.embedding_dim
        data = datasets_dict[dataset](embedding_dim = embedding_dim, cv = cv)
        device = args.device
        torch.cuda.set_device(int(device[-1]))
        torch.cuda.empty_cache()
        max_steps = args.max_steps
        lr = args.lr
        betas = (0.9,0.99)
        eps = 1e-6
        flat = args.flat
        k = 6
        alpha = 0.5
        model_layers = args.model_layers
        model_heads = args.heads
        rff_layers = args.rff_layers
        drop_out = args.drop_out
        batch_size = args.batch_size
        init_tradeoff = args.init_tradeoff
        clip = args.clip
        eval_every_n_th_epoch = args.eval_every_n_th_epoch
        print_every_n_th_epoch = args.print_every_n_th_epoch
        if print_every_n_th_epoch % eval_every_n_th_epoch != 0:
            raise Exception('need to eval the validation test on the epoch the results are printed')
        improvements_necessary = args.improvements_necessary
        evaluation_metrics = [item for item in args.evaluation_metrics.split(',')]
        params_dict = {'max_steps':max_steps,'lr':lr,'betas':betas,'eps':eps,'lr_scheduler':args.lr_scheduler,
                       'flat':flat,'k':k,'alpha':alpha,'model_layers':model_layers,'rff':rff_layers,
                       'heads':model_heads,'drop':drop_out,'batch_size':batch_size,'init_tradeoff':init_tradeoff,'clip':clip,
                       'eval_every_n_th_epoch':eval_every_n_th_epoch,'print_every_n_th_epoch':print_every_n_th_epoch,
                       'dataset':dataset,'label_mask_prob':label_mask_prob,'features_mask_prob':features_mask_prob,
                       'embedding_dim':embedding_dim,'cv':cv,'torch_seed':torch_seed,'numpy_seed':numpy_seed,
                       'evaluation_metrics':evaluation_metrics,'improvements_necessary':improvements_necessary}
        trainer = Trainer(params_dict=params_dict,data=data,device=device,experiment=args.experiment)
        trainer.run(data,batch_size,cv,experiment=args.experiment)

def custom_parser(args):
    args.flat = 0.5
    args.lr = 5e-4
    args.embedding_dim = 32
    args.experiment = 'duplicate'
    args.evaluation_metrics = 'nll'

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='breast_cancer')
    parser.add_argument('--amount_of_seeds',type=int,default=1)
    parser.add_argument('--seeds',type = str, default=None)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--max_steps',type = int,default=2000)
    parser.add_argument('--embedding_dim',type=int, default=128)
    parser.add_argument('--model_layers',type = int,default=4)
    parser.add_argument('--heads',type=int,default=8)
    parser.add_argument('--rff_layers',type=int,default=1)
    parser.add_argument('--drop_out', type=float, default=0.1)
    parser.add_argument('--label_mask',type=float,default=1)
    parser.add_argument('--features_mask',type=float,default=0.15)
    parser.add_argument('--init_tradeoff',type=float,default=1)
    parser.add_argument('--batch_size',type=int,default=-1)
    parser.add_argument('--evaluation_metrics',type=str,default='rmse')
    parser.add_argument('--device',type=str,default='cuda:1')
    parser.add_argument('--flat',type=float,default=0.7)
    parser.add_argument('--cv',type=int,default=None)
    parser.add_argument('--clip',type=int,default=1)
    parser.add_argument('--improvements_necessary',type=int,default=1)
    parser.add_argument('--eval_every_n_th_epoch',type=int,default=1)
    parser.add_argument('--print_every_n_th_epoch',type=int,default=1)
    parser.add_argument('--lr_scheduler',type=str,default='flat_then_anneal')
    parser.add_argument('--tradeoff_scheduler',type=str,default='cosine')
    parser.add_argument('--experiment',type=str,default=None)
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    custom_parser(args)
    main(args)
