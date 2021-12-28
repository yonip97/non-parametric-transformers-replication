import torch

from build_datasets import *
from training_constractor import Trainer
import argparse

datasets_dict = {'breast_cancer':breast_cancer_dataset, 'concrete':concrete_dataset, 'yacht':yacht_dataset,
                 'boston_housing':boson_housing_dataset,'protein':protein_dataset,'income':income_dataset
                 ,'forest_cover':forest_cover_dataset,'mnist':MNIST}


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
        cv = args.cv
        embedding_dim = args.embedding_dim
        data = datasets_dict[dataset](embedding_dim = embedding_dim, cv = cv)
        device = args.device
        torch.cuda.set_device(int(device[-1]))
        torch.cuda.empty_cache()
        batch_size = args.batch_size
        eval_every_n_th_epoch = args.eval_every_n_th_epoch
        print_every_n_th_epoch = args.print_every_n_th_epoch
        if print_every_n_th_epoch % eval_every_n_th_epoch != 0:
            raise Exception('need to eval the validation test on the epoch the results are printed')
        params_dict = vars(args)
        evaluation_metrics = [item for item in args.evaluation_metrics.split(',')]
        params_dict['evaluation_metrics'] = evaluation_metrics
        if args.models_paths is not None:
            loading_paths = [item for item in args.models_paths.split(',')]
        else:
            loading_paths = None
        trainer = Trainer(params_dict=params_dict,data=data,device=device,experiment=args.experiment)
        trainer.run(data, batch_size, cv, experiment=args.experiment,loading_paths=loading_paths)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='breast_cancer')
    parser.add_argument('--amount_of_seeds',type=int,default=1)
    parser.add_argument('--seeds',type = str, default=None)
    parser.add_argument('--lr',type=float,default=5e-4)
    parser.add_argument('--end_lr',type = float,default=1e-7)
    parser.add_argument('--power',type = float,default=1)
    parser.add_argument('--max_steps',type = int,default=2000)
    parser.add_argument('--embedding_dim',type=int, default=32)
    parser.add_argument('--model_layers',type = int,default=4)
    parser.add_argument('--heads',type=int,default=8)
    parser.add_argument('--rff_layers',type=int,default=1)
    parser.add_argument('--drop_out', type=float, default=0.1)
    parser.add_argument('--label_mask_prob',type=float,default=1)
    parser.add_argument('--features_mask_prob',type=float,default=0.15)
    parser.add_argument('--init_tradeoff',type=float,default=1)
    parser.add_argument('--batch_size',type=int,default=-1)
    parser.add_argument('--evaluation_metrics',type=str,default='auc-roc')
    parser.add_argument('--device',type=str,default='cuda:2')
    parser.add_argument('--warmup',type=float,default=0.5)
    parser.add_argument('--cv',type=int,default=None)
    parser.add_argument('--clip',type=int,default=1)
    parser.add_argument('--improvements_necessary',type=int,default=1)
    parser.add_argument('--eval_every_n_th_epoch',type=int,default=1)
    parser.add_argument('--print_every_n_th_epoch',type=int,default=1)
    parser.add_argument('--lr_scheduler',type=str,default='flat_then_anneal')
    parser.add_argument('--tradeoff_scheduler',type=str,default='cosine')
    parser.add_argument('--experiment',type=str,default=None)
    parser.add_argument('--use_image_patcher',type=bool,default=False)
    parser.add_argument('--image_n_patches',type = int,default=49)
    parser.add_argument('--model_image_n_channels',type= int,default=1)
    parser.add_argument('--models_paths',type =str,default=None)
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    main(args)
