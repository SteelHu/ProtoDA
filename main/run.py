import argparse
import sys
from pathlib import Path


def build_parser():
    p = argparse.ArgumentParser(description='ProtoDA')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train_ratio', type=float, default=0.8, help='random 80/20 split on windows (full code release later)')
    p.add_argument('--dataset', type=str, choices=['alfa', 'rflymad'], default='alfa')
    p.add_argument('--experiment_folder', type=str, default=None)
    p.add_argument('--dataset_path', type=str, default='datasets/ALFA')
    p.add_argument('--source', type=str, default='engine_failure')
    p.add_argument('--targets', type=str, nargs='+', default=['aileron_failure'])
    p.add_argument('--experiments_main_folder', type=str, default='results')
    p.add_argument('--log', type=str, default='train.log')
    p.add_argument('--num_val_iteration', type=int, default=50)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--num_transformer_layers', type=int, default=4)
    p.add_argument('--patch_size', type=int, default=10)
    p.add_argument('--num_prototypes', type=int, default=3)
    p.add_argument('--temperature', type=float, default=0.07)
    p.add_argument('--margin', type=float, default=1.0)
    p.add_argument('--num_epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--eval_batch_size', type=int, default=256)
    p.add_argument('--learning_rate', type=float, default=0.0001)
    p.add_argument('--weight_decay', type=float, default=0.0001)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--weight_loss_disc', type=float, default=0.5)
    p.add_argument('--weight_loss_pred', type=float, default=1.0)
    p.add_argument('--weight_loss_src_sup', type=float, default=0.1)
    p.add_argument('--weight_loss_trg_inj', type=float, default=0.1)
    p.add_argument('--weight_ratio', type=float, default=10.0)
    p.add_argument('--hidden_dim_MLP', type=int, default=256)
    p.add_argument('--queue_size', type=int, default=8192)
    p.add_argument('--momentum', type=float, default=0.99)
    p.add_argument('--window_size', type=int, default=100)
    p.add_argument('--window_stride', type=int, default=1)
    p.add_argument('--use_gpu', action='store_true', default=True)
    p.add_argument('--no_use_gpu', action='store_false', dest='use_gpu')
    p.add_argument('--gpu_type', type=str, default='cuda', choices=['cuda', 'mps'])
    return p


def main():
    build_parser().parse_args()
    root = Path(__file__).resolve().parent.parent
    sys.exit(1)


if (__name__ == '__main__'):
    main()
