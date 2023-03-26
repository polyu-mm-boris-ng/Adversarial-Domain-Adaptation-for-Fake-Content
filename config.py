import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mode', type=str, choices=[
        'TLnoDA', 'TLDA', 'TLDA_FT', 'cal_tran_score'
    ]) 
    parser.add_argument('--cuda', action='store_true', help='Enable CUDA')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    parser.add_argument('--model_path', type=str, default='', help='Model path for loading or saving')

    # dataset arguments
    parser.add_argument('--source_path', type=str, 
                        default='./datasets/source.csv', help='Source dataset path')
    parser.add_argument('--target_path', type=str,
                        default='./datasets/target.csv', help='Target dataset path')
    # training arguments
    parser.add_argument('--do_train', action='store_true', help='Enable training')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--embedding_dim', type=int, default=300, help='Embedding dimension')
    parser.add_argument('--n_filters', type=int, default=100, help='Number of filters')
    parser.add_argument('--filter_sizes', type=int, nargs='+', default=[7, 6, 5], help='Filter sizes')
    parser.add_argument('--pretrained_model_path', type=str, default='', help='Pre-trained model path for fine-tuning')

    # testing arguments
    parser.add_argument('--do_test', action='store_true', help='Enable testing')
    parser.add_argument('--test_model_epoch', type=int, default=0, help='Epoch of model for testing')
    
    # transferability score arguments
    parser.add_argument('--cal_tran_score', action='store_true', help='Enable calculating transferability score')
    parser.add_argument('--model_no_da_path', type=str, default='', help='Model without domain adptation path for calculating transferability score')
    parser.add_argument('--model_da_path', type=str, default='', help='Model with domain adptation path for calculating transferability score')
    parser.add_argument('--do_visualize', action='store_true', help='Enable visualizing transferability score')
    parser.add_argument('--fig_prefix', type=str, default='fig', help='Figure prefix')

    args = parser.parse_args()
    if args.cuda:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
        
    return args
