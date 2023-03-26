import os, sys
import datetime
import logging
import random
import warnings

from tqdm import tqdm, trange
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable

from config import get_args
from datasets import get_datasets
from models import CNNModel
from transferability_score import cal_tran_score, visualize

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
args = get_args()

warnings.filterwarnings("ignore")

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set as {seed}")

def train(source_train, source_train_iter, source_test_iter, TEXT, target_train_iter=None, target_test_iter=None):
    # Load model
    if args.mode == 'TLDA_FT':
        model = torch.load(args.pretrained_model_path)
        logger.info(f'Load pretrained model from {args.pretrained_model_path}')
    else:
        model = CNNModel(
            args.input_dim, args.embedding_dim, args.n_filters, args.filter_sizes, args.mode
        )
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    for p in model.parameters():
        p.requires_grad = True

    if args.mode == 'TLDA_FT':
        frozen_params = [
            "embedding.f_embed.weight",
            "feature.f_convs.0.weight",
            "feature.f_convs.0.bias",
            "feature.f_convs.1.weight",
            "feature.f_convs.1.bias",
            "feature.f_convs.2.weight",
            "feature.f_convs.2.bias",
            "domain_classifier.d_fc1.weight",
            "domain_classifier.d_fc1.bias",
            "domain_classifier.d_bn1.weight",
            "domain_classifier.d_bn1.bias",
            "domain_classifier.d_fc2.weight",
            "domain_classifier.d_fc2.bias",
            "domain_classifier.d_bn2.weight",
            "domain_classifier.d_bn2.bias",
            "domain_classifier.d_fc3.weight",
            "domain_classifier.d_fc3.bias"
        ]
        
        for n, p in model.named_parameters():
            if n in frozen_params:
                p.requires_grad = False
    
    pretrained_embeddings = TEXT.vocab.vectors
    model.init_weights(pretrained_embeddings, is_static=True)

    if args.cuda:
        model = model.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()

    logger.info(f'\n ** Start training with {args.mode} mode ** \n')
    model.train()
    for epoch in range(0, args.n_epoch):
        logger.info(f'\n==> Epoch: {epoch}\n')
        i = 0
        n_correct_class = 0
        n_total_class = 0
        if args.mode == 'TLnoDA':
            p = float(i + epoch * len(source_train)) / args.n_epoch / len(source_train)
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        if args.mode == 'TLDA':
            s_n_batch = len(source_train_iter)
            t_n_batch = len(target_train_iter)
            n_batch = min(s_n_batch, t_n_batch)
            n_correct_class1 = 0
            n_total_class1 = 0
            n_correct_domain = 0
            n_total_domain = 0
        if args.mode == 'TLDA_FT':
            alpha = 0

        start_time = datetime.datetime.now()
        
        if args.mode == 'TLnoDA':
            for data_source in source_train_iter:
                s_text = data_source.Text
                s_label = data_source.Label

                model.zero_grad()
                batch_size = len(s_label)

                if args.cuda:
                    s_text = s_text.cuda()
                    s_label = s_label.cuda()

                inputv_text = Variable(s_text)
                classv_label = Variable(s_label).long()

                class_output, _ = model(input_data=inputv_text, alpha=alpha)
                err_s_label = loss_class(class_output, classv_label)

                pred_class = class_output.data.max(1, keepdim=True)[1]
                n_correct_class += (
                    pred_class.eq(classv_label.data.view_as(pred_class)).cpu().sum().numpy()
                )
                n_total_class += batch_size
                accu_class = n_correct_class / n_total_class

                err_s_label.backward()
                optimizer.step()

                if i % 100 == 0:
                    end_time = datetime.datetime.now()
                    cal_time = end_time - start_time
                    logger.info(
                        f'Epoch: {epoch}, [iter: {i * batch_size} / all {len(source_train)}], accu_s_label: {accu_class}'
                    )
                    start_time = datetime.datetime.now()
                i += 1
                
        if args.mode == 'TLDA':
            while i < n_batch:
                p = float(i + epoch * n_batch) / args.n_epoch / n_batch
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                # Training model using source data
                data_source = next(iter(source_train_iter))
                s_text = data_source.Text
                s_label = data_source.Label

                model.zero_grad()
                batch_size = len(s_label)

                domain_label = torch.zeros(batch_size)
                domain_label = domain_label.long()

                if args.cuda:
                    s_text = s_text.cuda()
                    s_label = s_label.cuda()
                    domain_label = domain_label.cuda()

                inputv_text = Variable(s_text)
                classv_label = Variable(s_label).long()
                domainv_label = Variable(domain_label).long()

                class_output, domain_output = model(input_data=inputv_text, alpha=alpha)
                err_s_label = loss_class(class_output, classv_label)
                err_s_domain = loss_domain(domain_output, domainv_label)

                pred_class = class_output.data.max(1, keepdim=True)[1]
                n_correct_class += pred_class.eq(classv_label.data.view_as(pred_class)).cpu().sum().numpy()
                n_total_class += batch_size
                accu_class = n_correct_class / n_total_class

                pred_class1 = domain_output.data.max(1, keepdim=True)[1]
                n_correct_class1 += pred_class1.eq(domainv_label.data.view_as(pred_class1)).cpu().sum().numpy()
                n_total_class1 += batch_size
                accu_class1 = n_correct_class1 / n_total_class1

                # Training model using target data
                data_target = next(iter(target_train_iter))
                t_text = data_target.Text

                if args.cuda:
                    t_text = t_text.cuda()

                inputv_text = Variable(t_text)

                _, domain_output = model(input_data=inputv_text, alpha=alpha)

                batch_size = len(domain_output)

                domain_label = torch.ones(batch_size)
                domain_label = domain_label.long()

                if args.cuda:
                    domain_label = domain_label.cuda()            

                domainv_label = Variable(domain_label).long()
                err_t_domain = loss_domain(domain_output, domainv_label)
                err = err_t_domain + err_s_domain + err_s_label
                err.backward()
                optimizer.step()

                pred_domain = domain_output.data.max(1, keepdim=True)[1]
                n_correct_domain += pred_domain.eq(domainv_label.data.view_as(pred_domain)).cpu().sum().numpy()
                n_total_domain += batch_size
                accu_domain = n_correct_domain / n_total_domain

                i += 1

                if i % 100 == 0:
                    logger.info(f'Epoch: {epoch}, [iter: {i * batch_size} / all {n_batch * batch_size}], accu_s_label: {accu_class}, accu_s_domain: {accu_class1}, accu_t_domain: {accu_domain}') 
                                   
        if args.mode == 'TLDA_FT':
            for data_source in source_train_iter:                
                # Training model using source data
                s_text = data_source.Text
                s_label = data_source.Label

                model.zero_grad()

                if args.cuda:
                    s_text = s_text.cuda()
                    s_label = s_label.cuda()

                inputv_text = Variable(s_text)
                classv_label = Variable(s_label).long()

                class_output, _ = model(input_data=inputv_text, alpha=alpha)
                err_s_label = loss_class(class_output, classv_label)

                pred_class = class_output.data.max(1, keepdim=True)[1]
                n_correct_class += pred_class.eq(classv_label.data.view_as(pred_class)).cpu().sum().numpy()
                n_total_class += args.batch_size
                accu_class = n_correct_class / n_total_class

                err = err_s_label
                err.backward()
                optimizer.step()
                
                i += 1

        saved_model_path = os.path.join(args.model_path, f'epoch{epoch}.pth')
        torch.save(model, saved_model_path)
        logger.info(f'Save model at {saved_model_path}')
        logger.info(f'\n** Begin testing for epoch {epoch}... **')
        test(saved_model_path, source_test_iter, target_test_iter)
        end_time = datetime.datetime.now()
        cal_time = end_time - start_time
        logger.info(f'** Testing is done, testing time for current epoch: {cal_time} **\n')
        
        start_time = datetime.datetime.now()
        
    logger.info('Training is done!\n')
    
def test_for_TLDA(model_path, model, source_test_iter, dataset_type='source'):
    i = 0
    n_total = 0
    n_correct = 0
    alpha = 0
    loss_class = torch.nn.NLLLoss()
    n_total_domain = 0
    n_correct_domain = 0 
    err_label = 0
    err_domain = 0
    loss_domain = torch.nn.NLLLoss()
    for batch_st in source_test_iter:
        st_text = batch_st.Text
        st_label = batch_st.Label

        batch_size = len(st_label)

        if dataset_type == 'source':
            domain_label = torch.zeros(batch_size)
        else:
            domain_label = torch.ones(batch_size)
        domain_label = domain_label.long()

        if args.cuda:
            st_text = st_text.cuda()
            st_label = st_label.cuda()
            domain_label = domain_label.cuda()

        inputv_text = Variable(st_text)
        classv_label = Variable(st_label).long()
        domainv_label = Variable(domain_label).long()

        class_output, domain_output = model(input_data=inputv_text, alpha=alpha)
        err_label += loss_class(class_output, classv_label)
        err_domain += loss_domain(domain_output, domainv_label)

        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(classv_label.data.view_as(pred)).cpu().sum().numpy()
        n_total += batch_size

        pred_domain = domain_output.data.max(1, keepdim=True)[1]
        n_correct_domain += pred_domain.eq(domainv_label.data.view_as(pred_domain)).cpu().sum().numpy()
        n_total_domain += batch_size
        i += 1
        
    acc = n_correct / n_total
    accu_domain = n_correct_domain / n_total_domain

    err_label_avg = err_label / i
    err_domain_avg = err_domain / i     
    
    logger.info(f'[{dataset_type}] Model: {model_path}, predicted class acc: {acc}. predicted domain acc: {accu_domain}')
    logger.info(f'predicted class error: {err_label_avg}, predicted domain error: {err_domain_avg}')
    
def test(model_path, source_test_iter, target_test_iter=None):    
    model = torch.load(model_path)
    model = model.eval()
    if args.cuda:
        model = model.cuda()

    i = 0
    n_total = 0
    n_correct = 0
    alpha = 0

    # Test model using source data
    with torch.no_grad():
        if args.mode == 'TLnoDA' or args.mode == 'TLDA_FT':
            for batch_st in source_test_iter:
                st_text = batch_st.Text
                st_label = batch_st.Label

                batch_size = len(st_label)

                if args.cuda:
                    st_text = st_text.cuda()
                    st_label = st_label.cuda()

                inputv_text = Variable(st_text)
                classv_label = Variable(st_label).long()
                class_output, _ = model(input_data=inputv_text, alpha=alpha)
                pred = class_output.data.max(1, keepdim=True)[1]
                n_correct += pred.eq(classv_label.data.view_as(pred)).cpu().sum().numpy()
                n_total += batch_size
            acc = n_correct / n_total
            logger.info(f'Model: {model_path}, accuracy of the source dataset: {acc}')
            
        if args.mode == 'TLDA':
            test_for_TLDA(model_path, model, source_test_iter, dataset_type='source')
            test_for_TLDA(model_path, model, target_test_iter, dataset_type='target')
    
if __name__ == '__main__':
    set_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    
    logger.info('** Configuration **')
    logger.info(f'Mode: {args.mode}, random seed: {args.seed}')
    logger.info(f'Source dataset path: {args.source_path}')
    logger.info(f'Target dataset path: {args.target_path}')
    if args.do_train:
        logger.info(f'Batch size: {args.batch_size}')
        logger.info(f'Learning rate: {args.lr}')
        logger.info(f'Number of epochs: {args.n_epoch}')
        logger.info(f'Device: {args.device}')
        logger.info(f'Embedding dimension: {args.embedding_dim}')
        logger.info(f'Number of filters: {args.n_filters}')
        logger.info(f'Filter sizes: {args.filter_sizes}')
        logger.info(f'Model saving path: {args.model_path}')
    if args.do_test:
        logger.info(f'Model testing path: {args.model_path}')
    if args.cal_tran_score:
        logger.info(f'Model with domain adaptation path: {args.model_da_path}')
        logger.info(f'Model without domain adaptation path: {args.model_no_da_path}')
    logger.info('** End of configuration **')
    
    if args.mode != 'cal_tran_score':
        TEXT, source, target, source_train, source_test, \
        source_train_iter, source_test_iter, target_train_iter, target_test_iter = get_datasets(
            mode=args.mode,
            source_path=args.source_path,
            target_path=args.target_path,
            batch_size=args.batch_size,
            device=args.device,
        )
        args.input_dim = len(TEXT.vocab)
        logger.info(f'Input dimension: {args.input_dim}')
    else:
        source_out_iter, target_out_iter = get_datasets(
            mode=args.mode,
            source_path=args.source_path,
            target_path=args.target_path,
            batch_size=args.batch_size,
            device=args.device,
        )
    
    if args.do_train:
        train(source_train, source_train_iter, source_test_iter, TEXT, target_train_iter, target_test_iter)
    if args.do_test:
        test(os.path.join(args.model_path, f'epoch{args.test_model_epoch}.pth'), source_test_iter)
    if args.cal_tran_score:
        transferability_score, da_source_df, da_target_df, no_da_source_df, no_da_target_df = cal_tran_score(
            args.model_no_da_path, args.model_da_path, source_out_iter, target_out_iter, logger, args.cuda)
        # model_no_da, model_da, source_out_iter, target_out_iter, logger, cuda
        if args.do_visualize:
            visualize(da_source_df, da_target_df, no_da_source_df, no_da_target_df, args.fig_prefix)