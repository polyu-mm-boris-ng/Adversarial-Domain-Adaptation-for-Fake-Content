import torch.utils.data
from torchtext import data # torchtext version 0.8.1

def get_datasets(mode, source_path, target_path, batch_size, device):
    TEXT = data.Field(sequential=True, tokenize=lambda news: news.split(" "), lower=True)
    LABEL = data.Field(sequential=False, use_vocab=False)

    source = data.TabularDataset(
        path=source_path, format="csv", skip_header=True, fields=[("Text", TEXT), ("Label", LABEL)]
    )

    source_train, source_test = source.split(split_ratio=0.75)
    
    if mode != 'cal_tran_score':
        source_train_iter = data.BucketIterator(
            source_train,
            sort_key=lambda x: len(x.Text),
            batch_size=batch_size,
            device=torch.device(device),
            shuffle=True,
            sort_within_batch=True,
            repeat=False,
        )
        source_test_iter = data.BucketIterator(
            source_test,
            sort_key=lambda x: len(x.Text),
            batch_size=batch_size,
            device=torch.device(device),
            shuffle=True,
            sort_within_batch=True,
            repeat=False,
        )
        
        if mode == 'TLDA':
            target = data.TabularDataset(
                path=target_path, format="csv", skip_header=True, fields=[("Text", TEXT), ("Label", LABEL)]
            )
            target_train, target_test = target.split(split_ratio=0.75)
            TEXT.build_vocab(source, target, vectors="glove.6B.300d", max_size=25000)

            target_train_iter = data.BucketIterator(
                target_train,
                sort_key=lambda x: len(x.Text),
                batch_size=batch_size,
                device=torch.device(device),
                shuffle=True,
                sort_within_batch=True,
                repeat=False,
            )
        
            target_test_iter = data.BucketIterator(
                target_test,
                sort_key=lambda x: len(x.Text),
                batch_size=batch_size,
                device=torch.device(device),
                shuffle=True,
                sort_within_batch=True,
                repeat=False,
            )
            
        else:
            TEXT.build_vocab(source, vectors="glove.6B.300d", max_size=25000)
            target_train_iter = None
            target_test_iter = None

        return (
            TEXT,
            source,
            target,
            source_train,
            source_test,
            source_train_iter,
            source_test_iter,
            target_train_iter,
            target_test_iter
        )
    else:
        source_train, source_test = source.split(split_ratio=0.75)
        source_out_iter = data.Iterator(
            source_train, batch_size=256, device=torch.device(device),
            shuffle=False, sort = False, sort_within_batch=False, repeat=False
        )
        target = data.TabularDataset(
            path=target_path, format="csv", skip_header=True, fields=[("Text", TEXT)]
        )
        target_out_iter = data.Iterator(
            target, batch_size=256, device=torch.device(device),
            shuffle=False, sort = False, sort_within_batch=False, repeat=False
        )
        TEXT.build_vocab(source, vectors="glove.6B.300d", max_size=25000)
        return source_out_iter, target_out_iter
