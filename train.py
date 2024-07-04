from bert_model import BertWithLMHead
import os
from utils import create_words_batch, evaluate_acc, Vocab
import time
import torch.nn as nn
import torch

pad_token = 0
mask_token = 1
log_every_iter = 100
validate_every_iter = 1000


def read_train_data():
    with open("train_aug_masked.txt") as f:
        words = f.read().split('\n')
    return words[:-1]


def read_dev_data():
    with open("test_aug_masked.txt") as f:
        words = f.read().split('\n')
    return words[:-1]


def train():
    directory = 'checkpoints/'
    if not os.path.isdir(directory):
        os.mkdir(directory)
    model_save_path = 'bert_best.pth'
    model_save_path = os.path.join(directory, model_save_path)

    resume = False
    use_cuda = True
    device = torch.device("cuda:0" if use_cuda else "cpu")

    vocab = Vocab()
    vocab_size = len(vocab.char2id)
    embedding_dim = 256
    mlp_dim = 1024
    num_heads = 4
    n_layers = 6

    model = BertWithLMHead(vocab_size=vocab_size, embedding_dim=embedding_dim, mlp_dim=mlp_dim, dropout=0.1, device=device, n_layers=n_layers, num_heads=num_heads)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
    
    factor = 2
    warmup = 4000
    def lambda_step(step):
        return factor * (embedding_dim ** (-0.5) * min((step+1) ** (-0.5), (step+1) * warmup ** (-1.5)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_step)
    criterion = nn.KLDivLoss(reduction='batchmean')
    if use_cuda:
        criterion.cuda(device=device)

    train_data = read_train_data()
    dev_data = read_dev_data()

    batch_size = 1024
    global_step = period_loss = cum_loss = 0
    hist_valid_scores = []

    if resume:
        checkpoint = torch.load(model_save_path)
        current_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        current_step = checkpoint['global_step']
        print(f'Loading checkpoint from epoch {current_epoch}, iter {current_step}')
    else:
        current_epoch = 0
        current_step = 0

    for epoch in range(current_epoch, 50):
        print("=" * 42)
        model.train()

        start = time.time()
        train_data_iter = create_words_batch(train_data, vocab, mini_batch=batch_size, shuffle=True, device=model.device)
        for i, batch in enumerate(train_data_iter):
            if resume and global_step < current_step:
                global_step = current_step
                continue
            bs = batch.src.shape[0]
            out = model(batch.src, batch.src_mask)
            generator_mask = torch.zeros(bs, vocab_size, device=model.device)
            generator_mask = generator_mask.scatter_(1, batch.src, 1)
            out = model.generator(out, generator_mask)
            loss = criterion(out, batch.tgt)
            if loss.isnan():
                raise ValueError("Loss is NaN!")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            period_loss += loss.item()
            cum_loss += loss.item()

            global_step += 1

            if global_step % log_every_iter == 0:
                elapsed = time.time() - start
                print(f'epoch {epoch}, iter {global_step}, avg. loss {period_loss / log_every_iter:.2f} , cum. loss {cum_loss / global_step:.2f}, time elapsed {elapsed:.2f}sec')
                start = time.time()
                period_loss = 0

            if global_step % validate_every_iter == 0:
                print('Start Evaluation')
                acc = evaluate_acc(model, vocab, dev_data, device=model.device)
                print(f'validation: iter {global_step}, dev. acc {acc:.4f}')

                valid_metric = acc

                hist_valid_scores.append(valid_metric)

                if len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores):
                    print(f'Saving best model to {model_save_path}')
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': cum_loss,
                                'global_step': global_step,
                                }, model_save_path)

        if epoch % 5 == 0:
          torch.save({'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'loss': cum_loss,
                      'global_step': global_step,
                      }, os.path.join(directory, f'bertv6_{epoch}.pth'))

if __name__ == "__main__":

    train()
