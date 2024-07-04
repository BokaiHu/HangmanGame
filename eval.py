from bert_model import BertWithLMHead
import os
import torch
from utils import Vocab
from tqdm import tqdm
pad_token = 0
mask_token = 1

def read_dev_data():
    with open("test.txt") as f:
        words = f.read().split('\n')
    return words[:-1]

def play(model, word_list, vocab, device, verbose=False):
    success_cnt = 0
    print("Start Evaluation")
    for word in tqdm(word_list):
        word = word.strip()
        remain_lives = 6
        guess_word = "#" * len(word)
        target = word
        # guess_word = "#i#ti#cte#"
        # target = "distincter"
        guessed_letters = set("#_")
        if verbose:
            print(f"Guess: {guess_word}, Target: {target}")
        while remain_lives > 0 and "#" in guess_word:
            input_tensor = torch.tensor([vocab.char2id[c] for c in guess_word], device=device).unsqueeze(0)
            input_mask = ((input_tensor != pad_token)).unsqueeze(-2)
            output = model(input_tensor, input_mask)
            generator_mask = torch.zeros(1, len(vocab.char2id), device=model.device)
            generator_mask = generator_mask.scatter_(1, input_tensor, 1)
            
            p = model.generator(output, generator_mask)
            most_prob_letter = torch.argsort(-p, dim=1).detach().cpu().numpy()[0]
            for candidate in most_prob_letter:
                cand_chr = vocab.id2char[candidate]
                if cand_chr not in guessed_letters:
                    if cand_chr in target:
                        for t in range(len(target)):
                            if target[t] == cand_chr:
                                guess_word = guess_word[:t] + cand_chr + guess_word[t+1:]
                    else:
                        remain_lives -= 1
                    guessed_letters.add(cand_chr)
                    if verbose:
                        print(f"Guess char: {cand_chr}, now guessing: {guess_word}, remaining lives: {remain_lives}")
                    break
        if not "#" in guess_word:
            success_cnt += 1
    if verbose:
        print(f"{len(word_list)} games in total, {success_cnt} wins.")
    return success_cnt  / len(word_list)


def eval(verbose=False):
    directory = "checkpoints/"
    model_save_path = 'bert_best.pth'
    model_save_path = os.path.join(directory, model_save_path)
    use_cuda = True
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    vocab = Vocab()
    vocab_size = len(vocab.char2id)
    embedding_dim = 256
    mlp_dim = 1024
    num_heads = 4
    n_layers = 6
    
    model = BertWithLMHead(vocab_size=vocab_size, embedding_dim=embedding_dim, mlp_dim=mlp_dim, dropout=0.1, n_layers=n_layers, num_heads=num_heads)

    model = model.to(device)
    test_data = read_dev_data()
    
    checkpoint = torch.load(model_save_path)
    current_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    global_step = checkpoint['global_step']
    print(f'Loading checkpoint from epoch {current_epoch}, iter {global_step}')
    
    model.eval()
    success_rate = play(model, test_data, vocab, device, verbose=verbose)
    print(f"Success rate: {success_rate}")


if __name__ == "__main__":
    eval()