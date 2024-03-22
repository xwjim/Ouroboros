import torch
from tqdm import tqdm
import torch, random, time
from typing import List, Tuple
from .kv_cache_model import KVCacheModelLade, KVCacheModelSimpleWithGuess
from ouroboros.cache_engine import CacheEngine
from ouroboros.models.modeling_llama import  config_lade

@torch.no_grad()
def evaluate_posterior(
        target_logits: torch.Tensor, 
        candidates: torch.Tensor,
        candidate_tree_index: torch.Tensor,
        draft_logits: torch.Tensor = None,
        do_sample: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - draft_logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - target_logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidate_tree_index: the position index according to the candidate (batch_size, branch_max, length_max).
    - candidates (torch.Tensor): Candidate token sequences  (batch_size, branch_max, length_max).
    - logits_processor : Softmax temperature for probability scaling. A value of None indicates greedy decoding.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate (batch_size).
    - accept_length (torch.Tensor): Length of the accepted candidate sequence (batch_size).
    """
    # Greedy decoding based on temperature value
    candidate_logits = torch.gather(target_logits.unsqueeze(1), 2, candidate_tree_index.expand(-1,-1,-1,target_logits.shape[-1]))
    N_batch = candidate_logits.shape[0]
    next_new_token_prob = torch.zeros((N_batch,candidate_logits.shape[-1]), dtype=candidate_logits.dtype)
    if not do_sample:
        
        posterior_mask = (
                candidates.to(target_logits.device) == torch.argmax(candidate_logits, dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=-1)).sum(dim=-1)
        accept_length = candidates_accept_length.max(dim=1).values
        # Choose the best candidate
        best_candidate = torch.argmax(candidates_accept_length,dim=1).to(torch.long)
        # Default to the first candidate if none are accepted
        best_candidate.masked_fill_(accept_length == 0 ,0)

        for bt in range(accept_length.shape[0]):
            if accept_length[bt] < candidate_logits.shape[2]:
                next_new_token_prob[bt] = torch.argmax(candidate_logits[bt,best_candidate[bt],accept_length[bt]], dim=-1)
            
        return best_candidate, accept_length, next_new_token_prob
    else:
        candidate_draft_logits = torch.gather(draft_logits.unsqueeze(1), 2, candidate_tree_index.expand(-1,-1,-1,target_logits.shape[-1]))

        best_candidate = torch.zeros((N_batch), dtype=torch.long)
        accept_length = torch.zeros((N_batch), dtype=torch.long)
        
        for bt in range(N_batch):

            accept_candidate = None
            ac_length = 0
            ac_index = 0

            for step in range(candidates.shape[-1]):
            
                if step != ac_length:
                    break

                if step == 0:
                    is_eq = torch.ones((candidates.shape[1]) ,dtype=torch.long)
                else:
                    is_eq = (candidates[bt, :, :ac_length] == accept_candidate).all(dim=1)

                fi = torch.nonzero(is_eq, as_tuple=True)[0].tolist()
                tar_pro = torch.softmax(candidate_logits[bt],dim=-1)
                draft_pro = torch.softmax(candidate_draft_logits[bt],dim=-1)

                for candidate_index in fi:
                    # from IPython import embed; embed() 
                    r = random.random()
                    verify_token = candidates[bt,candidate_index,step]
                    acp = tar_pro[candidate_index, step, verify_token]/draft_pro[candidate_index, step, verify_token]
                    # print("step:{:d}. random seed:{:.3f}. acp pro:{:.3f}. verify pro:{:.3f}".format(step,r,acp,tar_pro[candidate_index, step, verify_token]))
                    if r <= acp:
                        ac_length += 1
                        ac_index = candidate_index
                        accept_candidate = torch.cat((accept_candidate,verify_token.reshape(1)),dim=0) if accept_candidate is not None else verify_token.reshape(1)
                        break
                    else:
                        tar_pro = tar_pro - draft_pro
                        tar_pro[tar_pro<0] = 0
                        tar_pro = tar_pro/tar_pro.sum()

            best_candidate[bt] = ac_index
            accept_length[bt] = ac_length
            if ac_length < candidate_logits.shape[2]:
                next_new_token_prob[bt] = tar_pro[ac_index,ac_length]
        return best_candidate, accept_length, next_new_token_prob


                # for token_set in fi:

@torch.no_grad()
def ouroboros(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, ngram_cache : CacheEngine = None,
                        max_len : int = 512 , gamma : int = 4, window_size = 20, guess_set_size = 20, lookahead_level = 7,
                        eos_token_id = 2, topk = 30, top_p = 0.9, do_sample = True, temperature=1) -> torch.Tensor:
    """
    Performs ouroboros with an approximate model and a target model to generate a sequence of tokens.

    Args:
        prefix (torch.Tensor): The initial sequence of tokens to start the generation from.
        approx_model (torch.nn.Module): The approximate model used for initial token generation. The model should support huggingface transformers model methods.
        target_model (torch.nn.Module): The target model used for refining the generated tokens. The model should support huggingface transformers model methods.
        ngram_cache (CacheEngine, optional): A cache engine for storing and retrieving n-gram predictions. Defaults to None, in which case a new cache engine is created.
        max_len (int, optional): The maximum length of the generated sequence. Defaults to 512.
        gamma (int, optional): The lookahead parameter for generation. Defaults to 4.
        window_size (int, optional): The window size used for n-gram generation. Defaults to 20. Currently, must be equal to guess_set_size.
        guess_set_size (int, optional): The size of the guess set for n-gram retrieving. Defaults to 20. Currently, must be equal to window_size.
        lookahead_level (int, optional): The level of lookahead decoding. Defaults to 7.
        eos_token_id (int, optional): The token id representing the end-of-sequence token. Defaults to 2. Should be given by tokenizer.eos_token_id.

    Returns:
        torch.Tensor: The generated sequence of tokens, including the initial prefix and any additional tokens generated by the function.
    """
    assert window_size == guess_set_size, "We only support same window_size and guess_set_size now. More combinations will be supported in the future."
    if ngram_cache == None:
        ngram_cache = CacheEngine(lookahead_level, guess_set_size)
    seq_len = prefix.shape[1]
    T = seq_len + max_len

    if do_sample:
        import os
        config_lade(LEVEL=lookahead_level, WINDOW_SIZE=window_size, GUESS_SET_SIZE=guess_set_size, POOL_FROM_PROMPT=False, DIST_WORKERS=len(os.environ.get("CUDA_VISIBLE_DEVICES").split(",")))
        # lade.config_lade(LEVEL=args.level, WINDOW_SIZE=args.window, GUESS_SET_SIZE=args.guess, DEBUG=1, USE_FLASH=args.use_flash, )
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    assert approx_model.device == target_model.device
    
    device = target_model.device

    guess_size = lookahead_level - 1
    
    approx_model_cache = KVCacheModelLade(approx_model, window_size=window_size, guess_set_size=guess_set_size, lookahead_level=lookahead_level)
    target_model_cache = KVCacheModelSimpleWithGuess(target_model, lookahead_level=lookahead_level)

    # target_model_cache = KVCacheModelSimple(target_model)
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0

    end_pos = None
    
    while prefix.shape[1] < T:
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        prefix_len = prefix.shape[1]

        x, out_len, guess = approx_model_cache.generate(prefix, ngram_cache, gamma, do_sample = do_sample, temperature=temperature, top_k=topk, top_p=top_p)
        target_model_cache._forward_with_kvcache(x, guess)

        key_tok = int(x[:,-1])

        gen_len = out_len - prefix_len
        
        n = prefix_len + gen_len - 1

        # print(prefix)
        # assert (torch.argmax(approx_model_cache.ctx["past_logits"][:, prefix_len-1:prefix_len-1+gen_len, :],dim=-1) == x[:, prefix_len:prefix_len+gen_len]).all()
        # print(target_model_cache._prob_history.device,approx_model_cache.ctx["past_logits"].device,device)
        best_candidate, accept_length, next_new_token_prob = evaluate_posterior(
            target_logits = target_model_cache._prob_history[:, prefix_len-1:, :].to(device),
            candidates = x[:, prefix_len:prefix_len+gen_len].unsqueeze(1),
            candidate_tree_index = torch.arange(gen_len)[None,None,:,None].to(device),
            draft_logits = approx_model_cache.ctx["past_logits"][:, prefix_len-1:prefix_len-1+gen_len, :].to(device),
            do_sample = do_sample,
        )

        accept_length = accept_length.item()
        accepted_count += accept_length
        n = prefix_len + accept_length - 1

        if accept_length != 0 and x[:,prefix_len+accept_length-1] == eos_token_id:
            end_pos = prefix_len + accept_length

        prefix = x[:, :n + 1]

        approx_model_cache.rollback(n+1)

        corr_ngram = [] # ngram corrected by target_model

        if n < prefix_len + gen_len - 1:
            if do_sample:
                t = torch.multinomial(next_new_token_prob[0], 1)[None,:]
            else:
                t = target_model_cache._prob_history[:, n, :].argmax(dim=-1, keepdim=True)
            if t == eos_token_id:
                end_pos = n + 2

            first_tok = int(target_model_cache._prob_history[:, prefix_len + gen_len - 1, :].argmax(dim=-1))
            beg_pos = prefix_len + gen_len
            guess_num = len(guess) // guess_size
            for i in range(guess_num):
                real_ngram = tuple([first_tok] + target_model_cache._prob_history[0, beg_pos + i * guess_size : beg_pos + i * guess_size + guess_size - 1, :].argmax(dim=-1).tolist())
                corr_ngram.append(real_ngram)
            if len(corr_ngram) > 0:
                approx_model_cache.update_in_place(key_tok, corr_ngram)


            target_model_cache.rollback(n+1)
            prefix = torch.cat((prefix, t.to(prefix.device)), dim=1)
        else:
            # find the longest guess
            guess = [item for sublist in guess for item in sublist]
            guess_num = len(guess) // guess_size
            first_tok = int(target_model_cache._prob_history[:, n, :].argmax(dim=-1))
            beg_pos = prefix_len + gen_len
            candidate = [first_tok]
            longest_can_len = 1
            candidate_idx = -1
            tmp_end_pos = n + 2 if first_tok == eos_token_id else None
            loc_end_pos = None
            for i in range(guess_num):
                real_ngram = [first_tok] + target_model_cache._prob_history[0, beg_pos + i * guess_size : beg_pos + i * guess_size + guess_size, :].argmax(dim=-1).tolist()
                corr_ngram.append(tuple(real_ngram[:-1]))
                pred_ngram = guess[i * guess_size : (i + 1) * guess_size]
                ml = 0
                for j in range(guess_size):
                    ml = j
                    if real_ngram[j] == eos_token_id:
                        loc_end_pos = j
                    if real_ngram[j] != pred_ngram[j]:
                        break
                if ml + 1 > longest_can_len:
                    candidate = real_ngram[:ml + 1]
                    longest_can_len = ml + 1
                    candidate_idx = i
                    tmp_end_pos = loc_end_pos
            if tmp_end_pos is not None:
                end_pos = beg_pos + candidate_idx * guess_size + tmp_end_pos + 1
            candidate = torch.tensor([candidate], device=prefix.device)
            prefix = torch.cat((prefix, candidate), dim=1)
            if len(corr_ngram) > 0:
                approx_model_cache.update_in_place(key_tok, corr_ngram)
            if candidate_idx != -1:
                target_model_cache.confirm(n+1, beg_pos + candidate_idx * guess_size, candidate.shape[-1] - 1) # cur_len, start_pos, length
            else:
                target_model_cache.rollback(n+1)
        if end_pos is not None:
            break
        


    if end_pos is not None:
        prefix = prefix[:, :end_pos]
    return prefix[:, :T]