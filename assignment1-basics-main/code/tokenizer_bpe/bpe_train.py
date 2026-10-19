
import regex as re
from collections import defaultdict
from tqdm import tqdm
import time
import multiprocessing as mp
def init_vocab(specialToken:list[str])->dict[int,bytes]:
    vocab = {}

    for i in range(256):
        vocab[i] = bytes([i])
    # vocab[256]=bytes(['<|endoftext|>'])
    next_id = 256
    for i in range(len(specialToken)):
        #加入specialToken
        vocab[i+next_id]=specialToken[i].encode('utf-8')
    # print(vocab[97])
    # print(vocab[256])
    # print(vocab[257])
    return vocab



def countVocab(specialTokens, text):
    #初始版本的countVocab
    escaped = [re.escape(tok) for tok in specialTokens]
    split_pattern = "|".join(escaped)

    # print("开始按 special token 切分……", flush=True)
    split_start = time.perf_counter()

    parts = re.split(split_pattern, text)

    # print(
    #     f"切分完成：{len(parts):,} 个片段，"
    #     f"耗时 {time.perf_counter() - split_start:.3f} 秒",
    #     flush=True,
    # )

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretoken_counts = defaultdict(int)

    #性能瓶颈,最好考虑并发实现
    for part in tqdm(parts, desc="正则预分词", unit="片段"):
        for match in re.finditer(PAT, part):
            pretoken = match.group()
            bs = pretoken.encode("utf-8")
            pretoken_bytes = tuple(bytes([b]) for b in bs)
            pretoken_counts[pretoken_bytes] += 1

    return dict(pretoken_counts)

def countVocab2(specialTokens, parts):



    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretoken_counts = defaultdict(int)

    #性能瓶颈,最好考虑并发实现
    for part in tqdm(parts, desc="正则预分词", unit="片段"):
        for match in re.finditer(PAT, part):
            pretoken = match.group()
            bs = pretoken.encode("utf-8")
            pretoken_bytes = tuple(bytes([b]) for b in bs)
            pretoken_counts[pretoken_bytes] += 1

    return dict(pretoken_counts)


# Pool.map 通常给 worker 传入一个参数，所以将多个参数打包成： args = (file_path, specialTokens)
def count_vocab_worker(args):
    special_tokens, part_chunk = args
    return countVocab2(special_tokens, part_chunk)

def count_vocab_parallel(special_tokens, text, num_workers=4):

    escaped = [re.escape(tok) for tok in special_tokens]
    split_pattern = "|".join(escaped)

    # print("开始按 special token 切分……", flush=True)
    split_start = time.perf_counter()

    parts = re.split(split_pattern, text)

    # print(
    #     f"切分完成：{len(parts):,} 个片段，"
    #     f"耗时 {time.perf_counter() - split_start:.3f} 秒",
    #     flush=True,
    # )

    pretoken_counts = defaultdict(int)
    num_workers=min(num_workers, len(parts)) #避免进程数大于parts数量，拆分导致失效
    num_parts = len(parts)//num_workers
    chunks = []
    for i in range(num_workers - 1):
        #多的数据全部塞最后一块，可能会导致最后一块特别大
        chunks.append(parts[i * num_parts:(i + 1) * num_parts])

    chunks.append(parts[(num_workers - 1) * num_parts:])
    tasks = [
    (special_tokens, chunk)
    for chunk in chunks
    ]
    #进程池运算
    with mp.Pool(processes=num_workers) as pool:
        partial_counts= pool.map(count_vocab_worker, tasks)

    for partial_count in partial_counts:
        for token_tuple, count in partial_count.items():
            pretoken_counts[token_tuple] += count
    return dict(pretoken_counts)



def countPairs(pretokens):
    pair_counts = defaultdict(int)
    for token_tuple, count in pretokens.items():
        for i in range(len(token_tuple)-1):
            pair_counts[(token_tuple[i],token_tuple[i+1])]+=count
    return pair_counts

def build_pair_to_pretokens(pretokens):
    """
    建立反向索引：

    pair -> 包含该 pair 的所有 token_tuple
    """
    pair_to_pretokens = defaultdict(set)

    for token_tuple in pretokens:
        for i in range(len(token_tuple) - 1):
            pair = (
                token_tuple[i],
                token_tuple[i + 1],
            )

            pair_to_pretokens[pair].add(token_tuple)

    return pair_to_pretokens

def get_best_pair(count):
    if not count:
        return None
    return max(count,key=lambda pair:(count[pair],pair))



def merge_token(token_tuple, pair):
    #合并后的token样式,这里用的简单匹配，用kmp应该会快一点? 这个想法是错误的，因为匹配长度固定为2，KMP只会增加开销
    # token_tuple是指合并几次后的序列，里面的元素可能不再是单个字
    merged_tokens = []
    i = 0

    while i < len(token_tuple):
        # 当前及下一个 token 正好组成目标 pair
        if (
            i + 1 < len(token_tuple)
            and token_tuple[i] == pair[0]
            and token_tuple[i + 1] == pair[1]
        ):
            merged_tokens.append(pair[0] + pair[1])
            i += 2
        else:
            merged_tokens.append(token_tuple[i])
            i += 1

    return tuple(merged_tokens)
'''
def merge_token2(token_tuple, pair,pair_count,count):
    #合并后的token样式,这里用的简单匹配，用kmp应该会快一点? 这个想法是错误的，因为匹配长度固定为2，KMP只会增加开销
    # token_tuple是指合并几次后的序列，里面的元素可能不再是单个字
    new_token = []
    i = 0
    new_pair_count = defaultdict(int, pair_count)
    
    while i < len(token_tuple):
        if (i <= len(token_tuple) - len(pair)):
            token=True
            for j in range( len(pair)):
                if token_tuple[i+j]!=pair[j]:
                    token=False #不能合并
                    break
            if token==True:
                new_token.append(pair[0] + pair[1])
                if i>0:
                    new_pair_count[(token_tuple[i-1],pair[0] + pair[1])]+=count
                    new_pair_count[(token_tuple[i-1],pair[0])]-=count
                    if new_pair_count[(token_tuple[i-1],pair[0])]==0:
                        del new_pair_count[(token_tuple[i-1],pair[0])]
                #合并了相邻的两个token,所以i要加2
                i += 2
                if i<len(token_tuple):
                    new_pair_count[(pair[0] + pair[1],token_tuple[i])]+=count
                    new_pair_count[(pair[1],token_tuple[i])]-=count
                    if new_pair_count[(pair[1],token_tuple[i])]==0:
                        del new_pair_count[(pair[1],token_tuple[i])]

            else:
                new_token.append(token_tuple[i])
                i += 1     

        else:
            #用于退出死循环
            new_token.append(token_tuple[i])
            i += 1
    
    return tuple(new_token),dict(new_pair_count)
    '''


def merge_token2(token_tuple, pair,frequency,new_pair_count):
    """
    合并 token_tuple 中从左到右、不重叠出现的 pair，
    同时统计合并后 token_tuple 的相邻 pair 频率。
    """
    merged_tokens=[]
    i=0
    left,right=pair
    merged_token=left+right

    while i < len(token_tuple):
        # 当前及下一个 token 正好组成目标 pair
        if (
            i + 1 < len(token_tuple)
            and token_tuple[i] == pair[0]
            and token_tuple[i + 1] == pair[1]
        ):
            
            current_token = merged_token
            i += 2
        else:
            current_token = token_tuple[i]
            i += 1

        # 当前输出 token 与前一个输出 token 组成新 pair
        if merged_tokens:
            new_pair = (
                merged_tokens[-1],
                current_token,
            )
            new_pair_count[new_pair] += frequency

        merged_tokens.append(current_token)

    return tuple(merged_tokens)



#整理成函数
def train_bpe_before(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_workers: int = 4,
) -> tuple[
    dict[int, bytes],
    list[tuple[bytes, bytes]],
]:
    #默认for 循环过不了 speed test,问题出在BPE merge上   
    vocab=init_vocab(special_tokens)

    start = time.perf_counter()
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    pretoken=count_vocab_parallel(special_tokens, text,num_workers=num_workers)

    print(f"countVocab耗时：{time.perf_counter() - start:.3f} 秒")
    merge=[]

    bpe_start = time.perf_counter()
    num_tokens_to_add = vocab_size - len(vocab)
    pair_count=countPairs(pretoken)
    print(pair_count)

    with tqdm(total=vocab_size-len(vocab), desc="训练 BPE") as progress:
        while len(vocab)<vocab_size:
            #每次都执行countPairs导致复杂度超了
            if not pair_count:
                break
#             更新 pretoken 中的 token 序列
#             更新 pair_count 中的相邻 pair 频率
            pair=get_best_pair(pair_count) 
            vocab[len(vocab)]=pair[0]+pair[1]
            merge.append((pair[0],pair[1]))

            #循环遍历所有pair
            new_pretokens=defaultdict(int)
            for token_tuple, count in pretoken.items():
                merged_tuple=merge_token(token_tuple,pair)
                new_pretokens[merged_tuple]+=count
            pretoken=new_pretokens
            pair_count=countPairs(pretoken)
            # print('-'*20)
            # print(pair_count)
            # print(new_pretokens)
            progress.update(1)
            

    elapsed = time.perf_counter() - bpe_start
    # print(f"产生{num_tokens_to_add}个新token共计耗时：{elapsed:.3f} 秒")
    # #合并后的词表，最终的tuple token,训练过程
    # print(pretoken)
    return vocab,merge

#新版，使用增量更新处理pair_count
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_workers: int = 4,
) -> tuple[
    dict[int, bytes],
    list[tuple[bytes, bytes]],
]:
    #默认for 循环过不了 speed test,问题出在BPE merge上   
    vocab=init_vocab(special_tokens)

    start = time.perf_counter()
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    pretoken=count_vocab_parallel(special_tokens, text,num_workers=num_workers)

    # print(f"countVocab耗时：{time.perf_counter() - start:.3f} 秒")
    merge=[]

    bpe_start = time.perf_counter()
    num_tokens_to_add = vocab_size - len(vocab)
    pair_count=countPairs(pretoken)
    # print(pair_count)

    with tqdm(total=vocab_size-len(vocab), desc="训练 BPE") as progress:
        while len(vocab)<vocab_size:
            #每次都执行countPairs导致复杂度超了
            if not pair_count:
                break
#             更新 pretoken 中的 token 序列
#             更新 pair_count 中的相邻 pair 频率
            pair=get_best_pair(pair_count) 
            vocab[len(vocab)]=pair[0]+pair[1]
            merge.append((pair[0],pair[1]))

            #循环遍历所有pair
            new_pretokens=defaultdict(int)
            new_pair_count = defaultdict(int)
            for token_tuple, frequency in pretoken.items():
                merged_tuple=merge_token2(token_tuple,pair,frequency,new_pair_count)
                new_pretokens[merged_tuple]+=frequency
            pretoken=new_pretokens
            pair_count=new_pair_count

    elapsed = time.perf_counter() - bpe_start
    # print(f"产生{num_tokens_to_add}个新token共计耗时：{elapsed:.3f} 秒")
    # #合并后的词表，最终的tuple token,训练过程
    # print(pretoken)
    return vocab,merge

if __name__ == "__main__":
    specialTokens=['<|endoftext|>','<|startoftext|>']
    text="hello<|endoftext|>world low low low lower"
    file="/nfs/tianjungang/assignment1-basics-main/data/TinyStoriesV2-GPT4-valid.txt"
    file="/nfs/tianjungang/assignment1-basics-main/data/1.txt"
    import time

    # pretoken=count_vocab_parallel(specialTokens,text,1)


    # # for num_workers in [1,2,4,8]:
    for num_workers in [8]:
        start = time.perf_counter()
        # print(f"使用 {num_workers} 个进程进行并行化处理：")
        vocab2, merge2 = train_bpe(file,258+5, specialTokens,  num_workers=num_workers)
        elapsed = time.perf_counter() - start
        # print(f"完整任务耗时：{elapsed:.3f} 秒\n")



    # print(vocab)
    # print(merge)


    # with open(file, 'r', encoding='utf-8') as f:
    #         text = f.read()
    # # 验证countVocab和并行化的count_vocab_parallel是否一致
    # preToken1=countVocab(specialTokens,text)
    # preToken2=count_vocab_parallel(specialTokens, text)
    # print(preToken1==preToken2)