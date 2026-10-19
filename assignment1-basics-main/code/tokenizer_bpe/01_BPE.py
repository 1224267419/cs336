import regex as re
from collections import defaultdict

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

specialTokens=['<|endoftext|>','<|startoftext|>']
vocab=init_vocab(specialTokens)
# print(vocab)
text="hello<|endoftext|>world"
# print(text.replace('<|endoftext|>','')) #不能这样做，这样子如demo所示会变成helloworld,
# 分词拆分会出问题

#按specialToken截断,escaped用于转义避免正则字符被转换
def countVocab(specialTokens,text):
    #按词计算频率(词频)

    escaped = [re.escape(tok) for tok in specialTokens]
    #正则|切分，取或的意思(遇到任意特殊字符就切分)
    split_pattern = "|".join(escaped)
    parts = re.split(split_pattern, text)
    # print("-------------------")
    # print(split_pattern)
    # print(parts)
    # print("-------------------")

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretoken_counts = defaultdict(int)
    for part in parts:
        for match in re.finditer(PAT,part):
            pretoken=match.group()
            bs=pretoken.encode('utf-8')
            pretoken_bytes=tuple(bytes([b]) for b in bs)
            pretoken_counts[pretoken_bytes]+=1
            print(match.group())

    # print("-------------------")
    # print(dict(pretoken_counts))
    # print("-------------------")

    return dict(pretoken_counts)
    


text="hello<|endoftext|>world low low low lower"
pretokens=countVocab(specialTokens,text)
def countPairs(pretokens):
    pair_counts = defaultdict(int)
    for token_tuple, count in pretokens.items():
        for i in range(len(token_tuple)-1):
            pair_counts[(token_tuple[i],token_tuple[i+1])]+=count
    return pair_counts

def get_bset_pair(count):
    if not count:
        return None
    return max(count,key=lambda pair:(count[pair],pair))

def merge_token(token_tuple, pair):
    #合并后的token样式,这里用的简单匹配，用kmp应该会快一点? 这个想法是错误的，因为匹配长度固定为2，KMP只会增加开销
    # token_tuple是指合并几次后的序列，里面的元素可能不再是单个字
    new_token = []
    i = 0

    while i < len(token_tuple):
        if (i <= len(token_tuple) - len(pair)):
            token=True
            for j in range( len(pair)):
                if token_tuple[i+j]!=pair[j]:
                    #不能合并
                    token=False
                    break
            if token==True:
                new_token.append(pair[0] + pair[1])
                #合并了相邻的两个token,所以i要加2
                i += 2
            else:
                new_token.append(token_tuple[i])
                i += 1     

        else:
            #用于退出死循环
            new_token.append(token_tuple[i])
            i += 1
    
    return tuple(new_token)



def loop_vocab_size(vocab_size,vocab,pretoken):
    # while len(vocab)<vocab_size+len(specialTokens):
    merge=[]
    while len(vocab)<vocab_size:
        pair_count=countPairs(pretoken)
        if not pair_count:
            break
        pair=get_bset_pair(pair_count)  
        vocab[len(vocab)]=pair[0]+pair[1]
        merge.append(pair[0]+pair[1])
        #循环遍历所有pair
        new_pretokens=defaultdict(int)
        for token_tuple, count in pretoken.items():
            merged_tuple=merge_token(token_tuple,pair)
            new_pretokens[merged_tuple]+=count
        pretoken=new_pretokens
        print(new_pretokens)
    #合并后的词表，最终的tuple token,训练过程
    return vocab,pretoken,merge


count=countPairs(pretokens)
print(count)
pair=get_bset_pair(count)
print(pair)


vocab,merged,merge=loop_vocab_size(len(vocab)+5,vocab,pretokens) #258+2+5

print(vocab)
print(merged)
print(merge)
print(len(merge))
#一次完整的合并

#默认的max函数是按字典序比较的，所以这里要用lambda函数指定按频率比较；想同时按频率和字典序比较可以用元组，先按频率再按字典序
print(max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
==get_bset_pair({("A", "B"): 5, ("A", "C"): 5, ("B", "ZZ"): 3, ("BA", "A"): 5}))

# print(pretokens)
# print(merged)

