import torch
import kge.model
import numpy as np
import random
import json
import sys
import pickle
from tqdm import tqdm
# download link for this checkpoint given under results above
model = kge.model.KgeModel.load_from_checkpoint('./local/fb15k-237-complex.pt')

result_dir = "/var/scratch2/uji300/OpenKE-results/"
test_file = result_dir+"/fb15k237/data/fb15k237-transe-test-topk-10.json"

ent_dict_file = "/var/scratch2/uji300/OpenKE-results/fb15k237/misc/fb15k237-id-to-entity.pkl"
rel_dict_file = "/var/scratch2/uji300/OpenKE-results/fb15k237/misc/fb15k237-id-to-relation.pkl"
def load_pickle(file_name):
    with open(file_name, 'rb') as fin:
        pkl = pickle.load(fin)
    return pkl

ent_dict = load_pickle(ent_dict_file)
rel_dict = load_pickle(rel_dict_file)

tail_ans_dict = {}
head_ans_dict = {}
def read_file_collect_triples(file_name):
    with open(file_name, "r") as fin:
        lines = fin.readlines()
        for line in lines[1:]:
            head = int(line.split()[0])
            tail = int(line.split()[1])
            rel  = int(line.split()[2].rstrip())
            if (head,rel) not in tail_ans_dict:
                tail_ans_dict[(head,rel)] = [tail]
            else:
                tail_ans_dict[(head,rel)].append(tail)

            if (tail,rel) not in head_ans_dict:
                head_ans_dict[(tail,rel)] = [head]
            else:
                head_ans_dict[(tail,rel)].append(head)

train_file = "/home/uji300/OpenKE/benchmarks/fb15k237/train2id.txt"
valid_file = "/home/uji300/OpenKE/benchmarks/fb15k237/valid2id.txt"

read_file_collect_triples(train_file)
read_file_collect_triples(valid_file)

with open(test_file, "r") as fin:
    records = json.loads(fin.read())

#query_string = open("test-queries-string.log", "w")
heads_for_unique_tail_queries = []
rels_for_unique_tail_queries  = []
tails_for_unique_head_queries = []
rels_for_unique_head_queries  = []

unique_pairs_hr = set()
unique_pairs_rt = set()
for r in records:
    head = r['head']
    tail = r['tail']
    rel  = r['rel']
    if (head, rel) not in unique_pairs_hr:
        unique_pairs_hr.add((head, rel))
        heads_for_unique_tail_queries.append(head)
        rels_for_unique_tail_queries.append(rel)
    if (tail, rel) not in unique_pairs_rt:
        unique_pairs_rt.add((tail,rel))
        tails_for_unique_head_queries.append(tail)
        rels_for_unique_head_queries.append(rel)

    #heads.append(r['head'])
    #tails.append(r['tail'])
    #rels.append(r['rel'])
    #h_str = model.dataset.entity_strings(r['head'])
    #t_str = model.dataset.entity_strings(r['tail'])
    #r_str = model.dataset.relation_strings(r['rel'])
    #if h_str is not None and t_str is not None and r_str is not None:
    #    print(h_str + "," + r_str + " => "+ t_str, file = query_string)
    #else:
    #    print("some IDs not present")

#query_string.close()
'''
s = torch.Tensor([0, 2,]).long()             # subject indexes
p = torch.Tensor([0, 1,]).long()             # relation indexes
scores = model.score_sp(s, p)                # scores of all objects for (s,p,?)
o = torch.argmax(scores, dim=-1)             # index of highest-scoring objects
print(model.dataset.entity_strings(s[0]), model.dataset.relation_strings(p[0]), "=> ", model.dataset.entity_strings(o[0]))
print(model.dataset.entity_strings(s[1]), model.dataset.relation_strings(p[1]), "=> ", model.dataset.entity_strings(o[1]))

'''
topk = 10
s = torch.Tensor(heads_for_unique_tail_queries).long()             # subject indexes
p = torch.Tensor(rels_for_unique_tail_queries).long()             # relation indexes
scores = model.score_sp(s, p)                # scores of all objects for (s,p,?)
o = torch.argsort(scores, dim=-1, descending = True)             # index of highest-scoring objects
#o = torch.argmax(scores, dim=-1)             # index of highest-scoring objects

log = open("complex-answers-tail.log", "w")
for index in tqdm(range(0, len(heads_for_unique_tail_queries))):
    #h = model.dataset.entity_strings(s[index])
    #r = model.dataset.relation_strings(p[index])

    filtered_answers = []
    if (s[index].item(), p[index].item()) not in tail_ans_dict.keys():
        for oi in o[index][:topk]:
            filtered_answers.append(oi.item())
    else:
        for oi in o[index]:
            if oi.item() not in tail_ans_dict[(s[index].item(),p[index].item())]:
                filtered_answers.append(oi.item())
                if len(filtered_answers) == topk:
                    break

    assert(len(filtered_answers) == topk)

    #t = model.dataset.entity_strings(torch.Tensor(filtered_answers).long())
    h_dict = ent_dict[s[index].item()]
    r_dict = rel_dict[p[index].item()]
    #print(h, ",", r , "=>", t, file = log)
    for i in range(topk):
        t_dict = ent_dict[filtered_answers[i]]
        print(h_dict, ",", r_dict, ",", t_dict, ";", s[index].item(), ",", p[index].item(), ",", filtered_answers[i], sep='', file = log)
    print("*" * 80, file = log)

log.close()
#'''

'''
    Head predictions
'''
o = torch.Tensor(tails_for_unique_head_queries).long()             # object indexes
p = torch.Tensor(rels_for_unique_head_queries).long()             # relation indexes
scores = model.score_po(p, o)                # scores of all subjects for (?,p,o)
s = torch.argsort(scores, dim=-1, descending = True)             # index of highest-scoring objects
#o = torch.argmax(scores, dim=-1)             # index of highest-scoring objects

log = open("complex-answers-head.log", "w")
for index in tqdm(range(0, len(tails_for_unique_head_queries))):
    #t = model.dataset.entity_strings(o[index])
    #r = model.dataset.relation_strings(p[index])

    filtered_answers = []
    if (o[index].item(), p[index].item()) not in head_ans_dict.keys():
        for si in s[index][:topk]:
            filtered_answers.append(si.item())
    else:
        for si in s[index]:
            if si.item() not in head_ans_dict[(o[index].item(),p[index].item())]:
                filtered_answers.append(si.item())
                if len(filtered_answers) == topk:
                    break

    assert(len(filtered_answers) == topk)

    #h = model.dataset.entity_strings(torch.Tensor(filtered_answers).long())
    t_dict = ent_dict[o[index].item()]
    r_dict = rel_dict[p[index].item()]
    #print(h, ",", r , "=>", t, file = log)
    for i in range(topk):
        h_dict = ent_dict[filtered_answers[i]]
        print(h_dict, ",", r_dict, ",", t_dict, ";", filtered_answers[i], ",", p[index].item(), ",", o[index].item(), sep='', file = log)
    print("*" * 80, file = log)

log.close()
