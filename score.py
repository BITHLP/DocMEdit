import re

from rouge_score import rouge_scorer
from rouge_score import scoring
import utils
from pprint import pprint
import difflib


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_additions(source, target, similarity_threshold=0.7):
    target_sentences = target.split('.')
    
    vectorizer = TfidfVectorizer().fit([source] + target_sentences)
    source_vector = vectorizer.transform([source])
    
    similarities = cosine_similarity(source_vector, vectorizer.transform(target_sentences)).flatten()
    
    additions = []
    for idx, similarity in enumerate(similarities):
        if similarity < similarity_threshold:  # Low similarity means a possible addition
            additions.append(target_sentences[idx].strip())
    
    return additions



def extract_remaining(source, target):
    """Extract text that remains in both source and target."""
    remaining_text = []
    for match in re.finditer(r"[^.]+\.?", target):
        if match.group(0) in source:
            remaining_text.append(match.group(0).strip())
    return remaining_text

def get_best_rouge_pairs(src, tgt):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    total_rouge_score = 0
    count = 0
    best_pairs = []
    total_rouge_score_ls = []
    for s in src:
        best_score = -float('inf')
        best_pair = None
        
        for t in tgt:
            score = scorer.score(t, s)
            rougeL_score = score['rougeL'].fmeasure
            avg_rouge = rougeL_score
            
            if avg_rouge > best_score:
                best_score = avg_rouge
                best_pair = (s, t)
        
        best_pairs.append((best_pair, best_score))
        total_rouge_score += best_score
        total_rouge_score_ls.append(best_score)
        count += 1

    avg_rouge_score = total_rouge_score / count if count > 0 else count
    return best_pairs, total_rouge_score_ls, avg_rouge_score
  
  
def edit_rouge_item(item):
  
  scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"])
  target_additions = extract_additions(
        source=item["normalized_inputs"],
        target=item["normalized_targets"],
    )
  target_additions_ls = " ".join(target_additions)
  prediction_additions = extract_additions(
      source=item["normalized_inputs"],
      target=item["output_processed"],
  )
  prediction_additions_ls = " ".join(prediction_additions)
  
  
  addition_scores = scorer.score(
      target=target_additions_ls,
      prediction=prediction_additions_ls,
  )
  return addition_scores

def edit_rouge_remaining_item(item):
  scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"])
  target_additions = extract_remaining(
        source=item["normalized_inputs"],
        target=item["normalized_targets"],
    )
  target_additions = " ".join(target_additions)
  prediction_additions = extract_remaining(
      source=item["normalized_inputs"],
      target=item.get("output_processed",item['output']),
  )
  prediction_additions = " ".join(prediction_additions)
  if target_additions=="" or item["normalized_inputs"] in item.get("output_processed",item['output']) :
  # if target_additions=="" :
    # gt没有需要保留的东西 或者 输出和期望完全一样
    target_additions = "a"
    prediction_additions = "a" ## return score==1
  addition_scores = scorer.score(
      target=target_additions,
      prediction=prediction_additions,
  )
  return addition_scores

def edit_rouge_any(inp,tgt,outp):
  scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"])
  target_additions = extract_additions(
        source=inp,
        target=tgt,
    )
  target_additions = " ".join(target_additions)
  prediction_additions = extract_additions(
      source=inp,
      target=outp,
  )
  prediction_additions = " ".join(prediction_additions)

  addition_scores = scorer.score(
      target=target_additions,
      prediction=prediction_additions,
  )
  return addition_scores

def rouge_evi(evi,inp,pre):
  prediction_additions = extract_additions(
      source=inp,
      target=pre,
  )
  scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"])
  if prediction_additions==[]:
    prediction_additions = [""]
  score_ls = [
      scorer.score(
        target=evi,
        prediction=a,
    )
    for a in prediction_additions
  ]
  max_s =  score_ls[0]
  for i in score_ls:
    if i['rougeL'].fmeasure > max_s['rougeL'].fmeasure:
      max_s = i
  return max_s

def rouge_evils(evils,inp,pre):
  prediction_additions = extract_additions(
      source=inp,
      target=pre,
  )
  _, s, _ = get_best_rouge_pairs(prediction_additions,evils)
  return s
def rouge_any(tgt,pre):
  
  scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"])
  addition_scores = scorer.score(
      target=tgt,
      prediction=pre,
  )
  return addition_scores


def edit_rouge(data):
  """Measures a variety of different ROUGE scores."""
  # We do not measure ROUGE-L for updates since LCS is likely entirely contained
  # in source.
  scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"])
  aggregator = scoring.BootstrapAggregator()

  for item in data:

    all_scores = {}

    target_additions = extract_additions(
        source=item["normalized_inputs"],
        target=item["normalized_targets"],
    )
    target_additions = " ".join(target_additions)
    prediction_additions = extract_additions(
        source=item["normalized_inputs"],
        target=item["output"],
    )
    prediction_additions = " ".join(prediction_additions)

    addition_scores = scorer.score(
        target=target_additions,
        prediction=prediction_additions,
    )
    # addition_scores = scorer.score(
    #     target=item["normalized_targets"],
    #     prediction=item["normalized_inputs"],
    # )
    if target_additions.strip() or prediction_additions.strip():
      all_scores.update({f"update_{k}": v for k, v in addition_scores.items()})
    else:
      all_scores.update(
          {f"update_{k}": 100.0 for k, _ in addition_scores.items()})

    aggregator.add_scores(all_scores)

  result = aggregator.aggregate()
  return {key: value.mid.fmeasure * 100 for key, value in result.items()}



def score_one(fn):
    data = utils.read_json(fn)
    score = edit_rouge(data)
    pprint(score)

def ret_tuple(ls):
  return [tuple(i) for i in ls]

from typing import List

def calculate_generation_quality(b: List[List[str]], c: List[List[str]]) -> dict:
    b_set = set(tuple(triplet) for triplet in b)
    c_set = set(tuple(triplet) for triplet in c)

    intersection = b_set.intersection(c_set)

    precision = len(intersection) / len(c_set) if len(c_set) > 0 else 0
    recall = len(intersection) / len(b_set) if len(b_set) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1


def compute_update_and_overlap(a, b, c,item):
    """
    Given three lists a, b, and c, this function computes the updates b1 and c1, 
    and calculates the overlap ratio of c1 in b1.
    
    Args:
    - a: List representing the original set.
    - b: List representing the updated set compared to a.
    - c: List representing the updated set compared to a.
    
    Returns:
    - overlap_ratio: The ratio of elements in c1 that overlap with b1.
    """
    if a!=[] and isinstance(a[0],list):
      a,b,c = ret_tuple(a), ret_tuple(b), ret_tuple(c)
    b1 = set(b) - set(a)
    if b1==set():
      return -1

    c1 = set(c) - set(a)
    overlap = len(c1 & b1)  # intersection of c1 and b1
    overlap_ratio = overlap / len(b1) if len(b1) > 0 else 0
    b1 = sorted(b1)
    c1 = sorted(c1) # for debug
    # c1.sort()
    return overlap_ratio
  
def compute_update_and_overlap_ei(evi, inp, tgt, outp):
    evi, inp, tgt, outp = ret_tuple(evi), ret_tuple(inp),ret_tuple(tgt), ret_tuple(outp)
    diff_tgt_inp = set(tgt) - set(inp)
    
    intersection_evi_tgt = set(evi) & set(diff_tgt_inp)
    if intersection_evi_tgt == set():
      return -1
    
    diff_outp_inp = set(outp) - set(inp)
    overlap_ratio = len(diff_outp_inp & intersection_evi_tgt) / len(intersection_evi_tgt) if len(intersection_evi_tgt) > 0 else 0

    return overlap_ratio
  
def compute_ratio_tse(a, b, c):
    """
    Given three lists a, b, and c, this function computes the ratio of elements in c 
    that are in a, with respect to the elements in b that are in a.
    
    Args:
    - a: List representing the original set.
    - b: List representing the comparison set.
    - c: List representing the comparison set.
    
    Returns:
    - ratio: The ratio of elements in c that are in a with respect to elements in b that are in a.
    """
    a,b,c = ret_tuple(a), ret_tuple(b), ret_tuple(c)
    b_in_a = set(b) & set(a)
    c_in_a = set(c) & set(a)
    c_in_a = c_in_a & b_in_a
    ratio = len(c_in_a) / len(b_in_a) if len(b_in_a) > 0 else 0

    return ratio

