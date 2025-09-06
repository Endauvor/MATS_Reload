from bisect import bisect_left
from typing import List, Dict, Callable
import numpy as np

def _positions_map(seq: List[int]) -> Dict[int, int]:
    """Карта: элемент -> его (первая) позиция."""
    return {x: i for i, x in enumerate(seq)}

def upvote_reward(first: List[int], second: List[int], reward_fn: Callable[[int, int], float] = None) -> Dict:
    
    N = len(first)
    pos_first = _positions_map(first)
    pos_second = _positions_map(second)

    common_items = [x for x in first if x in pos_second]
    first_list = [pos_first[x] for x in common_items]
    second_list = [pos_second[x] for x in common_items]

    if reward_fn is None:
        reward_fn = lambda k: 2*np.exp(-0.4*k)


    shifts = [(reward_fn(f) - reward_fn(i), f - i) for i, f in zip(first_list, second_list)]

    pos_rewards = [tup[0] for tup in shifts if tup[1] < 0]
    upvote_reward = np.sum(pos_rewards)
    
    return upvote_reward

def _lis_length_strict(seq: List[int]) -> int:
    """
    Длина строго возрастающей подпоследовательности (O(n log n)),
    алгоритм 'patience sorting'.
    """
    tails: List[int] = []
    for x in seq:
        j = bisect_left(tails, x)
        if j == len(tails):
            tails.append(x)
        else:
            tails[j] = x
    return len(tails)

def ulam_distance_ratio(first: List[int], second: List[int], allow_decreasing: bool = True) -> Dict:
    r"""
    Функция 2. Ulam distance на пересечении.

    P = (j_1,...,j_n), где j_k = pos_B(a_{i_k}), а i_k возрастают.
    lis_inc = LIS(P); при allow_decreasing=True также lis_dec = LIS(-P).
    best = max(lis_inc, lis_dec) при allow_decreasing, иначе lis_inc.
    U = n - best; ratio = U / n.

    Возвращает: n_common, lis_inc, lis_dec (или None), best, ulam, ratio, P.
    """

    N = len(first)
    pos_second = _positions_map(second)
    P = [pos_second[x] for x in first if x in pos_second]
    n = len(P)
    if n == 0:
        return {
            "n_common": 0, "lis_inc": 0, "lis_dec": 0 if allow_decreasing else None,
            "best": 0, "ulam": 0, "ratio": 0.0, "P": []
        }

    lis_inc = _lis_length_strict(P)
    if allow_decreasing:
        lis_dec = _lis_length_strict([-p for p in P])
        best = max(lis_inc, lis_dec)
    else:
        lis_dec = None
        best = lis_inc

    return  best / N

#def churn_rates(first: List[int], second: List[int]) -> Dict:
#    r"""
#    Функция 3. Доли исчезнувших/новых относительно размера первого списка L.
#    disappeared = {a_i \notin B}, appeared = {b_j \notin A}.
#    """
#    L = len(first)
#    set_first = set(first)
#    appeared  = [x for x in second if x not in set_first]
#
#    return {
#        "appeared_frac": len(appeared) / L if L else 0.0,
#    }

def newcomer_reward(first: List[int], second: List[int],
                    reward_fn: Callable[[int, int], float] = None) -> Dict:
    r"""
    Функция 4. Награда за новичков по их позициям в B.
    C = { k : b_k \notin A }.
    По умолчанию R(k) = L*(L-k). Reward = \sum_{k \in C} R(k).
    """
    L = len(first)
    set_first = set(first)
    if reward_fn is None:
        reward_fn = lambda k: 2*np.exp(-0.4*k)

    whole_reward = np.sum(np.array([reward_fn(k) for k in range(L)]))

    new_positions = [k for k, x in enumerate(second) if x not in set_first]
    per_pos = {k: reward_fn(k) for k in new_positions}

    reward = sum(per_pos.values()) / whole_reward

    return reward

def print_analysis_results(first: List[int], second: List[int]):
   
    
    print(f"A: {first}")
    print(f"B: {second}")
    print("-"*40)
    
    shifts_result = overlap_shift_stats(first, second)
    fractions = shifts_result['fractions']

    print(f"upvote :  {fractions['f_plus']}")
    print(f"downvote : {fractions['f_minus']}")
    print(f"no vote :  {fractions['f_zero']}")
    
    if shifts_result['avg_upvote'] > 0:
        print(f"avg upvote: {shifts_result['avg_upvote']} positions")
    if shifts_result['avg_downvote'] > 0:
        print(f"avg_downvote: {-shifts_result['avg_downvote']} positions")
    
    print("-"*40)
    ulam_result = ulam_distance_ratio(first, second)
    
    print(f"Order: {ulam_result['best']} ")
   
    
    
    
    print("-"*40)
    churn_result = churn_rates(first, second)
    
    
    print(f"New fraction: {churn_result['appeared_frac']}")
    print("-"*40)
    reward_result = newcomer_reward(first, second)
    print("Reward:", reward_result["total_reward"])
    
    #if reward_result['new_positions']:
        
    #    for pos in reward_result['new_positions']:
    #        reward = reward_result['per_position_reward'][pos]
    #        element = second[pos]
    #        print(f"  Position {pos}: элемент {element}, reward {reward}")
    
    print("\n" + "="*60)

def analysis_results(first: List[int], second: List[int]):

    upvote = upvote_reward(first, second)
    ulam_result = ulam_distance_ratio(first, second)
    new_reward = newcomer_reward(first, second)

    return upvote, ulam_result, new_reward
    





if __name__ == "__main__":
    
    A = [1,2,3,4,5,6,7,8,9,10]
    B = [1,8,2,4,7,10,6,5,10,21,22]
    
    analysis_results(A, B)

