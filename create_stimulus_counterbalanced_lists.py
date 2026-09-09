import numpy as np
def find_counterbalanced_quads(lexicon, min_sem_overlap=1):
    """
    Find sets of 4 words [a, b, c, d] where:
    - a & b are sem-related (not orth-related)
    - c & d are sem-related (not orth-related)
    - {a,b} and {c,d} share NO sem or orth features
    
    Soft constraint: prefer higher semantic overlap within pairs.
    """
    
    # shared_feats = lexicon.semfeatmatrix.T @ lexicon.semfeatmatrix
    shared_feats = np.load('./helper_txt_files/shared_feats.npy')
    diag_mask = np.ones(shared_feats.shape) - np.eye(shared_feats.shape[0])
    only_shared_features = shared_feats * diag_mask
    
    orthrelated_matrix = lexicon.orthmatrix.T @ lexicon.orthmatrix
    
    n_words = lexicon.size
    
    # Step 1: Find ALL valid sem-related pairs with their overlap strength
    sem_pairs = []
    
    for i in range(n_words):
        for j in range(i+1, n_words):
            sem_overlap = only_shared_features[i, j]
            orth_overlap = orthrelated_matrix[i, j]
            
            if sem_overlap >= min_sem_overlap and orth_overlap == 0:
                sem_pairs.append((i, j, sem_overlap))
    
    # Sort by semantic overlap (strongest first)
    sem_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"Found {len(sem_pairs)} valid sem-related pairs")
    print(f"Overlap distribution: "
          f"max={sem_pairs[0][2]}, "
          f"median={sem_pairs[len(sem_pairs)//2][2]}, "
          f"min={sem_pairs[-1][2]}")
    
    # Step 2: Check pair compatibility
    def pairs_are_unrelated(pair1, pair2):
        a, b = pair1[0], pair1[1]
        c, d = pair2[0], pair2[1]
        
        for x in [a, b]:
            for y in [c, d]:
                if only_shared_features[x, y] > 0:
                    return False
                if orthrelated_matrix[x, y] > 0:
                    return False
        return True
    
    # Step 3: Build compatibility matrix
    n_pairs = len(sem_pairs)
    print(f"Building compatibility matrix for {n_pairs} pairs...")
    
    compatible = np.zeros((n_pairs, n_pairs), dtype=bool)
    
    for i in range(n_pairs):
        for j in range(i+1, n_pairs):
            if pairs_are_unrelated(sem_pairs[i], sem_pairs[j]):
                compatible[i, j] = True
                compatible[j, i] = True
        
        if i % 500 == 0:
            print(f"  Processed {i}/{n_pairs} pairs...")
    
    # Step 4: Greedy selection - prioritize strongest pairs
    used_words = set()
    used_pairs = set()
    quads = []
    
    for pair_idx in range(n_pairs):
        if pair_idx in used_pairs:
            continue
            
        a, b, overlap_ab = sem_pairs[pair_idx]
        
        if a in used_words or b in used_words:
            continue
        
        # Find the strongest compatible partner
        compatible_indices = np.where(compatible[pair_idx])[0]
        
        best_partner = None
        best_overlap = -1
        
        for partner_idx in compatible_indices:
            if partner_idx in used_pairs:
                continue
            
            c, d, overlap_cd = sem_pairs[partner_idx]
            
            if c in used_words or d in used_words:
                continue
            
            if overlap_cd > best_overlap:
                best_partner = partner_idx
                best_overlap = overlap_cd
        
        if best_partner is not None:
            c, d, overlap_cd = sem_pairs[best_partner]
            
            quads.append({
                'words': (a, b, c, d),
                'sem_overlap_ab': overlap_ab,
                'sem_overlap_cd': overlap_cd
            })
            
            used_words.update([a, b, c, d])
            used_pairs.add(pair_idx)
            used_pairs.add(best_partner)
    
    print(f"\nFound {len(quads)} quads ({len(quads) * 4} words)")
    
    # Report
    all_overlaps = [q['sem_overlap_ab'] for q in quads] + [q['sem_overlap_cd'] for q in quads]
    print(f"Final semantic overlap stats:")
    print(f"  Mean: {np.mean(all_overlaps):.2f}")
    print(f"  Min: {np.min(all_overlaps)}")
    print(f"  Max: {np.max(all_overlaps)}")
    print(f"  Pairs with >1 shared feature: {sum(1 for o in all_overlaps if o > 1)}/{len(all_overlaps)}")
    
    return quads


def create_stimulus_lists(quads, lexicon):
    """From quads, create the three orderings."""
    standard = []
    sem_related = []
    unrelated = []
    
    for q in quads:
        a, b, c, d = q['words']
        
        standard.extend([a, b, c, d])
        sem_related.extend([b, a, d, c])
        unrelated.extend([c, d, a, b])
    
    return {
        'standard_idx': np.array(standard),
        'sem_related_idx': np.array(sem_related),
        'unrelated_idx': np.array(unrelated),
        'standard_words': [lexicon.words[i] for i in standard],
        'sem_related_words': [lexicon.words[i] for i in sem_related],
        'unrelated_words': [lexicon.words[i] for i in unrelated],
    }


import string
import numpy as np
import random


def word_to_onehot(word):
    """Convert a 4-letter word to a 104-dim one-hot vector, matching lexicon.orthmatrix encoding."""
    alphabet = string.ascii_lowercase
    word = str(word).lower()
    wordids = np.array([alphabet.index(c) for c in word])
    vec = np.zeros(len(word) * len(alphabet))
    indices = wordids + np.array([0, 1, 2, 3]) * 26
    vec[indices] = 1
    return vec


def compute_on_size(pseudo_vec, orth_matrix, threshold=3):
    """Count how many lexicon words share >= threshold letter positions with pseudo_vec."""
    overlaps = pseudo_vec @ orth_matrix  # (1568,)
    return int(np.sum(overlaps >= threshold))


def generate_pseudoword(base_word, lexicon_words_set):
    """Substitute one random letter at one random position. Return None if result is a real word."""
    alphabet = string.ascii_lowercase
    word = list(str(base_word).lower())
    pos = random.randint(0, len(word) - 1)
    letters = [c for c in alphabet if c != word[pos]]
    word[pos] = random.choice(letters)
    candidate = ''.join(word)
    if candidate in lexicon_words_set:
        return None
    return candidate


def generate_matched_pseudowords(standard_words, lexicon, max_attempts=50000):
    """
    For each word in standard_words, generate a pseudoword with the same
    orthographic neighborhood size (words sharing >= 3/4 letter positions).
    """
    lexicon_words_set = set(str(w) for w in lexicon.words)
    orth_matrix = lexicon.orthmatrix  # (104, 1568)
    standard_words = [str(w) for w in standard_words]

    # Precompute target ON sizes
    target_on_sizes = {word: compute_on_size(word_to_onehot(word), orth_matrix)
                       for word in set(standard_words)}

    pseudowords = []
    used_pseudowords = set()

    for word in standard_words:
        target_on = target_on_sizes[word]
        found = None

        for attempt in range(max_attempts):
            candidate = generate_pseudoword(word, lexicon_words_set | used_pseudowords)
            if candidate is None:
                continue

            pseudo_vec = word_to_onehot(candidate)
            pseudo_on = compute_on_size(pseudo_vec, orth_matrix)

            if pseudo_on == target_on:
                found = candidate
                break

        if found is None:
            print(f"WARNING: no match found for '{word}' (target ON={target_on})")
            pseudowords.append(None)
        else:
            used_pseudowords.add(found)
            pseudowords.append(found)
            print(f"{word} (ON={target_on}) -> {found}")

    return pseudowords


# Usage:
# pseudowords = generate_matched_pseudowords(stim_dict['standard_words'], lexicon)