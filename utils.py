import torch

def hamming_score(vector1, vector2):
    assert len(vector1) == len(vector2), "Vectors must be of the same length"
    vector1 = torch.tensor(vector1, dtype=torch.float32)
    vector2 = torch.tensor(vector2, dtype=torch.float32)
    # Calculate the number of differing bits
    differing_bits = (vector1 != vector2).float().sum()
    # Calculate the Hamming score
    hamming_score = differing_bits / len(vector1)
    return hamming_score.item()
