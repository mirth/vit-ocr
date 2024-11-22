import string


chars = ['|'] + list(string.ascii_lowercase + string.digits)

char_by_id = {i:c for i, c in enumerate(chars)}
id_by_char = {c:i for i, c in enumerate(chars)}

def encode(s, *, max_length):
    s = [id_by_char[c] for c in s]
    s += [0 for _ in range(max_length - len(s))]

    return s         

def decode(ids):
    return ''.join([char_by_id[id] for id in ids])