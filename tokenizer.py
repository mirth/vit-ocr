import string


assert len(string.ascii_lowercase) == len(string.ascii_uppercase)

id_by_char = {'|': 0}
char_by_id = {0 : '|'}

for i, (c, C) in enumerate(zip(string.ascii_lowercase, string.ascii_uppercase), 1):
    id_by_char[c] = i
    id_by_char[C] = i
    char_by_id[i] = c

for i, d in enumerate(string.digits, len(char_by_id)):
    id_by_char[d] = i
    char_by_id[d] = i

def encode(s, *, max_length):
    s = [id_by_char[c] for c in s]
    s += [0 for _ in range(max_length - len(s))]

    return s         

def decode(ids):
    return ''.join([char_by_id[id] for id in ids])

# print(id_by_char)
# print(char_by_id)
