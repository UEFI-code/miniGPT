def build_tokenizer(token_list, text):
    p = 0
    while p < len(text): # len(text) is Not contains the '\0'
        matchedToken = None
        for token in token_list[2:]: # 0 is EOF, 1 is Mask
            if text[p:].startswith(token):
                matchedToken = token
                break
        if matchedToken is not None:
            print(f'Found Token: {matchedToken}')
            p += len(matchedToken)
        else:
            # no token can be matched, insert new
            new_token = None
            for p_t in range(p, len(text)):
                if ord(text[p_t]) < 65:
                    if not p_t > p:
                        p_t += 1 # current char is special
                    new_token = text[p:p_t]
                    break
            if new_token is None:
                # couldn't find? assume it's 8 bytes as max
                new_token = text[p:p+8]
            print(f'New Token: {new_token}')
            token_list.append(new_token)
            p += len(new_token)

def query_or_update_tokenizer(token_list, text):
    results = []
    p = 0
    while p < len(text): # len(text) is Not contains the '\0'
        matchedToken = None
        for token_id in range(2, len(token_list)): # 0 is EOF, 1 is Mask
            if text[p:].startswith(token_list[token_id]):
                matchedToken = token_list[token_id]
                break
        if matchedToken is not None:
            print(f'Found Token: {matchedToken}')
            results.append(token_id)
            p += len(matchedToken)
        else:
            # no token can be matched, insert new
            new_token = None
            for p_t in range(p, len(text)):
                if ord(text[p_t]) < 65:
                    if not p_t > p:
                        p_t += 1 # current char is special
                    new_token = text[p:p_t]
                    break
            if new_token is None:
                # couldn't find? assume it's 8 bytes as max
                new_token = text[p:p+8]
            print(f'New Token: {new_token}')
            token_list.append(new_token)
            results.append(len(token_list) - 1)
            p += len(new_token)
    return results

def decode_back(token_list, encoded_tokens):
    decoded_tokens = []
    for token_idx in encoded_tokens:
        try:
            decoded_tokens.append(token_list[token_idx])
        except:
            decoded_tokens.append('UNK')
    return ''.join(decoded_tokens)

if __name__ == "__main__":
    token_list = ['EOF', 'MASK']
    #build_tokenizer(token_list, 'Hello, World! 233 Test')
    results = query_or_update_tokenizer(token_list, 'Hello, World! 233 Test')
    print(f'Token List: {token_list}')
    print(f'Encoded Results: {results}')
    results = decode_back(token_list, results)
    print(f'Decoded Results: {results}')