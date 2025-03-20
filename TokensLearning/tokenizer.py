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
                    if p_t > p:
                        pass # normal state
                    else:
                        p_t += 1 # current char is the special char
                    new_token = text[p:p_t]
                    break
            if new_token is None:
                # couldn't find? assume it's 8 bytes as max
                new_token = text[p:p+8]
            print(f'New Token: {new_token}')
            token_list.append(new_token)
            p += len(new_token)

if __name__ == "__main__":
    token_list = ['EOF', 'MASK']
    build_tokenizer(token_list, 'Hello, World! 233 Test')
    print(token_list)