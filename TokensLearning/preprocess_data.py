import tokenizer
import json
import os
import pickle

if __name__ == '__main__':
    token_file = 'demo_token_dataset/token_list.json'
    if os.path.exists(token_file):
        token_list = json.load(open(token_file, 'r'))
    else:
        token_list = ['EOF', 'MASK']
    for i in os.walk('demo_txt_dataset'):
        for j in i[2]:
            if j.endswith('.txt'):
                txt = open(os.path.join(i[0], j), 'r', encoding='utf-8').read()
                print(f'Processing {j}')
                res = tokenizer.query_or_update_tokenizer(token_list, txt)
                pickle.dump(res, open(os.path.join('demo_token_dataset/', j.replace('.txt', '.token.pkl')), 'wb'))
    json.dump(token_list, open(token_file, 'w'))
    print(f'Token list saved, total {len(token_list)} tokens')