import json
import tqdm


raw_train_data_file = 'webnlg_dataset/webnlg_release_v2.1_train.json'
raw_dev_data_file = 'webnlg_dataset/webnlg_release_v2.1_dev.json'
raw_test_data_file = 'webnlg_dataset/webnlg_release_v2.1_test.json'

processed_train_data_file = 'webnlg_dataset/webnlg_train.jsonl'
processed_dev_data_file = 'webnlg_dataset/webnlg_dev.jsonl'
processed_test_data_file = 'webnlg_dataset/webnlg_test.jsonl'


def generate_seq(data:list):
    parent = {}
    children = {}
    relations = {}
    for tri in data:
        sub, obj, rel = tri['subject'], tri['object'], tri['property']
        if sub == obj:
            return [sub, '[P]', obj], [rel]
        parent[obj] = sub
        if sub not in children:
            children[sub] = []
        children[sub].append(obj)
        relations[(sub, obj)] = rel
    root = list(parent.keys())[0]
    while root in parent:
        root = parent[root]
    seq = []
    relation_list = []
    def dfs(root:str):
        seq.append(root)
        if root in children:
            for child in children[root]:
                relation_list.append(relations[(root, child)])
                seq.append('[P]')
                dfs(child)
    dfs(root)
    return seq, relation_list


if __name__ == '__main__':
    file_dict = {'Train' : (raw_train_data_file, processed_train_data_file), 
                 'Dev' : (raw_dev_data_file, processed_dev_data_file), 
                 'Test' : (raw_test_data_file, processed_test_data_file)}
    for k, v in file_dict.items():
        print('Processing %s file' % k)
        raw_data = json.load(open(v[0]))['entries']
        data_samples = [data_sample[list(data_sample.keys())[0]] for data_sample in raw_data]
        
        new_data_samples = []
        for data_sample in tqdm.tqdm(data_samples):
            new_data_sample = data_sample.copy()
            unused_labels = ['originaltriplesets', 'xml_id', 'size', 'shape', 'shape_type']
            for label in unused_labels:
                new_data_sample.pop(label)
            triples = new_data_sample['modifiedtripleset']
            new_data_sample['input_seq'], new_data_sample['properties'] = generate_seq(triples)
            new_data_sample['target_sents'] = [sent['lex'] for sent in new_data_sample.pop('lexicalisations')]
            new_data_samples.append(new_data_sample)
        with open(v[1], 'w') as f_out:
            f_out.write('\n'.join([json.dumps(data) for data in new_data_samples]))