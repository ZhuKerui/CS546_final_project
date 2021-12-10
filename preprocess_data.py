import json
import tqdm
import os

data_folder = 'webnlg_dataset'
raw_folder = 'raw'
mix_folder = 'mix'
single_folder = 'single'

raw_train_data_file = os.path.join(data_folder, raw_folder, 'webnlg_release_v2.1_train.json')
raw_dev_data_file = os.path.join(data_folder, raw_folder, 'webnlg_release_v2.1_dev.json')
raw_test_data_file = os.path.join(data_folder, raw_folder, 'webnlg_release_v2.1_test.json')

if not os.path.exists(os.path.join(data_folder, mix_folder)):
    os.mkdir(os.path.join(data_folder, mix_folder))
    
mix_processed_train_data_file = os.path.join(data_folder, mix_folder, 'webnlg_train.jsonl')
mix_processed_dev_data_file = os.path.join(data_folder, mix_folder, 'webnlg_dev.jsonl')
mix_processed_test_data_file = os.path.join(data_folder, mix_folder, 'webnlg_test.jsonl')

if not os.path.exists(os.path.join(data_folder, single_folder)):
    os.mkdir(os.path.join(data_folder, single_folder))

unused_labels = ['originaltriplesets', 'xml_id', 'size', 'shape', 'shape_type']

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
    
    print('Collect mix data...')
    file_dict = {'Train' : (raw_train_data_file, mix_processed_train_data_file), 
                 'Dev' : (raw_dev_data_file, mix_processed_dev_data_file), 
                 'Test' : (raw_test_data_file, mix_processed_test_data_file)}
    for k, v in file_dict.items():
        print('Processing %s file' % k)
        raw_data = json.load(open(v[0]))['entries']
        data_samples = [data_sample[list(data_sample.keys())[0]] for data_sample in raw_data]
        
        new_data_samples = []
        for data_sample in tqdm.tqdm(data_samples):
            new_data_sample = data_sample.copy()
            for label in unused_labels:
                new_data_sample.pop(label)
            triples = new_data_sample['modifiedtripleset']
            new_data_sample['input_seq'], new_data_sample['properties'] = generate_seq(triples)
            new_data_sample['target_sents'] = [sent['lex'] for sent in new_data_sample.pop('lexicalisations')]
            new_data_samples.append(new_data_sample)
        with open(v[1], 'w') as f_out:
            f_out.write('\n'.join([json.dumps(data) for data in new_data_samples]))


    print('Collect single data...')
    file_dict = {'Train' : raw_train_data_file,
                 'Dev' : raw_dev_data_file,
                 'Test' : raw_test_data_file}
    label2sample = {}
    for k, v in file_dict.items():
        print('Processing %s file' % k)
        raw_data = json.load(open(v))['entries']
        data_samples = [data_sample[list(data_sample.keys())[0]] for data_sample in raw_data]
        
        for data_sample in tqdm.tqdm(data_samples):
            if data_sample['size'] != '1':
                continue
            new_data_sample = data_sample.copy()
            for label in unused_labels:
                new_data_sample.pop(label)
            triples = new_data_sample['modifiedtripleset']
            new_data_sample['input_seq'], new_data_sample['properties'] = generate_seq(triples)
            new_data_sample['target_sents'] = [sent['lex'] for sent in new_data_sample.pop('lexicalisations')]
            property_ = new_data_sample['properties'][0]
            if property_ not in label2sample:
                label2sample[property_] = {'Train' : [], 'Dev' : [], 'Test' : []}
            label2sample[property_][k].append(new_data_sample)
    for p in label2sample:
        if len(label2sample[p]['Train']) < 10 or len(label2sample[p]['Dev']) < 5 or len(label2sample[p]['Test']) < 3:
            continue
        temp_folder = os.path.join(data_folder, single_folder, p.split('/')[-1])
        if not os.path.exists(temp_folder):
            os.mkdir(temp_folder)
        for k, v in label2sample[p].items():
            temp_file = os.path.join(temp_folder, k+'.jsonl')
            with open(temp_file, 'w') as f_out:
                f_out.write('\n'.join([json.dumps(data) for data in label2sample[p][k]]))