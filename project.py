import argparse
import json
import logging
import yaml
import re
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModelWithLMHead, AutoTokenizer
from tqdm import tqdm
import os


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config", "--config", required=True, type=str, help="path to the config file"
    )
    args = vars(parser.parse_args())

    with open(args['config']) as config_in:
        config = yaml.safe_load(config_in)
        
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")
    
    return config


def set_logger(config):
    if config['debug']:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%d-%m-%Y:%H:%M:%S',
        filemode='w',
        filename=config['log_file'],
        level=level)
    logger = logging.getLogger(__name__)
    return logger


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def add_task_specific_tokens(tokenizer):
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[T]']
    })
    tokenizer.trigger_token = '[T]'
    tokenizer.trigger_token_id = tokenizer.convert_tokens_to_ids('[T]')


def load_pretrained(model_name):
    """
    Loads pretrained HuggingFace config/model/tokenizer, as well as performs required
    initialization steps to facilitate working with triggers.
    """
    pretrained_model_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    add_task_specific_tokens(tokenizer)
    return pretrained_model_config, model, tokenizer


def get_embeddings(model, pretrained_model_config):
    """Returns the wordpiece embedding module."""
    base_model = model.get_encoder()
    embeddings = base_model.embed_tokens
    return embeddings


class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient


def load_jsonl(fname):
    with open(fname, 'r') as f:
        for line in f:
            yield json.loads(line)


def generate_property_map(fname):
    property_map = defaultdict(lambda: 0)
    token_id = 1
    for x in load_jsonl(fname):
        for property in x['properties']:
            if property not in property_map:
                property_map[property] = token_id
                token_id += 1
    return property_map


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def split_label(label):
    label_split = re.split('[^a-zA-Z0-9\n\.]', label)
    return sum([camel_case_split(label2) for label2 in label_split],[])


class TriggerTemplatizer:
    """
    An object to facilitate creating transformers-friendly triggers inputs from a template.

    Parameters
    ==========
    template : str
        The template string, comprised of the following tokens:
            [T] to mark a trigger placeholder.
            [P] to mark a prediction placeholder.
            {fields} arbitrary fields instantiated from the dataset instances.
        For example a NLI template might look like:
            "[T] [T] [T] {premise} [P] {hypothesis}"
    tokenizer : PretrainedTokenizer
        A HuggingFace tokenizer. Must have special trigger and predict tokens.
    add_special_tokens : bool
        Whether or not to add special tokens when encoding. Default: False.
    """
    def __init__(self,
                 tokenizer,
                 prompt_len:int,
                 property_map:dict):
        if not hasattr(tokenizer, 'trigger_token'):
            raise ValueError(
                'Tokenizer missing special trigger and predict tokens in vocab.'
                'Use `utils.add_special_tokens` to add them.'
            )
        self._tokenizer = tokenizer
        self._prompt_len = prompt_len
        self._property_map = property_map


    def __call__(self, format_kwargs:dict):
        # Format the template string
        format_kwargs = format_kwargs.copy()
        label = random.choice(format_kwargs.pop('target_sents'))
        input_seq = format_kwargs.pop('input_seq')
        properties = format_kwargs.pop('properties')
        
        if input_seq.count('[P]') != len(properties) or label is None:
            raise Exception('Bad data')
        
        text = ' '.join([word.replace('_', ' ') if word != '[P]' else ''.join(['[T]'] * self._prompt_len) for word in input_seq])

        # Have the tokenizer encode the text and process the output to:
        # - Create a trigger and predict mask
        # - Replace the predict token with a mask token
        model_inputs = self._tokenizer(text, return_tensors='pt')
        input_ids = model_inputs['input_ids']
        trigger_mask = input_ids.eq(self._tokenizer.trigger_token_id)

        model_inputs['trigger_mask'] = trigger_mask
        model_inputs['properties'] = torch.LongTensor([[self._property_map[p] for p in properties]])

        # Encode the label(s)
        model_inputs['labels'] = self.encode_label(label)

        return model_inputs
    
    
    def encode_label(self, label):
        """
        Helper function for encoding labels. Deals with the subtleties of handling multiple tokens.
        """
        if isinstance(label, str):
            encoded = self._tokenizer(label, return_tensors='pt')['input_ids']
        elif isinstance(label, list):
            encoded = torch.tensor(self._tokenizer.convert_tokens_to_ids(label)).unsqueeze(0)
        elif isinstance(label, int):
            encoded = torch.tensor([[label]])
        return encoded


class Collator:
    """
    Collates transformer outputs.
    """
    def __init__(self, pad_token_id=0):
        self._pad_token_id = pad_token_id

    def __call__(self, features):
        # Separate the list of inputs and labels
        # Assume that all inputs have the same keys as the first
        proto_input = features[0]
        keys = list(proto_input.keys())
        padded_inputs = {}
        for key in keys:
            if key == 'input_ids':
                padding_value = self._pad_token_id
            else:
                padding_value = 0
            # NOTE: We need to squeeze to get rid of fake batch dim.
            sequence = [x[key] for x in features]
            padded = Collator.pad_squeeze_sequence(sequence, batch_first=True, padding_value=padding_value)
            padded_inputs[key] = padded
        return padded_inputs
    
    @staticmethod
    def pad_squeeze_sequence(sequence, *args, **kwargs):
        """Squeezes fake batch dimension added by tokenizer before padding sequence."""
        return pad_sequence([x.squeeze(0) for x in sequence], *args, **kwargs)


def load_trigger_dataset(fname, templatizer):
    return [templatizer(x) for x in load_jsonl(fname)]


class PredictWrapper:
    """
    PyTorch transformers model wrapper. Handles necc. preprocessing of inputs for triggers
    experiments.
    """
    def __init__(self, model):
        self._model = model

    def __call__(self, model_inputs:dict, property_prompt:torch.LongTensor):
        # Copy dict so pop operations don't have unwanted side-effects
        model_inputs = model_inputs.copy()
        trigger_mask = model_inputs.pop('trigger_mask')
        properties = model_inputs.pop('properties')
        # predict_mask = model_inputs.pop('predict_mask')
        model_inputs = PredictWrapper.replace_trigger_tokens(model_inputs, property_prompt[properties].squeeze(1), trigger_mask)
        if 'labels' in model_inputs:
            outputs = self._model(**model_inputs)
            return outputs.loss, outputs.logits
        else:
            outputs = self._model(**model_inputs)
            return outputs.logits
        # predict_logits = logits.masked_select(predict_mask.unsqueeze(-1)).view(logits.size(0), -1)
        
    @staticmethod
    def replace_trigger_tokens(model_inputs:dict, trigger_ids:torch.LongTensor, trigger_mask:torch.BoolTensor):
        """Replaces the trigger tokens in input_ids."""
        out = model_inputs.copy()
        input_ids = model_inputs['input_ids']
        # try:
        filled = input_ids.masked_scatter(trigger_mask, trigger_ids)
        # except RuntimeError:
        #     filled = input_ids
        out['input_ids'] = filled
        return out
    
    def generate(self, model_inputs:dict, property_prompt:torch.LongTensor):
        # Copy dict so pop operations don't have unwanted side-effects
        model_inputs = model_inputs.copy()
        trigger_mask = model_inputs.pop('trigger_mask')
        properties = model_inputs.pop('properties')
        if 'labels' in model_inputs:
            model_inputs.pop('labels')
        model_inputs = PredictWrapper.replace_trigger_tokens(model_inputs, property_prompt[properties].squeeze(1), trigger_mask)
        tokens = self._model.generate(**model_inputs)
        return tokens, model_inputs['input_ids']
        # predict_logits = logits.masked_select(predict_mask.unsqueeze(-1)).view(logits.size(0), -1)


def hotflip_attack(grad,
                   embedding_matrix,
                   num_candidates=1):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            grad,
            embedding_matrix.T
        )
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids, (gradient_dot_embedding_matrix != 0).any(dim=1)


def eval(dev_loader, predictor, property_prompt, logger):
    logger.info('Evaluating')
    for model_inputs in dev_loader:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        with torch.no_grad():
            dev_metric, predict_logits = predictor(model_inputs, property_prompt)
    logger.info(f'Dev metric: {dev_metric.item()}')
    return dev_metric

def dump_prompt(property_map, property_prompt, logger):
    logger.info('Dumping prompts')
    out_str = [] 
    property_map_sorted = sorted(property_map.items(),key=lambda x:x[1])
    for (label, index), prompt in zip(property_map_sorted, property_prompt[1:]):
        out_str.append(label+'\t'+tokenizer.decode(prompt))
    with open(config['log_file']+'.tok.txt','w') as f:
        f.write('\n'.join(out_str))
    return

def generate(dev_loader, predictor, property_prompt, logger):
    logger.info('Generating')
    with open(config['log_file']+'.gen.txt','w') as f:
        for model_inputs in dev_loader:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            with torch.no_grad():
                tokens, modified_input_ids = predictor.generate(model_inputs, property_prompt)
                out_str = []
                for i in range(model_inputs['labels'].shape[0]):
                    out_str.append('\n'.join([
                        tokenizer.decode(model_inputs['labels'][i], skip_special_tokens=False),
                        tokenizer.decode(tokens[i], skip_special_tokens=False)
                    ]))
                f.write('\n\n'.join(out_str)+'\n\n')
    return


def train(config, model, logger, train_loader, dev_loader, device, predictor, embedding_gradient, property_prompt, embeddings):
    # Evaluate the initial prompt
    dev_metric = eval(dev_loader, predictor, property_prompt, logger)
    logger.info(f'Init dev metric: {dev_metric}')
    
    best_dev_metric = dev_metric
    best_property_prompt = property_prompt.clone()
    for i in tqdm(range(config['iter'])):
        
        update_num = 0

        logger.info(f'Iteration: {i}')

        logger.info('Accumulating Gradient')

        # Shuttle inputs to GPU
        for model_inputs in train_loader:
            model.zero_grad()
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            loss, predict_logits = predictor(model_inputs, property_prompt)
            loss.backward()
            # dict_keys(['input_ids', 'attention_mask', 'trigger_mask', 'properties', 'labels'])
            grad = embedding_gradient.get()
            bsz, _, emb_dim = grad.size()
            selection_mask = model_inputs['trigger_mask'].unsqueeze(-1)
            grad = torch.masked_select(grad, selection_mask)
            grad = grad.view(-1, config['prompt_len'], emb_dim)
            properties = model_inputs['properties']
            properties = properties[properties != 0]
            properties_matrix = F.one_hot(properties, num_classes=property_prompt.size(0)) * 1.0
            grad_ = torch.swapaxes(torch.swapaxes(grad, 0, 1), 1, 2)
            grad_sum_ = torch.matmul(grad_, properties_matrix)

            # Generate candidates
            token_to_flip = torch.randint(0, config['prompt_len'], (property_prompt.size(0),)).to(device)
            token_to_flip_one_hot = (F.one_hot(token_to_flip, num_classes=config['prompt_len']) * 1.0)
            selected_grad = torch.matmul(torch.swapaxes(grad_sum_, 0, 2), token_to_flip_one_hot.unsqueeze(-1)).squeeze(-1)
            candidates, valid_properties = hotflip_attack(selected_grad, embeddings.weight, num_candidates=config['num_cand'])

            # Evaluate candidates
            
            # Get current score
            current_score = loss.detach().sum()
            # Initialize candidate scores
            candidate_scores = torch.zeros(config['num_cand'], device=device)
            
            labels = model_inputs['labels']
            denom = labels.size(0)

            # NOTE: Instead of iterating over tokens to flip we randomly change just one each
            # time so the gradients don't get stale.
            for i in range(candidates.size(1)):

                temp_property_prompt = property_prompt.clone()
                temp_property_prompt[valid_properties, token_to_flip[valid_properties]] = candidates[valid_properties, i]
                with torch.no_grad():
                    eval_metric, predict_logits = predictor(model_inputs, temp_property_prompt)

                candidate_scores[i] = eval_metric.sum()
            
            # TODO: Something cleaner. LAMA templates can't have mask tokens, so if
            # there are still mask tokens in the trigger then set the current score
            # to -inf.
            
            if property_prompt[valid_properties].eq(tokenizer.mask_token_id).any():
                current_score = float('inf')

            if (candidate_scores < current_score).any():
                best_candidate_score = candidate_scores.min()
                logger.info(f'Better trigger detected. Train metric: {best_candidate_score / (denom + 1e-13): 0.4f}')
                best_candidate_idx = candidate_scores.argmin()
                property_prompt[valid_properties, token_to_flip[valid_properties]] = candidates[valid_properties, best_candidate_idx]
                update_num += 1
            else:
                continue
        
        if not update_num:
            logger.info('No update, skip eval')
            continue
        
        dev_metric = eval(dev_loader, predictor, property_prompt, logger)

        # TODO: Something cleaner. LAMA templates can't have mask tokens, so if
        # there are still mask tokens in the trigger then set the current score
        # to -inf.
        
        
        if property_prompt.eq(tokenizer.mask_token_id).any():
            # best_dev_metric = float('inf')
            logger.info('MASK still exist')

        if dev_metric < best_dev_metric:
            logger.info(f'Best performance so far. Dev metric: {dev_metric: 0.4f}')
            best_property_prompt = property_prompt.clone()
            best_dev_metric = dev_metric


    logger.info(f'Best dev metric: {best_dev_metric}')
    torch.save(best_property_prompt, config['property_prompt_cp'])


if __name__ == '__main__':
    
    # Get the config
    config = get_config()
    
    
    # Basic setup
    os.environ["CUDA_VISIBLE_DEVICES"]=config['device']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config['seed'])
    
    
    # Setup logger
    logger = set_logger(config)
    
    
    # Setup pretrained model
    logger.info('Loading model, tokenizer, etc.')
    pretrained_model_config, model, tokenizer = load_pretrained(config['model_name'])
    model.to(device)
    
    
    # Get embeddings and embedding gradients from pretrained model
    embeddings = get_embeddings(model, pretrained_model_config)
    embedding_gradient = GradientStorage(embeddings)
    
    
    # Load property map
    if os.path.exists(config['property_map']):
        property_map = json.load(open(config['property_map']))
        logger.info(f"Property map: {property_map}")
    else:
        property_map = generate_property_map(config['train'])
        with open(config['property_map'], 'w') as f_out:
            json.dump(property_map, f_out)
        logger.info('New property map generated')
        
        
    # Load property prompt paramenter
    if os.path.exists(config['property_prompt_cp']):
        property_prompt = torch.load(config['property_prompt_cp'])
    else:
        property_prompt = torch.ones((len(property_map)+1, config['prompt_len']), dtype=torch.int64) * tokenizer.mask_token_id
        if config['propertyInit']:
            logger.info('Using Property Init')
            property_map_sorted = sorted(property_map.items(),key=lambda x:x[1])
            property_splits = sum(list(map(lambda l: (split_label(l[0])+[tokenizer.mask_token]*config['prompt_len'])[:config['prompt_len']],property_map_sorted)),[])
            property_prompt[1:,:] = torch.tensor(tokenizer.convert_tokens_to_ids(property_splits), dtype=torch.int64).reshape((len(property_map), config['prompt_len']))
        else:
            logger.info('Using Mask Init')
    property_prompt = property_prompt.to(device)
    
    
    # Setup trigger templatizer
    templatizer = TriggerTemplatizer(tokenizer, prompt_len=config['prompt_len'], property_map=property_map)
    
    
    # Loading datasets
    logger.info('Loading datasets')
    collator = Collator(pad_token_id=tokenizer.pad_token_id)
    
    train_dataset = load_trigger_dataset(config['train'], templatizer)
    train_loader = DataLoader(train_dataset, batch_size=config['bsz'], collate_fn=collator)

    dev_dataset = load_trigger_dataset(config['dev'], templatizer)
    dev_loader = DataLoader(dev_dataset, batch_size=config['eval_size'], shuffle=False, collate_fn=collator)


    # Setup predict wrapper
    predictor = PredictWrapper(model)
    
    # Start training
    train(config, 
          model=model, 
          logger=logger, 
          train_loader=train_loader, 
          dev_loader=dev_loader, 
          device=device, 
          predictor=predictor, 
          embedding_gradient=embedding_gradient, 
          property_prompt=property_prompt, 
          embeddings=embeddings)
    
    property_prompt = torch.load(config['property_prompt_cp'])
    dump_prompt(property_map, property_prompt, logger)
    generate(dev_loader, predictor, property_prompt, logger)
    
    logger.info('Done')
    