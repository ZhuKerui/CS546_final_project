import time
import argparse
import json
import logging
from pathlib import Path
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModelWithLMHead, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.INFO,
    filemode='w',
    filename='logs.txt')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=Path, required=True, help='Train data path')
parser.add_argument('--dev', type=Path, required=True, help='Dev data path')
parser.add_argument('--property-map', type=str, default=None, help='JSON object defining property map')
parser.add_argument('--property-prompt', type=str, default=None, help='Tokens for each property')

# LAMA-specific
parser.add_argument('--tokenize-labels', action='store_true',
                    help='If specified labels are split into word pieces.'
                            'Needed for LAMA probe experiments.')
parser.add_argument('--filter', action='store_true',
                    help='If specified, filter out special tokens and gold objects.'
                            'Furthermore, tokens starting with capital '
                            'letters will not appear in triggers. Lazy '
                            'approach for removing proper nouns.')
parser.add_argument('--print-lama', action='store_true',
                    help='Prints best trigger in LAMA format.')

parser.add_argument('--initial-trigger', nargs='+', type=str, default=None, help='Manual prompt')

parser.add_argument('--bsz', type=int, default=32, help='Batch size')
parser.add_argument('--eval-size', type=int, default=256, help='Eval size')
parser.add_argument('--iters', type=int, default=100,
                    help='Number of iterations to run trigger search algorithm')
parser.add_argument('--accumulation-steps', type=int, default=10)
parser.add_argument('--model-name', type=str, default='bert-base-cased',
                    help='Model name passed to HuggingFace AutoX classes.')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--limit', type=int, default=None)
parser.add_argument('--use-ctx', action='store_true',
                    help='Use context sentences for relation extraction only')
parser.add_argument('--perturbed', action='store_true',
                    help='Perturbed sentence evaluation of relation extraction: replace each object in dataset with a random other object')
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--num-cand', type=int, default=10)
parser.add_argument('--sentence-size', type=int, default=50)

parser.add_argument('--debug', action='store_true')
args = parser.parse_args(['--train', 'webnlg_dataset/webnlg_train.jsonl', '--dev', 'webnlg_dataset/webnlg_dev.jsonl', '--num-cand', '10', '--accumulation-steps', '1', '--model-name', 'facebook/bart-base', '--bsz', '56', '--eval-size', '56', '--iters', '1000', '--tokenize-labels', '--filter', '--print-lama'])

if args.debug:
    level = logging.DEBUG
else:
    level = logging.INFO
logging.basicConfig(level=level)



def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
set_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def add_task_specific_tokens(tokenizer):
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[T]', '[Y]']
    })
    tokenizer.trigger_token = '[T]'
    tokenizer.trigger_token_id = tokenizer.convert_tokens_to_ids('[T]')
    # NOTE: BERT and RoBERTa tokenizers work properly if [X] is not a special token...
    # tokenizer.lama_x = '[X]'
    # tokenizer.lama_x_id = tokenizer.convert_tokens_to_ids('[X]')
    tokenizer.lama_y = '[Y]'
    tokenizer.lama_x_id = tokenizer.convert_tokens_to_ids('[Y]')

def load_pretrained(model_name):
    """
    Loads pretrained HuggingFace config/model/tokenizer, as well as performs required
    initialization steps to facilitate working with triggers.
    """
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    add_task_specific_tokens(tokenizer)
    return config, model, tokenizer

logger.info('Loading model, tokenizer, etc.')
config, model, tokenizer = load_pretrained(args.model_name)
_ = model.to(device)



def get_embeddings(model, config):
    """Returns the wordpiece embedding module."""
    base_model = model.get_encoder()
    embeddings = base_model.embed_tokens
    return embeddings

embeddings = get_embeddings(model, config)



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
    
embedding_gradient = GradientStorage(embeddings)



def load_jsonl(fname):
    with open(fname, 'r') as f:
        for line in f:
            yield json.loads(line)

def default_prompt():
    return 0

def generate_property_map(fname):
    property_map = defaultdict(default_prompt)
    token_id = 1
    for x in load_jsonl(fname):
        for property in x['properties']:
            if property not in property_map:
                property_map[property] = token_id
                token_id += 1
    return property_map

if args.property_map is not None and args.property_prompt is not None:
    property_map = json.loads(args.property_map)
    property_prompt = torch.load(args.property_prompt).to(device)
    logger.info(f"Property map: {property_map}")
else:
    property_map = generate_property_map(args.train)
    property_prompt = torch.ones((len(property_map)+1, 3), dtype=torch.int64) * tokenizer.mask_token_id
    property_prompt = property_prompt.to(device)
    with open('property_map.json', 'w') as f_out:
        json.dump(property_map, f_out)
    logger.info('New property map generated')
best_property_prompt = property_prompt.clone()



def encode_label(tokenizer, label):
    """
    Helper function for encoding labels. Deals with the subtleties of handling multiple tokens.
    """
    if isinstance(label, str):
        encoded = tokenizer(label, return_tensors='pt')['input_ids']
    elif isinstance(label, list):
        encoded = torch.tensor(tokenizer.convert_tokens_to_ids(label)).unsqueeze(0)
    elif isinstance(label, int):
        encoded = torch.tensor([[label]])
    return encoded

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
        label_id = encode_label(
            tokenizer=self._tokenizer,
            label=label
        )
        
        model_inputs['labels'] = label_id

        return model_inputs

templatizer = TriggerTemplatizer(
    tokenizer,
    prompt_len=3,
    property_map=property_map
)



def pad_squeeze_sequence(sequence, *args, **kwargs):
    """Squeezes fake batch dimension added by tokenizer before padding sequence."""
    return pad_sequence([x.squeeze(0) for x in sequence], *args, **kwargs)

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
            padded = pad_squeeze_sequence(sequence, batch_first=True, padding_value=padding_value)
            padded_inputs[key] = padded
        return padded_inputs

logger.info('Loading datasets')
collator = Collator(pad_token_id=tokenizer.pad_token_id)



def load_trigger_dataset(fname, templatizer):
    return [templatizer(x) for x in load_jsonl(fname)]

# if args.perturbed:
#     train_dataset = load_augmented_trigger_dataset(args.train, templatizer, limit=args.limit)
# else:
train_dataset = load_trigger_dataset(args.train, templatizer)
train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)

# if args.perturbed:
#     dev_dataset = utils.load_augmented_trigger_dataset(args.dev, templatizer)
# else:
dev_dataset = load_trigger_dataset(args.dev, templatizer)
dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)



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
        model_inputs = replace_trigger_tokens(model_inputs, property_prompt[properties].squeeze(1), trigger_mask)
        if 'labels' in model_inputs:
            outputs = self._model(**model_inputs)
            return outputs.loss, outputs.logits
        else:
            outputs = self._model(**model_inputs)
            return outputs.logits
        # predict_logits = logits.masked_select(predict_mask.unsqueeze(-1)).view(logits.size(0), -1)
        
    
predictor = PredictWrapper(model)



logger.info('Evaluating')
for model_inputs in dev_loader:
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    with torch.no_grad():
        dev_metric, predict_logits = predictor(model_inputs, property_prompt)
logger.info(f'Dev metric: {dev_metric.item()}')

best_dev_metric = -float('inf')
# Measure elapsed time of trigger search
start = time.time()



def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   num_candidates=1):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            averaged_grad,
            embedding_matrix.T
        )
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids, (gradient_dot_embedding_matrix != 0).any(dim=1)

for i in tqdm(range(args.iters)):

    logger.info(f'Iteration: {i}')

    logger.info('Accumulating Gradient')
    model.zero_grad()

    train_iter = iter(train_loader)
    averaged_grad = None

    # Accumulate
    for step in range(args.accumulation_steps):

        # Shuttle inputs to GPU
        try:
            model_inputs = next(train_iter)
        except:
            logger.warning(
                'Insufficient data for number of accumulation steps. '
                'Effective batch size will be smaller than specified.'
            )
            break
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        loss, predict_logits = predictor(model_inputs, property_prompt)
        loss.backward()

        grad = embedding_gradient.get()
        bsz, _, emb_dim = grad.size()
        selection_mask = model_inputs['trigger_mask'].unsqueeze(-1)
        grad = torch.masked_select(grad, selection_mask)
        grad = grad.view(-1, 3, emb_dim)
        
        properties = model_inputs['properties']
        properties = properties[properties != 0]
        properties_matrix = F.one_hot(properties, num_classes=property_prompt.size(0)) * 1.0
        grad_ = torch.swapaxes(torch.swapaxes(grad, 0, 1), 1, 2)
        grad_sum_ = torch.matmul(grad_, properties_matrix)

        if averaged_grad is None:
            averaged_grad = grad_sum_ / args.accumulation_steps
        else:
            averaged_grad += grad_sum_ / args.accumulation_steps

    logger.info('Evaluating Candidates')
    train_iter = iter(train_loader)

    token_to_flip = torch.randint(0, 3, (property_prompt.size(0),)).to(device)
    token_to_flip_one_hot = (F.one_hot(token_to_flip, num_classes=3) * 1.0)
    selected_grad = torch.matmul(torch.swapaxes(grad_sum_, 0, 2), token_to_flip_one_hot.unsqueeze(-1)).squeeze(-1)
    candidates, valid_properties = hotflip_attack(selected_grad, embeddings.weight, num_candidates=args.num_cand)

    current_score = 0
    candidate_scores = torch.zeros(args.num_cand, device=device)
    # candidate_scores = torch.zeros(args.num_cand)
    denom = 0
    for step in range(args.accumulation_steps):

        try:
            model_inputs = next(train_iter)
        except:
            logger.warning(
                'Insufficient data for number of accumulation steps. '
                'Effective batch size will be smaller than specified.'
            )
            break
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = model_inputs['labels']
        with torch.no_grad():
            eval_metric, predict_logits = predictor(model_inputs, property_prompt)

        # Update current score
        current_score += eval_metric.sum()
        denom += labels.size(0)

        # NOTE: Instead of iterating over tokens to flip we randomly change just one each
        # time so the gradients don't get stale.
        for i in range(candidates.size(1)):

            temp_property_prompt = property_prompt.clone()
            temp_property_prompt[valid_properties, token_to_flip[valid_properties]] = candidates[valid_properties, i]
            with torch.no_grad():
                eval_metric, predict_logits = predictor(model_inputs, temp_property_prompt)

            candidate_scores[i] += eval_metric.sum()

    # TODO: Something cleaner. LAMA templates can't have mask tokens, so if
    # there are still mask tokens in the trigger then set the current score
    # to -inf.
    
    # if args.print_lama:
    #     if trigger_ids.eq(tokenizer.mask_token_id).any():
    #         current_score = float('-inf')

    if (candidate_scores > current_score).any():
        logger.info('Better trigger detected.')
        best_candidate_score = candidate_scores.max()
        best_candidate_idx = candidate_scores.argmax()
        property_prompt[valid_properties, token_to_flip[valid_properties]] = candidates[valid_properties, best_candidate_idx]
        logger.info(f'Train metric: {best_candidate_score / (denom + 1e-13): 0.4f}')
    else:
        logger.info('No improvement detected. Skipping evaluation.')
        continue

    logger.info('Evaluating')
    # numerator = 0
    # denominator = 0
    for model_inputs in dev_loader:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = model_inputs['labels']
        with torch.no_grad():
            dev_metric, predict_logits = predictor(model_inputs, property_prompt)
        # numerator += loss.sum().item()
        # denominator += labels.size(0)
    # dev_metric = numerator / (denominator + 1e-13)

    # logger.info(f'Trigger tokens: {tokenizer.convert_ids_to_tokens(trigger_ids.squeeze(0))}')
    logger.info(f'Dev metric: {dev_metric.item()}')

    # TODO: Something cleaner. LAMA templates can't have mask tokens, so if
    # there are still mask tokens in the trigger then set the current score
    # to -inf.
    
    # if args.print_lama:
    #     if best_trigger_ids.eq(tokenizer.mask_token_id).any():
    #         best_dev_metric = float('-inf')

    if dev_metric > best_dev_metric:
        logger.info('Best performance so far')
        best_property_prompt = property_prompt.clone()
        best_dev_metric = dev_metric



logger.info(f'Best dev metric: {best_dev_metric}')
torch.save(best_property_prompt, 'best_property_prompt.pt')
