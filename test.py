from transformers import BertTokenizer
from model import BertNER
import config
import torch


def sent2idx(sentence=None):
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)
    return tokenizer.encode(sentence)


def idx2sent(sentence=None):
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)
    return tokenizer.decode(sentence)


def idx2label(sentence=None):
    id2label = config.id2label
    res = []
    for i in sentence:
        res.append(id2label[i])
    return res


def label2idx(sentence=None):
    label2id = config.label2id
    res = []
    for i in sentence:
        res.append(label2id[i])
    return res


def labelRange(tag):
    labels_list = {'address': [], 'book': [], 'company': [], 'game': [], 'government': [],
                   'movie': [], 'name': [], 'organization': [], 'position': [], 'scene': []}
    for i in range(len(tag)):
        if tag[i] == 'O':
            continue
        else:
            pos = tag[i].split('-')[0]
            label = tag[i].split('-')[1]
            if pos == 'S':
                labels_list[label].append((i, i))
            elif pos == 'B':
                start_index = i
                i += 1
                pos = tag[i].split('-')[0]
                while pos == 'I':
                    i += 1
                    if i < len(tag):
                        pos = tag[i].split('-')[0]
                    else:
                        pos = 'end'
                i -= 1
                labels_list[label].append((start_index, i))
    return labels_list


def entityExtraction(sentence, tag):
    entities = {}
    entity_range = labelRange(tag)
    for key in entity_range:
        if len(entity_range[key]) != 0:
            arr = entity_range[key]
            for start, end in arr:
                entity = sentence[start:end + 1]
                entities[entity] = key
    return entities


def test_single_sentence(sentence):
    if config.model_dir is not None:
        model = BertNER.from_pretrained(config.model_dir)
        model.to(config.device)
    s_data = sent2idx(sentence)[:-1]
    s_start = [1 for _ in range(len(s_data))]
    s_start[0] = 0
    s_data = torch.tensor(s_data, dtype=torch.long)
    s_start = torch.tensor(s_start, dtype=torch.long)
    s_data = torch.unsqueeze(s_data, 0)
    s_start = torch.unsqueeze(s_start, 0)
    d_masks = s_data.gt(0)
    output = model((s_data, s_start),
                   token_type_ids=None, attention_mask=d_masks)[0]
    output = model.crf.decode(output)
    tag = idx2label(output[0])
    entities = entityExtraction(text, tag)
    print(entities)


if __name__ == '__main__':
    text = "周杰伦去上海开演唱会了"
    test_single_sentence(text)
