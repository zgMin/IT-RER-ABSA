import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor, nn
from typing import Optional, Tuple, Union
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer,
)
from transformers.models.dpr.modeling_dpr import DPREncoder
from transformers.modeling_outputs import BaseModelOutputWithPooling
import json
import re
import random
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs
from allennlp.nn.util import sequence_cross_entropy_with_logits
import heapq
import pandas as pd
def reconstruct_strings(df, col, num = 2):
    """
    Reconstruct strings to dictionaries when loading csv/xlsx files.
    """
    reconstructed_col = []
    for text in df[col]:
        if text != '[]' and isinstance(text, str):
            text = text.replace('[', '').replace(']', '').replace('{', '').replace('}', '').split(", '")
            req_list = []
            reconstructed_dict = {}
            for idx, pair in enumerate(text):
                if num == 2:
                    if idx % 2 == 0:
                        reconstructed_dict = {}
                        reconstructed_dict[pair.split(':')[0].replace("'", '')] = pair.split(':')[1].replace("'", '')
                    else:
                        reconstructed_dict[pair.split(':')[0].replace("'", '')] = pair.split(':')[1].replace("'", '')

                        req_list.append(reconstructed_dict)
                elif num > 2:
                    if idx % num == num-1:
                        reconstructed_dict[pair.split(':')[0].replace("'", '')] = pair.split(':')[1].replace("'",
                                                                                                             '')
                        req_list.append(reconstructed_dict)
                        reconstructed_dict = {}
                    else:
                        reconstructed_dict[pair.split(':')[0].replace("'", '')] = pair.split(':')[1].replace("'",
                                                                                                             '')
        else:
            req_list = text
        reconstructed_col.append(req_list)
    df[col] = reconstructed_col
    return df

def extract_rowwise_aspect_polarity(df, on, key, min_val=None):
    """
    Create duplicate records based on number of aspect term labels in the dataset.
    Extract each aspect term for each row for reviews with muliple aspect term entries. 
    Do same for polarities and create new column for the same.
    """
    try:
        df.iloc[0][on][0][key]
    except:
        df = reconstruct_strings(df, on)

    df['len'] = df[on].apply(lambda x: len(x))
    if min_val is not None:
        df.loc[df['len'] == 0, 'len'] = min_val
    df = df.loc[df.index.repeat(df['len'])]
    df['record_idx'] = df.groupby(df.index).cumcount()
    df['aspect'] = df[[on, 'record_idx']].apply(
        lambda x: (x[0][x[1]][key], x[0][x[1]]['polarity']) if len(x[0]) != 0 else ('', ''), axis=1)
    df['polarity'] = df['aspect'].apply(lambda x: x[-1])
    df['aspect'] = df['aspect'].apply(lambda x: x[0])
    df = df.drop(['len', 'record_idx'], axis=1).reset_index(drop=True)
    return df
def create_data_in_atsc_format(df, on='aspectTerms', key='term', text_col = 'raw_text', aspect_col='aspect'):
    """
    Prepare the data in the input format required.
    """
    if df is None:
        return
    tmp = extract_rowwise_aspect_polarity(df, on=on, key=key, min_val=1)
    res = pd.DataFrame(columns=['text', 'labels'], dtype=str)
    # res['labels'] = df[on].apply(lambda x: ', '.join([i[key] for i in x]))
    res['labels'] = tmp['polarity']
    res['text'] = tmp[[text_col, aspect_col]].apply(
        lambda x:  x[0] + " The aspect is" + x[1]+'.', axis=1)

    return res
def score_function(tokenizer,model,d,e,x,y):
    # e = e[:10]
    scores = []
    for ei in e:
        input_ = tokenizer(d + f'\nExample 1-\n' + ei + "\nNow complete the following example-" + "\ninput: " +x,return_tensors="pt").to(model.device)
        target = tokenizer(y,return_tensors="pt").input_ids.to(model.device)
        input_ids = model.prepare_inputs_for_generation(input_.input_ids)
        model.eval()
        with torch.no_grad():
            output = model(input_ids = input_.input_ids,**input_ids)
        model.train()
        # 填充对齐logits和target
        pad_length = output.logits.shape[1] - target.shape[1]
        pad_seq = torch.tensor([[tokenizer.pad_token_id] * pad_length]).to(model.device)
        target = torch.cat((target,pad_seq),dim=1)
        score = sequence_cross_entropy_with_logits(logits=output.logits,
                                                                targets=target,
                                                                weights=input_.attention_mask,
                                                                average=None)
        scores.append(score)
    return scores

class DPREncoder(DPREncoder):
    def __init__(self, encoder):
        nn.Module.__init__(self)
        # super(DPREncoder,self).__init__()
        print(encoder)
        self.bert_model = encoder
        if self.bert_model.config.hidden_size <= 0:
            raise ValueError("Encoder hidden_size can't be zero")
        self.projection_dim = 0
        if self.projection_dim > 0:
            self.encode_proj = nn.Linear(self.bert_model.config.hidden_size, config.projection_dim)
        # Initialize weights and apply final processing
        # self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> Union[BaseModelOutputWithPooling, Tuple[Tensor, ...]]:
        outputs = self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                # head_mask=head_mask,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        sequence_output = outputs[0]
        pooled_output = torch.mean(sequence_output,1)

        if self.projection_dim > 0:
            pooled_output = self.encode_proj(pooled_output)

        if not return_dict:
            return (sequence_output, pooled_output) + outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @property
    def embeddings_size(self) -> int:
        if self.projection_dim > 0:
            return self.encode_proj.out_features
        return self.bert_model.config.hidden_size
class RetrievalModel(RetrievalModel):
    def __init__(
            self,
            model_type=None,
            model_name=None,
            context_encoder_name=None,
            query_encoder_name=None,
            context_encoder_tokenizer=None,
            query_encoder_tokenizer=None,
            prediction_passages=None,
            args=None,
            use_cuda=True,
            cuda_device=-1,
            **kwargs,
    ):
        # super(RetrievalModel, self).__init__()
        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, RetrievalArgs):
            self.args = args

        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set `use_cuda=False`."
                )
        else:
            self.device = "cpu"

        self.results = {}

        if not use_cuda:
            self.args.fp16 = False

        if context_encoder_name:
            self.context_encoder = DPREncoder(context_encoder_name)
            self.context_tokenizer = context_encoder_tokenizer

        if query_encoder_name:
            self.query_encoder = DPREncoder(query_encoder_name)
            self.query_tokenizer = query_encoder_tokenizer


        # TODO: Add support for adding special tokens to the tokenizers

        self.args.model_type = model_type
        self.args.model_name = model_name

        if prediction_passages is not None:
            self.prediction_passages = self.get_updated_prediction_passages(
                prediction_passages
            )
        else:
            self.prediction_passages = None
class T5Generator:
    def __init__(self, model_checkpoint,question_encoder_name,context_encoder_name):
        model_args = RetrievalArgs()
        model_args.num_train_epochs = 4
        model_args.no_save = True
        model_args.include_title = False
        model_args.hard_negatives = True
        model_type = "dpr"
        # context_encoder_name = "outputs/context_encoder"#"facebook/dpr-ctx_encoder-single-nq-base"
        # question_encoder_name = "outputs/query_encoder"#"facebook/dpr-question_encoder-single-nq-base"


        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        self.retriever = RetrievalModel(
            args=model_args,
            model_type=model_type,
            context_encoder_name=self.model.encoder,
            query_encoder_name=self.model.encoder,
            context_encoder_tokenizer =self.tokenizer,
            query_encoder_tokenizer= self.tokenizer
        )
        self.retriever.args.overwrite_output_dir = True
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        self.device = 'cuda' if torch.has_cuda else ('mps' if torch.has_mps else 'cpu')
    def init_retriever(self,tr_df,ev_df,task_def,cache_dir = None, task ='aoste'): #预热，与之后的区别主要在于采样范围，预热是对全部候选集的观测
        self.passages = []
        if cache_dir != None:
            with open(f'./outputs/{task}_passages.json', 'r') as file:
                self.passages = json.load(file)
            return
        if task != 'atsc':
            tr_input = list(tr_df['raw_text'])
        if task == 'aoste':
            tr_output = list(tr_df['aspectTerms'].apply(lambda x: ', '.join([f"{i['term']}:{i['opinion']}:{i['polarity']}" for i in x])))
        if task == 'aspe':
            tr_output = list(tr_df['aspectTerms'].apply(lambda x: ', '.join([f"{i['term']}:{i['polarity']}" for i in x])))
        if task == 'ate':
            tr_output = list(tr_df['aspectTerms'].apply(lambda x: ', '.join([f"{i['term']}" for i in x])))
        if task == 'atsc':
            tmp = create_data_in_atsc_format(tr_df)
            tr_input = list(tmp['text'])
            tr_output = list(tmp['labels'])
        tr_num_sample = len(tr_input)
        for i in range(tr_num_sample):
            passage = 'input: ' + tr_input[i] + '\noutput: ' + tr_output[i]
            self.passages.append(passage)
        # if task != 'atsc':
        #     ev_input = list(ev_df['raw_text'])
        # if task == 'aoste':
        #     ev_output = list(ev_df['aspectTerms'].apply(lambda x: ', '.join([f"{i['term']}:{i['opinion']}:{i['polarity']}" for i in x])))
        # if task == 'aspe':
        #     ev_output = list(ev_df['aspectTerms'].apply(lambda x: ', '.join([f"{i['term']}:{i['polarity']}" for i in x])))
        # if task == 'ate':
        #     ev_output = list(ev_df['aspectTerms'].apply(lambda x: ', '.join([f"{i['term']}" for i in x])))
        # if task == 'atsc':
        #     tmp = create_data_in_atsc_format(ev_df)
        #     ev_input = list(tmp['text'])
        #     ev_output = list(tmp['labels'])
        # ev_num_sample = len(ev_input)
        # for i in range(ev_num_sample):
        #     passage = 'input: ' + ev_input[i] + '\noutput: ' + ev_output[i]
        #     self.passages.append(passage)
        with open(f'./outputs/{task}_passages.json','w') as file:
            json.dump(self.passages,file)
        return
        # init_train_data = []
        # # TODO: 分数计算相当慢，考虑用BM25做模型初始化
        # for i in range(tr_num_sample):
        #     # 计算每个样本与passages库的分数，最高的作为pos（排除自身），最低的（貌似随便一个也可以）作为neg
        #     print(i)
        #     scores = score_function(self.tokenizer, self.model, task_def, self.passages, tr_input[i], tr_output[i])
        #
        #     # 获取下标， 输出为[4, 5, 2]
        #     high_s = heapq.nlargest(3, range(len(scores)), scores.__getitem__)
        #     tmp = {}
        #     tmp['query_text'] = tr_input[i]
        #     for j in high_s:
        #         if self.passages[j] != tr_input[i]:
        #             tmp['gold_passage'] = self.passages[j]
        #             break
        #     low_s = heapq.nsmallest(3, range(len(scores)), scores.__getitem__)
        #     for j in low_s:
        #         if self.passages[j] != tr_input[i]:
        #             tmp['hard_negatives'] = self.passages[j]
        #             break
        #     init_train_data.append(tmp)
        # init_eval_data = []
        # for i in range(ev_num_sample):
        #     # 计算每个样本与passages库的分数，最高的作为pos（排除自身），最低的（貌似随便一个也可以）作为neg
        #     scores = score_function(self.tokenizer, self.model, task_def, self.passages, ev_input[i], ev_output[i])
        #
        #     # 获取下标， 输出为[4, 5, 2]
        #     high_s = heapq.nlargest(3, range(len(scores)), scores.__getitem__)
        #     tmp = {}
        #     tmp['query_text'] = ev_input[i]
        #     for j in high_s:
        #         if self.passages[j] != ev_input[i]:
        #             tmp['gold_passage'] = self.passages[j]
        #             break
        #     low_s = heapq.nsmallest(3, range(len(scores)), scores.__getitem__)
        #     for j in low_s:
        #         if self.passages[j] != ev_input[i]:
        #             tmp['hard_negatives'] = self.passages[j]
        #             break
        #     init_eval_data.append(tmp)
        # init_eval_df = pd.DataFrame(
        #     init_eval_data
        # )
        # init_train_df = pd.DataFrame(
        #     init_train_data
        # )
        # # Train the model
        # self.retriever.train_model(init_train_df, eval_data=init_eval_df)
        # # Evaluate the model
        # self.retriever.eval_model(init_eval_df)
    def train_retriever(self,tr_df,ev_df,task_def,r=0.1,task='aspe'):
        tr_df = tr_df.sample(frac = r)
        ev_df = ev_df.sample(frac=r)
        if task != 'atsc':
            tr_input = list(tr_df['raw_text'])

        if task == 'aoste':
            tr_output = list(tr_df['aspectTerms'].apply(lambda x: ', '.join([f"{i['term']}:{i['opinion']}:{i['polarity']}" for i in x])))
        if task == 'aspe':
            tr_output = list(tr_df['aspectTerms'].apply(lambda x: ', '.join([f"{i['term']}:{i['polarity']}" for i in x])))
        if task == 'ate':
            tr_output = list(tr_df['aspectTerms'].apply(lambda x: ', '.join([f"{i['term']}" for i in x])))

        if task == 'atsc':
            tmp = create_data_in_atsc_format(tr_df)
            tr_input = list(tmp['text'])
            tr_output = list(tmp['labels'])

        tr_num_sample = len(tr_input)
        if task != 'atsc':
            ev_input = list(ev_df['raw_text'])
        if task == 'aoste':
            ev_output = list(ev_df['aspectTerms'].apply(lambda x: ', '.join([f"{i['term']}:{i['opinion']}:{i['polarity']}" for i in x])))
        if task == 'aspe':
            ev_output = list(ev_df['aspectTerms'].apply(lambda x: ', '.join([f"{i['term']}:{i['polarity']}" for i in x])))
        if task == 'ate':
            ev_output = list(ev_df['aspectTerms'].apply(lambda x: ', '.join([f"{i['term']}" for i in x])))
        if task == 'atsc':
            tmp = create_data_in_atsc_format(ev_df)
            ev_input = list(tmp['text'])
            ev_output = list(tmp['labels'])
        ev_num_sample = len(ev_input)

        to_predict = tr_input
        predicted_passages, *_ = self.retriever.predict(to_predict, prediction_passages=self.passages,retrieve_n_docs=10)
        train_data = []
        for i in tqdm(range(tr_num_sample)):
            # 计算每个样本与passages库的分数，最高的作为pos（排除自身），最低的（貌似随便一个也可以）作为neg
            scores = score_function(self.tokenizer, self.model, task_def, predicted_passages[i], tr_input[i], tr_output[i])

            # 获取下标， 输出为[4, 5, 2]
            high_s = heapq.nlargest(3, range(len(scores)), scores.__getitem__)
            tmp = {}
            tmp['query_text'] = tr_input[i]
            key1 = 'input:'
            key2 = '\noutput:'
            regex = r'%s(.*?)%s' % (key1, key2)
            high_s = random.sample(high_s,3)
            for j in high_s:
                result = re.findall(regex, predicted_passages[i][j], re.S)[0].strip()
                if result != tr_input[i]:
                    tmp['gold_passage'] = predicted_passages[i][j]
                    break
            low_s = heapq.nsmallest(3, range(len(scores)), scores.__getitem__)
            low_s = random.sample(low_s,3)
            for j in low_s:
                result = re.findall(regex, predicted_passages[i][j], re.S)[0].strip()
                if result != tr_input[i]:
                    tmp['hard_negative'] = predicted_passages[i][j]
                    break
            train_data.append(tmp)
        to_predict = ev_input
        predicted_passages, *_ = self.retriever.predict(to_predict, prediction_passages=self.passages,retrieve_n_docs=10)
        eval_data = []
        for i in tqdm(range(ev_num_sample)):
            # 计算每个样本与passages库的分数，最高的作为pos（排除自身），最低的（貌似随便一个也可以）作为neg
            scores = score_function(self.tokenizer, self.model, task_def, predicted_passages[i], ev_input[i], ev_output[i])

            # 获取下标， 输出为[4, 5, 2]
            high_s = heapq.nlargest(3, range(len(scores)), scores.__getitem__)
            high_s = random.sample(high_s,3)
            tmp = {}
            tmp['query_text'] = ev_input[i]
            key1 = 'input:'
            key2 = '\noutput:'
            regex = r'%s(.*?)%s' % (key1, key2)
            for j in high_s:
                result = re.findall(regex, predicted_passages[i][j], re.S)[0].strip()
                if result != ev_input[i]:
                    tmp['gold_passage'] = predicted_passages[i][j]
                    break
            low_s = heapq.nsmallest(3, range(len(scores)), scores.__getitem__)
            low_s = random.sample(low_s,3)
            for j in low_s:
                result = re.findall(regex, predicted_passages[i][j], re.S)[0].strip()
                if result != ev_input[i]:
                    tmp['hard_negative'] = predicted_passages[i][j]
                    break
            eval_data.append(tmp)
        eval_df = pd.DataFrame(
            eval_data
        )
        train_df = pd.DataFrame(
            train_data
        )
        # Train the model
        self.retriever.train_model(train_df, eval_data=eval_df)
        # Evaluate the model
        self.retriever.eval_model(eval_df)
    def set_num_epoch_retriever(self,num):
        self.retriever.num_train_epochs = num
    def select_examples(self, raw_texts, k=1):
        to_predict = raw_texts
        predicted_passages, *_ = self.retriever.predict(to_predict, prediction_passages=self.passages,
                                                        retrieve_n_docs=k+1)
        res = []
        key1 = 'input:'
        key2 = '\noutput:'
        regex = r'%s(.*?)%s' % (key1, key2)
        for i in range(len(predicted_passages)):
            done = True
            passages = predicted_passages[i]
            tmp = ""
            cnt=k
            for p in range(k,0,-1):
                result = re.findall(regex, passages[p], re.S)[0].strip()
                if result != raw_texts[i]:
                    tmp = f'\nExample {cnt}-\n' + passages[p]+tmp
                    cnt-=1
                else:
                    done = False
            if not done:
                tmp = f'\nExample 1-\n' + passages[0]+ tmp
            res.append(tmp)
        return res
    def tokenize_function_inputs(self, sample):
        """
        Udf to tokenize the input dataset.
        """
        model_inputs = self.tokenizer(sample['text'], max_length=512, truncation=True)
        labels = self.tokenizer(sample["labels"], max_length=64, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        
    def train(self, tokenized_datasets, **kwargs):
        """
        Train the generative model.
        """
        #Set training arguments
        args = Seq2SeqTrainingArguments(
            **kwargs
        )

        # Define trainer object
        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"] if tokenized_datasets.get("test") is not None else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        print("Trainer device:", trainer.args.device)

        # Finetune the model
        torch.cuda.empty_cache()
        print('\nModel training started ....')
        trainer.train()

        # Save best model
        trainer.save_model()
        return trainer

    def get_labels(self, tokenized_dataset, batch_size = 4, max_length = 128, sample_set = 'train'):
        """
        Get the predictions from the trained model.
        """
        def collate_fn(batch):
            input_ids = [torch.tensor(example['input_ids']) for example in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            return input_ids
        
        dataloader = DataLoader(tokenized_dataset[sample_set], batch_size=batch_size, collate_fn=collate_fn)
        predicted_output = []
        self.model.to(self.device)
        print('Model loaded to: ', self.device)

        for batch in tqdm(dataloader):
            batch = batch.to(self.device)
            output_ids = self.model.generate(batch, max_length = max_length)
            output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for output_text in output_texts:
                predicted_output.append(output_text)
        return predicted_output
    
    def get_metrics(self, y_true, y_pred, is_triplet_extraction=False,task='atsc'):
        if task =='atsc':
            return precision_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='macro'), \
                   f1_score(y_true, y_pred, average='macro'), accuracy_score(y_true, y_pred)
        total_pred = 0
        total_gt = 0
        tp = 0

        if not is_triplet_extraction:
            for gt, pred in zip(y_true, y_pred):
                gt_list = gt.split(',')
                pred_list = pred.split(',')
                pred_list = [i.strip() for i in pred_list]
                total_pred+=len(list(set(pred_list)))
                total_gt+=len(gt_list)
                for gt_val in gt_list:
                    gt_clean = gt_val.split(':')
                    gt_clean = [i.strip() for i in gt_clean]
                    gt_clean = ":".join(gt_clean)
                    for pred_val in pred_list:
                        pred_clean = pred_val.split(':')
                        pred_clean = [i.strip() for i in pred_clean]
                        pred_clean = ":".join(pred_clean)
                        # print(gt_clean)
                        # print(pred_clean)
                        # input()
                        # if pred_clean.lower() in gt_clean.lower() or gt_clean.lower() in pred_clean.lower():
                        if pred_clean.lower() == gt_clean.lower():
                            tp+=1
                            break
                        # if pred_val in gt_val or gt_val in pred_val:
                        #     tp+=1
                        #     break

        else:
            for gt, pred in zip(y_true, y_pred):
                gt_list = gt.split(',')
                pred_list = pred.split(',')
                total_pred+=len(pred_list)
                total_gt+=len(gt_list)
                for gt_val in gt_list:
                    try:
                        gt_asp = gt_val.split(':')[0].strip()
                    except:
                        continue
                    try:
                        gt_op = gt_val.split(':')[1].strip()
                    except:
                        continue

                    try:
                        gt_sent = gt_val.split(':')[2].strip()
                    except:
                        continue
                    for pred_val in pred_list:
                        try:
                            pr_asp = pred_val.split(':')[0].strip()
                        except:
                            continue


                        try:
                            pr_op = pred_val.split(':')[1].strip()
                        except:
                            continue

                        try:
                            pr_sent = pred_val.split(':')[2].strip()
                        except:
                            continue
                        if pr_asp == gt_asp and pr_op == gt_op and gt_sent == pr_sent:
                            # print(gt_asp, pr_asp, ' --> ', gt_op, pr_op, ' --> ', gt_sent, pr_sent)
                            # print(1)
                            tp+=1
                            break

        p = tp/total_pred
        r = tp/total_gt
        return p, r, 2*p*r/(p+r),None


class T5Classifier:
    def __init__(self, model_checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, force_download = True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, force_download = True)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        self.device = 'cuda' if torch.has_cuda else ('mps' if torch.has_mps else 'cpu')

    def tokenize_function_inputs(self, sample):
        """
        Udf to tokenize the input dataset.
        """
        sample['input_ids'] = self.tokenizer(sample["text"], max_length = 512, truncation = True).input_ids
        sample['labels'] = self.tokenizer(sample["labels"], max_length = 64, truncation = True).input_ids
        return sample
        
    def train(self, tokenized_datasets, **kwargs):
        """
        Train the generative model.
        """

        # Set training arguments
        args = Seq2SeqTrainingArguments(
            **kwargs
            )

        # Define trainer object
        trainer = Trainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"] if tokenized_datasets.get("validation") is not None else None,
            tokenizer=self.tokenizer, 
            data_collator = self.data_collator 
        )
        print("Trainer device:", trainer.args.device)

        # Finetune the model
        torch.cuda.empty_cache()
        print('\nModel training started ....')
        trainer.train()

        # Save best model
        trainer.save_model()
        return trainer

    def get_labels(self, tokenized_dataset, batch_size = 4, sample_set = 'train'):
        """
        Get the predictions from the trained model.
        """
        def collate_fn(batch):
            input_ids = [torch.tensor(example['input_ids']) for example in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            return input_ids
        
        dataloader = DataLoader(tokenized_dataset[sample_set], batch_size=batch_size, collate_fn=collate_fn)
        predicted_output = []
        self.model.to(self.device)
        print('Model loaded to: ', self.device)

        for batch in tqdm(dataloader):
            batch = batch.to(self.device)
            output_ids = self.model.generate(batch)
            output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for output_text in output_texts:
                predicted_output.append(output_text)
        return predicted_output
    
    def get_metrics(self, y_true, y_pred):
        return precision_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='macro'), \
            f1_score(y_true, y_pred, average='macro'), accuracy_score(y_true, y_pred)