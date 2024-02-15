from datasets import Dataset
from datasets.dataset_dict import DatasetDict
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import random
class DatasetLoader:
    def __init__(self, train_df_id=None, test_df_id=None,
                 train_df_ood=None, test_df_ood=None, sample_size = 1, val_df_id=None, val_df_ood=None):
        
        self.train_df_id = train_df_id.sample(frac = sample_size, random_state = 1999) if train_df_id is not None else train_df_id
        self.test_df_id = test_df_id
        self.train_df_ood = train_df_ood
        self.test_df_ood = test_df_ood
        self.val_df_id = val_df_id
        self.val_df_ood = val_df_ood
    def reconstruct_strings(self, df, col, num = 2):
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

    def extract_rowwise_aspect_polarity(self, df, on, key, min_val = None):
        """
        Create duplicate records based on number of aspect term labels in the dataset.
        Extract each aspect term for each row for reviews with muliple aspect term entries. 
        Do same for polarities and create new column for the same.
        """
        try:
            df.iloc[0][on][0][key]
        except:
            df = self.reconstruct_strings(df, on)

        df['len'] = df[on].apply(lambda x: len(x))
        if min_val is not None:
            df.loc[df['len'] == 0, 'len'] = min_val
        df = df.loc[df.index.repeat(df['len'])]
        df['record_idx'] = df.groupby(df.index).cumcount()
        df['aspect'] = df[[on, 'record_idx']].apply(lambda x : (x[0][x[1]][key], x[0][x[1]]['polarity']) if len(x[0]) != 0 else ('',''), axis=1)
        df['polarity'] = df['aspect'].apply(lambda x: x[-1])
        df['aspect'] = df['aspect'].apply(lambda x: x[0])
        df = df.drop(['len', 'record_idx'], axis=1).reset_index(drop = True)
        return df
    def extract_rowwise_ao_polarity(self, df, on, key,key2, min_val = None):
        """
        Create duplicate records based on number of aspect term labels in the dataset.
        Extract each aspect term for each row for reviews with muliple aspect term entries.
        Do same for polarities and create new column for the same.
        """
        try:
            df.iloc[0][on][0][key]
        except:
            df = self.reconstruct_strings(df, on,num=3)

        df['len'] = df[on].apply(lambda x: len(x))
        if min_val is not None:
            df.loc[df['len'] == 0, 'len'] = min_val
        df = df.loc[df.index.repeat(df['len'])]
        df['record_idx'] = df.groupby(df.index).cumcount()
        df['aspect'] = df[[on, 'record_idx']].apply(lambda x : (x[0][x[1]][key],x[0][x[1]][key2], x[0][x[1]]['polarity']) if len(x[0]) != 0 else ('',''), axis=1)
        df['polarity'] = df['aspect'].apply(lambda x: x[-1])
        # df['aspect'] = df['aspect'].apply(lambda x: x[0])
        df = df.drop(['len', 'record_idx'], axis=1).reset_index(drop = True)
        return df
    
    def extract_rowwise_aspect_opinions(self, df, aspect_col, key2, key, min_val = None):
        """
        Create duplicate records based on number of aspect term labels in the dataset.
        Extract each aspect term for each row for reviews with muliple aspect term entries. 
        Do same for polarities and create new column for the same.
        """
        res = pd.DataFrame(columns=['raw_text','aspect', 'opinion_term'], dtype=str)
        num_total = len(list(df['raw_text']))
        # res['raw_text'] = df['raw_text']
        # df['len'] = df[aspect_col].apply(lambda x: len(x))
        # if min_val is not None:
        #     df.loc[df['len'] == 0, 'len'] = min_val
        # df = df.loc[df.index.repeat(df['len'])]
        # df['record_idx'] = df.groupby(df.index).cumcount()
        # df['aspect'] = df[[aspect_col, 'record_idx']].apply(lambda x : x[0][x[1]][key] if len(x[0]) != 0 else '', axis=1)
        # df['opinion_term'] = df[[aspect_col, 'record_idx']].apply(lambda x : (x[0][x[1]][key], x[0][x[1]][key2]) if len(x[0]) != 0 else '', axis=1)
        # df['aspect'] = df['aspect'].apply(lambda x: x)
        raw_texts = list(df['raw_text'])
        terms = list(df[aspect_col])
        texts =[]
        opinions = []
        aspects = []
        for i in range(num_total):
            text = raw_texts[i]
            pre = None
            tmp = []
            a,o =None,None
            for it in terms[i]:
                a,o = it[key],it[key2]
                if a != pre:
                    if ','.join(tmp) != '':
                        texts.append(text)
                        aspects.append(a)
                        opinions.append(','.join(tmp))
                    tmp = []
                    pre = a
                tmp.append(o)
            texts.append(text)
            aspects.append(a)
            opinions.append(','.join(tmp))
        res['raw_text'] = texts
        res['aspect'] = aspects
        res['opinion_term'] = opinions
        # df['opinion_term'] = df['opinion_term'].apply(lambda x: x[-1])
        # df = df.drop(['len', 'record_idx'], axis=1).reset_index(drop = True)
        return res

    def create_data_in_ate_format(self, df, key, text_col, aspect_col, bos_instruction = '', 
                    eos_instruction = ''):
        """
        Prepare the data in the input format required.
        """
        if df is None:
            return
        try:
            df.iloc[0][aspect_col][0][key]
        except:
            df = self.reconstruct_strings(df, aspect_col,num=2)
        res = pd.DataFrame(columns=['text', 'labels'], dtype=str)
        res['labels'] = df[aspect_col].apply(lambda x: ', '.join([i[key] for i in x]))
        res['text'] = df[text_col].apply(lambda x: bos_instruction + x + eos_instruction)
        return res
    def create_data_in_ote_format(self, df, key, text_col, aspect_col, bos_instruction = '',
                    eos_instruction = ''):
        """
        Prepare the data in the input format required.
        """
        if df is None:
            return
        try:
            df.iloc[0][aspect_col][0][key]
        except:
            df = self.reconstruct_strings(df, aspect_col,num=3)
        res = pd.DataFrame(columns=['text', 'labels'], dtype=str)
        res['labels'] = df[aspect_col].apply(lambda x: ', '.join([i[key] for i in x]))
        res['text'] = df[text_col].apply(lambda x: bos_instruction + x + eos_instruction)
        return res

    def create_data_in_aooe_format(self, df, on, key, key2,text_col, aspect_col, bos_instruction = '',
                    delim_instruction = '', eos_instruction = ''):
        """
        Prepare the data in the input format required.
        """
        if df is None:
            return
        try:
            df.iloc[0][aspect_col][0][key]
        except:
            df = self.reconstruct_strings(df, on,num=3)
        df = self.extract_rowwise_aspect_opinions(df, aspect_col=on, key2=key2, key=key, min_val=1)
        res = pd.DataFrame(columns=['text', 'labels'], dtype=str)
        res['labels'] = df['opinion_term']
        res['text'] = df[[text_col, 'aspect']].apply(lambda x: bos_instruction + x[0] + delim_instruction + x[1] + eos_instruction, axis=1)
        # df = df.rename(columns = {'opinion_term': 'labels'})
        return res

    def create_data_in_atsc_format(self, df, on, key, text_col, aspect_col, bos_instruction = '', 
                    delim_instruction = '', eos_instruction = ''):
        """
        Prepare the data in the input format required.
        """
        if df is None:
            return
        df = self.extract_rowwise_aspect_polarity(df, on=on, key=key, min_val=1)
        res = pd.DataFrame(columns=['text', 'labels','de'], dtype=str)
        # res['labels'] = df[on].apply(lambda x: ', '.join([i[key] for i in x]))
        tmp = df[[text_col, aspect_col]].apply(
            lambda x: x[0] + delim_instruction + x[1] + '.', axis=1)
        res['text'] = df[[text_col, aspect_col]].apply(lambda x: bos_instruction + x[0] + delim_instruction + x[1] + eos_instruction, axis=1)
        res['raw_text'] = tmp
        res['labels'] = df['polarity']
        res['de'] = df['de']
        return res

    def create_data_in_aspe_format(self, df, key, label_key, text_col, aspect_col, bos_instruction = '', 
                                         eos_instruction = '', num = 2):
        """
        Prepare the data in the input format required.
        """
        if df is None:
            return
        try:
            df.iloc[0][aspect_col][0][key]
        except:
            df = self.reconstruct_strings(df, aspect_col, num=num)
        res = pd.DataFrame(columns=['text', 'labels'], dtype=str)
        res['labels'] = df[aspect_col].apply(lambda x: ', '.join([f"{i[key]}:{i[label_key]}" for i in x]))
        res['text'] = df[text_col].apply(lambda x: bos_instruction + x + eos_instruction)
        return res

    def create_data_in_aope_format(self, df, key, key2, text_col, aspect_col, bos_instruction = '',
                                         eos_instruction = '',):
        """
        Prepare the data in the input format required.
        """
        if df is None:
            return
        try:
            df.iloc[0][aspect_col][0][key]
        except:
            df = self.reconstruct_strings(df, aspect_col,num=3)
        res = pd.DataFrame(columns=['text', 'labels'], dtype=str)
        res['labels'] = df[aspect_col].apply(lambda x: ', '.join([f"{i[key]}:{i[key2]}" for i in x]))
        res['text'] = df[text_col].apply(lambda x: bos_instruction + x + eos_instruction)
        return res
    def create_data_in_aos_format(self, df, key, label_key, text_col, aspect_col, key2,
                                         bos_instruction = '', eos_instruction = ''):
        """
        Prepare the data in the input format required.
        """
        df = self.reconstruct_strings(df, aspect_col, num=3)
        if df is None:
            return
        df = self.extract_rowwise_ao_polarity(df, on=aspect_col, key=key,key2=key2, min_val=1)
        res = pd.DataFrame(columns=['text', 'labels'], dtype=str)
        # res['labels'] = df[on].apply(lambda x: ', '.join([i[key] for i in x]))
        res['text'] = df[[text_col, 'aspect']].apply(
            lambda x: bos_instruction + x[0] + f'. The aspect-opinion pair is: ({x[1][0]},{x[1][1]})' + eos_instruction, axis=1)
        res['labels'] = df['polarity']
        # print(res)
        return res
    def create_data_in_aoste_format(self, df, key, label_key, text_col, aspect_col, key2,
                                         bos_instruction = '', eos_instruction = ''):
        """
        Prepare the data in the input format required.
        """
        # label_map = {'POS':'positive', 'NEG':'negative', 'NEU':'neutral'}

        df = self.reconstruct_strings(df, aspect_col, num=3)

        df['labels'] = df[aspect_col].apply(lambda x: ', '.join([f"{i[key]}:{i[key2]}:{i[label_key]}" for i in x]))#f"{i[key]}:{j[key]}:{i[label_key]}" for i, j in zip(x[0], x[1])
        df['text'] = df[text_col].apply(lambda x: bos_instruction + x + eos_instruction)
        # res.to_csv('a.csv')
        return df
    def create_data_in_unify_format(self, df, key, label_key, text_col, aspect_col, key2,
                                         bos_instruction = '', eos_instruction = '', mode='test',tt=0,_random=True,ex_data = None):
        """
        Prepare the data in the input format required.
        """
        bos_instruction = mode.aoste['bos_instruct1']
        eos_instruction = mode.aoste['eos_instruct']
        res = self.create_data_in_aoste_format(df, key, label_key, text_col, aspect_col, key2, bos_instruction, eos_instruction)
        res = pd.concat([res, self.create_data_in_aos_format(df, key, label_key, text_col, aspect_col, key2, mode.aos['bos_instruct1'], mode.aos['eos_instruct'])])
        # res = pd.concat(
        #     [res, self.create_data_in_ate_format(ex_data, key, text_col, aspect_col, mode.ate['bos_instruct1'],
        #                                                      mode.ate['eos_instruct'])])
        res = pd.concat([res, self.create_data_in_ate_format(df, key, text_col, aspect_col, mode.ate['bos_instruct1'],
                    mode.ate['eos_instruct'])])
        res = pd.concat([res, self.create_data_in_ote_format(df, key2, text_col, aspect_col, mode.ote['bos_instruct1'],
                    mode.ote['eos_instruct'])])
        res = pd.concat([res, self.create_data_in_aooe_format(df, aspect_col, key,key2, text_col, "aspect",
                    mode.aooe['bos_instruct1'],mode.aooe['delim_instruct'], mode.aooe['eos_instruct'])])
        res = pd.concat([res, self.create_data_in_atsc_format(df, aspect_col, key, text_col, "aspect", mode.atsc['bos_instruct1'],
                    mode.atsc['delim_instruct'], mode.atsc['eos_instruct'])])
        # res = pd.concat(
        #     [res, self.create_data_in_atsc_format(ex_data, aspect_col, key, text_col, "aspect", mode.atsc['bos_instruct1'],
        #                                           mode.atsc['delim_instruct'], mode.atsc['eos_instruct'])])
        res = pd.concat([res, self.create_data_in_aope_format(df, key, key2, text_col, aspect_col, mode.aope['bos_instruct1'],
                    mode.aope['eos_instruct'])])
        res = pd.concat([res, self.create_data_in_aspe_format(df, key, label_key, text_col, aspect_col,mode.aspe['bos_instruct1'],
                    mode.aspe['eos_instruct'],num=3)])

        # res = pd.concat(
        #     [res, self.create_data_in_aspe_format(ex_data, key, label_key, text_col, aspect_col, mode.aspe['bos_instruct1'],
        #                                           mode.aspe['eos_instruct'])])

        # res.to_csv('a.csv')
        if _random:
            return res.sample(frac = 1, random_state = 1999)
        else:
            return res
    def set_data_for_training_semeval(self, tokenize_function):
        """
        Create the training and test dataset as huggingface datasets format.
        """
        # Define train and test sets
        dataset_dict_id, dataset_dict_ood = {}, {}

        if self.train_df_id is not None:
            dataset_dict_id['train'] = Dataset.from_pandas(self.train_df_id)
        if self.test_df_id is not None:
            dataset_dict_id['test'] = Dataset.from_pandas(self.test_df_id)
        # if self.val_df_id is not None:
        #     dataset_dict_id['validation'] = Dataset.from_pandas(self.val_df_id)
        if len(dataset_dict_id) > 1:
            indomain_dataset = DatasetDict(dataset_dict_id)
            # print(indomain_dataset)
            indomain_tokenized_datasets = indomain_dataset.map(tokenize_function, batched=True)
        else:
            indomain_dataset = {}
            indomain_tokenized_datasets = {}

        if self.train_df_ood is not None:
            dataset_dict_ood['train'] = Dataset.from_pandas(self.train_df_ood)
        if self.test_df_ood is not None:
            dataset_dict_ood['test'] = Dataset.from_pandas(self.test_df_ood)
        # if self.val_df_ood is not None:
        #     dataset_dict_ood['validation'] = Dataset.from_pandas(self.val_df_ood)
        if len(dataset_dict_id) > 1:
            other_domain_dataset = DatasetDict(dataset_dict_ood)
            other_domain_tokenized_dataset = other_domain_dataset.map(tokenize_function, batched=True)
        else:
            other_domain_dataset = {}
            other_domain_tokenized_dataset = {}

        return indomain_dataset, indomain_tokenized_datasets, other_domain_dataset, other_domain_tokenized_dataset
        