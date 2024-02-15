import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

import torch
from InstructABSA.data_prep import DatasetLoader
from InstructABSA.utils import T5Generator, T5Classifier
from InstructABSA.config import Config
from instructions import InstructionsHandler


try:
    use_mps = True if torch.has_mps else False
except:
    use_mps = False

# Set Global Values
config = Config()
instruct_handler = InstructionsHandler(config = config)
if config.inst_type == 1:
    instruct_handler.load_instruction_set1()
else:
    instruct_handler.load_instruction_set2()

print('Task: ', config.task)

if config.mode == 'train':
    if config.id_tr_data_path is None:
        raise Exception('Please provide training data path for mode=training.')
    
if config.mode == 'eval':
    if config.id_te_data_path is None and config.ood_te_data_path is None:
        raise Exception('Please provide testing data path for mode=eval.')

if config.experiment_name is not None and config.mode == 'train':
    print('Experiment Name: ', config.experiment_name)
    model_checkpoint = config.model_checkpoint
    model_out_path = config.output_dir
    model_out_path = os.path.join(model_out_path, config.task, f"{model_checkpoint.replace('/', '')}-{config.experiment_name}")
else:
    model_checkpoint = config.model_checkpoint
    model_out_path = config.model_checkpoint

print('Mode set to: ', 'training' if config.mode == 'train' else ('inference' if config.mode == 'eval' \
                                                                  else 'Individual sample inference'))

# Load the data
id_tr_data_path = config.id_tr_data_path
ood_tr_data_path = config.ood_tr_data_path
id_te_data_path = config.id_te_data_path
ood_te_data_path = config.ood_te_data_path

if config.mode != 'cli':
    id_tr_df,  id_te_df = None, None
    ood_tr_df,  ood_te_df = None, None
    if id_tr_data_path is not None:
        id_tr_df = pd.read_csv(id_tr_data_path)
    if id_te_data_path is not None:
        id_te_df = pd.read_csv(id_te_data_path)
    if ood_tr_data_path is not None:
        ood_tr_df = pd.read_csv(ood_tr_data_path)
    if ood_te_data_path is not None:
        ood_te_df = pd.read_csv(ood_te_data_path)
    print('Loaded data...')
else:
    print('Running inference on input: ', config.test_input)


# Training arguments
training_args = {
                'output_dir': model_out_path,
                'evaluation_strategy': config.evaluation_strategy if config.id_te_data_path is not None else 'no',
                'learning_rate': config.learning_rate,
                'per_device_train_batch_size': config.per_device_train_batch_size if config.per_device_train_batch_size is not None else None,
                'per_device_eval_batch_size': config.per_device_eval_batch_size,
                'num_train_epochs': config.num_train_epochs if config.num_train_epochs is not None else None,
                'weight_decay': config.weight_decay,
                'warmup_ratio': config.warmup_ratio,
                'save_strategy': config.save_strategy,
                'load_best_model_at_end': config.load_best_model_at_end,
                'push_to_hub': config.push_to_hub,
                'eval_accumulation_steps': config.eval_accumulation_steps,
                'predict_with_generate': config.predict_with_generate,
                'use_mps_device': use_mps
            }

# Create T5 model object
print(config.set_instruction_key)
if config.set_instruction_key == 1:
    indomain = 'bos_instruct1'
    outdomain = 'bos_instruct2'
else:
    indomain = 'bos_instruct2'
    outdomain = 'bos_instruct1'

if config.task == 'ate':
    t5_exp = T5Generator(model_checkpoint)
    bos_instruction_id = instruct_handler.ate[indomain]
    if ood_tr_data_path is not None or ood_te_data_path is not None:
        bos_instruction_ood = instruct_handler.ate[outdomain]
    eos_instruction = instruct_handler.ate['eos_instruct']
if config.task == 'aos':
    t5_exp = T5Generator(model_checkpoint)
    bos_instruction_id = instruct_handler.ate[indomain]
    bos_instruction_id1 = instruct_handler.ote[indomain]
    bos_instruction_id2 = instruct_handler.aoste[indomain]
    if ood_tr_data_path is not None or ood_te_data_path is not None:
        bos_instruction_ood = instruct_handler.ate[outdomain]
        bos_instruction_ood1 = instruct_handler.ote[outdomain]
        bos_instruction_ood2 = instruct_handler.aoste[outdomain]
    eos_instruction = instruct_handler.ate['eos_instruct']
    eos_instruction1 = instruct_handler.ote['eos_instruct']
    eos_instruction2 = instruct_handler.aoste['eos_instruct']
if config.task == 'ote':
    t5_exp = T5Generator(model_checkpoint)
    bos_instruction_id = instruct_handler.ote[indomain]
    if ood_tr_data_path is not None or ood_te_data_path is not None:
        bos_instruction_ood = instruct_handler.ote[outdomain]
    eos_instruction = instruct_handler.ote['eos_instruct']
if config.task == 'atsc':
    t5_exp = T5Classifier(model_checkpoint)
    bos_instruction_id = instruct_handler.atsc[indomain]
    if ood_tr_data_path is not None or ood_te_data_path is not None:
        bos_instruction_ood = instruct_handler.atsc[outdomain]
    delim_instruction = instruct_handler.atsc['delim_instruct']
    eos_instruction = instruct_handler.atsc['eos_instruct']
if config.task == 'aspe':
    t5_exp = T5Generator(model_checkpoint)
    bos_instruction_id = instruct_handler.aspe[indomain]
    if ood_tr_data_path is not None or ood_te_data_path is not None:
        bos_instruction_ood = instruct_handler.aspe[outdomain]
    eos_instruction = instruct_handler.aspe['eos_instruct']
if config.task == 'aoste' or config.task == 'unify':
    t5_exp = T5Generator(model_checkpoint)
    bos_instruction_id = instruct_handler.aoste[indomain]
    if ood_tr_data_path is not None or ood_te_data_path is not None:
        bos_instruction_ood = instruct_handler.aoste[outdomain]
    eos_instruction = instruct_handler.aoste['eos_instruct']

if config.mode != 'cli':
    # Define function to load datasets and tokenize datasets
    loader = DatasetLoader(id_tr_df, id_te_df, ood_tr_df, ood_te_df, config.sample_size, id_te_df, ood_te_df)
    train_data_path={}
    train_data_path['lap14'] = "../Dataset/SemEval14/Train/Laptops_Train.csv"
    train_data_path['res14'] = "../Dataset/SemEval14/Train/Restaurants_Train.csv"
    train_data_path['res15'] = "../Dataset/SemEval15/Train/Restaurants_Train.csv"
    train_data_path['res16'] = "../Dataset/SemEval16/Train/Restaurants_Train.csv"
    if config.task == 'ate':
        if loader.train_df_id is not None:
            loader.train_df_id = loader.create_data_in_ate_format(loader.train_df_id, 'term', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
        if loader.test_df_id is not None:
            loader.test_df_id = loader.create_data_in_ate_format(loader.test_df_id, 'term', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
        if loader.train_df_ood is not None:
            loader.train_df_ood = loader.create_data_in_ate_format(loader.train_df_ood, 'term', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)
        if loader.test_df_ood is not None:
            loader.test_df_ood = loader.create_data_in_ate_format(loader.test_df_ood, 'term', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)
    elif config.task == 'ote':
        if loader.train_df_id is not None:
            loader.train_df_id = loader.create_data_in_ote_format(loader.train_df_id, 'opinion', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
        if loader.test_df_id is not None:
            loader.test_df_id = loader.create_data_in_ote_format(loader.test_df_id, 'opinion', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
        if loader.train_df_ood is not None:
            loader.train_df_ood = loader.create_data_in_ote_format(loader.train_df_ood, 'opinion', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)
        if loader.test_df_ood is not None:
            loader.test_df_ood = loader.create_data_in_ote_format(loader.test_df_ood, 'opinion', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)

    elif config.task == 'atsc':
        if loader.train_df_id is not None:
            loader.train_df_id = loader.create_data_in_atsc_format(loader.train_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_id, delim_instruction, eos_instruction)
        if loader.test_df_id is not None:
            loader.test_df_id = loader.create_data_in_atsc_format(loader.test_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_id, delim_instruction, eos_instruction)
        if loader.train_df_ood is not None:
            loader.train_df_ood = loader.create_data_in_atsc_format(loader.train_df_ood, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_ood, delim_instruction, eos_instruction)
        if loader.test_df_ood is not None:
            loader.test_df_ood = loader.create_data_in_atsc_format(loader.test_df_ood, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_ood, delim_instruction, eos_instruction)

    elif config.task == 'aspe':
        if loader.train_df_id is not None:
            loader.train_df_id = loader.create_data_in_aspe_format(loader.train_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
        if loader.test_df_id is not None:
            loader.test_df_id = loader.create_data_in_aspe_format(loader.test_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
        if loader.train_df_ood is not None:
            loader.train_df_ood = loader.create_data_in_aspe_format(loader.train_df_ood, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)
        if loader.test_df_ood is not None:
            loader.test_df_ood = loader.create_data_in_aspe_format(loader.test_df_ood, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)
    elif config.task == 'aos':
        loader1 = DatasetLoader(id_tr_df, id_te_df, ood_tr_df, ood_te_df, config.sample_size, id_te_df, ood_te_df)
        loader2 = DatasetLoader(id_tr_df, id_te_df, ood_tr_df, ood_te_df, config.sample_size, id_te_df, ood_te_df)
        if loader.train_df_id is not None:
            loader.train_df_id = loader.create_data_in_ate_format(loader.train_df_id, 'term', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
            loader1.train_df_id = loader1.create_data_in_ote_format(loader1.train_df_id, 'opinion', 'raw_text', 'aspectTerms',
                                                                  bos_instruction_id1, eos_instruction1)
            loader2.train_df_id = loader2.create_data_in_aoste_format(loader2.train_df_id, 'term', 'polarity',
                                                                      'raw_text', 'aspectTerms', 'opinion',
                                                                      bos_instruction_id2, eos_instruction2, tt=config.k)
        if loader.test_df_id is not None:
            loader.test_df_id = loader.create_data_in_ate_format(loader.test_df_id, 'term', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
            # loader1.test_df_id = loader1.create_data_in_ote_format(loader1.test_df_id, 'opinion', 'raw_text', 'aspectTerms',
            #                                                      bos_instruction_id1, eos_instruction1)
            loader1.test_df_ood = loader1.create_data_in_aope_format(loader1.test_df_id, 'term','opinion', 'raw_text',
                                                                     'aspectTerms', instruct_handler.aope['bos_instruct1'],
                                                                     instruct_handler.aope['eos_instruct'])
            loader2.test_df_id = loader2.create_data_in_aoste_format(loader2.test_df_id, 'term', 'polarity', 'raw_text',
                                                                     'aspectTerms', 'opinion', bos_instruction_id2,
                                                                     eos_instruction2, tt=config.k)
        if loader.train_df_ood is not None:
            loader.train_df_ood = loader.create_data_in_ate_format(loader.train_df_ood, 'term', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)
            loader1.train_df_ood = loader1.create_data_in_ote_format(loader1.train_df_ood, 'opinion', 'raw_text',
                                                                   'aspectTerms', bos_instruction_ood1, eos_instruction1)
            loader2.train_df_ood = loader2.create_data_in_aoste_format(loader2.train_df_ood, 'term', 'polarity',
                                                                       'raw_text', 'aspectTerms', 'opinion',
                                                                       bos_instruction_ood2, eos_instruction2,
                                                                       tt=config.k)
        if loader.test_df_ood is not None:
            loader.test_df_ood = loader.create_data_in_ate_format(loader.test_df_ood, 'term', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)
            loader1.test_df_ood = loader1.create_data_in_ote_format(loader1.test_df_ood, 'opinion', 'raw_text', 'aspectTerms',
                                                                  bos_instruction_ood1, eos_instruction1)
            loader2.test_df_ood = loader2.create_data_in_aoste_format(loader2.test_df_ood, 'term', 'polarity',
                                                                      'raw_text', 'aspectTerms', 'opinion',
                                                                      bos_instruction_ood2, eos_instruction2, tt=config.k)

        id_ds1, id_tokenized_ds1, ood_ds1, ood_tokenized_ds1 = loader1.set_data_for_training_semeval(
            t5_exp.tokenize_function_inputs)
        id_ds2, id_tokenized_ds2, ood_ds2, ood_tokenized_ds2 = loader2.set_data_for_training_semeval(
            t5_exp.tokenize_function_inputs)
    elif config.task == 'aoste':
        if loader.train_df_id is not None:
            loader.train_df_id = loader.create_data_in_aoste_format(loader.train_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', 'opinion', bos_instruction_id, eos_instruction,tt=config.k)
        if loader.test_df_id is not None:
            loader.test_df_id = loader.create_data_in_aoste_format(loader.test_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', 'opinion', bos_instruction_id, eos_instruction,tt=config.k)
        if loader.train_df_ood is not None:
            loader.train_df_ood = loader.create_data_in_aoste_format(loader.train_df_ood, 'term', 'polarity', 'raw_text', 'aspectTerms', 'opinion', bos_instruction_ood, eos_instruction,tt=config.k)
        if loader.test_df_ood is not None:
            loader.test_df_ood = loader.create_data_in_aoste_format(loader.test_df_ood, 'term', 'polarity', 'raw_text', 'aspectTerms', 'opinion', bos_instruction_ood, eos_instruction,tt=config.k)
    elif config.task == 'unify':
        _random = False
        if config.mode == 'train':
            _random = True
        ex_data = pd.read_csv(train_data_path['lap14'])
        if loader.train_df_id is not None:
            loader.train_df_id = loader.create_data_in_unify_format(loader.train_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', 'opinion', bos_instruction_id, eos_instruction,mode = instruct_handler,tt=config.k,_random=_random,ex_data=ex_data)
        if loader.test_df_id is not None:
            loader.test_df_id = loader.create_data_in_unify_format(loader.test_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', 'opinion', bos_instruction_id, eos_instruction,mode = instruct_handler,tt=config.k,_random=_random,ex_data=ex_data)
        if loader.train_df_ood is not None:
            loader.train_df_ood = loader.create_data_in_unify_format(loader.train_df_ood, 'term', 'polarity', 'raw_text', 'aspectTerms', 'opinion', bos_instruction_ood, eos_instruction,mode = instruct_handler,tt=config.k,_random=_random,ex_data=ex_data)
        if loader.test_df_ood is not None:
            loader.test_df_ood = loader.create_data_in_unify_format(loader.test_df_ood, 'term', 'polarity', 'raw_text', 'aspectTerms', 'opinion', bos_instruction_ood, eos_instruction,mode = instruct_handler,tt=config.k,_random=_random,ex_data=ex_data)
    # Tokenize dataset
    id_ds, id_tokenized_ds, ood_ds, ood_tokenized_ds = loader.set_data_for_training_semeval(t5_exp.tokenize_function_inputs) 

    if config.mode == 'train':
        # Train model
        model_trainer = t5_exp.train(id_tokenized_ds, **training_args)
        print('Model saved at: ', model_out_path)
    elif config.mode == 'eval':
        # Get prediction labels
        print('Model loaded from: ', model_checkpoint)
        if id_tokenized_ds.get("train") is not None:

            id_tr_pred_labels = t5_exp.get_labels(tokenized_dataset = id_tokenized_ds, sample_set = 'train',
                                                  batch_size=config.per_device_eval_batch_size, 
                                                  max_length = config.max_token_length)
            id_tr_df = pd.DataFrame(id_ds['train'])[['text', 'labels']]
            id_tr_df['labels'] = id_tr_df['labels'].apply(lambda x: x.strip())
            id_tr_df['pred_labels'] = id_tr_pred_labels
            id_tr_df.to_csv(os.path.join(config.output_path, f'{config.experiment_name}_id_train.csv'), index=False)
            print('*****Train Metrics*****')
            precision, recall, f1, accuracy = t5_exp.get_metrics(id_tr_df['labels'], id_tr_pred_labels)
            print('Precision: ', precision)
            print('Recall: ', recall)
            print('F1-Score: ', f1)
            if config.task == 'atsc':
                print('Accuracy: ', accuracy)


        if id_tokenized_ds.get("test") is not None:
            if config.task == 'aos':
                # ate->aooe(ote valid)->aos(atsc otsc)
                raw_text = id_te_df['raw_text']
                # ate_labels = t5_exp.get_labels(tokenized_dataset=id_tokenized_ds, sample_set='test',
                #                                batch_size=config.per_device_eval_batch_size,
                #                                max_length=config.max_token_length)
                # ote_labels = t5_exp.get_labels(tokenized_dataset=id_tokenized_ds1, sample_set='test',
                #                                batch_size=config.per_device_eval_batch_size,
                #                                max_length=config.max_token_length)
                ao_labels = t5_exp.get_labels(tokenized_dataset=id_tokenized_ds1, sample_set='test',
                                               batch_size=config.per_device_eval_batch_size,
                                               max_length=config.max_token_length)
                id_te_pred_labels =[]
                for i in range(len(ao_labels)):
                    ao = ao_labels[i].split(',')
                    ao = [x.strip() for x in ao]
                    tri_preds = []
                    for j in ao:
                        tmp = j.split(':')
                        tmp = [x.strip() for x in tmp]
                        if "\"" not in tmp[1]:
                            if tmp[0].lower() not in raw_text[i].lower() or tmp[1].lower() not in raw_text[i].lower():
                                continue
                        model_input = instruct_handler.atsc['bos_instruct1'] + raw_text[
                                            i] + f'. The aspect is: {tmp[0]}' + instruct_handler.atsc['eos_instruct']
                        input_ids = t5_exp.tokenizer(model_input, return_tensors="pt").input_ids.to(t5_exp.model.device)
                        outputs = t5_exp.model.generate(input_ids, max_length=config.max_token_length)
                        s = t5_exp.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                        tri_preds.append(':'.join([tmp[0], tmp[1], s]))
                    tri_output = ','.join(list(set(tri_preds)))
                    id_te_pred_labels.append(tri_output)
                # for i in range(len(ate_labels)):
                #     aspects = ate_labels[i].split(',')
                #     aspects = [tmp.strip() for tmp in aspects]
                #     aspects = list(set(aspects))
                #
                #     opinions = ote_labels[i].split(',')
                #     opinions = [tmp.strip() for tmp in opinions]
                #
                #     # aooe
                #     tri_preds = []
                #     for a in aspects:
                #         model_input =  instruct_handler.aooe['bos_instruct1'] + raw_text[i] + f'. The aspect term is: {a}' + instruct_handler.aooe['eos_instruct']
                #         input_ids = t5_exp.tokenizer(model_input, return_tensors="pt").input_ids.to(t5_exp.model.device)
                #         outputs = t5_exp.model.generate(input_ids, max_length=config.max_token_length)
                #         o_list = t5_exp.tokenizer.decode(outputs[0], skip_special_tokens=True).split(',')
                #         #ote fliter
                #         o_list = [o for o in o_list if o in opinions]
                #         #(a,o)->(a,o,s)
                #
                #         for o in o_list:
                #             model_input = instruct_handler.aos['bos_instruct1'] + raw_text[
                #                 i] + f'. The aspect-opinion pair is: ({a}, {o})' + instruct_handler.aos['eos_instruct']
                #             input_ids = t5_exp.tokenizer(model_input, return_tensors="pt").input_ids.to(t5_exp.model.device)
                #             outputs = t5_exp.model.generate(input_ids, max_length=config.max_token_length)
                #             s = t5_exp.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                #             tri_preds.append(':'.join([a,o,s]))
                #     tri_output = ','.join(tri_preds)
                #     id_te_pred_labels.append(tri_output)


                id_te_df = pd.DataFrame(id_ds2['test'])[['text', 'labels']]
            else:
                id_te_pred_labels = t5_exp.get_labels(tokenized_dataset = id_tokenized_ds, sample_set = 'test',
                                                      batch_size=config.per_device_eval_batch_size,
                                                      max_length = config.max_token_length)
                id_te_df = pd.DataFrame(id_ds['test'])[['text', 'labels']]
            id_te_df['labels'] = id_te_df['labels'].apply(lambda x: x.strip())
            id_te_df['pred_labels'] = id_te_pred_labels
            id_te_df.to_csv(os.path.join(config.output_path, f'{config.experiment_name}_id_test.csv'), index=False)
            print('*****Test Metrics*****')

            precision, recall, f1, accuracy = t5_exp.get_metrics(id_te_df['labels'], id_te_pred_labels)
            print('Precision: ', precision)
            print('Recall: ', recall)
            print('F1-Score: ', f1)
            if config.task == 'atsc':
                print('Accuracy: ', accuracy)

        if ood_tokenized_ds.get("train") is not None:
            ood_tr_pred_labels = t5_exp.get_labels(tokenized_dataset = ood_tokenized_ds, sample_set = 'train', 
                                                   batch_size=config.per_device_eval_batch_size, 
                                                   max_length = config.max_token_length)
            ood_tr_df = pd.DataFrame(ood_ds['train'])[['text', 'labels']]
            ood_tr_df['labels'] = ood_tr_df['labels'].apply(lambda x: x.strip())
            ood_tr_df['pred_labels'] = ood_tr_pred_labels
            ood_tr_df.to_csv(os.path.join(config.output_path, f'{config.experiment_name}_ood_train.csv'), index=False)
            print('*****Train Metrics - OOD*****')

            precision, recall, f1, accuracy = t5_exp.get_metrics(id_tr_df['labels'], id_tr_pred_labels)
            print('Precision: ', precision)
            print('Recall: ', precision)
            print('F1-Score: ', precision)
            if config.task == 'atsc':
                print('Accuracy: ', accuracy)
            
        if ood_tokenized_ds.get("test") is not None:
            ood_te_pred_labels = t5_exp.get_labels(tokenized_dataset = ood_tokenized_ds, sample_set = 'test', 
                                                   batch_size=config.per_device_eval_batch_size, 
                                                   max_length = config.max_token_length)
            ood_te_df = pd.DataFrame(ood_ds['test'])[['text', 'labels']]
            ood_te_df['labels'] = ood_te_df['labels'].apply(lambda x: x.strip())
            ood_te_df['pred_labels'] = ood_te_pred_labels
            ood_te_df.to_csv(os.path.join(config.output_path, f'{config.experiment_name}_ood_test.csv'), index=False)
            print('*****Test Metrics - OOD*****')
            precision, recall, f1, accuracy = t5_exp.get_metrics(id_te_df['labels'], id_te_pred_labels)
            print('Precision: ', precision)
            print('Recall: ', precision)
            print('F1-Score: ', precision)
            if config.task == 'atsc':
                print('Accuracy: ', accuracy)
else:
    print('Model loaded from: ', model_checkpoint)
    if config.task == 'atsc':
        config.test_input, aspect_term = config.test_input.split('|')[0], config.test_input.split('|')[1]
        model_input = bos_instruction_id + config.test_input + f'. The aspect term is: {aspect_term}' + eos_instruction
    else:
        model_input = bos_instruction_id + config.test_input + eos_instruction
    input_ids = t5_exp.tokenizer(model_input, return_tensors="pt").input_ids
    outputs = t5_exp.model.generate(input_ids, max_length = config.max_token_length)
    print('Model output: ', t5_exp.tokenizer.decode(outputs[0], skip_special_tokens=True))
