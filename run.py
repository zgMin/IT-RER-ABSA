import os
import warnings
os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings('ignore')
import pandas as pd

import torch
from InstructABSA.data_prep import DatasetLoader
from InstructABSA.utils import T5Generator, T5Classifier,reconstruct_strings,create_data_in_atsc_format
from InstructABSA.config import Config
from instructions import InstructionsHandler

try:
    use_mps = True if torch.has_mps else False
except:
    use_mps = False

# Set Global Values
config = Config()
instruct_handler = InstructionsHandler(config=config)
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
    question_encoder_name = config.question_encoder_name
    context_encoder_name = config.context_encoder_name
    model_out_path = config.output_dir
    model_out_path = os.path.join(model_out_path, config.task,
                                  f"{model_checkpoint.replace('/', '')}-{config.experiment_name}")
else:
    model_checkpoint = config.model_checkpoint
    question_encoder_name =config.question_encoder_name
    context_encoder_name= config.context_encoder_name
    model_out_path = config.model_checkpoint

print('Mode set to: ', 'training' if config.mode == 'train' else ('inference' if config.mode == 'eval' \
                                                                      else 'Individual sample inference'))

# Load the data
id_tr_data_path = config.id_tr_data_path
ood_tr_data_path = config.ood_tr_data_path
id_te_data_path = config.id_te_data_path
ood_te_data_path = config.ood_te_data_path

if config.mode != 'cli':
    id_tr_df, id_te_df = None, None
    ood_tr_df, ood_te_df = None, None
    if id_tr_data_path is not None:
        id_tr_df = pd.read_csv(id_tr_data_path)
        id_tr_df = reconstruct_strings(id_tr_df, 'aspectTerms')
    if id_te_data_path is not None:
        id_te_df = pd.read_csv(id_te_data_path)
        id_te_df = reconstruct_strings(id_te_df, 'aspectTerms')
    if ood_tr_data_path is not None:
        ood_tr_df = pd.read_csv(ood_tr_data_path)
        ood_tr_df = reconstruct_strings(ood_tr_df, 'aspectTerms')
    if ood_te_data_path is not None:
        ood_te_df = pd.read_csv(ood_te_data_path)
        ood_te_df = reconstruct_strings(ood_te_df, 'aspectTerms')
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
    # 'use_mps_device': use_mps
}

# Create T5 model object
t5_exp = T5Generator(model_checkpoint,question_encoder_name,context_encoder_name)
print(config.set_instruction_key)
if config.set_instruction_key == 1:
    indomain = 'bos_instruct1'
    outdomain = 'bos_instruct2'
else:
    indomain = 'bos_instruct2'
    outdomain = 'bos_instruct1'
if config.task == 'ate':
    definition = instruct_handler.ate['definition']
if config.task == 'ote':
    definition = instruct_handler.ote['definition']
if config.task == 'atsc':
    definition = instruct_handler.atsc['definition']
if config.task == 'aspe':
    definition = instruct_handler.aspe['definition']
if config.task == 'aoste':
    definition = instruct_handler.aoste['definition']
if config.task == 'ate':
    bos_instruction_id = instruct_handler.ate[indomain]
    if ood_tr_data_path is not None or ood_te_data_path is not None:
        bos_instruction_ood = instruct_handler.ate[outdomain]
if config.task == 'ote':
    bos_instruction_id = instruct_handler.ote[indomain]
    if ood_tr_data_path is not None or ood_te_data_path is not None:
        bos_instruction_ood = instruct_handler.ote[outdomain]
if config.task == 'atsc':
    bos_instruction_id = instruct_handler.atsc[indomain]
    if ood_tr_data_path is not None or ood_te_data_path is not None:
        bos_instruction_ood = instruct_handler.atsc[outdomain]
if config.task == 'aspe':
    bos_instruction_id = instruct_handler.aspe[indomain]
    if ood_tr_data_path is not None or ood_te_data_path is not None:
        bos_instruction_ood = instruct_handler.aspe[outdomain]
if config.task == 'aoste' or config.task == 'unify':
    bos_instruction_id = instruct_handler.aoste[indomain]
    if ood_tr_data_path is not None or ood_te_data_path is not None:
        bos_instruction_ood = instruct_handler.aoste[outdomain]




def create_dataset(id_tr_df, id_ev_df, ood_tr_df, ood_ev_df, config, id_te_df, ood_te_df,instruct_handler,bos_instruction_id):# 封装一下创建训练数据的过程
    loader = DatasetLoader(id_tr_df, id_ev_df, ood_tr_df, ood_ev_df, config.sample_size, id_te_df, ood_te_df)
    eos_instruction = '\noutput: '
    delim_instruction = ' The aspect is '
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
            loader.train_df_id = loader.create_data_in_atsc_format(loader.train_df_id, 'aspectTerms', 'term',
                                                                   'raw_text', 'aspect', bos_instruction_id, delim_instruction, eos_instruction)
        if loader.test_df_id is not None:
            loader.test_df_id = loader.create_data_in_atsc_format(loader.test_df_id, 'aspectTerms', 'term', 'raw_text',
                                                                  'aspect', bos_instruction_id, delim_instruction, eos_instruction)
        if loader.train_df_ood is not None:
            loader.train_df_ood = loader.create_data_in_atsc_format(loader.train_df_ood, 'aspectTerms', 'term',
                                                                    'raw_text', 'aspect', bos_instruction_ood, delim_instruction, eos_instruction)
        if loader.test_df_ood is not None:
            loader.test_df_ood = loader.create_data_in_atsc_format(loader.test_df_ood, 'aspectTerms', 'term',
                                                                   'raw_text', 'aspect', bos_instruction_ood, delim_instruction, eos_instruction)
    elif config.task == 'aspe':
        if loader.train_df_id is not None:
            loader.train_df_id = loader.create_data_in_aspe_format(loader.train_df_id, 'term', 'polarity', 'raw_text',
                                                                   'aspectTerms', bos_instruction_id, eos_instruction)
        if loader.test_df_id is not None:
            loader.test_df_id = loader.create_data_in_aspe_format(loader.test_df_id, 'term', 'polarity', 'raw_text',
                                                                  'aspectTerms', bos_instruction_id, eos_instruction)
        if loader.train_df_ood is not None:
            loader.train_df_ood = loader.create_data_in_aspe_format(loader.train_df_ood, 'term', 'polarity', 'raw_text',
                                                                    'aspectTerms', bos_instruction_ood, eos_instruction)
        if loader.test_df_ood is not None:
            loader.test_df_ood = loader.create_data_in_aspe_format(loader.test_df_ood, 'term', 'polarity', 'raw_text',
                                                                   'aspectTerms', bos_instruction_ood, eos_instruction)
    elif config.task == 'aoste':
        if loader.train_df_id is not None:
            loader.train_df_id = loader.create_data_in_aoste_format(loader.train_df_id, 'term', 'polarity', 'raw_text',
                                                                    'aspectTerms', 'opinion', bos_instruction_id, eos_instruction)
        if loader.test_df_id is not None:
            loader.test_df_id = loader.create_data_in_aoste_format(loader.test_df_id, 'term', 'polarity', 'raw_text',
                                                                   'aspectTerms', 'opinion', bos_instruction_id, eos_instruction)
        if loader.train_df_ood is not None:
            loader.train_df_ood = loader.create_data_in_aoste_format(loader.train_df_ood, 'term', 'polarity', 'raw_text', 'aspectTerms', 'opinion',
                                                                     bos_instruction_ood, eos_instruction)
        if loader.test_df_ood is not None:
            loader.test_df_ood = loader.create_data_in_aoste_format(loader.test_df_ood, 'term', 'polarity', 'raw_text',
                                                                    'aspectTerms', 'opinion', bos_instruction_ood, eos_instruction)
    elif config.task == 'unify':
        _random = False
        if config.mode == 'train':
            _random = True
        ex_data = pd.read_csv(train_data_path['lap14'])
        if loader.train_df_id is not None:
            loader.train_df_id = loader.create_data_in_unify_format(loader.train_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', 'opinion', bos_instruction_id,
                                                                    eos_instruction, mode=instruct_handler, tt=config.k, _random=_random, ex_data=ex_data)
        if loader.test_df_id is not None:
            loader.test_df_id = loader.create_data_in_unify_format(loader.test_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', 'opinion', bos_instruction_id,
                                                                   eos_instruction, mode=instruct_handler, tt=config.k, _random=_random, ex_data=ex_data)
        if loader.train_df_ood is not None:
            loader.train_df_ood = loader.create_data_in_unify_format(loader.train_df_ood, 'term', 'polarity', 'raw_text', 'aspectTerms', 'opinion', bos_instruction_ood, eos_instruction,
                                                                     mode=instruct_handler, tt=config.k, _random=_random, ex_data=ex_data)
        if loader.test_df_ood is not None:
            loader.test_df_ood = loader.create_data_in_unify_format(loader.test_df_ood, 'term', 'polarity', 'raw_text', 'aspectTerms', 'opinion', bos_instruction_ood, eos_instruction, mode=instruct_handler, tt=config.k, _random=_random, ex_data=ex_data)
    return loader
if config.mode == 'train':
# te_df = pd.read_csv("../Dataset/ASTE/lap14/dev.csv")
# te_df = reconstruct_strings(te_df, 'aspectTerms')
    t5_exp.init_retriever(id_tr_df, id_te_df, definition,task = config.task)
    # t5_exp.train_retriever(id_tr_df,id_te_df,definition)
else:
    t5_exp.init_retriever(id_tr_df, id_te_df, definition,cache_dir = True,task = config.task)
if config.mode != 'cli':
    de = t5_exp.select_examples(list(id_tr_df['raw_text']),k=config.k)
    for i in range(len(de)):
        de[i] = definition + '\n'+de[i]
    id_tr_df['de'] = de
    de = t5_exp.select_examples(list(id_te_df['raw_text']), k=config.k)
    for i in range(len(de)):
        de[i] = definition + '\n'+de[i]
    id_te_df['de'] = de

    loader = create_dataset(id_tr_df, id_te_df, ood_tr_df, ood_te_df, config, id_te_df, ood_te_df,instruct_handler,bos_instruction_id)
    if config.task == 'atsc':
        loader.train_df_id['text'] = loader.train_df_id[['raw_text', 'de']].apply(
            lambda x: x[1] + "\nNow complete the following example-" + "\ninput: " + x[0] + '\noutput: ', axis=1)
        loader.test_df_id['text'] = loader.test_df_id[['raw_text', 'de']].apply(
            lambda x: x[1] + "\nNow complete the following example-" + "\ninput: " + x[0] + '\noutput: ', axis=1)
    else:
        loader.train_df_id['text'] = id_tr_df[['raw_text','de']].apply(lambda x:  x[1] + "\nNow complete the following example-" + "\ninput: " +x[0] + '\noutput: ' , axis=1)
        loader.test_df_id['text'] = id_te_df[['raw_text', 'de']].apply(lambda x: x[1] + "\nNow complete the following example-" + "\ninput: "+x[0] +'\noutput: ', axis=1)
    # Tokenize dataset
    id_ds, id_tokenized_ds, ood_ds, ood_tokenized_ds = loader.set_data_for_training_semeval(
        t5_exp.tokenize_function_inputs)

    if config.mode == 'train':
        # Train model
        for _ in range(1):
            t5_exp.train_retriever(id_tr_df, id_te_df, definition)
            de = t5_exp.select_examples(list(id_tr_df['raw_text']), k=config.k)
            for i in range(len(de)):
                de[i] = definition + '\n'+de[i]
            id_tr_df['de'] = de
            de = t5_exp.select_examples(list(id_te_df['raw_text']), k=config.k)
            for i in range(len(de)):
                de[i] = definition + '\n'+de[i]
            id_te_df['de'] = de
            loader = create_dataset(id_tr_df, id_te_df, ood_tr_df, ood_te_df, config, id_te_df, ood_te_df, instruct_handler, bos_instruction_id)
            if config.task == 'atsc':
                loader.train_df_id['text'] = loader.train_df_id[['raw_text', 'de']].apply(
                    lambda x: x[1] + "\nNow complete the following example-" + "\ninput: " + x[0] + '\noutput: ',
                    axis=1)
                loader.test_df_id['text'] = loader.test_df_id[['raw_text', 'de']].apply(
                    lambda x: x[1] + "\nNow complete the following example-" + "\ninput: " + x[0] + '\noutput: ',
                    axis=1)
            else:
                loader.train_df_id['text'] = id_tr_df[['raw_text', 'de']].apply(
                    lambda x: x[1] + "\nNow complete the following example-" + "\ninput: " + x[0] + '\noutput: ', axis=1)
                loader.test_df_id['text'] = id_te_df[['raw_text', 'de']].apply(
                    lambda x: x[1] + "\nNow complete the following example-" + "\ninput: " + x[0] + '\noutput: ', axis=1)
            # Tokenize dataset
            id_ds, id_tokenized_ds, ood_ds, ood_tokenized_ds = loader.set_data_for_training_semeval(t5_exp.tokenize_function_inputs)
            model_trainer = t5_exp.train(id_tokenized_ds, **training_args)

        print('Model saved at: ', model_out_path)
    elif config.mode == 'eval':
        # Get prediction labels
        print('Model loaded from: ', model_checkpoint)
        if id_tokenized_ds.get("train") is not None:

            id_tr_pred_labels = t5_exp.get_labels(tokenized_dataset=id_tokenized_ds, sample_set='train',
                                                  batch_size=config.per_device_eval_batch_size,
                                                  max_length=512)#config.max_token_length)
            id_tr_df = pd.DataFrame(id_ds['train'])[['text', 'labels']]
            id_tr_df['labels'] = id_tr_df['labels'].apply(lambda x: x.strip())
            id_tr_df['pred_labels'] = id_tr_pred_labels
            id_tr_df.to_csv(os.path.join(config.output_path, f'{config.experiment_name}_id_train.csv'), index=False)
            print('*****Train Metrics*****')
            precision, recall, f1, accuracy = t5_exp.get_metrics(id_tr_df['labels'], id_tr_pred_labels,task =config.task)
            print('Precision: ', precision)
            print('Recall: ', recall)
            print('F1-Score: ', f1)
            if config.task == 'atsc':
                print('Accuracy: ', accuracy)

        if id_tokenized_ds.get("test") is not None:
            id_te_pred_labels = t5_exp.get_labels(tokenized_dataset=id_tokenized_ds, sample_set='test',
                                                      batch_size=config.per_device_eval_batch_size, max_length=512)#config.max_token_length)
            id_te_df = pd.DataFrame(id_ds['test'])[['text', 'labels']]
            id_te_df['labels'] = id_te_df['labels'].apply(lambda x: x.strip())
            id_te_df['pred_labels'] = id_te_pred_labels
            id_te_df.to_csv(os.path.join(config.output_path, f'{config.experiment_name}_id_test.csv'), index=False)
            print('*****Test Metrics*****')

            precision, recall, f1, accuracy = t5_exp.get_metrics(id_te_df['labels'], id_te_pred_labels,task =config.task)
            print('Precision: ', precision)
            print('Recall: ', recall)
            print('F1-Score: ', f1)
            if config.task == 'atsc':
                print('Accuracy: ', accuracy)

        if ood_tokenized_ds.get("train") is not None:
            ood_tr_pred_labels = t5_exp.get_labels(tokenized_dataset=ood_tokenized_ds, sample_set='train',
                                                   batch_size=config.per_device_eval_batch_size,
                                                   max_length=config.max_token_length)
            ood_tr_df = pd.DataFrame(ood_ds['train'])[['text', 'labels']]
            ood_tr_df['labels'] = ood_tr_df['labels'].apply(lambda x: x.strip())
            ood_tr_df['pred_labels'] = ood_tr_pred_labels
            ood_tr_df.to_csv(os.path.join(config.output_path, f'{config.experiment_name}_ood_train.csv'), index=False)
            print('*****Train Metrics - OOD*****')

            precision, recall, f1, accuracy = t5_exp.get_metrics(id_tr_df['labels'], id_tr_pred_labels,task =config.task)
            print('Precision: ', precision)
            print('Recall: ', precision)
            print('F1-Score: ', precision)
            if config.task == 'atsc':
                print('Accuracy: ', accuracy)

        if ood_tokenized_ds.get("test") is not None:
            ood_te_pred_labels = t5_exp.get_labels(tokenized_dataset=ood_tokenized_ds, sample_set='test',
                                                   batch_size=config.per_device_eval_batch_size,
                                                   max_length=config.max_token_length)
            ood_te_df = pd.DataFrame(ood_ds['test'])[['text', 'labels']]
            ood_te_df['labels'] = ood_te_df['labels'].apply(lambda x: x.strip())
            ood_te_df['pred_labels'] = ood_te_pred_labels
            ood_te_df.to_csv(os.path.join(config.output_path, f'{config.experiment_name}_ood_test.csv'), index=False)
            print('*****Test Metrics - OOD*****')
            precision, recall, f1, accuracy = t5_exp.get_metrics(id_te_df['labels'], id_te_pred_labels,task =config.task)
            print('Precision: ', precision)
            print('Recall: ', precision)
            print('F1-Score: ', precision)
            if config.task == 'atsc':
                print('Accuracy: ', accuracy)
else:
    print('Model loaded from: ', model_checkpoint)
    model_input = bos_instruction_id + config.test_input + eos_instruction
    input_ids = t5_exp.tokenizer(model_input, return_tensors="pt").input_ids
    outputs = t5_exp.model.generate(input_ids, max_length=512)#config.max_token_length)
    print('Model output: ', t5_exp.tokenizer.decode(outputs[0], skip_special_tokens=True))
