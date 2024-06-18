import os
import numpy as np
from modeling.neuralmodeling.exp.Experiments import Experiment
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler

def sample_stride(spec):
    patch_len = spec.config.patch_len
    return np.random.randint(1, patch_len + 1)
search_space = {
    "model_name" : 'Linear',
    "random_seed" : 2021,
    "seq_len" : 10, 
    "label_len" : 0,
    "pred_len" : 6,
    "e_layers" : ray.tune.randint(1, 3),
    "d_layers" : 1,
    "d_model" : tune.choice([128, 256, 512]),
    "n_heads": tune.choice([2, 4, 8]),
    "d_ff" : tune.choice([128, 256, 512]),
    "learning_rate" : 0.01,
    "is_training" : 1,
    "root_path" : os.path.join(os.getcwd(), 'datasets', 'covid') + os.sep,
    "data_path" : 'national_covid_weekly.csv',
    "model_id" : 'covid_19',
    "data" : 'custom',
    "features" : 'M',
    "factor" : 5,
    "enc_in" : 1,
    "dec_in" : 1,
    "c_out" :1,
    "des" : 'Exp',
    "itr" : 1,
    "checkpoints" : os.path.join(os.getcwd(), 'checkpoints') + os.sep,
    "distil" : False,
    "activation" : 'gelu',
    "output_attention" : False,
    "train_epochs" : 100,
    "patience" : 6,
    "batch_size" : tune.choice([16]),
    "loss" : 'mse',
    "vis_type" : 'Original',
    "lradj" : tune.choice(["constant", "type1", "type2", "type3", "type5", "type6"]),
    "use_amp" : False,
    "gpu" : 0,
    "use_multi_gpu" : True,
    "devices" : '0',
    "use_gpu" : True,
    "moving_avg" : 25,
    "num_workers" : 0,
    "pct_start" : 0.3,
    "target" : 'new_deaths',
    "embed" : 'timeF',
    "individual" : True,
    "dropout" : 0.0,
    "fc_dropout" : 0.0,
    "head_dropout" : 0.0,
    "horizon_plot" : tune.sample_from(lambda spec: spec.config.pred_len - 1),
    "patch_len": tune.randint(7, 21),
    "stride": tune.sample_from(sample_stride),
    "patch_len_dec": 1,
    "levels" : 3,
    "iterations" : 6,
    "tf_ratio" : tune.uniform(0.0, 1.0),
    "use_tf" : False,
    "device_ids" : ['0'],
    "device_ids" : [0],
    "gpu" : 0,
    "use_temporal": False,
    "embed_type": 0,
    "freq": 'h',
    "revin": True,
    "dataset": 'Covid-19',
    "first_diff": False,
    "nlags": 10,
    "retraining_freq": 1,
    "padding_patch": 'end',
    "affine": 0,
    "subtract_last": 0,
    "decomposition": 0,
    "kernel_size": 25,
    "version": 'Fourier',
    "mode_select": 'random',
    "modes": 64,
    "rnn_layer_dim": tune.choice([1,2,3]),
    "task_name": 'long_term_forecast',
    "horizons": [1, 2, 3, 4, 5, 6],
    "validation": 'In-sample Validation',
    "use_original_data": False,
    "in_sample": True
}

def objective(args):
    Exp = Experiment(args)
    Exp.train()
trainable_with_resources = tune.with_resources(objective, {"cpu": 6, "gpu": 0.5})

scheduler = AsyncHyperBandScheduler(grace_period = 5, max_t = 100)

tuner = tune.Tuner( 
    trainable_with_resources,
    tune_config=tune.TuneConfig(
        metric="vali_loss",
        mode="min",
        scheduler = scheduler,
        num_samples = 20
    ),
    param_space=search_space,
)
results = tuner.fit()

print("Best configs are:", results.get_best_result().config)

best_results = results.get_best_result().config
best_results['patience'] = best_results['iterations']
best_results = results.get_best_result().config
Exp = Experiment(best_results)
setting = 'id{}_m{}_d{}_f{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_dff{}_do{}_lr{}'.format(
        best_results["model_id"], best_results["model_name"], best_results["data"], best_results["features"], best_results["seq_len"], best_results["label_len"],
        best_results["pred_len"], best_results["d_model"], best_results["n_heads"], best_results["e_layers"], best_results["d_layers"], best_results["d_ff"],
        best_results["dropout"], best_results["learning_rate"])
Exp.train(setting)
trues_original, preds_original = Exp.test(setting)

def sample_stride(spec):
    patch_len = spec.config.patch_len
    return np.random.randint(1, patch_len + 1)
search_space = {
    "model_name" : 'DLinear',
    "random_seed" : 2021,
    "seq_len" : 10, 
    "label_len" : 0,
    "pred_len" : 6,
    "e_layers" : ray.tune.randint(1, 3),
    "d_layers" : 1,
    "d_model" : tune.choice([128, 256, 512]),
    "n_heads": tune.choice([2, 4, 8]),
    "d_ff" : tune.choice([128, 256, 512]),
    "learning_rate" : 0.01,
    "is_training" : 1,
    "root_path" : os.path.join(os.getcwd(), 'datasets', 'covid') + os.sep,
    "data_path" : 'national_covid_weekly.csv',
    "model_id" : 'covid_19',
    "data" : 'custom',
    "features" : 'M',
    "factor" : 5,
    "enc_in" : 1,
    "dec_in" : 1,
    "c_out" : 1,
    "des" : 'Exp',
    "itr" : 1,
    "checkpoints" : os.path.join(os.getcwd(), 'checkpoints') + os.sep,
    "distil" : False,
    "activation" : 'gelu',
    "output_attention" : False,
    "train_epochs" : 100,
    "patience" : 6,
    "batch_size" : tune.choice([16]),
    "loss" : 'mse',
    "vis_type" : 'Original',
    "lradj" : tune.choice(["constant", "type1", "type2", "type3", "type5", "type6", "type7"]),
    "use_amp" : False,
    "gpu" : 0,
    "use_multi_gpu" : True,
    "devices" : '0',
    "use_gpu" : True,
    "moving_avg" : 25,
    "num_workers" : 0,
    "pct_start" : 0.3,
    "target" : 'new_deaths',
    "embed" : 'timeF',
    "individual" : True,
    "dropout" : 0.0,
    "fc_dropout" : 0.0,
    "head_dropout" : 0.0,
    "horizon_plot" : 0,
    "patch_len": tune.randint(7, 21),
    "stride": tune.sample_from(sample_stride),
    "patch_len_dec": 1,
    "levels" : 3,
    "iterations" : 6,
    "tf_ratio" : tune.uniform(0.0, 1.0),
    "use_tf" : False,
    "device_ids" : ['0'],
    "device_ids" : [0],
    "gpu" : 0,
    "use_temporal": False,
    "embed_type": 0,
    "freq": 'h',
    "revin": True,
    "dataset": 'Covid-19',
    "first_diff": False,
    "nlags": 10,
    "retraining_freq": 1,
    "padding_patch": 'end',
    "affine": 0,
    "subtract_last": 0,
    "decomposition": 0,
    "kernel_size": 25,
    "version": 'Fourier',
    "mode_select": 'random',
    "modes": 64,
    "rnn_layer_dim": tune.choice([1,2,3]),
    "task_name": 'long_term_forecast',
    "horizons": [1, 2, 3, 4, 5, 6],
    "validation": 'In-sample Validation',
    "use_original_data": False,
    "in_sample": True
}

def objective(args):
    Exp = Experiment(args)
    Exp.train()
trainable_with_resources = tune.with_resources(objective, {"cpu": 6, "gpu": 0.5})

scheduler = AsyncHyperBandScheduler(max_t = 100, grace_period = 5)

tuner = tune.Tuner( 
    trainable_with_resources,
    tune_config=tune.TuneConfig(
        metric="vali_loss",
        mode="min",
        scheduler = scheduler,
        num_samples = 20
    ),
    param_space=search_space,
)
results = tuner.fit()

print("Best configs are:", results.get_best_result().config)

best_results = results.get_best_result().config
best_results['patience'] = best_results['iterations']
best_results = results.get_best_result().config
Exp = Experiment(best_results)
setting = 'id{}_m{}_d{}_f{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_dff{}_do{}_lr{}'.format(
        best_results["model_id"], best_results["model_name"], best_results["data"], best_results["features"], best_results["seq_len"], best_results["label_len"],
        best_results["pred_len"], best_results["d_model"], best_results["n_heads"], best_results["e_layers"], best_results["d_layers"], best_results["d_ff"],
        best_results["dropout"], best_results["learning_rate"])
Exp.train(setting)
trues_original, preds_original = Exp.test(setting)


def sample_stride(spec):
    patch_len = spec.config.patch_len
    return np.random.randint(1, patch_len + 1)
search_space = {
    "model_name" : 'FFNN',
    "random_seed" : 2021,
    "seq_len" : 10, 
    "label_len" : 0,
    "pred_len" : 6,
    "e_layers" : ray.tune.randint(1, 3),
    "d_layers" : 1,
    "d_model" : tune.choice([128, 256, 512]),
    "n_heads": tune.choice([2, 4, 8]),
    "d_ff" : tune.choice([128, 256, 512]),
    "learning_rate" : 0.0001,
    "is_training" : 1,
    "root_path" : os.path.join(os.getcwd(), 'datasets', 'covid') + os.sep,
    "data_path" : 'national_covid_weekly.csv',
    "model_id" : 'covid_19',
    "data" : 'custom',
    "features" : 'M',
    "factor" : 5,
    "enc_in" : 1,
    "dec_in" : 1,
    "c_out" : 1,
    "des" : 'Exp',
    "itr" : 1,
    "checkpoints" : os.path.join(os.getcwd(), 'checkpoints') + os.sep,
    "distil" : False,
    "activation" : 'gelu',
    "output_attention" : False,
    "train_epochs" : 100,
    "patience" : 6,
    "batch_size" : tune.choice([16]),
    "loss" : 'mse',
    "vis_type" : 'Original',
    "lradj" : tune.choice(["constant", "type1", "type2", "type3", "type5", "type6", "type7"]),
    "use_amp" : False,
    "gpu" : 0,
    "use_multi_gpu" : True,
    "devices" : '0',
    "use_gpu" : True,
    "moving_avg" : 25,
    "num_workers" : 0,
    "pct_start" : 0.3,
    "target" : 'new_deaths',
    "embed" : 'timeF',
    "individual" : True,
    "dropout" : 0.0,
    "fc_dropout" : 0.0,
    "head_dropout" : 0.0,
    "horizon_plot" : tune.sample_from(lambda spec: spec.config.pred_len - 1),
    "patch_len": tune.randint(7, 21),
    "stride": tune.sample_from(sample_stride),
    "patch_len_dec": 1,
    "levels" : 3,
    "iterations" : 6,
    "tf_ratio" : tune.uniform(0.0, 1.0),
    "use_tf" : False,
    "device_ids" : ['0'],
    "device_ids" : [0],
    "gpu" : 0,
    "use_temporal": False,
    "embed_type": 0,
    "freq": 'h',
    "revin": True,
    "dataset": 'Covid-19',
    "first_diff": False,
    "nlags": 10,
    "retraining_freq": 1,
    "padding_patch": 'end',
    "affine": 0,
    "subtract_last": 0,
    "decomposition": 0,
    "kernel_size": 25,
    "version": 'Fourier',
    "mode_select": 'random',
    "modes": 64,
    "rnn_layer_dim": tune.choice([1,2,3]),
    "task_name": 'long_term_forecast',
    "horizons": [1, 2, 3, 4, 5, 6],
    "validation": 'In-sample Validation',
    "use_original_data": False,
    "in_sample": True
}

def objective(args):
    Exp = Experiment(args)
    Exp.train()
trainable_with_resources = tune.with_resources(objective, {"cpu": 6, "gpu": 0.5})

scheduler = AsyncHyperBandScheduler(max_t = 100, grace_period = 5)

tuner = tune.Tuner( 
    trainable_with_resources,
    tune_config=tune.TuneConfig(
        metric="vali_loss",
        mode="min",
        scheduler = scheduler,
        num_samples = 20
    ),
    param_space=search_space,
)
results = tuner.fit()

print("Best configs are:", results.get_best_result().config)

best_results = results.get_best_result().config
best_results['patience'] = best_results['iterations']
best_results = results.get_best_result().config
Exp = Experiment(best_results)
setting = 'id{}_m{}_d{}_f{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_dff{}_do{}_lr{}'.format(
        best_results["model_id"], best_results["model_name"], best_results["data"], best_results["features"], best_results["seq_len"], best_results["label_len"],
        best_results["pred_len"], best_results["d_model"], best_results["n_heads"], best_results["e_layers"], best_results["d_layers"], best_results["d_ff"],
        best_results["dropout"], best_results["learning_rate"])
Exp.train(setting)
trues_original, preds_original = Exp.test(setting)


def sample_stride(spec):
    patch_len = spec.config.patch_len
    return np.random.randint(1, patch_len + 1)
search_space = {
    "model_name" : 'LSTM_FFNN',
    "random_seed" : 2021,
    "seq_len" : 10, 
    "label_len" : 0,
    "pred_len" : 6,
    "e_layers" : ray.tune.randint(1, 3),
    "d_layers" : 1,
    "d_model" : tune.choice([128, 256, 512]),
    "n_heads": tune.choice([2, 4, 8]),
    "d_ff" : tune.choice([128, 256, 512]),
    "learning_rate" : 0.0001,
    "is_training" : 1,
    "root_path" : os.path.join(os.getcwd(), 'datasets', 'covid') + os.sep,
    "data_path" : 'national_covid_weekly.csv',
    "model_id" : 'covid_19',
    "data" : 'custom',
    "features" : 'M',
    "factor" : 5,
    "enc_in" : 1,
    "dec_in" : 1,
    "c_out" : 1,
    "des" : 'Exp',
    "itr" : 1,
    "checkpoints" : os.path.join(os.getcwd(), 'checkpoints') + os.sep,
    "distil" : False,
    "activation" : 'gelu',
    "output_attention" : False,
    "train_epochs" : 100,
    "patience" : 6,
    "batch_size" : tune.choice([16]),
    "loss" : 'mse',
    "vis_type" : 'Original',
    "lradj" : tune.choice(["constant", "type1", "type2", "type3", "type5", "type6", "type7"]),
    "use_amp" : False,
    "gpu" : 0,
    "use_multi_gpu" : True,
    "devices" : '0',
    "use_gpu" : True,
    "moving_avg" : 25,
    "num_workers" : 0,
    "pct_start" : 0.3,
    "target" : 'new_deaths',
    "embed" : 'timeF',
    "individual" : True,
    "dropout" : 0.0,
    "fc_dropout" : 0.0,
    "head_dropout" : 0.0,
    "horizon_plot" : tune.sample_from(lambda spec: spec.config.pred_len - 1),
    "patch_len": tune.randint(7, 21),
    "stride": tune.sample_from(sample_stride),
    "patch_len_dec": 1,
    "levels" : 3,
    "iterations" : 6,
    "tf_ratio" : tune.uniform(0.0, 1.0),
    "use_tf" : False,
    "device_ids" : ['0'],
    "device_ids" : [0],
    "gpu" : 0,
    "use_temporal": False,
    "embed_type": 0,
    "freq": 'h',
    "revin": True,
    "dataset": 'Covid-19',
    "first_diff": False,
    "nlags": 10,
    "retraining_freq": 1,
    "padding_patch": 'end',
    "affine": 0,
    "subtract_last": 0,
    "decomposition": 0,
    "kernel_size": 25,
    "version": 'Fourier',
    "mode_select": 'random',
    "modes": 64,
    "rnn_layer_dim": tune.choice([1,2,3]),
    "task_name": 'long_term_forecast',
    "horizons": [1, 2, 3, 4, 5, 6],
    "validation": 'In-sample Validation',
    "use_original_data": False,
    "in_sample": True
}

def objective(args):
    Exp = Experiment(args)
    Exp.train()
trainable_with_resources = tune.with_resources(objective, {"cpu": 6, "gpu": 0.5})

scheduler = AsyncHyperBandScheduler(max_t = 100, grace_period = 5)

tuner = tune.Tuner( 
    trainable_with_resources,
    tune_config=tune.TuneConfig(
        metric="vali_loss",
        mode="min",
        scheduler = scheduler,
        num_samples = 20
    ),
    param_space=search_space,
)
results = tuner.fit()

print("Best configs are:", results.get_best_result().config)

best_results = results.get_best_result().config
best_results['patience'] = best_results['iterations']
best_results = results.get_best_result().config
Exp = Experiment(best_results)
setting = 'id{}_m{}_d{}_f{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_dff{}_do{}_lr{}'.format(
        best_results["model_id"], best_results["model_name"], best_results["data"], best_results["features"], best_results["seq_len"], best_results["label_len"],
        best_results["pred_len"], best_results["d_model"], best_results["n_heads"], best_results["e_layers"], best_results["d_layers"], best_results["d_ff"],
        best_results["dropout"], best_results["learning_rate"])
Exp.train(setting)
trues_original, preds_original = Exp.test(setting)


def sample_stride(spec):
    patch_len = spec.config.patch_len
    return np.random.randint(1, patch_len + 1)
search_space = {
    "model_name" : 'LSTM_Seq2Seq',
    "random_seed" : 2021,
    "seq_len" : 10, 
    "label_len" : 0,
    "pred_len" : 6,
    "e_layers" : ray.tune.randint(1, 3),
    "d_layers" : 1,
    "d_model" : tune.choice([128, 256, 512]),
    "n_heads": tune.choice([2, 4, 8]),
    "d_ff" : tune.choice([128, 256, 512]),
    "learning_rate" : 0.0001,
    "is_training" : 1,
    "root_path" : os.path.join(os.getcwd(), 'datasets', 'covid') + os.sep,
    "data_path" : 'national_covid_weekly.csv',
    "model_id" : 'covid_19',
    "data" : 'custom',
    "features" : 'M',
    "factor" : 5,
    "enc_in" : 1,
    "dec_in" : 1,
    "c_out" : 1,
    "des" : 'Exp',
    "itr" : 1,
    "checkpoints" : os.path.join(os.getcwd(), 'checkpoints') + os.sep,
    "distil" : False,
    "activation" : 'gelu',
    "output_attention" : False,
    "train_epochs" : 100,
    "patience" : 6,
    "batch_size" : tune.choice([16]),
    "loss" : 'mse',
    "vis_type" : 'Original',
    "lradj" : tune.choice(["constant", "type1", "type2", "type3", "type5", "type6", "type7"]),
    "use_amp" : False,
    "gpu" : 0,
    "use_multi_gpu" : True,
    "devices" : '0',
    "use_gpu" : True,
    "moving_avg" : 25,
    "num_workers" : 0,
    "pct_start" : 0.3,
    "target" : 'new_deaths',
    "embed" : 'timeF',
    "individual" : True,
    "dropout" : 0.0,
    "fc_dropout" : 0.0,
    "head_dropout" : 0.0,
    "horizon_plot" : tune.sample_from(lambda spec: spec.config.pred_len - 1),
    "patch_len": tune.randint(7, 21),
    "stride": tune.sample_from(sample_stride),
    "patch_len_dec": 1,
    "levels" : 3,
    "iterations" : 6,
    "tf_ratio" : tune.uniform(0.0, 1.0),
    "use_tf" : False,
    "device_ids" : ['0'],
    "device_ids" : [0],
    "gpu" : 0,
    "use_temporal": False,
    "embed_type": 0,
    "freq": 'h',
    "revin": True,
    "dataset": 'Covid-19',
    "first_diff": False,
    "nlags": 10,
    "retraining_freq": 1,
    "padding_patch": 'end',
    "affine": 0,
    "subtract_last": 0,
    "decomposition": 0,
    "kernel_size": 25,
    "version": 'Fourier',
    "mode_select": 'random',
    "modes": 64,
    "rnn_layer_dim": tune.choice([1,2,3]),
    "task_name": 'long_term_forecast',
    "horizons": [1, 2, 3, 4, 5, 6],
    "validation": 'In-sample Validation',
    "use_original_data": False,
    "in_sample": True
}

def objective(args):
    Exp = Experiment(args)
    Exp.train()
trainable_with_resources = tune.with_resources(objective, {"cpu": 6, "gpu": 0.5})

scheduler = AsyncHyperBandScheduler(max_t = 100, grace_period = 5)

tuner = tune.Tuner( 
    trainable_with_resources,
    tune_config=tune.TuneConfig(
        metric="vali_loss",
        mode="min",
        scheduler = scheduler,
        num_samples = 20
    ),
    param_space=search_space,
)
results = tuner.fit()

print("Best configs are:", results.get_best_result().config)

best_results = results.get_best_result().config
best_results['patience'] = best_results['iterations']
best_results = results.get_best_result().config
Exp = Experiment(best_results)
setting = 'id{}_m{}_d{}_f{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_dff{}_do{}_lr{}'.format(
        best_results["model_id"], best_results["model_name"], best_results["data"], best_results["features"], best_results["seq_len"], best_results["label_len"],
        best_results["pred_len"], best_results["d_model"], best_results["n_heads"], best_results["e_layers"], best_results["d_layers"], best_results["d_ff"],
        best_results["dropout"], best_results["learning_rate"])
Exp.train(setting)
trues_original, preds_original = Exp.test(setting)


def sample_stride(spec):
    patch_len = spec.config.patch_len
    return np.random.randint(1, patch_len + 1)
search_space = {
    "model_name" : 'Autoformer',
    "random_seed" : 2021,
    "seq_len" : 10, 
    "label_len" : tune.sample_from(lambda spec: spec.config.seq_len//2),
    "pred_len" : 6,
    "e_layers" : ray.tune.randint(1, 3),
    "d_layers" : 1,
    "d_model" : tune.choice([128, 256, 512]),
    "n_heads": tune.choice([2, 4, 8]),
    "d_ff" : tune.choice([128, 256, 512]),
    "dropout" : 0.0,
    "fc_dropout" : 0.0,
    "head_dropout" : 0.0,
    "learning_rate" : 0.0001,
    "is_training" : 1,
    "root_path" : os.path.join(os.getcwd(), 'datasets', 'covid') + os.sep,
    "data_path" : 'national_covid_weekly.csv',
    "model_id" : 'covid_19',
    "data" : 'custom',
    "features" : 'M',
    "factor" : 2,
    "enc_in" : 1,
    "dec_in" : 1,
    "c_out" : 1,
    "des" : 'Exp',
    "itr" : 1,
    "checkpoints" : os.path.join(os.getcwd(), 'checkpoints') + os.sep,
    "distil" : False,
    "activation" : 'gelu',
    "output_attention" : False,
    "train_epochs" : 100,
    "patience" : 6,
    "batch_size" : tune.choice([16]),
    "loss" : 'mse',
    "vis_type" : 'Original',
    "lradj" : tune.choice(["constant", "type1", "type2", "type3", "type5", "type6", "type7"]),
    "use_amp" : False,
    "gpu" : 0,
    "use_multi_gpu" : True,
    "devices" : '0',
    "use_gpu" : True,
    "moving_avg" : 25,
    "num_workers" : 0,
    "pct_start" : 0.3,
    "target" : 'new_deaths',
    "embed" : 'timeF',
    "individual" : True,
    "dropout" : tune.uniform(0.0, 0.3),
    "fc_dropout" : tune.uniform(0.0, 0.3),
    "head_dropout" : tune.uniform(0.0, 0.3),
    "horizon_plot" : tune.sample_from(lambda spec: spec.config.pred_len - 1),
    "patch_len": tune.randint(7, 21),
    "stride": tune.sample_from(sample_stride),
    "patch_len_dec": 1,
    "levels" : 3,
    "iterations" : 6,
    "tf_ratio" : tune.uniform(0.0, 1.0),
    "use_tf" : False,
    "device_ids" : ['0'],
    "device_ids" : [0],
    "gpu" : 0,
    "use_temporal": False,
    "embed_type": 0,
    "freq": 'h',
    "revin": True,
    "dataset": 'Covid-19',
    "first_diff": False,
    "nlags": 10,
    "retraining_freq": 1,
    "padding_patch": 'end',
    "affine": 0,
    "subtract_last": 0,
    "decomposition": 0,
    "kernel_size": 25,
    "version": 'Fourier',
    "mode_select": 'random',
    "modes": 64,
    "rnn_layer_dim": tune.choice([1,2,3]),
    "task_name": 'long_term_forecast',
    "horizons": [1, 2, 3, 4, 5, 6],
    "validation": 'In-sample Validation',
    "use_original_data": False,
    "in_sample": True
}

def objective(args):
    Exp = Experiment(args)
    Exp.train()
trainable_with_resources = tune.with_resources(objective, {"cpu": 6, "gpu": 0.5})

scheduler = AsyncHyperBandScheduler(max_t = 100, grace_period = 5)

tuner = tune.Tuner( 
    trainable_with_resources,
    tune_config=tune.TuneConfig(
        metric="vali_loss",
        mode="min",
        scheduler = scheduler,
        num_samples = 20
    ),
    param_space=search_space,
)
results = tuner.fit()

print("Best configs are:", results.get_best_result().config)

best_results = results.get_best_result().config
best_results['patience'] = best_results['iterations']
best_results = results.get_best_result().config
Exp = Experiment(best_results)
setting = 'id{}_m{}_d{}_f{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_dff{}_do{}_lr{}'.format(
        best_results["model_id"], best_results["model_name"], best_results["data"], best_results["features"], best_results["seq_len"], best_results["label_len"],
        best_results["pred_len"], best_results["d_model"], best_results["n_heads"], best_results["e_layers"], best_results["d_layers"], best_results["d_ff"],
        best_results["dropout"], best_results["learning_rate"])
Exp.train(setting)
trues_original, preds_original = Exp.test(setting)

def sample_stride(spec):
    patch_len = spec.config.patch_len
    return np.random.randint(1, patch_len + 1)
search_space = {
    "model_name" : 'FEDformer',
    "random_seed" : 2021,
    "seq_len" : 10, 
    "label_len" : tune.sample_from(lambda spec: spec.config.seq_len//2),
    "pred_len" : 6,
    "e_layers" : ray.tune.randint(1, 3),
    "d_layers" : 1,
    "d_model" : tune.choice([128, 256, 512]),
    "n_heads": 8,
    "d_ff" : tune.choice([128, 256, 512]),
    "learning_rate" : 0.0001,
    "is_training" : 1,
    "root_path" : os.path.join(os.getcwd(), 'datasets', 'covid') + os.sep,
    "data_path" : 'national_covid_weekly.csv',
    "model_id" : 'covid_19',
    "data" : 'custom',
    "features" : 'M',
    "factor" : 5,
    "enc_in" : 1,
    "dec_in" : 1,
    "c_out" : 1,
    "des" : 'Exp',
    "itr" : 1,
    "checkpoints" : os.path.join(os.getcwd(), 'checkpoints') + os.sep,
    "distil" : False,
    "activation" : 'gelu',
    "output_attention" : False,
    "train_epochs" : 100,
    "patience" : 6,
    "batch_size" : tune.choice([16]),
    "loss" : 'mse',
    "vis_type" : 'Original',
    "lradj" : tune.choice(["constant", "type1", "type2", "type3", "type5", "type6", "type7"]),
    "use_amp" : False,
    "gpu" : 0,
    "use_multi_gpu" : True,
    "devices" : '0',
    "use_gpu" : True,
    "moving_avg" : 25,
    "num_workers" : 0,
    "pct_start" : 0.3,
    "target" : 'new_deaths',
    "embed" : 'timeF',
    "individual" : True,
    "dropout" : 0.0,
    "fc_dropout" : 0.0,
    "head_dropout" : 0.0,
    "horizon_plot" : tune.sample_from(lambda spec: spec.config.pred_len - 1),
    "patch_len": tune.randint(7, 21),
    "stride": tune.sample_from(sample_stride),
    "patch_len_dec": 1,
    "levels" : 3,
    "iterations" : 6,
    "tf_ratio" : tune.uniform(0.0, 1.0),
    "use_tf" : False,
    "device_ids" : ['0'],
    "device_ids" : [0],
    "gpu" : 0,
    "use_temporal": False,
    "embed_type": 0,
    "freq": 'h',
    "revin": True,
    "dataset": 'Covid-19',
    "first_diff": False,
    "nlags": 10,
    "retraining_freq": 1,
    "padding_patch": 'end',
    "affine": 0,
    "subtract_last": 0,
    "decomposition": 0,
    "kernel_size": 25,
    "version": 'Fourier',
    "mode_select": 'random',
    "modes": 64,
    "rnn_layer_dim": tune.choice([1,2,3]),
    "task_name": 'long_term_forecast',
    "horizons": [1, 2, 3, 4, 5, 6],
    "validation": 'In-sample Validation',
    "use_original_data": False,
    "in_sample": True
}

def objective(args):
    Exp = Experiment(args)
    Exp.train()
trainable_with_resources = tune.with_resources(objective, {"cpu": 6, "gpu": 0.5})

scheduler = AsyncHyperBandScheduler(max_t = 100, grace_period = 5)

tuner = tune.Tuner( 
    trainable_with_resources,
    tune_config=tune.TuneConfig(
        metric="vali_loss",
        mode="min",
        scheduler = scheduler,
        num_samples = 20
    ),
    param_space=search_space,
)
results = tuner.fit()

print("Best configs are:", results.get_best_result().config)

best_results = results.get_best_result().config
best_results['patience'] = best_results['iterations']
best_results = results.get_best_result().config
Exp = Experiment(best_results)
setting = 'id{}_m{}_d{}_f{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_dff{}_do{}_lr{}'.format(
        best_results["model_id"], best_results["model_name"], best_results["data"], best_results["features"], best_results["seq_len"], best_results["label_len"],
        best_results["pred_len"], best_results["d_model"], best_results["n_heads"], best_results["e_layers"], best_results["d_layers"], best_results["d_ff"],
        best_results["dropout"], best_results["learning_rate"])
Exp.train(setting)
trues_original, preds_original = Exp.test(setting)


def sample_stride(spec):
    patch_len = spec.config.patch_len
    return np.random.randint(1, patch_len + 1)
search_space = {
    "model_name" : 'PatchTST',
    "random_seed" : 2021,
    "seq_len" : 10, 
    "label_len" : 0,
    "pred_len" : 6,
    "e_layers" : ray.tune.randint(1, 3),
    "d_layers" : 1,
    "d_model" : tune.choice([128, 256, 512]),
    "n_heads": tune.choice([2, 4, 8]),
    "d_ff" : tune.choice([128, 256, 512]),
    "learning_rate" : 0.0001,
    "is_training" : 1,
    "root_path" : os.path.join(os.getcwd(), 'datasets', 'covid') + os.sep,
    "data_path" : 'national_covid_weekly.csv',
    "model_id" : 'covid_19',
    "data" : 'custom',
    "features" : 'M',
    "factor" : 5,
    "enc_in" : 1,
    "dec_in" : 1,
    "c_out" : 1,
    "des" : 'Exp',
    "itr" : 1,
    "checkpoints" : os.path.join(os.getcwd(), 'checkpoints') + os.sep,
    "distil" : False,
    "activation" : 'gelu',
    "output_attention" : False,
    "train_epochs" : 100,
    "patience" : 6,
    "batch_size" : tune.choice([16]),
    "loss" : 'mse',
    "vis_type" : 'Original',
    "lradj" : tune.choice(["constant", "type1", "type2", "type3", "type5", "type6", "type7"]),
    "use_amp" : False,
    "gpu" : 0,
    "use_multi_gpu" : True,
    "devices" : '0',
    "use_gpu" : True,
    "moving_avg" : 25,
    "num_workers" : 0,
    "pct_start" : 0.3,
    "target" : 'new_deaths',
    "embed" : 'timeF',
    "individual" : True,
    "dropout" : 0.0,
    "fc_dropout" : 0.0,
    "head_dropout" : 0.0,
    "horizon_plot" : tune.sample_from(lambda spec: spec.config.pred_len - 1),
    "patch_len": tune.sample_from(lambda spec: np.random.randint(1, spec.config.seq_len)),
    "stride": tune.sample_from(sample_stride),
    "patch_len_dec": 1,
    "levels" : 3,
    "iterations" : 6,
    "tf_ratio" : tune.uniform(0.0, 1.0),
    "use_tf" : False,
    "device_ids" : ['0'],
    "device_ids" : [0],
    "gpu" : 0,
    "use_temporal": False,
    "embed_type": 0,
    "freq": 'h',
    "revin": True,
    "dataset": 'Covid-19',
    "first_diff": False,
    "nlags": 10,
    "retraining_freq": 1,
    "padding_patch": 'end',
    "affine": 0,
    "subtract_last": 0,
    "decomposition": 0,
    "kernel_size": 25,
    "version": 'Fourier',
    "mode_select": 'random',
    "modes": 64,
    "rnn_layer_dim": tune.choice([1,2,3]),
    "task_name": 'long_term_forecast',
    "horizons": [1, 2, 3, 4, 5, 6],
    "validation": 'In-sample Validation',
    "use_original_data": False,
    "in_sample": True
}

def objective(args):
    Exp = Experiment(args)
    Exp.train()
trainable_with_resources = tune.with_resources(objective, {"cpu": 6, "gpu": 0.5})

scheduler = AsyncHyperBandScheduler(max_t = 100, grace_period = 5)

tuner = tune.Tuner( 
    trainable_with_resources,
    tune_config=tune.TuneConfig(
        metric="vali_loss",
        mode="min",
        scheduler = scheduler,
        num_samples = 20
    ),
    param_space=search_space,
)
results = tuner.fit()

print("Best configs are:", results.get_best_result().config)

best_results = results.get_best_result().config
best_results['patience'] = best_results['iterations']
best_results = results.get_best_result().config
Exp = Experiment(best_results)
setting = 'id{}_m{}_d{}_f{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_dff{}_do{}_lr{}'.format(
        best_results["model_id"], best_results["model_name"], best_results["data"], best_results["features"], best_results["seq_len"], best_results["label_len"],
        best_results["pred_len"], best_results["d_model"], best_results["n_heads"], best_results["e_layers"], best_results["d_layers"], best_results["d_ff"],
        best_results["dropout"], best_results["learning_rate"])
Exp.train(setting)
trues_original, preds_original = Exp.test(setting)


def sample_stride(spec):
    patch_len = spec.config.patch_len
    return np.random.randint(1, patch_len + 1)
search_space = {
    "model_name" : 'Koopa',
    "random_seed" : 2021,
    "seq_len" : 10, 
    "label_len" : 0,
    "pred_len" : 6,
    "e_layers" : ray.tune.randint(1, 3),
    "d_layers" : 1,
    "d_model" : tune.choice([128, 256, 512]),
    "n_heads": tune.choice([2, 4, 8]),
    "d_ff" : tune.choice([128, 256, 512]),
    "learning_rate" : 0.001,
    "is_training" : 1,
    "root_path" : os.path.join(os.getcwd(), 'datasets', 'covid') + os.sep,
    "data_path" : 'national_covid_weekly.csv',
    "model_id" : 'covid_19',
    "data" : 'custom',
    "features" : 'M',
    "factor" : 5,
    "enc_in" : 1,
    "dec_in" : 1,
    "c_out" : 1,
    "des" : 'Exp',
    "itr" : 1,
    "checkpoints" : os.path.join(os.getcwd(), 'checkpoints') + os.sep,
    "distil" : False,
    "activation" : 'gelu',
    "output_attention" : False,
    "train_epochs" : 100,
    "patience" : 6,
    "batch_size" : tune.choice([16]),
    "loss" : 'mse',
    "vis_type" : 'Original',
    "lradj" : tune.choice(["constant", "type1", "type2", "type3", "type5", "type6", "type7"]),
    "use_amp" : False,
    "gpu" : 0,
    "use_multi_gpu" : True,
    "devices" : '0',
    "use_gpu" : True,
    "moving_avg" : 25,
    "num_workers" : 0,
    "pct_start" : 0.3,
    "target" : 'new_deaths',
    "embed" : 'timeF',
    "individual" : True,
    "dropout" : 0.0,
    "fc_dropout" : 0.0,
    "head_dropout" : 0.0,
    "horizon_plot" : tune.sample_from(lambda spec: spec.config.pred_len - 1),
    "patch_len": tune.randint(7, 21),
    "stride": tune.sample_from(sample_stride),
    "patch_len_dec": 1,
    "levels" : 3,
    "iterations" : 6,
    "tf_ratio" : tune.uniform(0.0, 1.0),
    "use_tf" : False,
    "device_ids" : ['0'],
    "device_ids" : [0],
    "gpu" : 0,
    "use_temporal": False,
    "embed_type": 0,
    "freq": 'h',
    "revin": True,
    "dataset": 'Covid-19',
    "first_diff": False,
    "nlags": 10,
    "retraining_freq": 1,
    "padding_patch": 'end',
    "affine": 0,
    "subtract_last": 0,
    "decomposition": 0,
    "kernel_size": 25,
    "version": 'Fourier',
    "mode_select": 'random',
    "modes": 64,
    "rnn_layer_dim": tune.choice([1,2,3]),
    "task_name": 'long_term_forecast',
    "horizons": [1, 2, 3, 4, 5, 6],
    "validation": 'In-sample Validation',
    "use_original_data": False,
    "in_sample": True
}

def objective(args):
    Exp = Experiment(args)
    Exp.train()
trainable_with_resources = tune.with_resources(objective, {"cpu": 6, "gpu": 0.5})

scheduler = AsyncHyperBandScheduler(max_t = 100, grace_period = 5)

tuner = tune.Tuner( 
    trainable_with_resources,
    tune_config=tune.TuneConfig(
        metric="vali_loss",
        mode="min",
        scheduler = scheduler,
        num_samples = 20
    ),
    param_space=search_space,
)
results = tuner.fit()

print("Best configs are:", results.get_best_result().config)

best_results = results.get_best_result().config
best_results['patience'] = best_results['iterations']
best_results = results.get_best_result().config
Exp = Experiment(best_results)
setting = 'id{}_m{}_d{}_f{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_dff{}_do{}_lr{}'.format(
        best_results["model_id"], best_results["model_name"], best_results["data"], best_results["features"], best_results["seq_len"], best_results["label_len"],
        best_results["pred_len"], best_results["d_model"], best_results["n_heads"], best_results["e_layers"], best_results["d_layers"], best_results["d_ff"],
        best_results["dropout"], best_results["learning_rate"])
Exp.train(setting)
trues_original, preds_original = Exp.test(setting)

