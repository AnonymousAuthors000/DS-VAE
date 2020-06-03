import os
import wandb, os, sys
from argparse import Namespace
import numpy as np
import pandas as pd
import argparse

import tensorflow as tf
import tensorflow_hub as hub
from disentanglement_lib.config import reproduce
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.visualize import visualize_model
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.data.ground_truth import cars3d
from disentanglement_lib.data.ground_truth import norb

from data_settings import get_dataset, get_image
from data_helpers import DataProvider
from classifier_settings import get_classifier
from optims_lr import get_optimizers
from graph_functions import cgan_graph, enc_graph, mi_penalty_graph, infogan_penalty_graph, cgan_graph_twovae_dlib
from misc import *

import gin.tf
from sklearn.metrics import log_loss
import keras
from keras import metrics
import tensorflow.keras.backend as K

from scipy.misc import imsave as ims
from scipy.misc import imread as imr

import shutil
import cv2

os.environ["DISENTANGLEMENT_LIB_DATA"]="/hdd_c/data/disentanglement_lib"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu)

parser = argparse.ArgumentParser()
parser.add_argument("--id", help="run_id", default='uyhvojef', type=str)
parser.add_argument("--dataset", help="dataset", default='norb', type=str)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--project_name", default='norb-tcvae-twovae-vary-beta-c10-z10', type=str)
parser.add_argument("--chkpt_path", type=str)
parser.add_argument("--dlib_model_path", type=str)
args = parser.parse_args()


# wandb path
with open('../source_tf/run_data/wandb_api_key.txt', 'r') as f:
    key = f.read().split('\n')[0]
os.environ['WANDB_API_KEY'] = key
api = wandb.Api()

run_id = args.id
project_name = args.project_name
run_path_list = list(api.runs("../../results/"+project_name))

path = project_name + '/'+ run_id
results_path = '../../results/'+path
chkpt_path = args.chkpt_path 
dlib_model_path = args.dlib_model_path

for (i, run) in enumerate(run_path_list):
    # Run settings
    settings_dict = dict.fromkeys(run.config.keys())
    settings_dict.pop('_wandb')
    for key in settings_dict.keys():
        if type(run.config[key]) is dict:
            settings_dict[key] = run.config[key]['value']
        else:
            settings_dict[key] = run.config[key]
    settings_dict['run_id'] = run.id
    settings_dict['run_state'] = run.state
    if i==0:
        df = pd.DataFrame(settings_dict, index=[0])
    else:
        tmp = pd.DataFrame(settings_dict, index=[i])
        df = pd.concat((df, tmp))


run = api.run("../../results/" + path)
settings_dict = dict.fromkeys(run.config.keys())
settings_dict.pop('_wandb')

for key in settings_dict.keys():
    if type(run.config[key]) is dict:
        settings_dict[key] = run.config[key]['value']
    else:
        settings_dict[key] = run.config[key]
        
settings_dict['dlib_train_new_model'] = False
settings_dict['dlib_model_path'] = dlib_model_path
settings_dict['dataroot'] = dlib_model_path

settings = Namespace(**settings_dict)
sys.path.insert(0, '../source_tf/utils')

dlib = 'dlib' in settings.dataname
if settings.dlib_train_new_model:
    dlib_path = os.path.join(results_dir, "dlib")
    dlib_overwrite = True
    data_dir = os.path.join(dlib_path, "model", "tfhub")
elif dlib and not settings.dlib_train_new_model:
    data_dir = settings.dlib_model_path
else:
    pass

if settings.showplot or settings.saveplot:
    import matplotlib

    if not settings.showplot:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

# get ground truth factors
if args.dataset == 'cars3d':
    dta = cars3d.Cars3D()
    factors = get_cars_factors()
elif args.dataset == 'norb':
    dta = norb.SmallNORB()
    factors = get_norb_factors()
else:
    raise Error

with hub.eval_function_for_module(dlib_model_path) as f:
    # Save reconstructions.
    inputs = dta.images
    if inputs.ndim < 4:
        inputs = np.expand_dims(inputs,3)
    
    targets = f(
        dict(images=inputs), signature="reconstructions",
        as_dict=True)["images"]

if settings.dataname == 'faces' or settings.dataname == 'faces2' or settings.dataname == 'planes' or settings.dataname == 'cars' or settings.dataname == 'chairs' or settings.dataname == 'dlib_cars3d' or settings.dataname == 'dlib_faces3d':        
    input_shape = [64, 64, 3]
elif settings.dataname == 'dlib_smallnorb':
    input_shape = [64, 64, 1]
else:
    input_shape = list(tr_data_loader.inputs[0].shape)
print('--- Created Dataset ---')

#####################################################
###################### Models #######################
#####################################################
if not dlib: simvae_model = get_classifier(settings, 'simvae')
gen_model = get_classifier(settings, 'gen') #Decoder for the second VAE
residual_enc_model = get_classifier(settings, 'residual_enc') #Encoder for the second VAE
if settings.add_encoder: enc_model = get_classifier(settings, 'enc')
if settings.add_mi_penalty: mi_disc_model = get_classifier(settings, 'mi_disc')
print('--- Created Models ---')

######################################################
############### Learning Rate and Optimizer ##########
######################################################
optimizer_gen = get_optimizers(settings, 'gen')
if settings.add_encoder: optimizer_enc = get_optimizers(settings, 'enc')
if settings.add_mi_penalty: optimizer_mi_disc = get_optimizers(settings, 'mi_disc')
if settings.add_infogan_penalty: optimizer_infogan = get_optimizers(settings, 'infogan_penalty')
print('--- Created Optimizers ---')

#######################################################
##################### Create Graph ####################
#######################################################
batchsize = settings.batchsize
if not dlib: latent_code_dim = len(settings.latent_code_indices)
latent_noise_dim = settings.latent_noise_dim
tdw_img = tf.placeholder(tf.float32, [batchsize] + input_shape, name="tdw_images")
if not dlib:
    latent_code_input = tf.placeholder(tf.float32, [batchsize] + [latent_code_dim], name="code")
else:
    E_enc = tf.placeholder(tf.float32, [batchsize] + input_shape, name="dlib_images") #y
    E_enc_randY = tf.placeholder(tf.float32, [batchsize] + input_shape, name="dlib_images_randY") # randY
latent_noise_input = tf.placeholder(tf.float32, [batchsize] + [latent_noise_dim], name="noise")


G_dec_sampler = gen_model(E_enc, latent_noise_input, input_shape)

z_mu, z_sigma = residual_enc_model(tdw_img, E_enc, settings.latent_noise_dim)
epsilon = tf.random_normal(tf.shape(z_mu))
latent_noise_input_ = z_mu + (z_sigma) * epsilon

G_dec = gen_model(E_enc, latent_noise_input_, input_shape, reuse=True)
G_dec_randY = gen_model(E_enc_randY, latent_noise_input_, input_shape, reuse=True)# randY

z_sigma_sq = tf.square(z_sigma)
z_log_sigma_sq = tf.log(z_sigma_sq+1e-10)
kld_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + z_log_sigma_sq 
                                       - tf.square(z_mu) 
                                       - tf.exp(z_log_sigma_sq), 1))
gloss = tf.reduce_mean(64*64*3*tf.reduce_mean(metrics.binary_crossentropy(tdw_img, G_dec))+kld_loss)

if settings.add_encoder: eloss, G_dec_e = enc_graph_simgan(settings, enc_model, simvae_model, gen_model, tdw_img, latent_code_input)
if settings.add_mi_penalty: closs, gloss_ = mi_penalty_graph(settings, enc_model, mi_disc_model, G_dec_e, latent_code_dim)
if settings.add_infogan_penalty: infogan_loss = infogan_penalty_graph(settings, zbar, latent_noise_input)

t_vars = tf.trainable_variables()
sim_vars = [var for var in t_vars if 'simvae' in var.name]
d_vars = [var for var in t_vars if 'dec' in var.name] #simgan_decoder
c_vars = [var for var in t_vars if (('disc' in var.name) & ('d_z_lin' not in var.name))]
res_enc_vars = [var for var in t_vars if 'residual_enc' in var.name]
e_vars = [var for var in t_vars if 'enc' in var.name]
class_vars = [var for var in t_vars if 'class' in var.name]
info_vars = [var for var in t_vars if 'd_z_lin' in var.name]

if not dlib: vae_optim0 = optimizer_gen.minimize(simloss, var_list=sim_vars)
vae_optim1 = optimizer_gen.minimize(gloss, var_list=d_vars+res_enc_vars)
optim_list = [vae_optim1]
eval_list = [E_enc, G_dec]
loss_dict = {'gloss': gloss}
if settings.add_encoder:
    vae_optim3 = optimizer_enc.minimize(eloss, var_list=e_vars)
    eval_list.append(G_dec_e)
    optim_list.append(vae_optim3)
    loss_dict['eloss'] = eloss
if settings.add_mi_penalty:
    vae_optim4 = optimizer_mi_disc.minimize(closs, var_list=class_vars)
    vae_optim1_ = optimizer_gen.minimize(gloss_, var_list=d_vars)
    optim_list.append(vae_optim4)
    optim_list.append(vae_optim1_)
    loss_dict['closs'] = closs
    loss_dict['gloss_'] = gloss_
if settings.add_infogan_penalty:
    vae_optim5 = optimizer_infogan.minimize(infogan_loss, var_list=[d_vars, info_vars])
    optim_list.append(vae_optim5)
    loss_dict['iloss'] = infogan_loss
print('--- Created Graph ---')

config = tf.ConfigProto()
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(d_vars+c_vars+e_vars+class_vars+info_vars)
saver.restore(sess, chkpt_path)

batch_size = 32
n_batch = int(factors.shape[0]/batch_size)

if args.dataset == 'cars3d':
    x_rand = np.zeros((n_batch*batch_size, 64, 64,3))
    x_enc = np.zeros((n_batch*batch_size, 64, 64,3))
    x_randY = np.zeros((n_batch*batch_size, 64, 64,3))
elif args.dataset == 'norb':
    x_rand = np.zeros((n_batch*batch_size, 64, 64,1))
    x_enc = np.zeros((n_batch*batch_size, 64, 64,1))
    x_randY = np.zeros((n_batch*batch_size, 64, 64,1))
else:
    raise NotImplementedError
    
    
for i in range(n_batch):
    x = inputs[i*batch_size:(i+1)*batch_size]
    y = targets[i*batch_size:(i+1)*batch_size]
    randY = np.random.permutation(y) # randY
    latent_noise_batch = np.random.randn(settings.batchsize, settings.latent_noise_dim).astype(np.float32)
    
    latent_noise_batch = np.random.randn(settings.batchsize, settings.latent_noise_dim).astype(np.float32)
    feed_dict = {tdw_img: x, latent_noise_input: latent_noise_batch, E_enc: y, E_enc_randY: randY} # randY
    sdata, sdata1, sdata2, sdata3, sdata4 = sess.run([G_dec_sampler, G_dec, E_enc, tdw_img, G_dec_randY],feed_dict = feed_dict)

    if args.dataset == 'cars3d':
        assert sdata.shape == (batch_size, 64, 64, 3)
    elif args.dataset == 'norb':
        assert sdata.shape == (batch_size, 64, 64, 1)
    else:
        raise NotImplementedError
    # sdata
    x_rand[i*batch_size:(i+1)*batch_size] = sdata
    x_enc[i*batch_size:(i+1)*batch_size] = sdata1
    x_randY[i*batch_size:(i+1)*batch_size] = sdata4

x_baseline =targets[:n_batch*batch_size]
c=factors[:n_batch*batch_size]
np.savez_compressed('../../results/eval_output/{}.npz'.format(run_id), c=factors[:n_batch*batch_size], x_baseline =targets[:n_batch*batch_size], x_rand=x_rand, x_enc=x_enc, x_randY=x_randY)
            


def get_eval_res(uid):   
    data = np.load('../../results/eval_output/{}.npz'.format(uid))
    c = data['c']
    x_rand=data['x_rand']
    x_enc=data['x_enc']
    x_randY=data['x_randY']
    x_baseline_logits=data['x_baseline']
    x_baseline=sigmoid(x_baseline_logits)

    model_dir = dlib_model_path[:-6]
    output_directory = '../../results/eval_output/{}/'.format(uid)
    config = '/hdd_c/data/disentanglement_lib/disentanglement_lib/config/unsupervised_study_v1/postprocess_configs/mean.gin'
    study = reproduce.STUDIES['unsupervised_study_v1']

    random_state = np.random.RandomState(0)
    postprocess_config_files = sorted(study.get_postprocess_config_files())
    for config in postprocess_config_files:
        post_name = os.path.basename(config).replace(".gin", "")
        #logging.info("Extracting representation %s...", post_name)
        post_dir = os.path.join(output_directory, "postprocessed", post_name)
        postprocess_bindings = [
            "postprocess.random_seed = {}".format(random_state.randint(2**32)),
            "postprocess.name = '{}'".format(post_name)
        ]
        postprocess.postprocess_with_gin(model_dir, post_dir, True,
                                         [config], postprocess_bindings)

    post_processed_dir = post_dir+'/tfhub'
    with hub.eval_function_for_module(post_processed_dir) as f:
        # Save reconstructions.
        inputs = dta.images
        if inputs.ndim < 4:
            inputs = np.expand_dims(inputs,3)
        inputs = inputs[:c.shape[0]]
        assert inputs.shape == x_baseline.shape    
        inputs_c = f(
            dict(images=inputs), signature="representation",
           as_dict=True)["default"]
        baseline_c = f(
            dict(images=x_baseline), signature="representation",
            as_dict=True)["default"]
        x_rand_c = f(
            dict(images=x_rand), signature="representation",
            as_dict=True)["default"]
        x_enc_c = f(
            dict(images=x_enc), signature="representation",
            as_dict=True)["default"]
        x_randY_c = f(
            dict(images=x_randY), signature="representation",
            as_dict=True)["default"]
        
    eval_bindings = [
      "evaluation.random_seed = {}".format(random_state.randint(2**32)),
      "evaluation.name = 'MI'"
    ]
    gin_config_files = ['/hdd_c/data/disentanglement_lib/disentanglement_lib/config/unsupervised_study_v1/metric_configs/mig.gin']

    gin.parse_config_files_and_bindings(gin_config_files, eval_bindings)

    def compute_mi_matrix(mus_train, ys_train, need_discretized_1=False, need_discretized_2=False):
      score_dict = {}
      if need_discretized_1:
          mus_train = utils.make_discretizer(mus_train)
      if need_discretized_2:
          ys_train = utils.make_discretizer(ys_train)
      m = utils.discrete_mutual_info(mus_train, ys_train)
      assert m.shape[0] == mus_train.shape[0]
      assert m.shape[1] == ys_train.shape[0]
      # m is [num_latents, num_factors]
      entropy = utils.discrete_entropy(ys_train)

      return m, entropy
    # compute MI matrix
    x_rand_mi_matrix, x_rand_entropy = compute_mi_matrix(np.transpose(x_rand_c), np.transpose(inputs_c), True, True)
    x_enc_mi_matrix, x_enc_entropy = compute_mi_matrix(np.transpose(x_enc_c), np.transpose(inputs_c), True, True)
    baseline_mi_matrix, baseline_entropy = compute_mi_matrix(np.transpose(baseline_c), np.transpose(inputs_c), True, True)
    x_randY_mi_matrix, x_randY_entropy=compute_mi_matrix(np.transpose(x_randY_c), np.transpose(inputs_c), True, True)
    
    x_enc_mi_matrix_gd, x_enc_entropy_gd = compute_mi_matrix(np.transpose(x_enc_c), np.transpose(c), True, False)
    baseline_mi_matrix_gd, baseline_entropy_gd = compute_mi_matrix(np.transpose(baseline_c), np.transpose(c), True, False)
    
    # compute MI and MIG
    x_enc_mi_average=(np.trace(np.divide(x_enc_mi_matrix, baseline_entropy))/float(baseline_mi_matrix.shape[0]))
    x_enc_m = np.divide(x_enc_mi_matrix_gd, baseline_entropy_gd)
    sorted_x_enc_m = np.sort(x_enc_m, axis=0)[::-1]
    x_enc_MIG=(np.mean(sorted_x_enc_m[0,:]-sorted_x_enc_m[1,:]))
   
    x_rand_mi_average=(np.trace(np.divide(x_rand_mi_matrix, baseline_entropy))/float(baseline_mi_matrix.shape[0]))

    randY_mi_average=(np.trace(np.divide(x_randY_mi_matrix, x_randY_entropy))/float(x_randY_mi_matrix.shape[0]))

    baseline_mi_average=(np.trace(np.divide(baseline_mi_matrix, baseline_entropy))/float(baseline_mi_matrix.shape[0]))
    baseline_m = baseline_mi_matrix_gd
    sorted_baseline_m = np.sort(baseline_m, axis=0)[::-1]
    baseline_MIG=(np.mean(np.divide(sorted_baseline_m[0,:]-sorted_baseline_m[1,:], baseline_entropy_gd)))


    def get_fid_with_uid(uid):
        convert2image_path = '../../results/eval_output/converted_images/'
        originaldata_path = '../../dataset/{}'.format(args.dataset)
        data = np.load('../../results/eval_output/{}.npz'.format(uid))
        path2inceptionnet = '../../inception'
        c = data['c']
        x_rand=data['x_rand']
        x_enc=data['x_enc']
        x_baseline_logits=data['x_baseline']
        x_baseline=sigmoid(x_baseline_logits)
        fid_list = get_fid_from_array([x_baseline,x_rand,x_enc], convert2image_path, originaldata_path, path2inceptionnet)
        return fid_list
    
    fid_list = get_fid_with_uid(args.id)
    
    print('Evaluation results:')
    print('Beta-TCVAE FID: {}'.format(fid_list[0]))
    print('DS-VAE FID (Random Y): {}'.format(fid_list[1]))
    print('DS-VAE FID: {}'.format(fid_list[2]))
    print('DS-VAE MIG: {}'.format(x_enc_MIG))
    print('Beta-TCVAE MIG: {}'.format(baseline_MIG))
    print('DS-VAE MI: {}'.format(x_enc_mi_average))
    print('DS-VAE MI (Random Z): {}'.format(x_rand_mi_average))
    print('DS-VAE MI (Random Y): {}'.format(randY_mi_average))
    print('Beta-TCVAE MI: {}'.format(baseline_mi_average))
 

get_eval_res(args.id) 