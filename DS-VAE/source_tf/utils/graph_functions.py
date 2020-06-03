import tensorflow as tf
import numpy as np
from keras import metrics
slim = tf.contrib.slim

def cgan_graph_twovae_dlib(settings, input_shape, gen_model, enc_model, tdw_img, E_enc, latent_noise_input):

    batchsize = settings.batchsize
    latent_noise_dim = settings.latent_noise_dim

    G_dec_sampler = gen_model(E_enc, latent_noise_input, input_shape)

    z_mu, z_sigma = enc_model(tdw_img, E_enc, settings.latent_noise_dim)
    epsilon = tf.random_normal(tf.shape(z_mu))
    latent_noise_input_ = z_mu + (z_sigma) * epsilon

    G_dec = gen_model(E_enc, latent_noise_input_, input_shape, reuse=True)

    z_sigma_sq = tf.square(z_sigma)
    z_log_sigma_sq = tf.log(z_sigma_sq+1e-10)
    kld_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + z_log_sigma_sq 
                                           - tf.square(z_mu) 
                                           - tf.exp(z_log_sigma_sq), 1))
    #Gen Loss
    gloss = tf.reduce_mean(64*64*3*tf.reduce_mean(metrics.binary_crossentropy(tdw_img, G_dec))+kld_loss)
    
    return gloss, G_dec, G_dec_sampler

def cgan_graph_simgan_dlib(settings, input_shape, gen_model, disc_model, tdw_img, E_enc, latent_noise_input):

    batchsize = settings.batchsize
    latent_noise_dim = settings.latent_noise_dim

    #Decoder
    G_dec = gen_model(E_enc, latent_noise_input, input_shape)

    #Discriminator 
    DL,xin = disc_model(tdw_img, E_enc)
    GL,xin_gen = disc_model(G_dec, E_enc, reuse=True)

    #Disc Loss
    dr = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=DL,labels=tf.ones_like(DL)))
    df = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=GL,labels=tf.zeros_like(GL)))
    dloss = (dr + df)/2.

    gloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=GL,labels=tf.ones_like(GL)))

    return gloss, dloss, G_dec

def cgan_graph_simgan(settings, input_shape, simvae_model, gen_model, disc_model, tdw_img, latent_code_input, latent_noise_input):

    batchsize = settings.batchsize
    latent_code_dim = len(settings.latent_code_indices)
    latent_noise_dim = settings.latent_noise_dim

    #Encoder
    E_enc = simvae_model(latent_code_input, input_shape)

    #Decoder
    G_dec = gen_model(E_enc, latent_noise_input, input_shape)

    #Discriminator 
    DL,xin = disc_model(tdw_img, E_enc)
    GL,xin_gen = disc_model(G_dec, E_enc, reuse=True)


    #Disc Loss
    dr = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=DL,labels=tf.ones_like(DL)))
    df = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=GL,labels=tf.zeros_like(GL)))
    dloss = (dr + df)/2.

    gloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=GL,labels=tf.ones_like(GL)))

    simloss = tf.reduce_mean(metrics.mean_squared_error(tdw_img, E_enc))

    return simloss, gloss, dloss, G_dec, E_enc

def enc_graph_simgan(settings, enc_model, simvae_model, gen_model, tdw_img, latent_code_input):
    z_dim = latent_code_input.shape[1] + settings.latent_noise_dim
    input_shape = tdw_img.shape[1:]
    c_dim = latent_code_input.shape[1]
    
    z_mean_enc, _ = enc_model(tdw_img, z_dim, i=1)

    E_enc_e = simvae_model(z_mean_enc[:,:c_dim], input_shape, reuse=True)
    G_dec_e = gen_model(E_enc_e, z_mean_enc[:, c_dim:], input_shape, reuse=True)

    if settings.enc_loss_name=='reconstr_ce':
        reconstr_loss2_pix = 64 * 64 * 1 * metrics.binary_crossentropy(G_dec_e, tdw_img)
        eloss = tf.reduce_mean(reconstr_loss2_pix)
    elif settings.enc_loss_name=='reconstr_mse':
        reconstr_loss2_pix = metrics.mean_squared_error(G_dec_e, tdw_img)
        eloss = tf.reduce_mean(reconstr_loss2_pix)
    elif settings.enc_loss_name=='supervised_mse':
        supervised_mse_loss = metrics.mean_squared_error(z_mu, latent_code_input)
        eloss = tf.reduce_mean(supervised_mse_loss) 
    else:
        raise NotImplementedError

    return eloss, G_dec_e


def cgan_graph(settings, input_shape, gen_model, disc_model, tdw_img, latent_code_input, latent_noise_input):

    batchsize = settings.batchsize
    latent_code_dim = len(settings.latent_code_indices)
    latent_noise_dim = settings.latent_noise_dim

    z_input_ = tf.concat([latent_code_input,latent_noise_input], 1)

    if settings.gen_model_name == 'adaIn':
        G_dec = gen_model(latent_code_input, latent_noise_input, input_shape)
    else:
        G_dec = gen_model(z_input_, input_shape)
        
    if settings.add_infogan_penalty:
        DL,xin, _ = disc_model(tdw_img,latent_code_input, settings.latent_noise_dim)
        GL,xin_gen, zbar = disc_model(G_dec,latent_code_input, settings.latent_noise_dim, reuse=True)
    else:
        DL,xin = disc_model(tdw_img,latent_code_input)
        GL,xin_gen = disc_model(G_dec,latent_code_input,reuse=True)
        zbar=None

    dr = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=DL,labels=tf.ones_like(DL)))
    df = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=GL,labels=tf.zeros_like(GL)))
    dloss = (dr + df)/2.

    gloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=GL,labels=tf.ones_like(GL)))

    return gloss, dloss, tdw_img, z_input_, G_dec, zbar

def enc_graph(settings, enc_model, gen_model, tdw_img, latent_code_input):
    z_dim = latent_code_input.shape[1] + settings.latent_noise_dim
    input_shape = tdw_img.shape[1:]
    c_dim = latent_code_input.shape[1]
    
    z_mean_enc, _ = enc_model(tdw_img, z_dim, i=1)
    if settings.gen_model_name=='adaIn':
        G_dec_e = gen_model(z_mean_enc[:,:c_dim], z_mean_enc[:, c_dim:], input_shape, reuse=True)
    else:
        G_dec_e = gen_model(z_mean_enc, input_shape, reuse=True)
    z_mu = tf.gather(z_mean_enc, settings.latent_code_indices, axis=1)

    if settings.enc_loss_name=='reconstr_ce':
        reconstr_loss2_pix = 64 * 64 * 1 * metrics.binary_crossentropy(G_dec_e, tdw_img)
        eloss = tf.reduce_mean(reconstr_loss2_pix)
    elif settings.enc_loss_name=='reconstr_mse':
        reconstr_loss2_pix = metrics.mean_squared_error(G_dec_e, tdw_img)
        eloss = tf.reduce_mean(reconstr_loss2_pix)
    elif settings.enc_loss_name=='supervised_mse':
        supervised_mse_loss = metrics.mean_squared_error(z_mu, latent_code_input)
        eloss = tf.reduce_mean(supervised_mse_loss) 
    else:
        raise NotImplementedError

    return eloss, G_dec_e

def mi_penalty_graph(settings, enc_model, mi_disc_model, G_dec_e, latent_code_dim):
    z_dim = latent_code_dim + settings.latent_noise_dim
    
    joint_sample, _ = enc_model(G_dec_e, z_dim, i=1, reuse=True)
    marg_sample = tf.concat([joint_sample[:,:latent_code_dim], tf.gather(joint_sample[:,latent_code_dim:], tf.random.shuffle(tf.range(tf.shape(joint_sample[:,latent_code_dim:])[0])))], axis=1)
    joint, fj = mi_disc_model(joint_sample)
    marg, fm = mi_disc_model(marg_sample,reuse=True)

    cr = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=joint,labels=tf.ones_like(joint)))
    cf = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=marg,labels=tf.zeros_like(marg)))
    closs = settings.mi_penalty_weight*(cr + cf)/2.
    gloss_ = settings.mi_penalty_weight*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=joint,labels=tf.zeros_like(joint)))

    if settings.add_moments_mi_penalty:
        data_moments = tf.reduce_mean(fm, axis = 0)
        sample_moments = tf.reduce_mean(fj, axis = 0)
        gloss_ += tf.reduce_mean(tf.square(data_moments-sample_moments))

    return closs, gloss_

def mi_penalty_graph_prior(settings, enc_model, mi_disc_model, G_dec_e, z_input_):
    z_dim = z_input_.shape[1]
    
    joint_sample, _ = enc_model(G_dec_e, z_dim, i=1, reuse=True)
    joint, fj = mi_disc_model(joint_sample)
    marg, fm = mi_disc_model(z_input_,reuse=True)

    cr = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=joint,labels=tf.ones_like(joint)))
    cf = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=marg,labels=tf.zeros_like(marg)))
    closs = (cr + cf)/2.
    gloss_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=joint,labels=tf.zeros_like(joint)))

    if settings.add_moments_mi_penalty:
        data_moments = tf.reduce_mean(fm, axis = 0)
        sample_moments = tf.reduce_mean(fj, axis = 0)
        gloss_ += tf.reduce_mean(tf.square(data_moments-sample_moments))

    return closs, gloss_

def infogan_penalty_graph(settings, zbar, latent_noise_input):
    infogan_loss = settings.infogan_penalty_weight*tf.reduce_mean(metrics.mean_squared_error(latent_noise_input, zbar))
    return infogan_loss
