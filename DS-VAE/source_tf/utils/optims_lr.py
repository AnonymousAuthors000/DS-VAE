import tensorflow as tf

def get_lr_optimname(settings, model_use):
    if model_use=='gen':
        lr = settings.gen_lr
        optimname = settings.gen_optimname
    elif model_use=='disc':
        lr = settings.disc_lr
        optimname = settings.disc_optimname
    elif model_use=='enc':
        lr = settings.enc_lr
        optimname = settings.enc_optimname
    elif model_use=='mi_disc':
        lr = settings.mi_disc_lr
        optimname = settings.mi_disc_optimname
    elif model_use=='infogan_penalty':
        lr = settings.gen_lr
        optimname = settings.gen_optimname
    else:
        raise NotImplementedError
    return lr, optimname

def get_optimizers(settings, model_use):
    lr, optimname = get_lr_optimname(settings, model_use)
    if optimname=='adam':
        optim = tf.train.AdamOptimizer(lr, beta1=settings.beta1)
    elif optimname=='sgd':
        optim = tf.train.GradientDescentOptimizer(lr)
    elif optimname=='momentum':
        optim = tf.train.MomentumOptimizer(lr, momentum=settings.momentum)
    else:
        raise NotImplementedError

    return optim
