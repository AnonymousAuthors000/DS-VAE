import sys

def get_classifier(settings, model_use):

    if settings.dlaas:
        sys.path.insert(0, 'source_tf/models')
    else:
        sys.path.insert(0, '../source_tf/models')

    from basic_models import decoder, encoder, classifier, discriminator, discriminator_infogan
    from adaptive_instnorm_models import decoder_adaIN
    from simgan_models import simvae, simgan_decoder, simgan_discriminator, residual_encoder
    
    if model_use=='gen':
        if settings.gen_model_name=='adaIn':
            model = decoder_adaIN
        elif settings.gen_model_name=='simgan':
            model = simgan_decoder
        else:
            model = decoder
    elif model_use=='disc':
        if settings.add_infogan_penalty:
            model = discriminator_infogan
        elif settings.gen_model_name=='simgan':
            model = simgan_discriminator
        else:
            model = discriminator
    elif model_use=='residual_enc':
        model = residual_encoder
    elif model_use=='mi_disc':
        model = classifier
    elif model_use=='enc':
        model = encoder
    elif model_use=='simvae':
        model = simvae
    else:
        raise NotImplementedError

    return model
