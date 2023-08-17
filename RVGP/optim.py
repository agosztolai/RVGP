#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from tqdm import trange

# def optimize_GPR(model, train_steps):
#     adam_opt = tf.optimizers.Adam()
#     adam_opt.minimize(loss=model.training_loss, var_list=model.trainable_variables)

#     t = trange(train_steps - 1)
#     for step in t:
#         adam_opt.minimize(model.training_loss, var_list=model.trainable_variables)
#         if step % 200 == 0:
#             t.set_postfix({'likelihood': -model.training_loss().numpy()})