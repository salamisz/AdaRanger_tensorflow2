from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras
from keras.src import ops


@tf.function(jit_compile=True)
def centralized_gradient(x):
    #in tensorflow te gradient layout for conv layer is as follows: kernel_height, kernel_width, output_channels, input_channels
    grad_len = len(x.shape)
    if grad_len > 1:
        return x - tf.math.reduce_mean(input_tensor=x, axis=list(range(grad_len-1)), keepdims=True)
    else:
        return x

@tf.function(jit_compile=True)
def custom_softplus(x, beta = 50.0):
    return 1.0 / beta * tf.math.log(1.0 + tf.math.exp(beta * x))

@tf.function(jit_compile=True)
def normalize_gradient(x, epsilon, use_channels=False):
    """  use stdev to normalize gradients """
    #for mutitask networks
    size = len(x.shape)
    # print(f"size = {size}")

    if (size > 1) and use_channels:
        s = tf.math.reduce_std(input_tensor=x, axis=list(range(size - 1)), keepdim=True) + epsilon
        # print(f"s = {s}")
        x = ops.divide(x, s)

    elif tf.experimental.numpy.size(x) > 2:
        s = ops.std(x) + epsilon
        x = ops.divide(x, s)
    return x


class TFRanger(keras.src.optimizers.Optimizer):

    def __init__(
            self,
            max_iterations,
            learning_rate=0.001,
            beta_0 = 0.9,
            beta_1 = 0.9,
            beta_2 = 0.999,
            weight_decay = None,
            stable_decay=1e-11,
            epsilon=1e-8,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            name: str = "TFRanger",
            agc_clipping_value = 0.1,
            agc_eps = 1e-3,
            #good defaults
            # learning_rate=0.001,
            # beta_0=0.9,
            # beta_1=0.9,
            # beta_2=0.999,
            # weight_decay=None,
            # stable_decay=1e-8,
            # epsilon=1e-8,
            # clipnorm=None,
            # clipvalue=None,
            # global_clipnorm=None,
            # use_ema=False,
            # ema_momentum=0.99,
            # ema_overwrite_frequency=None,
            # name: str = "TFRanger",
            # agc_clipping_value = 0.1,
            # agc_eps=1e-3,
            #use_agc=False,#this parameters does not actually do anything it's just a reminder what components are active
           # gc_loc=False,#this parameters does not actually do anything it's just a reminder what components are active
           # warmup=True,#this parameters does not actually do anything it's just a reminder what components are active
            #warmdown=True,#this parameters does not actually do anything it's just a reminder what components are active
            warmup_proportion = 0.25,
            warmdown_proportion = 0.25,
            **kwargs):
        super().__init__(learning_rate=learning_rate, name=name, weight_decay=weight_decay, clipnorm=clipnorm, clipvalue=clipvalue,global_clipnorm=global_clipnorm,
                        use_ema=use_ema, ema_momentum=ema_momentum, ema_overwrite_frequency=ema_overwrite_frequency, **kwargs)
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.agc_clipping_value = agc_clipping_value
        self.agc_eps = agc_eps
        self.max_iterations = max_iterations
        self.warmdown_proportion = warmdown_proportion
        self.warmup_proportion = warmup_proportion
        self.stable_decay = stable_decay


    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._momentums = []
        self._velocities = []
        self._neg_m = []
        self._var_list_m = []
        self._slow = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="momentum"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="velocity"
                )
            )
            self._neg_m.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="neg_m"
                )
            )
            self._var_list_m.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="var_list_m"
                )
            )


    @tf.function(jit_compile=True)
    def unit_norm(self, x):
        if len(x.shape) <= 1:  # Scalars and vectors
            axis = 0
        elif len(x.shape) in [2, 3]:  # Linear layers of shape IO or multihead linear
            axis = -1
        elif len(x.shape) == 4:  # Conv kernels of shape HWIO
            axis = [-1,-2]
        else:
            axis = tuple([x for x in range(1, len(x.shape))])
        return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(x), axis=axis, keepdims=True))

    @tf.function(jit_compile=True)
    def agc(self, original_grad, original_var, agc_eps, clip_factor):
            grad = original_grad
            var = original_var
            norm_var = self.unit_norm(var)
            norm_grad = self.unit_norm(grad)
            normalized_grad = clip_factor * (tf.math.maximum(norm_var, agc_eps) / norm_grad) * grad
            new_grad = tf.where(norm_grad / tf.math.maximum(norm_var, agc_eps) > clip_factor, normalized_grad, grad)
            #print(clipped_grad.shape)
            return new_grad


    @tf.function(jit_compile=True)
    def leaning_rate_schedule(self, local_step, max_steps, warmup_proportion, beta_2, warmdown_proportion, lr):
        #the first part of this equation might be useless
        #warmup = ops.maximum(((1.0 - beta_2) / 2.0) * local_step, local_step / (max_steps * warmup_proportion))#the division by 2 in the first part might be an uneccesary operation
        #warmup = local_step / (max_steps * warmup_proportion)
        warmup = (max_steps - local_step) / (max_steps * (1 - warmup_proportion))
        warmdown = (max_steps - local_step) / (max_steps * warmdown_proportion)

        #return ops.minimum(1.0, tf.math.minimum(warmup, warmdown)) * lr
        #instead of a constant 1.0 try 1 + 0.1 * sin(local_step)
        return tf.where(local_step < max_steps*warmup_proportion, warmup, ops.minimum(1, warmdown)) * lr

    def update_step(self, gradient, variable, learning_rate):
        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        agc_eps = ops.cast(self.agc_eps, variable.dtype)
        agc_clipping_value = ops.cast(self.agc_clipping_value, variable.dtype)
        max_steps = ops.cast(self.max_iterations, variable.dtype)
        warmup_proportion = ops.cast(self.warmup_proportion, variable.dtype)
        warmdown_proportion = ops.cast(self.warmdown_proportion, variable.dtype)
        local_step = ops.cast(self.iterations + 1, variable.dtype)

        beta_1_power = ops.power(
            ops.cast(self.beta_1, variable.dtype), local_step
        )
        beta_2_power = ops.power(
            ops.cast(self.beta_2, variable.dtype), local_step
        )
        m = self._momentums[self._get_variable_index(variable)]
        v = self._velocities[self._get_variable_index(variable)]
        neg_m = self._neg_m[self._get_variable_index(variable)]
        var_list_m = self._var_list_m[self._get_variable_index(variable)]

        lr_t = self.leaning_rate_schedule(local_step, max_steps, warmup_proportion, self.beta_2, warmdown_proportion, lr)


        gradient = self.agc(gradient, variable, agc_eps, agc_clipping_value)  # uncomment for adaptive gradient clipping
        gradient = centralized_gradient(gradient)  # uncomment for gradient centralization. seems to have no impact on improving results.
        m_t = ops.square(self.beta_1) * neg_m + (1.0 - ops.square(beta_1_power)) * gradient
        m_corr_t = ((1.0 + self.beta_0) * m_t - self.beta_0 * m) / (1.0 - beta_1_power)
        self.assign(m, m_t)
        self.assign(neg_m, tf.where(local_step // 2 == local_step / 2, neg_m, m))

        v_t = self.beta_2 * v + (1.0 - self.beta_2) * ops.square(gradient - m_t) + self.epsilon
        self.assign(v, v_t)
        self.assign(var_list_m, ops.maximum(v_t, var_list_m))
        v_corr_t = var_list_m / (1.0 - beta_2_power)

        #since learning rate is not constant it might be necassary to adjust the stable_decay value accordingly
        decay = (-1 * (lr_t / tf.math.sqrt(tf.math.reduce_mean(v_corr_t)) + self.epsilon)) * self.stable_decay * variable
        update = m_corr_t / (ops.sqrt(tf.math.square(1 + self.beta_0) + ops.square(self.beta_0)) * (ops.sqrt(v_corr_t) + self.epsilon))
        #weight decay is usually multiplied by learning rate before applying it to the variable.
        var_t = variable - lr_t * update - decay
        self.assign(variable, var_t)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta_0": self.beta_0,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "agc_eps": self.agc_eps,
                "max_iterations": self.max_iterations,
                "agc_clipping_value": self.agc_clipping_value,
                "warmdown_proportion": self.warmdown_proportion,
                "warmup_proportion": self.warmup_proportion,
            }
        )
        return config
