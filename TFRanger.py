from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

@tf.function
def centralized_gradient(x, gc_conv_only=False):
    grad_len = len(x.shape)
    if gc_conv_only:
        if grad_len > 3:
            x = x - tf.math.reduce_mean(input_tensor=x ,axis=list(range(grad_len-1)), keepdims=True)
    else:
        if grad_len > 1:
            x = x - tf.math.reduce_mean(input_tensor=x,axis=list(range(grad_len-1)), keepdims=True)
    return x
@tf.function
def normalize_gradient(x, epsilon ,use_channels=False):
    """  use stdev to normalize gradients """
    size = len(x.shape)
    # print(f"size = {size}")

    if (size > 1) and use_channels:
        s = tf.math.reduce_std(input_tensor=x, axis=list(range(size - 1)), keepdim=True) + epsilon
        # print(f"s = {s}")
        x = tf.math.divide(x, s)

    elif tf.experimental.numpy.size(x) > 2:
        s = tf.math.reduce_std(x) + epsilon
        x = tf.math.divide(x, s)
    return x

class TFRanger(tf.keras.optimizers.Optimizer):

    def __init__(
        self,
        learning_rate=0.001,
        beta_0 = 0.9,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-14,
        weight_decay=0.0,
        min_lr=0.0,
        name="TFRanger",
        agc_clipping_value=1e-2,
        agc_eps=1e-3,
        use_agc=True,
        gc_loc = True,
        use_gn = False,
        warmup=False,
        max_iterations = 0,
        warmup_proportion = 0.22,
        warmdown_proportion = 0.28,
        **kwargs):
        super().__init__(name, **kwargs)

        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta_0", beta_0)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("weight_decay", weight_decay)
        self._set_hyper("min_lr", min_lr)
        self._set_hyper("agc_clipping_value", agc_clipping_value)
        self._set_hyper("ags_epsilon", agc_eps)
        self._set_hyper("max_iterations", max_iterations)
        self._set_hyper("warmdown_proportion", warmdown_proportion)
        self._set_hyper("warmup_proportion", warmup_proportion)
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        self._has_weight_decay = weight_decay != 0.0
        self.agc_active = use_agc
        self.use_gn = use_gn
        self.gc_loc = gc_loc
        self.warmup = warmup

    @tf.function
    def unit_norm(self, x):
        if len(x.shape) <= 1:  # Scalars and vectors
            axis = None
            keepdims = False
        elif len(x.shape) in [2, 3]:  # Linear layers of shape IO or multihead linear
            axis = 0
            keepdims = True
        elif len(x.shape) == 4:  # Conv kernels of shape HWIO
            axis = [0, 1, 2, ]
            keepdims = True
        else:
            raise ValueError(f"Got a parameter with shape not in [1, 2, 4]! {x}")
        return tf.math.reduce_sum(tf.math.abs(x) ** 2.0, axis=axis, keepdims=keepdims) ** 0.5 #0.5 is because {(1 / ord)} ord is 2 in this case

    @tf.function
    def agc(self, grad, var,agc_eps, clip_factor):
        p_norm = self.unit_norm(var)
        max_norm = tf.math.maximum(p_norm, agc_eps) * clip_factor
        grad_norm = self.unit_norm(grad)
        clipped_grad = grad * (max_norm / tf.math.maximum(grad_norm, 1e-6))
        new_grad = tf.where(grad_norm < max_norm, grad, clipped_grad)
        grad += new_grad
        return grad

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
        for var in var_list:
            self.add_slot(var, "v")
        for var in var_list:
            self.add_slot(var, "neg_m")
        for var in var_list:
            self.add_slot(var, "v_max")


    def set_weights(self, weights):
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[: len(params)]
        super().set_weights(weights)

    def _resource_apply_dense(self, grad, var):
        #notes p.grad is grad
        #p is var and p.data is also var

        decay = 0
        var_dtype = var.dtype.base_dtype
        wd_t = self._get_hyper("weight_decay", var_dtype)
        lr_t = self._decayed_lr(var_dtype)
        v = self.get_slot(var, "v")
        m = self.get_slot(var, "m")
        v_max = self.get_slot(var, "v_max")
        neg_m = self.get_slot(var, "neg_m")
        neg_m = neg_m.assign(m)
        beta_0 = self._get_hyper("beta_0", var_dtype)
        beta_1 = self._get_hyper("beta_1", var_dtype)
        beta_2 = self._get_hyper("beta_2", var_dtype)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.math.pow(beta_1, local_step)
        beta_2_power = tf.math.pow(beta_2, local_step)
        agc_eps = self._get_hyper("ags_epsilon", var_dtype)
        agc_clipping_value = self._get_hyper("agc_clipping_value", var_dtype)
        if self.warmup:
            max_steps = self._get_hyper("max_iterations", var_dtype)
            warmup_proportion = self._get_hyper("warmup_proportion", var_dtype)
            warmdown_proportion = self._get_hyper("warmdown_proportion", var_dtype)

            lr_warmup = tf.math.maximum(tf.math.divide(1-beta_2, 2) * local_step, local_step / (max_steps * warmup_proportion))
            lr_ex = (max_steps - local_step) / (max_steps * warmdown_proportion)

            lr_n = tf.math.minimum(lr_warmup, lr_ex)
            lr_n = tf.math.minimum(1.0, lr_n)
            lr_t *= lr_n

            lr_t = tf.math.minimum(1.0,
                                   tf.math.maximum(tf.math.divide(1-beta_2, 2) * local_step, local_step / (max_steps * warmup_proportion)),
                                   ((max_steps - local_step) / (max_steps * warmdown_proportion))) * lr_t

        if self.agc_active:
            grad = self.agc(grad, var, agc_eps, agc_clipping_value)

        if self.gc_loc:
            grad = centralized_gradient(grad)
        if self.use_gn:
            grad = normalize_gradient(grad, epsilon_t)

        m_t =  m.assign(tf.math.square(beta_1) * m + (1.0 - tf.math.square(beta_1)) * grad, use_locking=self._use_locking)
        m_corr_t = ((1 + beta_0)*m_t - beta_0*neg_m) / (1 - beta_1_power)

        v_t = v.assign(
            beta_2 * v + (1.0 - beta_2) * tf.math.square(grad - m_t) + epsilon_t,
            use_locking=self._use_locking,
        )

        v_max = tf.math.maximum(v_t, v_max)
        v_corr_t = v_max / (1 - beta_2_power)

        var_t = m_corr_t / (tf.math.sqrt(tf.math.square(1 + beta_0) + tf.math.square(beta_0)) * (tf.math.sqrt(v_corr_t) + epsilon_t))

        if self._has_weight_decay:
            decay = (-1 *(lr_t/tf.math.sqrt(tf.math.reduce_mean(v_corr_t)))) * wd_t * var

        m.assign(tf.where(local_step % 2 == 0, m, neg_m))
        neg_m.assign(tf.where(local_step % 2 == 0, neg_m, m))

        var_update = var.assign(var - lr_t*var_t - lr_t*decay, use_locking=self._use_locking)

        updates = [var_update, m_t, v_t]
        return tf.group(*updates)

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "decay": self._serialize_hyperparameter("decay"),
                "weight_decay": self._serialize_hyperparameter("weight_decay"),
                "sma_threshold": self._serialize_hyperparameter("sma_threshold"),
                "epsilon": self.epsilon,
                "total_steps": self._serialize_hyperparameter("total_steps"),
                "warmup_proportion": self._serialize_hyperparameter(
                    "warmup_proportion"
                ),
                "min_lr": self._serialize_hyperparameter("min_lr"),
            }
        )
        return config