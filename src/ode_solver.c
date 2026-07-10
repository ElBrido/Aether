#include "aether_core.h"
#include <stdlib.h>
#include <math.h>

ODEFunc* ode_func_create(int hidden_dim, int input_dim) {
    ODEFunc* f = (ODEFunc*)malloc(sizeof(ODEFunc));
    f->hidden_dim = hidden_dim;
    f->input_dim = input_dim;

    f->W_hh = tensor_create(hidden_dim, hidden_dim);
    f->W_xh = tensor_create(hidden_dim, input_dim);
    f->bias = tensor_create(hidden_dim, 1);

    f->buf_Whh_h = tensor_create(hidden_dim, 1);
    f->buf_Wxh_x = tensor_create(hidden_dim, 1);
    f->buf_dhdt = tensor_create(hidden_dim, 1);

    tensor_randomize(f->W_hh, -0.1f, 0.1f);
    tensor_randomize(f->W_xh, -0.1f, 0.1f);
    tensor_zero(f->bias);

    return f;
}

void ode_func_free(ODEFunc* f) {
    if (!f) return;
    tensor_free(f->W_hh);
    tensor_free(f->W_xh);
    tensor_free(f->bias);
    tensor_free(f->buf_Whh_h);
    tensor_free(f->buf_Wxh_x);
    tensor_free(f->buf_dhdt);
    free(f);
}

void ode_func_forward(const ODEFunc *f, const Tensor *h, float t, const Tensor *x, Tensor *dhdt) {
    // Unused param 't' here, but reserved for time-varying ODEs
    (void)t;

    // dh/dt = tanh(W_hh*h + W_xh*x + bias)
    tensor_matmul(f->buf_Whh_h, f->W_hh, (Tensor*)h);
    tensor_matmul(f->buf_Wxh_x, f->W_xh, (Tensor*)x);

    tensor_add(dhdt, f->buf_Whh_h, f->buf_Wxh_x);
    tensor_add(dhdt, dhdt, f->bias);

    // Apply tanh
    int size = dhdt->rows * dhdt->cols;
    for (int i = 0; i < size; i++) {
        dhdt->data[i] = tanhf(dhdt->data[i]);
    }
}

// Integrator RK4
void rk4_step(const ODEFunc *f, Tensor *h, const Tensor *x, float dt) {
    int size = h->rows * h->cols;

    Tensor* k1 = tensor_create(h->rows, h->cols);
    Tensor* k2 = tensor_create(h->rows, h->cols);
    Tensor* k3 = tensor_create(h->rows, h->cols);
    Tensor* k4 = tensor_create(h->rows, h->cols);
    Tensor* h_temp = tensor_create(h->rows, h->cols);

    // k1 = f(h, t)
    ode_func_forward(f, h, 0.0f, x, k1);

    // k2 = f(h + k1*dt/2, t + dt/2)
    tensor_copy(h_temp, k1);
    tensor_scale(h_temp, dt / 2.0f);
    tensor_add(h_temp, h_temp, h);
    ode_func_forward(f, h_temp, dt / 2.0f, x, k2);

    // k3 = f(h + k2*dt/2, t + dt/2)
    tensor_copy(h_temp, k2);
    tensor_scale(h_temp, dt / 2.0f);
    tensor_add(h_temp, h_temp, h);
    ode_func_forward(f, h_temp, dt / 2.0f, x, k3);

    // k4 = f(h + k3*dt, t + dt)
    tensor_copy(h_temp, k3);
    tensor_scale(h_temp, dt);
    tensor_add(h_temp, h_temp, h);
    ode_func_forward(f, h_temp, dt, x, k4);

    // h = h + dt/6 * (k1 + 2k2 + 2k3 + k4)
    for (int i = 0; i < size; i++) {
        h->data[i] += (dt / 6.0f) * (k1->data[i] + 2.0f * k2->data[i] + 2.0f * k3->data[i] + k4->data[i]);
    }

    tensor_free(k1);
    tensor_free(k2);
    tensor_free(k3);
    tensor_free(k4);
    tensor_free(h_temp);
}
