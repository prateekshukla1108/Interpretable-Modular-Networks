#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <random>
#include <limits>
#include <chrono>

struct Hyperparameters {
    int batch_size = 64;
    int num_modules = 3;
    int input_size = 3072;
    int output_dim = 10;
    float lambda1 = 0.1f; // Initial lambda1
    float lambda3 = 0.01f;
    int num_epochs = 100;
    float lr = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    std::string train_images_path = "./data/cifar10_train_images.bin";
    std::string train_labels_path = "./data/cifar10_train_labels.bin";
    std::string test_images_path = "./data/cifar10_test_images.bin";
    std::string test_labels_path = "./data/cifar10_test_labels.bin";
};

// --- Tensor and Parameter Classes ---

class Tensor {
public:
    void* data;
    int rows, cols, size;
    bool is_half;
    Tensor(int r, int c, bool half) : rows(r), cols(c), size(r * c), is_half(half) {
        if (is_half) {
            cudaMalloc(&data, size * sizeof(__half));
        } else {
            cudaMalloc(&data, size * sizeof(float));
        }
        cudaMemset(data, 0, size * (is_half ? sizeof(__half) : sizeof(float)));
    }
    ~Tensor() { if (data) cudaFree(data); }
    void to_device(const float* host_data, cudaStream_t stream, bool normalize = false) {
        std::vector<float> temp(size);
        memcpy(temp.data(), host_data, size * sizeof(float));
        if (normalize) {
            for (int i = 0; i < size; ++i) {
                temp[i] = (temp[i] / 255.0f - 0.5f) / 0.5f;
            }
        }
        if (is_half) {
            std::vector<__half> h_data(size);
            for (int i = 0; i < size; ++i) {
                h_data[i] = __float2half(temp[i]);
            }
            cudaMemcpyAsync(data, h_data.data(), size * sizeof(__half), cudaMemcpyHostToDevice, stream);
        } else {
            cudaMemcpyAsync(data, temp.data(), size * sizeof(float), cudaMemcpyHostToDevice, stream);
        }
    }
    void to_host(float* host_data, cudaStream_t stream) const {
        if (is_half) {
            std::vector<__half> h_data(size);
            cudaMemcpyAsync(h_data.data(), data, size * sizeof(__half), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            for (int i = 0; i < size; ++i) {
                host_data[i] = __half2float(h_data[i]);
            }
        } else {
            cudaMemcpyAsync(host_data, data, size * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
    }
};

class Parameter {
public:
    Tensor data, grad, m, v;
    Parameter(int rows, int cols, bool is_half)
        : data(rows, cols, is_half), grad(rows, cols, is_half), m(rows, cols, is_half), v(rows, cols, is_half) {}
};

// --- Utility Functions ---

void update_parameter(Parameter& param, float lr, float beta1, float beta2,
                     float epsilon, cudaStream_t stream);

// --- Kernels ---

__global__ void matmul_kernel_half(const __half* A, const __half* B, __half* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum = __hfma(__half2float(A[row * k + i]), __half2float(B[i * n + col]), sum);
        }
        C[row * n + col] = __float2half(sum);
    }
}

__global__ void bias_add_kernel_half(__half* output, const __half* bias, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n) {
        int col = idx % n;
        output[idx] = __hadd(output[idx], bias[col]);
    }
}

__global__ void relu_forward_kernel_half(const __half* input, __half* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(input[idx]);
        output[idx] = __float2half(val > 0 ? val : 0.0f);
    }
}

__global__ void softmax_kernel_half(const __half* scores, __half* weights, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < n; i++) {
            float val = __half2float(scores[i]);
            if (val > max_val) max_val = val;
        }
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            float exp_val = expf(__half2float(scores[i]) - max_val);
            weights[i] = __float2half(exp_val);
            sum += exp_val;
        }
        for (int i = 0; i < n; i++) {
            weights[i] = __float2half(__half2float(weights[i]) / sum);
        }
    }
}

__global__ void alignment_loss_backward_kernel_half(const __half* attn_weights, __half* grad_attn, int n, float lambda1, float lambda3) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += __half2float(attn_weights[i]);
        }
        float mean = sum / n;
        for (int i = 0; i < n; i++) {
            float p = __half2float(attn_weights[i]);
            float grad_variance = 2.0f * lambda1 * (p - mean);
            float log_p = (p > 0.0f) ? logf(p) : -1e6f;
            float grad_entropy = -lambda3 * (log_p + 1.0f);
            grad_attn[i] = __float2half(grad_variance + grad_entropy);
        }
    }
}

__global__ void softmax_backward_kernel_half(
    const __half* weights, const __half* grad_weights, __half* grad_scores, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        float wi = __half2float(weights[idx]);
        for (int j = 0; j < n; j++) {
            float delta = (idx == j) ? 1.0f : 0.0f;
            float wj = __half2float(weights[j]);
            float gwj = __half2float(grad_weights[j]);
            sum += gwj * wj * (delta - wi);
        }
        grad_scores[idx] = __float2half(sum);
    }
}

__global__ void scale_and_add_kernel_half(
    const __half* input, __half scale, __half* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __hadd(output[idx], __hmul(input[idx], scale));
    }
}

__global__ void scale_kernel_half(const __half* a, __half scale, __half* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hmul(a[idx], scale);
    }
}

__global__ void compute_dot_product_kernel_half(
    const __half* a, const __half* b, __half* c, int n) {
    __shared__ float sum;
    if (threadIdx.x == 0) sum = 0.0f;
    __syncthreads();
    int idx = threadIdx.x;
    while (idx < n) {
        atomicAdd(&sum, __half2float(a[idx]) * __half2float(b[idx]));
        idx += blockDim.x;
    }
    __syncthreads();
    if (threadIdx.x == 0) *c = __float2half(sum);
}

__global__ void add_kernel_half(const __half* a, const __half* b, __half* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

__global__ void matmul_transB_kernel_half(
    const __half* A, const __half* B, __half* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += __half2float(A[row * k + i]) * __half2float(B[col * k + i]);
        }
        C[row * n + col] = __float2half(sum);
    }
}

__global__ void matmul_transA_kernel_half(
    const __half* A, const __half* B, __half* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += __half2float(A[i * m + row]) * __half2float(B[i * n + col]);
        }
        C[row * n + col] = __float2half(sum);
    }
}

__global__ void sum_over_batch_kernel_half(
    const __half* grad_output, __half* grad_biases, int batch_size, int output_dim) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < output_dim) {
        float sum = 0.0f;
        for (int row = 0; row < batch_size; row++) {
            sum += __half2float(grad_output[row * output_dim + col]);
        }
        grad_biases[col] = __float2half(sum);
    }
}

__global__ void relu_backward_kernel_half(
    const __half* input, const __half* grad_output, __half* grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float in = __half2float(input[idx]);
        grad_input[idx] = (in > 0) ? grad_output[idx] : __float2half(0.0f);
    }
}

__global__ void adam_update_kernel_half(
    __half* param, const __half* grad, __half* m, __half* v, int size,
    float lr, float beta1, float beta2, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float p = __half2float(param[idx]);
        float g = __half2float(grad[idx]);
        float m_val = __half2float(m[idx]);
        float v_val = __half2float(v[idx]);
        m_val = beta1 * m_val + (1 - beta1) * g;
        v_val = beta2 * v_val + (1 - beta2) * g * g;
        float m_hat = m_val / (1 - beta1); // Simplified, assumes step correction elsewhere
        float v_hat = v_val / (1 - beta2);
        p -= lr * m_hat / (sqrtf(v_hat) + epsilon);
        param[idx] = __float2half(p);
        m[idx] = __float2half(m_val);
        v[idx] = __float2half(v_val);
    }
}

__global__ void cross_entropy_loss_forward_kernel_half(const __half* predictions, const int* labels, float* loss, int batch_size, int num_classes) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float total_loss = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            int label = labels[i];
            float log_prob = -std::numeric_limits<float>::infinity();
            float max_val = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < num_classes; ++j) {
                max_val = fmaxf(max_val, __half2float(predictions[i * num_classes + j]));
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < num_classes; ++j) {
                sum_exp += expf(__half2float(predictions[i * num_classes + j]) - max_val);
            }
            log_prob = __half2float(predictions[i * num_classes + label]) - max_val - logf(sum_exp);
            total_loss -= log_prob;
        }
        *loss = total_loss / batch_size;
    }
}

__global__ void cross_entropy_loss_backward_kernel_half(const __half* predictions, const int* labels, __half* grad_predictions, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num_classes) {
        int row = idx / num_classes;
        int col = idx % num_classes;
        int label = labels[row];
        float max_val = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < num_classes; ++j) {
            max_val = fmaxf(max_val, __half2float(predictions[row * num_classes + j]));
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            sum_exp += expf(__half2float(predictions[row * num_classes + j]) - max_val);
        }
        float softmax_output = expf(__half2float(predictions[row * num_classes + col]) - max_val) / sum_exp;
        grad_predictions[idx] = __float2half(softmax_output - (col == label ? 1.0f : 0.0f));
    }
}

// --- Utility Functions Implementation ---
void update_parameter(Parameter& param, float lr, float beta1, float beta2,
                     float epsilon, cudaStream_t stream) {
    int threads = 256;
    int blocks = (param.data.size + threads - 1) / threads;
    adam_update_kernel_half<<<blocks, threads, 0, stream>>>(
        static_cast<__half*>(param.data.data),
        static_cast<const __half*>(param.grad.data),
        static_cast<__half*>(param.m.data),
        static_cast<__half*>(param.v.data),
        param.data.size, lr, beta1, beta2, epsilon
    );
}

// --- Linear Layer Class ---
class Linear {
public:
    Parameter weights, biases;
    int input_dim, output_dim;
    Linear(int in_dim, int out_dim, bool is_half)
        : input_dim(in_dim), output_dim(out_dim), weights(in_dim, out_dim, is_half), biases(1, out_dim, is_half) {
        float limit = sqrtf(6.0f / (in_dim + out_dim));
        std::vector<float> host_weights(in_dim * out_dim);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(-limit, limit);
        for (int i = 0; i < in_dim * out_dim; i++) {
            host_weights[i] = dis(gen);
        }
        weights.data.to_device(host_weights.data(), 0);
        std::vector<float> host_biases(out_dim, 0.0f);
        biases.data.to_device(host_biases.data(), 0);
    }
    void forward(const Tensor& input, Tensor& output, cudaStream_t stream) {
        dim3 blockDim(16, 16);
        dim3 gridDim((output.cols + blockDim.x - 1) / blockDim.x, (input.rows + blockDim.y - 1) / blockDim.y);
        matmul_kernel_half<<<gridDim, blockDim, 0, stream>>>(
            static_cast<const __half*>(input.data),
            static_cast<const __half*>(weights.data.data),
            static_cast<__half*>(output.data),
            input.rows, output_dim, input_dim
        );
        int total = output.rows * output.cols;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        bias_add_kernel_half<<<blocks, threads, 0, stream>>>(
            static_cast<__half*>(output.data),
            static_cast<const __half*>(biases.data.data),
            output.rows, output_dim
        );
    }
    void backward(const Tensor& input, const Tensor& grad_output,
                    Tensor& grad_input, cudaStream_t stream) {
        dim3 blockDim(16, 16);
        dim3 gridDim((input_dim + blockDim.x - 1) / blockDim.x,
                     (grad_output.rows + blockDim.y - 1) / blockDim.y);
        matmul_transB_kernel_half<<<gridDim, blockDim, 0, stream>>>(
            static_cast<const __half*>(grad_output.data),
            static_cast<const __half*>(weights.data.data),
            static_cast<__half*>(grad_input.data),
            grad_output.rows, input_dim, output_dim
        );
        gridDim = dim3((output_dim + blockDim.x - 1) / blockDim.x,
                       (input_dim + blockDim.y - 1) / blockDim.y);
        matmul_transA_kernel_half<<<gridDim, blockDim, 0, stream>>>(
            static_cast<const __half*>(input.data),
            static_cast<const __half*>(grad_output.data),
            static_cast<__half*>(weights.grad.data),
            input.rows, output_dim, input_dim
        );
        int threads = 256;
        int blocks = (biases.data.cols + threads - 1) / threads;
        sum_over_batch_kernel_half<<<blocks, threads, 0, stream>>>(
            static_cast<const __half*>(grad_output.data),
            static_cast<__half*>(biases.grad.data),
            grad_output.rows, output_dim
        );
    }
};


// --- Layer Class (Now with ReLU and Linear) ---
class Layer {
public:
    Linear linear;
    int input_dim, output_dim;
    Layer(int in_dim, int out_dim, bool is_half)
        : input_dim(in_dim), output_dim(out_dim), linear(in_dim, out_dim, is_half) {}

    void forward(const Tensor& input, Tensor& output, cudaStream_t stream) {
        Tensor linear_output(input.rows, linear.output_dim, input.is_half);
        linear.forward(input, linear_output, stream);
        int total = linear_output.rows * linear_output.cols;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        relu_forward_kernel_half<<<blocks, threads, 0, stream>>>(
            static_cast<const __half*>(linear_output.data),
            static_cast<__half*>(output.data), // Output is now separate after linear
            total
        );
         cudaMemcpyAsync(output.data, linear_output.data, total * sizeof(__half), cudaMemcpyDeviceToDevice, stream); // Copy ReLU result to output
    }
    void backward(const Tensor& input, const Tensor& grad_output,
                    Tensor& grad_input, cudaStream_t stream) {
        Tensor grad_linear_input(grad_output.rows, grad_output.cols, true);
        int threads = 256;
        int blocks = (grad_output.size + threads - 1) / threads;
        relu_backward_kernel_half<<<blocks, threads, 0, stream>>>(
            static_cast<const __half*>(input.data), // Use input for ReLU backward condition (pre-relu output)
            static_cast<const __half*>(grad_output.data), // grad_output is grad after ReLU
            static_cast<__half*>(grad_linear_input.data),
            grad_output.size
        );
        linear.backward(input, grad_linear_input, grad_input, stream);
    }
};


class Module {
public:
    Layer layer1, layer2;
    Tensor hidden;
    Module(int in_dim, int hidden_dim, int out_dim, bool is_half)
        : layer1(in_dim, hidden_dim, is_half), layer2(hidden_dim, out_dim, is_half),
          hidden(64, hidden_dim, is_half) {}
    void forward(const Tensor& input, Tensor& output, cudaStream_t stream) {
        layer1.forward(input, hidden, stream);
        layer2.forward(hidden, output, stream);
    }
     void backward(const Tensor& input, const Tensor& grad_output,
                     Tensor& grad_input, cudaStream_t stream) {
        Tensor grad_hidden(hidden.rows, hidden.cols, true);
        layer2.backward(hidden, grad_output, grad_hidden, stream);
        layer1.backward(input, grad_hidden, grad_input, stream);
    }
};

class AttentionLayer {
public:
    Parameter attention_scores;
    Tensor attention_weights;
    AttentionLayer(int num_mods, bool is_half)
        : attention_scores(1, num_mods, is_half), attention_weights(1, num_mods, is_half) {
        std::vector<float> host_scores(num_mods);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(-0.01f, 0.01f);
        for (int i = 0; i < num_mods; ++i) {
            host_scores[i] = dis(gen);
        }
        attention_scores.data.to_device(host_scores.data(), 0);
    }
    void forward(cudaStream_t stream) {
        softmax_kernel_half<<<1, 1, 0, stream>>>(
            static_cast<const __half*>(attention_scores.data.data),
            static_cast<__half*>(attention_weights.data),
            attention_scores.data.cols
        );
    }
    void backward(const Tensor& grad_attention_weights, cudaStream_t stream) {
        softmax_backward_kernel_half<<<1, attention_scores.data.cols, 0, stream>>>( // blockDim = n for shared memory in kernel
            static_cast<const __half*>(attention_weights.data),
            static_cast<const __half*>(grad_attention_weights.data),
            static_cast<__half*>(attention_scores.grad.data),
            attention_scores.data.cols
        );
    }
};

class AlignmentLoss {
public:
    float compute(const Tensor& attention_weights, cudaStream_t stream, const Hyperparameters& hp) {
        std::vector<float> h_weights(attention_weights.cols);
        attention_weights.to_host(h_weights.data(), stream);
        cudaStreamSynchronize(stream);
        float sum = 0.0f;
        for (int i = 0; i < attention_weights.cols; i++) {
            float p = h_weights[i];
            sum += p;
        }
        float mean = sum / attention_weights.cols;
        float sum_sq_diff = 0.0f;
        for (int i = 0; i < attention_weights.cols; i++) {
            float p = h_weights[i];
            float diff = p - mean;
            sum_sq_diff += diff * diff;
        }
        return hp.lambda1 * sum_sq_diff; // Modified Alignment Loss: Variance only
    }
};

class Network {
public:
    std::vector<Module> modules;
    AttentionLayer attention;
    std::vector<Tensor> module_outputs;
    Tensor predictions;
    std::vector<bool> frozen_modules;
    Network(int num_mods, const Hyperparameters& hp)
        : attention(num_mods, true), predictions(hp.batch_size, hp.output_dim, true), frozen_modules(num_mods, false) {
        for (int i = 0; i < num_mods; ++i) {
            modules.emplace_back(hp.input_size, 512, hp.output_dim, true);
            module_outputs.emplace_back(hp.batch_size, hp.output_dim, true);
            frozen_modules.push_back(false);
        }
    }
    void forward(const Tensor& input, cudaStream_t stream) {
        for (int m = 0; m < modules.size(); ++m) {
            modules[m].forward(input, module_outputs[m], stream);
        }
        attention.forward(stream);
        cudaMemsetAsync(predictions.data, 0, predictions.size * sizeof(__half), stream);
        for (int m = 0; m < modules.size(); ++m) {
            dim3 blockDim(256);
            dim3 gridDim((predictions.size + blockDim.x - 1) / blockDim.x);
            scale_and_add_kernel_half<<<gridDim, blockDim, 0, stream>>>(
                static_cast<const __half*>(module_outputs[m].data),
                static_cast<const __half*>(attention.attention_weights.data)[m],
                static_cast<__half*>(predictions.data),
                predictions.size
            );
        }
    }
    void backward(const Tensor& input, const Tensor& grad_predictions, cudaStream_t stream, const Hyperparameters& hp) {
        std::vector<Tensor> grad_module_outputs;
        Tensor grad_attention_weights_from_main(1, modules.size(), true);

        for (int m = 0; m < modules.size(); ++m) {
            grad_module_outputs.push_back(Tensor(module_outputs[m].rows, module_outputs[m].cols, true));
            dim3 blockDim(256);
            dim3 gridDim((grad_module_outputs.back().size + blockDim.x - 1) / blockDim.x);
            scale_kernel_half<<<gridDim, blockDim, 0, stream>>>(
                static_cast<const __half*>(grad_predictions.data),
                static_cast<const __half*>(attention.attention_weights.data)[m],
                static_cast<__half*>(grad_module_outputs.back().data),
                grad_module_outputs.back().size
            );
            compute_dot_product_kernel_half<<<1, 256, 0, stream>>>(
                static_cast<const __half*>(grad_predictions.data),
                static_cast<const __half*>(module_outputs[m].data),
                static_cast<__half*>(grad_attention_weights_from_main.data) + m,
                grad_predictions.size
            );
        }

        Tensor grad_from_alignment(1, modules.size(), true);
        alignment_loss_backward_kernel_half<<<1, 1, 0, stream>>>(
            static_cast<const __half*>(attention.attention_weights.data),
            static_cast<__half*>(grad_from_alignment.data),
            attention.attention_weights.cols,
            hp.lambda1, hp.lambda3
        );

        Tensor total_grad_attention_weights(1, modules.size(), true);
        dim3 blockDim_add(256);
        dim3 gridDim_add((modules.size() + blockDim_add.x - 1) / blockDim_add.x);
        add_kernel_half<<<gridDim_add, blockDim_add, 0, stream>>>(
            static_cast<const __half*>(grad_attention_weights_from_main.data),
            static_cast<const __half*>(grad_from_alignment.data),
            static_cast<__half*>(total_grad_attention_weights.data),
            modules.size()
        );

        attention.backward(total_grad_attention_weights, stream);

        for (int m = modules.size() - 1; m >= 0; --m) {
            if (!frozen_modules[m]) {
                Tensor grad_input(input.rows, input.cols, true);
                modules[m].backward(input, grad_module_outputs[m], grad_input, stream);
            }
        }
    }
    void update_parameters(float lr, float beta1, float beta2, float epsilon, cudaStream_t stream) {
        for (int m = 0; m < modules.size(); ++m) {
            if (!frozen_modules[m]) {
                update_parameter(modules[m].layer1.linear.weights, lr, beta1, beta2, epsilon, stream);
                update_parameter(modules[m].layer1.linear.biases, lr, beta1, beta2, epsilon, stream);
                update_parameter(modules[m].layer2.linear.weights, lr, beta1, beta2, epsilon, stream);
                update_parameter(modules[m].layer2.linear.biases, lr, beta1, beta2, epsilon, stream);
            }
        }
        update_parameter(attention.attention_scores, lr, beta1, beta2, epsilon, stream);
    }

    void set_attention_weight(int module_index, float weight, cudaStream_t stream) {
        if (module_index >= 0 && module_index < attention.attention_scores.data.cols) {
            std::vector<float> h_scores(attention.attention_scores.data.cols);
            attention.attention_scores.data.to_host(h_scores.data(), stream);
            h_scores[module_index] = weight;
            attention.attention_scores.data.to_device(h_scores.data(), stream);
            attention.forward(stream);
            std::cout << "Network: Manually set attention weight for module " << module_index << " to " << weight << std::endl;
        } else {
            throw std::out_of_range("Module index out of range.");
        }
    }

    void freeze_module(int module_index) {
        if (module_index >= 0 && module_index < modules.size()) {
            frozen_modules[module_index] = true;
            std::cout << "Network: Module " << module_index << " is now frozen." << std::endl;
        } else {
            throw std::out_of_range("Module index out of range.");
        }
    }
    void unfreeze_module(int module_index) {
        if (module_index >= 0 && module_index < modules.size()) {
            frozen_modules[module_index] = false;
            std::cout << "Network: Module " << module_index << " is now unfrozen." << std::endl;
        } else {
            throw std::out_of_range("Module index out of range.");
        }
    }
};

// --- Data Loading and Main Function ---

void read_binary(const std::string& path, void* data, size_t size) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    file.read(static_cast<char*>(data), size);
    if (file.gcount() != size) {
        throw std::runtime_error("Error reading file: " + path + ". Expected to read " + std::to_string(size) + " bytes but read " + std::to_string(file.gcount()) + " bytes.");
    }
    file.close();
}

int main() {
    Hyperparameters hp;
    float* h_train_images;
    int* h_train_labels;
    cudaMallocHost(&h_train_images, 50000 * hp.input_size * hp.num_modules * sizeof(float)); // Adjusted size for num_modules duplication
    cudaMallocHost(&h_train_labels, 50000 * sizeof(int));
    std::cout << "Main: Loading training data..." << std::endl;
    read_binary(hp.train_images_path, h_train_images, 50000 * hp.input_size * hp.num_modules * sizeof(float)); // Adjusted size for num_modules duplication
    read_binary(hp.train_labels_path, h_train_labels, 50000 * sizeof(int));
    float* h_test_images;
    int* h_test_labels;
    cudaMallocHost(&h_test_images, 10000 * hp.input_size * hp.num_modules * sizeof(float)); // Adjusted size for num_modules duplication
    cudaMallocHost(&h_test_labels, 10000 * sizeof(int));
    std::cout << "Main: Loading test data..." << std::endl;
    read_binary(hp.test_images_path, h_test_images, 10000 * hp.input_size * hp.num_modules * sizeof(float)); // Adjusted size for num_modules duplication
    read_binary(hp.test_labels_path, h_test_labels, 10000 * sizeof(int));

    Network network(hp.num_modules, hp);
    AlignmentLoss align_loss;
    cudaStream_t stream = 0;
    int num_batches = 50000 / hp.batch_size;
    float best_val_accuracy = 0.0f;
    float current_lambda1 = hp.lambda1;

    std::cout << "Main: Starting training..." << std::endl;
    for (int epoch = 0; epoch < hp.num_epochs; ++epoch) {
        auto epoch_start_time = std::chrono::high_resolution_clock::now();
        float epoch_loss = 0.0f;
        std::cout << "Epoch " << epoch + 1 << "/" << hp.num_epochs << " Training..." << std::endl;
        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            auto batch_start_time = std::chrono::high_resolution_clock::now();
            std::cout << "  Batch " << batch_idx << " start" << std::endl; // ADD THIS LINE
            Tensor input(hp.batch_size, hp.input_size, true);
            Tensor labels_tensor(hp.batch_size, 1, false);
            int* h_batch_labels = h_train_labels + batch_idx * hp.batch_size;
            float* h_batch_images = h_train_images + batch_idx * hp.batch_size * hp.input_size * hp.num_modules; // Adjusted pointer for num_modules duplication

            std::cout << "  Batch " << batch_idx << " before to_device" << std::endl; // ADD THIS LINE
            input.to_device(h_batch_images, stream, true);
            std::cout << "  Batch " << batch_idx << " after to_device" << std::endl; // ADD THIS LINE

            std::cout << "  Batch " << batch_idx << " before network.forward" << std::endl; // ADD THIS LINE
            network.forward(input, stream);
            std::cout << "  Batch " << batch_idx << " after network.forward" << std::endl; // ADD THIS LINE

            Tensor grad_predictions(hp.batch_size, hp.output_dim, true);
            float batch_loss_val;

            std::cout << "  Batch " << batch_idx << " before cross_entropy_forward" << std::endl; // ADD THIS LINE
            cross_entropy_loss_forward_kernel_half<<<1, 1, 0, stream>>>(
                static_cast<const __half*>(network.predictions.data),
                h_batch_labels,
                &batch_loss_val,
                hp.batch_size,
                hp.output_dim
            );
            std::cout << "  Batch " << batch_idx << " after cross_entropy_forward" << std::endl; // ADD THIS LINE


            cross_entropy_loss_backward_kernel_half<<<hp.batch_size * hp.output_dim / 256 + 1, 256, 0, stream>>>(
                static_cast<const __half*>(network.predictions.data),
                h_batch_labels,
                static_cast<__half*>(grad_predictions.data),
                hp.batch_size,
                hp.output_dim
            );

            hp.lambda1 = current_lambda1;
            float align_loss_val = align_loss.compute(network.attention.attention_weights, stream, hp);
            batch_loss_val += align_loss_val;
            epoch_loss += batch_loss_val;

            std::cout << "  Batch " << batch_idx << " before network.backward" << std::endl; // ADD THIS LINE
            network.backward(input, grad_predictions, stream, hp);
            std::cout << "  Batch " << batch_idx << " after network.backward" << std::endl; // ADD THIS LINE

            std::cout << "  Batch " << batch_idx << " before network.update_parameters" << std::endl; // ADD THIS LINE
            network.update_parameters(hp.lr, hp.beta1, hp.beta2, hp.epsilon, stream);
            std::cout << "  Batch " << batch_idx << " after network.update_parameters" << std::endl; // ADD THIS LINE


            cudaStreamSynchronize(stream);
            std::cout << "  Batch " << batch_idx << " after synchronize" << std::endl; // ADD THIS LINE

            if (batch_idx % 100 == 0) {
                std::vector<float> h_attention_weights(hp.num_modules);
                    network.attention.attention_weights.to_host(h_attention_weights.data(), stream);
                    cudaStreamSynchronize(stream);
                    std::cout << "Epoch " << epoch + 1 << " Batch " << batch_idx << "/" << num_batches << " Loss: " << batch_loss_val << " Attention Weights: ";
                    for (int m = 0; m < hp.num_modules; ++m) {
                        std::cout << "M" << m << ":" << h_attention_weights[m] << " ";
                    }
                    std::cout << std::endl;
            }
            auto batch_end_time = std::chrono::high_resolution_clock::now();
            auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end_time - batch_start_time);
        }
        auto epoch_end_time = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end_time - epoch_start_time);
        std::cout << "Epoch " << epoch + 1 << " Training Loss: " << epoch_loss / num_batches << " Epoch Time: " << epoch_duration.count() << " seconds." << std::endl;

        if (epoch == 5) {
            std::cout << "Main: Manually adjusting attention for module 0..." << std::endl;
            network.set_attention_weight(0, 1.0f, stream);
            network.set_attention_weight(1, -1.0f, stream);
        }
        if (epoch == 20) {
            network.freeze_module(1);
        }
        if (epoch == 30) {
            network.unfreeze_module(1);
        }
        if (epoch == 10) {
            current_lambda1 *= 2.0f;
            std::cout << "Main: Adjusting lambda1 to: " << current_lambda1 << std::endl;
        }


        int num_test_batches = 10000 / hp.batch_size;
        float total_correct = 0.0f;
        std::cout << "Epoch " << epoch + 1 << " Validation..." << std::endl;
        for (int test_batch_idx = 0; test_batch_idx < num_test_batches; ++test_batch_idx) {
            Tensor test_input(hp.batch_size, hp.input_size, true);
            float* h_test_batch = h_test_images + test_batch_idx * hp.batch_size * hp.input_size * hp.num_modules;
            test_input.to_device(h_test_batch, stream, true);
            network.forward(test_input, stream);
            Tensor h_predictions(hp.batch_size, hp.output_dim, false);
            network.predictions.to_host(static_cast<float*>(h_predictions.data), stream); // Corrected line: Cast to float*
            cudaStreamSynchronize(stream);
            int* h_test_batch_labels = h_test_labels + test_batch_idx * hp.batch_size;
            float* h_predictions_host_data = static_cast<float*>(h_predictions.data); // Create a correctly typed pointer
            for (int i = 0; i < hp.batch_size; ++i) {
                int predicted_label = 0;
                float max_val = -std::numeric_limits<float>::infinity();
                for (int j = 0; j < hp.output_dim; ++j) {
                    if (h_predictions_host_data[i * hp.output_dim + j] > max_val) { // Use the correctly typed pointer
                        max_val = h_predictions_host_data[i * hp.output_dim + j]; // Use the correctly typed pointer
                        predicted_label = j;
                    }
                }
                if (predicted_label == h_test_batch_labels[i]) {
                    total_correct += 1.0f;
                }
            }
        }
        float val_accuracy = total_correct / 10000.0f;
        std::cout << "Epoch " << epoch + 1 << " Validation Accuracy: " << val_accuracy * 100.0f << "%" << std::endl;
        if (val_accuracy > best_val_accuracy) {
            best_val_accuracy = val_accuracy;
            std::cout << "Epoch " << epoch + 1 << " Best Validation Accuracy: " << best_val_accuracy * 100.0f << "% - Model improved!" << std::endl;
        }
    }
    std::cout << "Main: Training finished." << std::endl;

    cudaFreeHost(h_train_images);
    cudaFreeHost(h_train_labels);
    cudaFreeHost(h_test_images);
    cudaFreeHost(h_test_labels);
    return 0;
}
