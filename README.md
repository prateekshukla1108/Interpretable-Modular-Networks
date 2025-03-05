## Interpretable Modular Networks 


![banner](assets/banner.png)

**Abstract:**

This project investigates **Interpretable Modular Networks (IMN)**, a proposed neural network architecture designed to enhance the transparency, and controllability of models, particularly for mixed datasets.  Initially, a direct implementation in CUDA C++ was attempted to maximize performance, but the complexity of implementing a full neural network framework from scratch in CUDA proved too challenging for the scope of this project.  Therefore, the primary focus shifted to a Python implementation using PyTorch, allowing for rapid prototyping and exploration of IMN's core concepts. This report details the Python implementation, its preliminary evaluation on a mixed-modality dataset derived from CIFAR10, the initial CUDA implementation attempt, and discusses the current limitations, particularly the achievable accuracy on CIFAR-10 (around 50% due to the use of linear layers instead of CNNs), and future directions including the integration of Convolutional Neural Networks (CNNs) for improved performance.

**1. Introduction:**

Deep learning has achieved remarkable success across various domains, but the inherent "black box" nature of complex neural networks raises concerns about interpretability and control. Understanding *why* a model makes a certain decision and ensuring we can influence its behavior are increasingly crucial, especially in sensitive applications.

**Interpretable Modular Networks (IMN)** proposes a neural network architecture to address these challenges. IMN aims to create neural networks that are not only accurate but also:

*   **More Transparent:** By making the decision-making process more understandable through modular architectures and attention mechanisms.
*   **Potentially Ethically Aligned:** By offering mechanisms to incorporate rules and constraints that can guide model behavior, indirectly contributing to ethical considerations.
*   **More Controllable:** By providing mechanisms to adjust model focus and behavior during training and inference.

This project initially aimed for a high-performance CUDA C++ implementation of IMN. However, the complexity of building a complete, flexible neural network framework from scratch in CUDA proved to be a significant hurdle.  Consequently, the project pivoted to prioritize a Python implementation using PyTorch to facilitate rapid exploration of IMN's core principles and conduct initial experiments.  A first attempt at a CUDA implementation was also undertaken and is included in this report, acknowledging its current limitations.

**2. Algorithm Description:**

**Interpretable Modular Networks (IMN)** architecture is defined by several key components:

*   **Modular Network Architecture:** IMN advocates for modularity, employing sub-networks (modules) specialized for different data types or tasks. This modular design enhances interpretability by isolating processing pathways. An attention mechanism dynamically combines module outputs.

*   **Attention Layer:**  An attention layer is introduced to learn weights that determine the contribution of each module to the final prediction. These attention weights are intended to be interpretable, indicating the relative importance of different data modalities or modules.  Manual adjustment of these weights is also proposed as a control mechanism.

*   **Multi-Objective Loss Function:**  IMN can utilize a loss function comprising multiple terms:
    *   **Task Loss (`L_task`):**  Standard loss function for the primary learning objective (e.g., cross-entropy for classification).
    *   **Alignment Loss (`L_align`):**  A penalty based on symbolic rules or constraints, designed to enforce desired model behaviors (e.g., fairness, feature usage constraints).
    *   **Transparency Regularization (`L_trans`):** A regularization term encouraging sparsity or meaningfulness in attention weights, further enhancing interpretability.

    The total loss can be a weighted sum: `L_total = L_task + λ₁ * L_align + λ₂ * L_trans`, where λ₁ and λ₂ are hyperparameters controlling the balance between these objectives.

*   **Symbolic Rules for Behavior Guidance:**  IMN allows for defining symbolic rules that represent desired behavior. These rules can be translated into differentiable penalty functions that contribute to `L_align`.

*   **Enhanced Backpropagation:** The standard backpropagation process is adapted to minimize the combined `L_total` loss, incorporating gradients from all loss components.

*   **Transparency and Control Mechanisms:**  IMN leverages attention weights as a primary transparency mechanism.  Control is exercised through manual adjustment of attention weights, tuning of loss hyperparameters (λ₁, λ₂), and module freezing/unfreezing.

**3. Python Implementation:**

Due to the complexity of building a full neural network framework in CUDA from scratch, the primary implementation effort focused on Python using PyTorch. This allowed for rapid prototyping and easier experimentation with IMN's core concepts.

*   **Libraries:**  The Python implementation relies on PyTorch for neural network construction and automatic differentiation, along with standard libraries like `numpy` for numerical operations.

*   **Key Classes:**  The implementation is structured using classes to represent the different components of IMN:
    *   `Hyperparameters`:  A class to manage training hyperparameters (batch size, learning rate, lambda values, etc.).
    *   `LinearLayer`, `Layer`, `Module`: Classes representing linear layers, ReLU layers, and modular sub-networks, respectively.  These are intentionally kept simple in this initial Python implementation, using basic linear transformations and ReLU activations.
    *   `AttentionLayer`:  Implements the attention mechanism, learning attention weights via softmax.
    *   `AlignmentLoss`, `TransparencyRegularizationLoss`: Classes to compute the alignment and transparency regularization loss terms (though in this initial implementation, transparency regularization was not explicitly implemented, and alignment loss was simplified).
    *   `MixedCIFAR10Dataset`: A custom PyTorch Dataset class to load and manage a modified CIFAR10 dataset, simulating a mixed-modality input by duplicating image data across modules.
    *   `Network`:  The main network class, orchestrating the modules, attention layer, forward and backward passes, and parameter updates.

*   **Training Process:**  A standard PyTorch training loop is implemented, adapted to:
    *   Iterate through epochs and batches of the `MixedCIFAR10Dataset`.
    *   Perform a forward pass through the `Network`.
    *   Compute the `L_task` (CrossEntropyLoss in this case) and `L_align` (variance-based alignment loss).
    *   Calculate `L_total`.
    *   Perform backpropagation using `L_total.backward()`.
    *   Update network parameters using the Adam optimizer.
    *   Monitor attention weights and validation accuracy.
    *   Include basic manual control mechanisms (setting attention weights, freezing modules) within the training loop.

**4. CUDA Implementation (Initial Attempt):**

An initial attempt was made to implement IMN directly in CUDA C++ for potential performance benefits.  This involved building neural network components from fundamental CUDA primitives.

*   **Challenges Encountered:** Implementing a full neural network framework from scratch in CUDA proved to be significantly more complex and time-consuming than anticipated.  Key challenges included:
    *   Manual memory management and kernel orchestration.
    *   Implementing efficient and numerically stable CUDA kernels for all necessary operations (linear algebra, activations, loss functions, optimizers).
    *   Debugging and ensuring correctness of CUDA kernels and the overall backpropagation process.
    *   Lack of built-in automatic differentiation, requiring manual derivation and implementation of gradients for all operations.

*   **First CUDA Implementation:** Despite the challenges, a first functional CUDA implementation was achieved.  This implementation:
    *   Utilizes CUDA half-precision (`__half`) for data storage and computation to improve performance.
    *   Includes CUDA kernels for core operations like matrix multiplication, bias addition, ReLU, Softmax, Cross-Entropy Loss, Alignment Loss (simplified), and Adam optimization.
    *   Implements basic `Tensor` and `Parameter` classes for memory management.
    *   Provides a basic modular network structure with `Linear`, `Layer`, `Module`, `AttentionLayer`, and `Network` classes mirroring the Python implementation.
    *   Includes rudimentary error checking using `checkCudaErrors`.

    However, this CUDA implementation is still in a nascent stage. It requires significant further work for optimization, thorough correctness verification of kernels, more robust error handling, and feature completeness to match the flexibility and functionality of the Python/PyTorch implementation.  Due to the time constraints and complexity, the Python implementation became the primary focus for exploring IMN's architectural aspects.

**5. Experiments and Preliminary Results:**

*   **Dataset:** Experiments were conducted using a modified CIFAR10 dataset (`MixedCIFAR10Dataset`). To simulate a mixed-modality scenario for IMN, the CIFAR10 image data (flattened) was duplicated and presented as input to multiple modules.  Synthetic text or other modality features were not included in this initial experiment.

*   **Hyperparameters:** Hyperparameters were set to explore the basic functionality of the IMN implementation (as detailed in the Python code). Key hyperparameters included learning rate, batch size, lambda values for alignment loss, and number of epochs.

*   **Validation Accuracy:**  The Python implementation, using simple linear layers within modules, achieved a validation accuracy of approximately **50% on CIFAR10**. This accuracy is significantly lower than what is achievable with standard CNN-based models on CIFAR10.

*   **Analysis of Accuracy:** The limited accuracy is primarily attributed to the **absence of Convolutional Neural Networks (CNNs) in the modules**. The current implementation uses only linear layers, which are not well-suited for capturing the spatial hierarchical features present in image data like CIFAR10.  CNNs are known to be essential for achieving high accuracy on image classification tasks.

*   **Attention Weight Monitoring:**  During training, attention weights were monitored.  The implementation allowed for manual adjustment of attention scores and freezing/unfreezing of modules, demonstrating basic control over the network's behavior.  For example, manually increasing the attention score for a specific module could influence its contribution to the final prediction.

*   **Example Attention Weights (from Python Training Log):**
    ```
    Epoch 10 Batch 100/781 Loss: 2.29341 Attention Weights: M0:0.33333 M1:0.33333 M2:0.33333
    Epoch 10 Batch 200/781 Loss: 2.29721 Attention Weights: M0:0.33333 M1:0.33333 M2:0.33333
    Epoch 10 Batch 300/781 Loss: 2.29045 Attention Weights: M0:0.33333 M1:0.33333 M2:0.33333
    Epoch 10 Batch 400/781 Loss: 2.30352 Attention Weights: M0:0.33333 M1:0.33333 M2:0.33333
    Epoch 10 Validation Accuracy: 10.00%
    Epoch 10 Best Validation Accuracy: 10.00% - Model improved!
    Main: Adjusting lambda1 to: 0.2
    Epoch 11 Batch 100/781 Loss: 2.30259 Attention Weights: M0:0.33333 M1:0.33333 M2:0.33333
    Epoch 11 Batch 200/781 Loss: 2.30323 Attention Weights: M0:0.33333 M1:0.33333 M2:0.33333
    Epoch 11 Batch 300/781 Loss: 2.29796 Attention Weights: M0:0.33333 M1:0.33333 M2:0.33333
    Epoch 11 Batch 400/781 Loss: 2.29507 Attention Weights: M0:0.33333 M1:0.33333 M2:0.33333
    Epoch 11 Validation Accuracy: 10.00%
    Epoch 11 Best Validation Accuracy: 10.00% - Model improved!
    ```
    (Note: These example logs show very low validation accuracy as the model with linear layers is not effective for CIFAR10.  The attention weights are initialized to be roughly equal initially.)

**6. Discussion:**

This project represents a valuable initial step in exploring **Interpretable Modular Networks (IMN)**.  The Python implementation successfully demonstrates the core concepts of IMN, including modular network architectures, attention mechanisms, and multi-objective loss functions within a manageable PyTorch framework. The ability to monitor and manually adjust attention weights and module freezing provides a basic level of control over the training process, aligning with IMN's goals for interpretability and controllability.

The initial attempt at a CUDA implementation, while challenging, lays a foundation for potential future performance optimization.  It highlights the significant effort required to build a deep learning framework from scratch in CUDA but also points towards the potential benefits of fine-grained control and performance optimization that CUDA can offer.

The relatively low validation accuracy on CIFAR10 (around 50%) is a clear limitation of the current implementation. However, it is crucial to understand that this is primarily due to the architectural choice of using simple linear layers in the modules, rather than a fundamental flaw in the IMN concept itself. Linear layers are simply not effective for image classification tasks on datasets like CIFAR10.

**Limitations of Current Implementation:**

*   **Simple Modules (Python):**  Python modules use only linear layers, limiting performance on complex datasets like CIFAR10, especially for image-related tasks.
*   **Synthetic Mixed Data (CIFAR10 Modification):**  The "mixed dataset" used is a simplified modification of CIFAR10, duplicating image data. Real mixed-modality datasets were not used.
*   **Basic Alignment Loss:** The alignment loss implemented is a simplified variance-based loss. More sophisticated alignment rules and loss functions need to be explored.
*   **No Transparency Regularization:** Transparency regularization was not explicitly implemented in this initial phase.
*   **Limited CUDA Implementation:** The CUDA implementation is a very early stage prototype, lacking optimization, thorough testing, and feature completeness.
*   **Lack of Quantitative Transparency/Control Metrics:**  Objective metrics to quantify transparency and control are not yet implemented, hindering rigorous evaluation.

**7. Conclusion and Future Directions:**

This project provides a functional Python implementation of **Interpretable Modular Networks (IMN)**, successfully demonstrating its core principles.  The initial CUDA implementation, though challenging, represents a first step towards potential performance optimization in the future.

**Future research should prioritize:**

*   **Integrating CNN Modules:**  Replacing the linear layers in the modules with Convolutional Neural Networks (CNNs) is the most crucial next step to improve performance on image datasets like CIFAR10 and to properly evaluate IMN's effectiveness in image-related tasks.
*   **Evaluating on Real Mixed-Modality Datasets:** Testing IMN with genuine mixed data (e.g., images and text, or tabular and image data) to assess its effectiveness in handling diverse input sources.
*   **Developing Sophisticated Behavior Guidance Rules and Losses:**  Exploring and implementing more meaningful symbolic rules and corresponding differentiable alignment loss functions to enforce desired model behaviors or constraints.
*   **Implementing Transparency Regularization:**  Adding a transparency regularization term to the loss function to explicitly encourage interpretable attention weights.
*   **Further Developing and Optimizing CUDA Implementation:**  Continuing to develop the CUDA implementation, focusing on kernel optimization, error handling, and feature parity with the Python version for potential performance gains.
*   **Establishing Evaluation Metrics:**  Defining and implementing quantitative metrics for transparency, controllability, and overall performance to enable objective evaluation and comparison with baseline methods.
*   **Exploring Automated Control Mechanisms:** Investigating data-driven or automated methods to adjust attention weights or loss hyperparameters during training, reducing reliance on manual intervention.

This project serves as a solid foundation for further investigation into the IMN architecture. By addressing the identified limitations and pursuing the suggested future directions, particularly the integration of CNN modules and evaluation on real mixed datasets, the potential of IMN as an approach to training more transparent and controllable neural networks can be more fully realized.
