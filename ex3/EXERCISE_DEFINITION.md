# Feature Map Explorer

---

##  Learning Objectives

By completing this exercise, you will:

1. **Understand CNN forward pass** - Track how data transforms through Conv→Activation→Pool layers
2. **Visualize feature maps** - See how dimensions change at each layer
3. **Calculate receptive fields** - Understand how much of the input each neuron "sees"
4. **Build layer pipelines** - Chain multiple CNN operations together
5. **Extension: Trace influence** - Determine exactly which input pixels affect a specific output neuron

---

##  The Scenario

You are a **Neural Network Architect** at DeepVision Labs. Your team is designing a new CNN for medical image analysis, but before training, you need to build a **Feature Map Explorer** tool that:

- Simulates the forward pass through CNN layers
- Tracks how the spatial dimensions shrink at each stage
- Calculates the receptive field - how much context each neuron uses
- Generates detailed layer-by-layer reports

The medical team needs to understand exactly how much of the original scan each feature "sees" before making predictions. Your tool will help them design the optimal architecture!

---

---

## 📋 Detailed Instructions

### Part 1: Convolution Layer with Multiple Filters (10 minutes)

Implement `apply_conv_layer()` that applies **multiple filters** to create feature maps.

**Key Concepts:**
- A Conv layer has multiple filters (e.g., 8 filters → 8 output channels)
- Each filter produces one feature map (output channel)
- Input shape: `(H, W, C_in)` - Height, Width, Input Channels
- Output shape: `(H_out, W_out, C_out)` - where C_out = number of filters

**Requirements:**
- Support `stride` parameter (default 1)
- Each filter is shape `(kH, kW, C_in)` - must match input channels
- Apply convolution for each filter separately
- Stack results to create multi-channel output

### Part 2: Activation and Pooling Layers (8 minutes)

#### 2a. Implement `apply_activation()`
Support three activation functions:
- `"relu"`: max(0, x)
- `"leaky_relu"`: x if x > 0 else 0.01 * x
- `"sigmoid"`: 1 / (1 + exp(-x))

#### 2b. Implement `apply_pool_layer()`
- Max pooling with configurable `pool_size` and `stride`
- Apply pooling to each channel independently
- Output shape: `(H // stride, W // stride, C)`

### Part 3: CNN Pipeline Builder (7 minutes)

Implement `build_cnn_pipeline()` that:
- Takes layer configurations as input
- Processes image through each layer sequentially
- Tracks dimensions at each stage
- Returns detailed report

**Layer Configuration Format:**
```python
layer_config = {
    "type": "conv",           # "conv", "activation", or "pool"
    "filters": 8,             # Number of filters (conv only)
    "kernel_size": 3,         # Filter size (conv only)
    "stride": 1,              # Stride (conv and pool)
    "activation": "relu",     # Activation type (activation only)
    "pool_size": 2            # Pool window size (pool only)
}
```

**Output Format (CRITICAL):**
```python
{
    "PIPELINE_REPORT": {
        "input_shape": tuple,
        "output_shape": tuple,
        "total_layers": int,
        "layer_sequence": [
            {
                "layer_index": int,
                "layer_type": str,
                "input_shape": tuple,
                "output_shape": tuple,
                "parameters": int  # 0 for activation/pool
            },
            ...
        ]
    },
    "feature_maps": np.ndarray  # Final output
}
```

### Part 4: Receptive Field Calculator (5 minutes)

Implement `calculate_receptive_field()` that computes how large the receptive field is at each layer.

**Receptive Field Formula:**
For each layer, the receptive field grows:
```
RF_new = RF_prev + (kernel_size - 1) * jump_prev
jump_new = jump_prev * stride
```

Where:
- `RF` = receptive field size
- `jump` = how many input pixels one output pixel movement corresponds to
- Initial: `RF = 1`, `jump = 1`

**Output Format:**
```python
{
    "receptive_fields": [
        {"layer": 0, "type": "input", "rf_size": 1, "jump": 1},
        {"layer": 1, "type": "conv", "rf_size": 3, "jump": 1},
        ...
    ],
    "final_rf": int  # Total receptive field at output
}
```

---

### Required Function Signatures (DO NOT MODIFY):

```python
def apply_conv_layer(feature_map: np.ndarray, filters: np.ndarray, 
                     stride: int = 1) -> np.ndarray:
    pass

def apply_activation(feature_map: np.ndarray, 
                     activation_type: str) -> np.ndarray:
    pass

def apply_pool_layer(feature_map: np.ndarray, pool_size: int = 2, 
                     stride: int = 2) -> np.ndarray:
    pass

def build_cnn_pipeline(image: np.ndarray, 
                       layer_configs: list) -> dict:
    pass

def calculate_receptive_field(layer_configs: list) -> dict:
    pass
```

### Data Format:
- All feature maps use shape `(Height, Width, Channels)`
- Filters array shape: `(num_filters, kernel_height, kernel_width, input_channels)`
- Use `np.float64` for all computations

---

## Testing Your Code

Run the built-in tests:
```bash
python code_starter_for_stud.py
```

Expected output format:
```
[TEST] apply_conv_layer: PASSED
[TEST] apply_activation - relu: PASSED
...
[PIPELINE] 3 layers processed, output shape: (6, 6, 16)
[RECEPTIVE FIELD] Final RF: 7x7 pixels
```

---

##  Submission Requirements

1. Submit your completed `code_starter_for_stud.py` file
2. All functions must pass the automated tests
3. Do NOT rename functions or change signatures
4. Do NOT import additional libraries beyond numpy

---

## 📊 Example: LeNet-5 Style Pipeline

```python
layer_configs = [
    {"type": "conv", "filters": 6, "kernel_size": 5, "stride": 1},
    {"type": "activation", "activation": "relu"},
    {"type": "pool", "pool_size": 2, "stride": 2},
    {"type": "conv", "filters": 16, "kernel_size": 5, "stride": 1},
    {"type": "activation", "activation": "relu"},
    {"type": "pool", "pool_size": 2, "stride": 2},
]

# Input: 32x32x1 grayscale image
# After Conv1: 28x28x6
# After Pool1: 14x14x6
# After Conv2: 10x10x16
# After Pool2: 5x5x16
```
