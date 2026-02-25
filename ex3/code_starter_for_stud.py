"""
============================================================================
FEATURE MAP EXPLORER - Student Implementation File
============================================================================
CNN Forward Pass Visualization & Receptive Field Calculator

Instructions:
- Complete all functions marked with TODO
- Do NOT modify function signatures
- Do NOT import additional libraries
- Run tests at the bottom to verify your implementation

Data Format:
- Feature maps: (Height, Width, Channels)
- Filters: (num_filters, kernel_height, kernel_width, input_channels)
============================================================================
"""

import numpy as np

# ============================================================================
# HELPER FUNCTION - Single 2D Convolution (PROVIDED)
# ============================================================================

def convolve_2d_single(image_2d: np.ndarray, kernel_2d: np.ndarray, 
                       stride: int = 1) -> np.ndarray:
    """
    Apply 2D convolution to a single-channel image with a single kernel.
    This is provided to help you - use it in apply_conv_layer!
    """
    H, W = image_2d.shape
    kH, kW = kernel_2d.shape
    
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    
    output = np.zeros((out_H, out_W), dtype=np.float64)
    
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            w_start = j * stride
            region = image_2d[h_start:h_start + kH, w_start:w_start + kW]
            output[i, j] = np.sum(region * kernel_2d)
    
    return output


# ============================================================================
# PART 1: CONVOLUTION LAYER WITH MULTIPLE FILTERS
# ============================================================================

def apply_conv_layer(feature_map: np.ndarray, filters: np.ndarray, 
                     stride: int = 1) -> np.ndarray:
    """
    Apply convolution layer with multiple filters to create feature maps.
    
    Parameters:
    -----------
    feature_map : np.ndarray
        Input of shape (H, W, C_in)
    filters : np.ndarray
        Filters of shape (num_filters, kH, kW, C_in)
    stride : int
        Step size for convolution (default 1)
    
    Returns:
    --------
    np.ndarray
        Output of shape (H_out, W_out, num_filters)
        where H_out = (H - kH) // stride + 1
    
    Note: For each filter, convolve across ALL input channels and sum results.
    """
    # Extract dimensions and verify channel compatibility
    H, W, C_in = feature_map.shape
    num_filters, kH, kW, f_C_in = filters.shape

    if C_in != f_C_in:
        raise ValueError(f"Channel mismatch: Input has {C_in}, filter expects {f_C_in}")
    
    # Calculate output dimensions and initialize output array
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    
    # Output shape is (Height, Width, Number of Filters)
    output = np.zeros((out_H, out_W, num_filters), dtype=np.float64)
    
    # For each filter, accumulate convolutions across all input channels
    for f in range(num_filters):
        # We start with a blank slate for this filter's output
        filter_map = np.zeros((out_H, out_W), dtype=np.float64)
        
        # What is C_in? It is the number of input channels.
        for c in range(C_in):
            # 1. Get the specific channel from the input image
            input_channel = feature_map[:, :, c]
            # 2. Get the specific channel kernel from the current filter
            kernel_channel = filters[f, :, :, c]
            
            # 3. Convolve them using the helper
            conv_result = convolve_2d_single(input_channel, kernel_channel, stride)
            
            # 4. ACCUMULATE: Add this channel's contribution to the total
            filter_map += conv_result
            
        # Store the final accumulated result
        output[:, :, f] = filter_map
        
    return output


# ============================================================================
# PART 2: ACTIVATION AND POOLING LAYERS
# ============================================================================

def apply_activation(feature_map: np.ndarray, activation_type: str) -> np.ndarray:
    """
    Apply activation function element-wise to feature map.
    
    Parameters:
    -----------
    feature_map : np.ndarray - Input of any shape
    activation_type : str - One of: "relu", "leaky_relu", "sigmoid"
    
    Returns:
    --------
    np.ndarray - Activated output (same shape as input)
    
    Formulas:
    - relu: max(0, x)
    - leaky_relu: x if x > 0 else 0.01 * x
    - sigmoid: 1 / (1 + exp(-x))
    
    Raises:
    -------
    ValueError: If activation_type is not recognized
    """
    if activation_type == "relu":
        return np.maximum(0, feature_map)
    
    elif activation_type == "leaky_relu":
        # Returns feature_map where it's > 0, otherwise 0.01 * feature_map
        return np.where(feature_map > 0, feature_map, 0.01 * feature_map)
    
    elif activation_type == "sigmoid":
        return 1 / (1 + np.exp(-feature_map))
    
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")


def apply_pool_layer(feature_map: np.ndarray, pool_size: int = 2, 
                     stride: int = 2) -> np.ndarray:
    """
    Apply max pooling to each channel of the feature map.
    
    Parameters:
    -----------
    feature_map : np.ndarray - Input of shape (H, W, C)
    pool_size : int - Size of pooling window (default 2)
    stride : int - Step between pooling windows (default 2)
    
    Returns:
    --------
    np.ndarray - Pooled output of shape (H_out, W_out, C)
    
    Note: Apply pooling independently to each channel!
    """
    H, W, C = feature_map.shape
    
    # 1. Calculate output dimensions
    # Standard formula: floor((Input - Filter) / Stride) + 1
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    
    # 2. Initialize output with zeros
    output = np.zeros((out_H, out_W, C), dtype=np.float64)
    
    # 3. Apply Max Pooling independently to each channel
    for c in range(C):
        for i in range(out_H):
            for j in range(out_W):
                # Define the boundaries of the current pooling window
                h_start = i * stride
                w_start = j * stride
                h_end = h_start + pool_size
                w_end = w_start + pool_size
                
                # Extract the 2D window for the current channel
                window = feature_map[h_start:h_end, w_start:w_end, c]
                
                # Take the maximum value in this window
                output[i, j, c] = np.max(window)
                
    return output


# ============================================================================
# PART 3: CNN PIPELINE BUILDER
# ============================================================================

def build_cnn_pipeline(image: np.ndarray, layer_configs: list) -> dict:
    """
    Process an image through a sequence of CNN layers and track transformations.
    
    Parameters:
    -----------
    image : np.ndarray - Input image of shape (H, W, C)
    layer_configs : list of dict
        Conv: {"type": "conv", "filters": int, "kernel_size": int, "stride": int}
        Activation: {"type": "activation", "activation": str}
        Pool: {"type": "pool", "pool_size": int, "stride": int}
    
    Returns:
    --------
    dict with structure:
        {
            "PIPELINE_REPORT": {
                "input_shape": tuple,
                "output_shape": tuple,
                "total_layers": int,
                "layer_sequence": [
                    {"layer_index": int, "layer_type": str, 
                     "input_shape": tuple, "output_shape": tuple, "parameters": int}
                ]
            },
            "feature_maps": np.ndarray
        }
    
    Parameter count for conv: kernel_size^2 * in_channels * num_filters + num_filters
    """
    current_map = image.copy()
    layer_sequence = []
    
    for idx, config in enumerate(layer_configs):
        layer_type = config["type"]
        input_shape = current_map.shape
        params = 0
        
        if layer_type == "conv":
            num_filters = config["filters"]
            k_size = config["kernel_size"]
            stride = config.get("stride", 1)
            in_channels = input_shape[2]
            
            # 1. Create random filters for simulation
            filters = np.random.randn(num_filters, k_size, k_size, in_channels).astype(np.float64) * 0.1
            
            # 2. Apply convolution
            current_map = apply_conv_layer(current_map, filters, stride)
            
            # 3. Calculate parameters: (k*k * cin * cout) + cout biases
            params = (k_size * k_size * in_channels * num_filters) + num_filters
            
        elif layer_type == "activation":
            act_type = config["activation"]
            current_map = apply_activation(current_map, act_type)
            params = 0 # Activation has no weights
            
        elif layer_type == "pool":
            p_size = config["pool_size"]
            stride = config.get("stride", 2)
            current_map = apply_pool_layer(current_map, p_size, stride)
            params = 0 # Pooling has no weights
            
        # Record this layer's statistics
        layer_sequence.append({
            "layer_index": idx,
            "layer_type": layer_type,
            "input_shape": input_shape,
            "output_shape": current_map.shape,
            "parameters": params
        })
        
    return {
        "PIPELINE_REPORT": {
            "input_shape": image.shape,
            "output_shape": current_map.shape,
            "total_layers": len(layer_configs),
            "layer_sequence": layer_sequence
        },
        "feature_maps": current_map
    }


# ============================================================================
# PART 4: RECEPTIVE FIELD CALCULATOR
# ============================================================================

def calculate_receptive_field(layer_configs: list) -> dict:
    """
    Calculate the receptive field size at each layer of the CNN.
    
    The receptive field tells us how many input pixels influence one output pixel.
    
    Formulas:
        RF_new = RF_prev + (kernel_size - 1) * jump_prev
        jump_new = jump_prev * stride
    
    Initial values: RF = 1, jump = 1
    
    For layer types:
    - conv: kernel_size from config, stride from config (default 1)
    - pool: kernel_size = pool_size, stride from config
    - activation: kernel_size = 1, stride = 1 (doesn't change RF)
    
    Returns:
    --------
    dict with structure:
        {
            "receptive_fields": [
                {"layer": int, "type": str, "rf_size": int, "jump": int}
            ],
            "final_rf": int
        }
    """
    # Initial values for the input layer
    rf = 1
    jump = 1
    
    # Start with the input layer (Layer 0)
    results = [
        {"layer": 0, "type": "input", "rf_size": 1, "jump": 1}
    ]
    
    for idx, config in enumerate(layer_configs):
        layer_type = config["type"]
        
        # Determine kernel size (k) and stride (s) for this layer type
        if layer_type == "conv":
            k = config["kernel_size"]
            s = config.get("stride", 1)
        elif layer_type == "pool":
            k = config["pool_size"]
            s = config.get("stride", 2)  # Instructions say "stride from config", assuming default 2
        elif layer_type == "activation":
            k = 1
            s = 1
        else:
            k, s = 1, 1
            
        # Update Receptive Field (BEFORE updating jump)
        rf = rf + (k - 1) * jump
        
        # Update Jump (effective stride in the input image)
        jump = jump * s
        
        results.append({
            "layer": idx + 1,
            "type": layer_type,
            "rf_size": int(rf),
            "jump": int(jump)
        })
        
    return {
        "receptive_fields": results,
        "final_rf": int(rf)
    }


# ============================================================================
# TEST FUNCTIONS - DO NOT MODIFY
# ============================================================================

def run_all_tests():
    """Run all verification tests for Feature Map Explorer."""
    print("=" * 60)
    print("FEATURE MAP EXPLORER - AUTOMATED TEST SUITE")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Conv layer basics
    print("\n[TEST 1] Testing apply_conv_layer...")
    try:
        test_input = np.random.rand(8, 8, 3).astype(np.float64)
        test_filters = np.random.randn(4, 3, 3, 3).astype(np.float64) * 0.1
        
        result = apply_conv_layer(test_input, test_filters, stride=1)
        expected_shape = (6, 6, 4)
        
        if result is None:
            print("    FAILED: Function returned None")
            all_passed = False
        elif result.shape != expected_shape:
            print(f"    FAILED: Expected shape {expected_shape}, got {result.shape}")
            all_passed = False
        else:
            print(f"    PASSED: Output shape {result.shape} correct")
    except Exception as e:
        print(f"    FAILED: Exception - {e}")
        all_passed = False
    
    # Test 2: Conv layer with stride
    print("\n[TEST 2] Testing apply_conv_layer with stride=2...")
    try:
        test_input = np.random.rand(8, 8, 1).astype(np.float64)
        test_filters = np.random.randn(2, 3, 3, 1).astype(np.float64) * 0.1
        
        result = apply_conv_layer(test_input, test_filters, stride=2)
        expected_shape = (3, 3, 2)
        
        if result is None:
            print("    FAILED: Function returned None")
            all_passed = False
        elif result.shape != expected_shape:
            print(f"    FAILED: Expected shape {expected_shape}, got {result.shape}")
            all_passed = False
        else:
            print(f"    PASSED: Stride=2 output shape {result.shape} correct")
    except Exception as e:
        print(f"    FAILED: Exception - {e}")
        all_passed = False
    
    # Test 3: Activation functions
    print("\n[TEST 3] Testing apply_activation...")
    try:
        test_input = np.array([[[-1, 2], [0, -3]]]).astype(np.float64)
        
        relu_result = apply_activation(test_input, "relu")
        relu_expected = np.array([[[0, 2], [0, 0]]]).astype(np.float64)
        
        if relu_result is None:
            print("    FAILED: ReLU returned None")
            all_passed = False
        elif not np.allclose(relu_result, relu_expected):
            print(f"    FAILED: ReLU incorrect")
            all_passed = False
        else:
            print("    PASSED: ReLU activation correct")
        
        leaky_result = apply_activation(test_input, "leaky_relu")
        if leaky_result is not None and leaky_result[0, 0, 0] == -0.01:
            print("    PASSED: Leaky ReLU activation correct")
        else:
            print("    FAILED: Leaky ReLU incorrect")
            all_passed = False
        
        sig_input = np.array([[[0]]]).astype(np.float64)
        sig_result = apply_activation(sig_input, "sigmoid")
        if sig_result is not None and np.isclose(sig_result[0, 0, 0], 0.5):
            print("    PASSED: Sigmoid activation correct")
        else:
            print("    FAILED: Sigmoid incorrect (sigmoid(0) should be 0.5)")
            all_passed = False
            
    except Exception as e:
        print(f"    FAILED: Exception - {e}")
        all_passed = False
    
    # Test 4: Unknown activation error
    print("\n[TEST 4] Testing unknown activation handling...")
    try:
        apply_activation(np.zeros((2, 2, 1)), "unknown_activation")
        print("    FAILED: Should raise ValueError")
        all_passed = False
    except ValueError as e:
        if "Unknown activation" in str(e):
            print("    PASSED: Correctly raises ValueError")
        else:
            print(f"    FAILED: Wrong error message - {e}")
            all_passed = False
    except Exception as e:
        print(f"    FAILED: Wrong exception type - {type(e).__name__}")
        all_passed = False
    
    # Test 5: Max pooling
    print("\n[TEST 5] Testing apply_pool_layer...")
    try:
        test_input = np.array([
            [[1, 10], [2, 20], [3, 30], [4, 40]],
            [[5, 50], [6, 60], [7, 70], [8, 80]],
            [[9, 90], [10, 100], [11, 110], [12, 120]],
            [[13, 130], [14, 140], [15, 150], [16, 160]]
        ], dtype=np.float64)
        
        result = apply_pool_layer(test_input, pool_size=2, stride=2)
        
        if result is None:
            print("    FAILED: Function returned None")
            all_passed = False
        elif result.shape != (2, 2, 2):
            print(f"    FAILED: Expected shape (2, 2, 2), got {result.shape}")
            all_passed = False
        elif result[0, 0, 0] != 6 or result[0, 0, 1] != 60:
            print(f"    FAILED: Values incorrect. Top-left should be [6, 60]")
            all_passed = False
        else:
            print(f"    PASSED: Max pooling shape {result.shape} and values correct")
    except Exception as e:
        print(f"    FAILED: Exception - {e}")
        all_passed = False
    
    # Test 6: Pipeline builder
    print("\n[TEST 6] Testing build_cnn_pipeline...")
    try:
        test_image = np.random.rand(16, 16, 1).astype(np.float64)
        layer_configs = [
            {"type": "conv", "filters": 4, "kernel_size": 3, "stride": 1},
            {"type": "activation", "activation": "relu"},
            {"type": "pool", "pool_size": 2, "stride": 2}
        ]
        
        result = build_cnn_pipeline(test_image, layer_configs)
        
        if "PIPELINE_REPORT" not in result:
            print("    FAILED: Missing 'PIPELINE_REPORT' key")
            all_passed = False
        elif "feature_maps" not in result:
            print("    FAILED: Missing 'feature_maps' key")
            all_passed = False
        elif result["PIPELINE_REPORT"]["total_layers"] != 3:
            print("    FAILED: total_layers should be 3")
            all_passed = False
        else:
            report = result["PIPELINE_REPORT"]
            print(f"    PASSED: Pipeline processed {report['total_layers']} layers")
            print(f"    [PIPELINE] Output shape: {report['output_shape']}")
    except Exception as e:
        print(f"    FAILED: Exception - {e}")
        all_passed = False
    
    # Test 7: Receptive field calculator
    print("\n[TEST 7] Testing calculate_receptive_field...")
    try:
        layer_configs = [
            {"type": "conv", "filters": 8, "kernel_size": 3, "stride": 1},
            {"type": "pool", "pool_size": 2, "stride": 2},
            {"type": "conv", "filters": 16, "kernel_size": 3, "stride": 1}
        ]
        
        result = calculate_receptive_field(layer_configs)
        
        if "receptive_fields" not in result:
            print("    FAILED: Missing 'receptive_fields' key")
            all_passed = False
        elif "final_rf" not in result:
            print("    FAILED: Missing 'final_rf' key")
            all_passed = False
        else:
            expected_final_rf = 8
            if result["final_rf"] == expected_final_rf:
                print(f"    PASSED: Final receptive field = {result['final_rf']}")
            else:
                print(f"    FAILED: Expected final RF {expected_final_rf}, got {result['final_rf']}")
                all_passed = False
    except Exception as e:
        print(f"    FAILED: Exception - {e}")
        all_passed = False
    
    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED! Your Feature Map Explorer is ready!")
    else:
        print("SOME TESTS FAILED. Review your implementation.")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
