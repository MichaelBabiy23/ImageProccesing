import cv2
import numpy as np
import pandas as pd
import os
from collections import deque

STRONG = 255
WEAK = 50


def load_image(image_path):
    """
    Loads an image from a file and returns it as a grayscale image.
    Parameters:
    -----------
    image_path : str
        The path to the image file.

    Returns:
    --------
    image : numpy.ndarray
        The grayscale image.
    """
    # Load and convert to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        # Raise an error if the image is not loaded
        raise ValueError(f"Could not load image from {image_path}")
    return image


def crop_edges(image, margin=40):
    """
    Crop a margin from all edges of the image to remove scanner/border artifacts.

    Parameters:
    -----------
    image : numpy.ndarray
        The input image (grayscale or color).
    margin : int
        Number of pixels to crop from each edge (default 40).

    Returns:
    --------
    cropped : numpy.ndarray
        The cropped image.
    """
    if image.ndim == 2:
        # Grayscale
        return image[margin:-margin, margin:-margin]
    else:
        # Color
        return image[margin:-margin, margin:-margin]


def gaussian_kernel(size, sigma=1.0):
    """
    Generates a 2D Gaussian kernel.

    Parameters:
    -----------
    size : int
        The size of the kernel (e.g., 3 for a 3x3 kernel).
        Must be an odd number.
    sigma : float
        The standard deviation of the Gaussian distribution.

    Returns:
    --------
    kernel : numpy.ndarray
        The 2D Gaussian kernel.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)


def convolve(image, kernel):
    """
    Applies convolution (cross-correlation) to an image with any odd-sized kernel.

    Parameters:
    -----------
    image : numpy.ndarray
        The grayscale image (2D array).
    kernel : numpy.ndarray
        The kernel to apply (must have odd dimensions).

    Returns:
    --------
    convolved_image : numpy.ndarray
        The convolved image with the same dimensions as the input.
    """
    k_height = kernel.shape[0]
    pad = k_height // 2

    convolved_image = np.zeros_like(image, dtype=np.float64)
    for i in range(pad, image.shape[0] - pad):
        for j in range(pad, image.shape[1] - pad):
            roi = image[i - pad: i + pad + 1, j - pad: j + pad + 1]
            convolved_image[i, j] = np.sum(roi * kernel)
    return convolved_image


def sobel_filter(blurred_image):
    """
    Applies the Sobel filter to a blurred (grayscale) image.

    Parameters:
    -----------
    blurred_image : numpy.ndarray
        The blurred grayscale image (2D array).

    Returns:
    --------
    G : numpy.ndarray
        Gradient magnitude of the image.
    D : numpy.ndarray
        Gradient direction (angle) of the image in radians.
    """
    Kx = np.array([[-1,  0,  1],
                   [-2,  0,  2],
                   [-1,  0,  1]])
    Ky = np.array([[-1, -2, -1],
                   [0,  0,  0],
                   [1,  2,  1]])
    Gx = convolve(blurred_image, Kx)
    Gy = convolve(blurred_image, Ky)
    M = np.hypot(Gx, Gy)
    D = np.arctan2(Gy, Gx)
    return M, D


def supression(M, D):
    """
    Performs non-maximum suppression (edge thinning) on the gradient magnitude image.

    Parameters:
    -----------
    M : numpy.ndarray
        The gradient magnitude (2D array).
    D : numpy.ndarray
        The gradient direction in radians (2D array).

    Returns:
    --------
    suppressed : numpy.ndarray
        The suppressed (thinned) image.
    """
    rows, cols = M.shape
    suppressed = np.zeros((rows, cols), dtype=M.dtype)
    angle = (np.rad2deg(D) + 180) % 180

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            q = 255
            r = 255

            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] < 180):
                q = M[i, j + 1]
                r = M[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = M[i - 1, j + 1]
                r = M[i + 1, j - 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = M[i - 1, j]
                r = M[i + 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = M[i - 1, j - 1]
                r = M[i + 1, j + 1]

            if (M[i, j] >= q) and (M[i, j] >= r):
                suppressed[i, j] = M[i, j]
            else:
                suppressed[i, j] = 0

    return suppressed


def threshold(image, low, high):
    """
    Applies double thresholding to an image.

    Parameters:
    -----------
    image : numpy.ndarray
        The input grayscale image.
    low : float or int
        The low threshold value.
    high : float or int
        The high threshold value.

    Returns:
    --------
    res : numpy.ndarray
        Thresholded image with STRONG, WEAK, or 0 values.
    """
    if low < 1.0 and high <= 1.0:
        highThreshold = image.max() * high
        lowThreshold = image.max() * low
    else:
        highThreshold = high
        lowThreshold = low

    M, N = image.shape
    res = np.zeros((M, N), dtype=np.uint8)

    strong_i, strong_j = np.where(image >= highThreshold)
    weak_i, weak_j = np.where((image >= lowThreshold) & (image < highThreshold))

    res[strong_i, strong_j] = STRONG
    res[weak_i, weak_j] = WEAK

    return res


def tracking(image):
    """
    Performs edge tracking by hysteresis for the Canny edge detection algorithm.

    Parameters:
    -----------
    image : numpy.ndarray
        Thresholded image (output from threshold function).

    Returns:
    --------
    image : numpy.ndarray
        Final edge map after edge tracking.
    """
    rows, cols = image.shape
    q = deque(zip(*np.where(image == STRONG)))
    while q:
        i, j = q.popleft()
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    if image[ni, nj] == WEAK:
                        image[ni, nj] = STRONG
                        q.append((ni, nj))

    image[image == WEAK] = 0
    return image


def rgb_to_grayscale(image):
    """
    Convert RGB/BGR image to grayscale using luminosity method.

    Parameters:
    -----------
    image : numpy.ndarray
        Color image (H, W, 3).

    Returns:
    --------
    gray : numpy.ndarray
        Grayscale image (H, W).
    """
    # BGR weights (OpenCV loads as BGR)
    return (0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]).astype(np.uint8)


def dilate_manual(image, kernel_size=3, iterations=1):
    """
    Manual morphological dilation.

    Parameters:
    -----------
    image : numpy.ndarray
        Binary image (0 and 255 values).
    kernel_size : int
        Size of the structuring element (must be odd).
    iterations : int
        Number of times to apply dilation.

    Returns:
    --------
    dilated : numpy.ndarray
        Dilated image.
    """
    result = image.copy()
    pad = kernel_size // 2

    for _ in range(iterations):
        padded = np.pad(result, pad, mode='constant', constant_values=0)
        output = np.zeros_like(result)

        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                region = padded[i:i + kernel_size, j:j + kernel_size]
                output[i, j] = np.max(region)

        result = output

    return result


def erode_manual(image, kernel_size=3, iterations=1):
    """
    Manual morphological erosion.

    Parameters:
    -----------
    image : numpy.ndarray
        Binary image (0 and 255 values).
    kernel_size : int
        Size of the structuring element (must be odd).
    iterations : int
        Number of times to apply erosion.

    Returns:
    --------
    eroded : numpy.ndarray
        Eroded image.
    """
    result = image.copy()
    pad = kernel_size // 2

    for _ in range(iterations):
        padded = np.pad(result, pad, mode='constant', constant_values=0)
        output = np.zeros_like(result)

        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                region = padded[i:i + kernel_size, j:j + kernel_size]
                output[i, j] = np.min(region)

        result = output

    return result


def morphological_close(image, kernel_size=3, iterations=1):
    """
    Manual morphological closing (dilation followed by erosion).

    Parameters:
    -----------
    image : numpy.ndarray
        Binary image.
    kernel_size : int
        Size of the structuring element.
    iterations : int
        Number of times to apply.

    Returns:
    --------
    closed : numpy.ndarray
        Closed image.
    """
    dilated = dilate_manual(image, kernel_size, iterations)
    closed = erode_manual(dilated, kernel_size, iterations)
    return closed


def connected_components(binary_image):
    """
    Find connected components in a binary image using flood fill.

    Parameters:
    -----------
    binary_image : numpy.ndarray
        Binary image (0 and 255 values).

    Returns:
    --------
    num_labels : int
        Number of labels (including background as 0).
    labels : numpy.ndarray
        Label matrix where each connected component has a unique label.
    stats : list
        List of (x, y, width, height, area) for each component.
    """
    h, w = binary_image.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 0
    stats = []

    def flood_fill(start_i, start_j, label):
        """BFS flood fill to label a connected component."""
        stack = [(start_i, start_j)]
        pixels = []

        while stack:
            i, j = stack.pop()
            if i < 0 or i >= h or j < 0 or j >= w:
                continue
            if labels[i, j] != 0 or binary_image[i, j] == 0:
                continue

            labels[i, j] = label
            pixels.append((i, j))

            # 8-connectivity
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    stack.append((i + di, j + dj))

        return pixels

    for i in range(h):
        for j in range(w):
            if binary_image[i, j] > 0 and labels[i, j] == 0:
                current_label += 1
                pixels = flood_fill(i, j, current_label)

                if pixels:
                    ys, xs = zip(*pixels)
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    width = max_x - min_x + 1
                    height = max_y - min_y + 1
                    area = len(pixels)
                    stats.append((min_x, min_y, width, height, area))

    return current_label + 1, labels, stats


def draw_line_manual(image, x1, y1, x2, y2, color, thickness=1):
    """
    Draw a line on an image using Bresenham's algorithm.

    Parameters:
    -----------
    image : numpy.ndarray
        The image to draw on.
    x1, y1, x2, y2 : int
        Line endpoints.
    color : tuple
        Color of the line (B, G, R) for color images, or int for grayscale.
    thickness : int
        Line thickness.
    """
    h, w = image.shape[:2]

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    x, y = x1, y1

    while True:
        # Draw with thickness
        for ti in range(-thickness//2, thickness//2 + 1):
            for tj in range(-thickness//2, thickness//2 + 1):
                px, py = x + tj, y + ti
                if 0 <= px < w and 0 <= py < h:
                    image[py, px] = color

        if x == x2 and y == y2:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return image


def canny_edge_detection(image, kernel_size=5, sigma=1.0, low_threshold=50, high_threshold=150):
    """
    Applies Canny edge detection to an image.
    Without cv2 functions.

    Parameters:
    -----------
    image : numpy.ndarray
        The grayscale image.

    Returns:
    --------
    edges : numpy.ndarray
        The edges of the image.
    """
    # Convert to grayscale if not already
    if image.ndim == 3:
        image = rgb_to_grayscale(image)

    # Apply Gaussian blur
    blurred_image = convolve(image, gaussian_kernel(size=kernel_size, sigma=sigma))

    # Apply the Sobel filter
    M, D = sobel_filter(blurred_image=blurred_image)

    # Non-maximum suppression
    supressed_image = supression(M, D)

    # Double thresholding
    threshold_image = threshold(supressed_image, low_threshold, high_threshold)

    # Edge tracking by hysteresis
    tracked_image = tracking(threshold_image)

    return tracked_image


def create_hough_accumulator(image_shape, num_thetas=180):
    """
    Create and initialize the Hough accumulator array for line detection.

    Args:
        image_shape: (height, width) of the input image
        num_thetas: Number of theta bins (default 180 for 1-degree resolution)

    Returns:
        Tuple of (accumulator, rhos, thetas)
    """
    height, width = image_shape
    diagonal = int(np.ceil(np.sqrt(height**2 + width**2)))
    thetas = np.deg2rad(np.arange(-90, 89 + 1))
    rhos = np.arange(-diagonal, diagonal + 1)
    num_rhos = len(rhos)
    accumulator = np.zeros((num_rhos, num_thetas), dtype=np.int32)
    return accumulator, rhos, thetas


def compute_rho(x: int, y: int, theta: float) -> float:
    """
    Compute the rho value for a point (x, y) at angle theta.

    Args:
        x: x-coordinate (column) of the point
        y: y-coordinate (row) of the point
        theta: angle in RADIANS

    Returns:
        rho value (can be negative)
    """
    rho = x * np.cos(theta) + y * np.sin(theta)
    return rho


def weighted_vote(gradient_value: float,
                  max_gradient: float = 255.0,
                  min_weight: float = 0.1) -> float:
    """
    Calculate weighted vote based on gradient magnitude.

    Args:
        gradient_value: Gradient magnitude at the edge pixel (0-255)
        max_gradient: Maximum possible gradient value (default 255)
        min_weight: Minimum vote weight (default 0.1)

    Returns:
        Vote weight between min_weight and 1.0
    """
    weight = max(gradient_value / max_gradient, min_weight)
    return weight


def hough_line_transform(edge_image, use_weighted=True, gradient_magnitude=None):
    """
    Perform Hough Transform for line detection.

    Args:
        edge_image: Binary image where edge pixels have value > 0
        use_weighted: Whether to use weighted voting
        gradient_magnitude: Gradient magnitude for weighted voting

    Returns:
        Tuple of (accumulator, rhos, thetas)
    """
    accumulator, rhos, thetas = create_hough_accumulator(edge_image.shape)

    if accumulator is None:
        raise ValueError("create_hough_accumulator returned None")

    edge_y, edge_x = np.nonzero(edge_image)

    for i in range(len(edge_x)):
        x = edge_x[i]
        y = edge_y[i]
        for theta_idx, theta in enumerate(thetas):
            rho = compute_rho(x, y, theta)
            rho_idx = int(round(rho)) + len(rhos) // 2

            if 0 <= rho_idx < len(rhos):
                if use_weighted and gradient_magnitude is not None:
                    vote = weighted_vote(gradient_magnitude[y, x])
                else:
                    vote = 1
                accumulator[rho_idx, theta_idx] += vote

    return accumulator, rhos, thetas


def find_hough_peaks(accumulator, rhos, thetas, threshold, neighborhood_size=50):
    """
    Find peaks in the Hough accumulator using non-maximum suppression.

    Args:
        accumulator: The Hough accumulator array.
        rhos: Array of rho values.
        thetas: Array of theta values.
        threshold: The minimum vote count to be considered a peak.
        neighborhood_size: Size for non-maximum suppression.

    Returns:
        List of (rho, theta) pairs for detected lines.
    """
    peaks = []

    while True:
        max_val = accumulator.max()

        if max_val < threshold:
            break

        max_idx = np.unravel_index(accumulator.argmax(), accumulator.shape)
        rho_idx, theta_idx = max_idx

        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        peaks.append((rho, theta))

        rho_min = max(0, rho_idx - neighborhood_size // 2)
        rho_max = min(accumulator.shape[0], rho_idx + neighborhood_size // 2 + 1)
        theta_min = max(0, theta_idx - neighborhood_size // 2)
        theta_max = min(accumulator.shape[1], theta_idx + neighborhood_size // 2 + 1)

        accumulator[rho_min:rho_max, theta_min:theta_max] = 0

    return peaks


def filter_horizontal_vertical_lines(lines, angle_tolerance=5):
    """
    Filter lines to keep only horizontal and vertical lines.

    Parameters:
    -----------
    lines : list
        List of (rho, theta) tuples from Hough transform.
    angle_tolerance : float
        Tolerance in degrees.

    Returns:
    --------
    filtered_lines : list
        List of (rho, theta) tuples that are horizontal or vertical.
    """
    filtered_lines = []
    tolerance_rad = np.deg2rad(angle_tolerance)

    for rho, theta in lines:
        is_vertical = abs(theta) < tolerance_rad
        is_horizontal = abs(abs(theta) - np.pi / 2) < tolerance_rad

        if is_vertical or is_horizontal:
            filtered_lines.append((rho, theta))

    return filtered_lines


def snap_line_to_straight(x1, y1, x2, y2, image_shape):
    """
    Snap a nearly horizontal or vertical line to be perfectly straight.

    Parameters:
    -----------
    x1, y1, x2, y2 : int
        Line endpoints.
    image_shape : tuple
        (height, width) of the image.

    Returns:
    --------
    tuple : (x1, y1, x2, y2) snapped to be perfectly horizontal or vertical.
    """
    height, width = image_shape
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if dx > dy:
        avg_y = (y1 + y2) // 2
        return (0, avg_y, width - 1, avg_y)
    else:
        avg_x = (x1 + x2) // 2
        return (avg_x, 0, avg_x, height - 1)


def clip_line_to_dark_regions(x1, y1, x2, y2, binary_mask, min_length=40):
    """
    Clips a line so it only exists in the DARK (Gap) regions of the mask.
    """
    h, w = binary_mask.shape
    segments = []

    if x1 == x2:
        x = max(0, min(w-1, x1))
        col = binary_mask[:, x]

        is_gap = (col == 0)

        padded = np.concatenate(([False], is_gap, [False]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for s, e in zip(starts, ends):
            if e - s >= min_length:
                segments.append((x, s, x, e))

    elif y1 == y2:
        y = max(0, min(h-1, y1))
        row = binary_mask[y, :]

        is_gap = (row == 0)

        padded = np.concatenate(([False], is_gap, [False]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for s, e in zip(starts, ends):
            if e - s >= min_length:
                segments.append((s, y, e, y))

    return segments


def merge_nearby_lines(lines, image_shape, distance_threshold=30):
    """
    Merge lines that are close together and parallel.

    Parameters:
    -----------
    lines : list
        List of (x1, y1, x2, y2) tuples.
    image_shape : tuple
        (height, width) of the image.
    distance_threshold : int
        Maximum distance between lines to merge them.

    Returns:
    --------
    merged_lines : list
        List of merged (x1, y1, x2, y2) tuples.
    """
    if not lines:
        return []

    horizontal = []
    vertical = []

    for line in lines:
        x1, y1, x2, y2 = line
        if y1 == y2:
            horizontal.append(line)
        elif x1 == x2:
            vertical.append(line)

    horizontal.sort(key=lambda l: l[1])
    merged_horizontal = []
    i = 0
    while i < len(horizontal):
        group = [horizontal[i]]
        y_sum = horizontal[i][1]
        j = i + 1
        while j < len(horizontal) and horizontal[j][1] - horizontal[i][1] < distance_threshold:
            group.append(horizontal[j])
            y_sum += horizontal[j][1]
            j += 1

        avg_y = y_sum // len(group)
        min_x = min(l[0] for l in group)
        max_x = max(l[2] for l in group)
        merged_horizontal.append((min_x, avg_y, max_x, avg_y))
        i = j

    vertical.sort(key=lambda l: l[0])
    merged_vertical = []
    i = 0
    while i < len(vertical):
        group = [vertical[i]]
        x_sum = vertical[i][0]
        j = i + 1
        while j < len(vertical) and vertical[j][0] - vertical[i][0] < distance_threshold:
            group.append(vertical[j])
            x_sum += vertical[j][0]
            j += 1

        avg_x = x_sum // len(group)
        min_y = min(l[1] for l in group)
        max_y = max(l[3] for l in group)
        merged_vertical.append((avg_x, min_y, avg_x, max_y))
        i = j

    return merged_horizontal + merged_vertical


def get_line_endpoints_from_polar(rho, theta, image_shape):
    """
    Given a line in polar coordinates (rho, theta) and an image shape,
    compute its two endpoints within image boundaries.

    Args:
        rho: Distance from the origin to the line.
        theta: Angle in radians.
        image_shape: (height, width) tuple.

    Returns:
        Tuple (x1, y1, x2, y2) or None if line is outside image.
    """
    height, width = image_shape
    a = np.cos(theta)
    b = np.sin(theta)

    points = []
    tol = 1e-5

    if abs(b) < tol:
        x = rho
        if 0 <= x < width:
            points.append((int(x), 0))
            points.append((int(x), height-1))

    elif abs(a) < tol:
        y = rho
        if 0 <= y < height:
            points.append((0, int(y)))
            points.append((width-1, int(y)))

    else:
        y0 = rho / b
        if 0 <= y0 < height:
            points.append((0, int(y0)))

        y_w = (rho - (width - 1) * a) / b
        if 0 <= y_w < height:
            points.append((width - 1, int(y_w)))

        x0 = rho / a
        if 0 <= x0 < width:
            points.append((int(x0), 0))

        x_h = (rho - (height - 1) * b) / a
        if 0 <= x_h < width:
            points.append((int(x_h), height - 1))

    unique_points = sorted(list(set(points)))
    if len(unique_points) >= 2:
        return unique_points[0] + unique_points[-1]
    return None


def draw_lines_from_coords(image, lines, color=(0, 0, 255), thickness=3):
    """
    Draw lines on an image from coordinate tuples.

    Args:
        image: The image to draw on (should be a color image).
        lines: A list of (x1, y1, x2, y2) tuples.
        color: The color of the lines.
        thickness: The thickness of the lines.
    """
    for line in lines:
        x1, y1, x2, y2 = line
        draw_line_manual(image, int(x1), int(y1), int(x2), int(y2), color, thickness)

    return image


def save_crops_from_lines(image, lines, mask, output_dir, base_filename):
    """
    Uses the detected separator lines to 'cut' the mask, then finds connected
    components to crop the actual documents.
    """
    h, w = mask.shape

    cut_mask = mask.copy()

    for x1, y1, x2, y2 in lines:
        draw_line_manual(cut_mask, x1, y1, x2, y2, 0, thickness=4)

    num_labels, labels, stats = connected_components(cut_mask)

    count = 0
    min_area = (h * w) * 0.01

    for i in range(len(stats)):
        x, y, cw, ch, area = stats[i]

        if area > min_area:
            crop = image[y:y+ch, x:x+cw]
            crop_name = f"{base_filename}_crop_{count}.jpg"
            cv2.imwrite(os.path.join(output_dir, crop_name), crop)
            count += 1

    print(f"  > Saved {count} crops to {output_dir}")


def process_single_image(image_path, debug_dir=None):
    image_name = os.path.basename(image_path)
    stem = os.path.splitext(image_name)[0]
    print(f"\nProcessing {image_name}...")

    # 1. Load Image
    try:
        gray_image = load_image(image_path)
    except Exception as e:
        print(f"Skipping {image_name}: {e}")
        return None, None

    crop_margin = 10
    gray_image = crop_edges(gray_image, margin=crop_margin)
    height, width = gray_image.shape

    # ---------------------------------------------------------
    # STEP 2: PREPROCESSING (The "Perfected" Mask)
    # ---------------------------------------------------------
    print("  > Generating Blob Mask...")

    # A. Brightness Mask (Base layer)
    g_kernel = gaussian_kernel(size=9, sigma=1.5)
    blurred = convolve(gray_image, g_kernel)

    blob_mask = np.zeros_like(blurred, dtype=np.uint8)
    blob_mask[blurred < 230] = 255

    # B. Edge Mask (Texture layer) - Using our canny_edge_detection
    edges_for_mask = canny_edge_detection(gray_image, kernel_size=5, sigma=1.0, low_threshold=30, high_threshold=100)

    # IMPROVEMENT: Use Morphological CLOSE instead of just Dilate
    # Using manual morphological operations
    edges_closed = morphological_close(edges_for_mask.astype(np.uint8), kernel_size=15, iterations=1)

    # Mild dilation
    edges_dilated = dilate_manual(edges_closed, kernel_size=3, iterations=2)

    # C. Combine using numpy
    blob_mask = np.maximum(blob_mask, edges_dilated)

    # D. Final Cleanup
    blob_mask = morphological_close(blob_mask, kernel_size=5, iterations=3)
    blob_mask = erode_manual(blob_mask, kernel_size=3, iterations=1)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{image_name}_mask.jpg"), blob_mask)

    # ---------------------------------------------------------
    # STEP 3: Edge Detection & Hough
    # ---------------------------------------------------------
    print("  > Hough Transform...")
    canny_final = canny_edge_detection(blob_mask, kernel_size=5, low_threshold=30, high_threshold=100)
    accumulator, rhos, thetas = hough_line_transform(canny_final, use_weighted=False)

    all_lines = find_hough_peaks(accumulator, rhos, thetas, threshold=65, neighborhood_size=45)
    hv_lines = filter_horizontal_vertical_lines(all_lines, angle_tolerance=10)

    detected_segments = []

    # 4. Process Lines: Snap -> CLIP -> Collect
    for rho, theta in hv_lines:
        pts = get_line_endpoints_from_polar(rho, theta, gray_image.shape)
        if pts:
            x1, y1, x2, y2 = pts
            snapped = snap_line_to_straight(x1, y1, x2, y2, gray_image.shape)
            sx1, sy1, sx2, sy2 = snapped

            valid_segments = clip_line_to_dark_regions(sx1, sy1, sx2, sy2, blob_mask, min_length=80)

            detected_segments.extend(valid_segments)

    # 5. Add Image Borders
    detected_segments.append((0, 0, width-1, 0))
    detected_segments.append((0, height-1, width-1, height-1))
    detected_segments.append((0, 0, 0, height-1))
    detected_segments.append((width-1, 0, width-1, height-1))

    # 6. Merge nearby lines
    final_lines = merge_nearby_lines(detected_segments, gray_image.shape, distance_threshold=140)

    print(f"  Final detected lines: {len(final_lines)}")

    # 7. Output Data
    line_data = []
    for line in final_lines:
        x1, y1, x2, y2 = line
        line_data.append([image_name, x1 + crop_margin, y1 + crop_margin, x2 + crop_margin, y2 + crop_margin])

    lines_df = pd.DataFrame(line_data, columns=['filename', 'x1', 'y1', 'x2', 'y2'])

    original_color = cv2.imread(image_path)
    annotated_image = draw_lines_from_coords(original_color, [
        (x1+crop_margin, y1+crop_margin, x2+crop_margin, y2+crop_margin) for x1,y1,x2,y2 in final_lines
    ], color=(0, 0, 255), thickness=10)

    # 9. CROP
    if debug_dir:
        crop_dir = os.path.join(debug_dir, "crops")
        os.makedirs(crop_dir, exist_ok=True)
        original_cropped = crop_edges(cv2.imread(image_path), margin=crop_margin)
        save_crops_from_lines(original_cropped, final_lines, blob_mask, crop_dir, stem)

    return lines_df, annotated_image


if __name__ == "__main__":
    base_dir = "/kaggle/input/images"
    annotated_dir = "annotated_images"
    os.makedirs(annotated_dir, exist_ok=True)

    # Find all images
    image_files = sorted([f for f in os.listdir(base_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if len(image_files) > 0:
        print(f"Processing all {len(image_files)} images.")

    print(f"Files to process: {image_files}")

    all_lines = []
    for image_file in image_files:
        image_path = os.path.join(base_dir, image_file)

        lines_df, annotated_image = process_single_image(image_path, debug_dir=annotated_dir)

        if lines_df is not None:
            all_lines.append(lines_df)
            stem = os.path.splitext(image_file)[0]
            annotated_path = os.path.join(annotated_dir, f"{stem}_annotated.jpg")
            cv2.imwrite(annotated_path, annotated_image)
            print(f"  Saved result: {annotated_path}")

    # Save CSV if successful
    if all_lines:
        combined_df = pd.concat(all_lines, ignore_index=True)
        combined_df.to_csv("lines_data.csv", index=False)
        print("\nDone. Check the 'annotated_images' folder for debug steps!")
    else:
        print("No images were processed successfully.")
