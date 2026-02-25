"""
Scale-Ratio Verification in RANSAC - Student Starter Code
==========================================================

This module implements a RANSAC variant that uses SIFT scale information
to improve inlier detection. In addition to the standard reprojection
error check, we verify that the scale ratio between matched keypoints
is consistent with the local scaling induced by the homography.

Your task: Complete the TODO sections below.

Time limit: 30 minutes
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


def extract_sift_features(image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Extract SIFT keypoints and descriptors from an image.
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        keypoints: List of detected keypoints
        descriptors: Nx128 array of descriptors
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return keypoints, descriptors


def match_features(desc1: np.ndarray, desc2: np.ndarray, 
                   ratio_threshold: float = 0.75) -> List[cv2.DMatch]:
    """
    Match features using BFMatcher with Lowe's ratio test.
    
    Args:
        desc1: Descriptors from first image
        desc2: Descriptors from second image
        ratio_threshold: Lowe's ratio threshold
        
    Returns:
        good_matches: List of good matches passing ratio test
    """
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    
    return good_matches


def compute_jacobian_scale(H: np.ndarray, point: np.ndarray) -> float:
    """
    Compute the local scale factor induced by homography H at a given point.
    
    The local scale is computed from the Jacobian determinant of the
    homography transformation at the specified point.
    
    For homography: [x', y', w']^T = H @ [x, y, 1]^T
    Actual coords: (x'/w', y'/w')
    
    The Jacobian J of this transformation is a 2x2 matrix:
    J = (1/w'^2) * [[h11*w' - h31*x', h12*w' - h32*x'],
                    [h21*w' - h31*y', h22*w' - h32*y']]
    
    The local scale factor is sqrt(|det(J)|).
    
    Args:
        H: 3x3 homography matrix
        point: 2D point (x, y) as numpy array
        
    Returns:
        scale_factor: Local scale factor at the given point
    """
    x, y = point[0], point[1]
    
    # Extract homography elements
    h11, h12, h13 = H[0, 0], H[0, 1], H[0, 2]
    h21, h22, h23 = H[1, 0], H[1, 1], H[1, 2]
    h31, h32, h33 = H[2, 0], H[2, 1], H[2, 2]
    
    w_prime = h31*x + h32*y + h33*1
    
    # Avoid division by zero
    if abs(w_prime) < 1e-10:
        return 1.0
    
    x_prime = h11*x + h12*y + h13
    y_prime = h21*x + h22*y + h23
    
    # Hint: j11 = (h11*w' - h31*x') / w'^2
    w_prime_sq = w_prime**2 # Pre-calculate for efficiency
    j11 = (h11*w_prime - h31*x_prime) / w_prime_sq
    j12 = (h12*w_prime - h32*x_prime) / w_prime_sq
    j21 = (h21*w_prime - h31*y_prime) / w_prime_sq
    j22 = (h22*w_prime - h32*y_prime) / w_prime_sq
    
    det_J = j11 * j22 - j12 * j21
    
    scale_factor = np.sqrt(abs(det_J))
    
    return scale_factor


def verify_scale_consistency(H: np.ndarray, 
                              kp1: cv2.KeyPoint, 
                              kp2: cv2.KeyPoint,
                              scale_threshold: float) -> Tuple[bool, float]:
    """
    Verify that the scale ratio between matched keypoints is consistent
    with the local scaling predicted by homography H.
    
    SIFT keypoint.size represents the diameter of the meaningful keypoint
    neighborhood (approximately 2*sigma).
    
    Args:
        H: 3x3 homography matrix
        kp1: Keypoint from first image
        kp2: Keypoint from second image (matched to kp1)
        scale_threshold: Maximum allowed log-scale error
        
    Returns:
        is_consistent: True if scale is consistent
        scale_error: The computed scale error (log-space)
    """
    scale1 = kp1.size
    scale2 = kp2.size
    
    # Avoid division by zero
    if scale1 < 1e-10 or scale2 < 1e-10:
        return False, float('inf')
    
    observed_ratio = scale2 / scale1
    
    # Hint: Use compute_jacobian_scale with kp1's position
    point = np.array(kp1.pt)
    predicted_ratio = compute_jacobian_scale(H, point)
    
    # Avoid log of zero
    if predicted_ratio < 1e-10 or observed_ratio < 1e-10:
        return False, float('inf')
    
    # Hint: |log(observed) - log(predicted)|
    scale_error = abs(np.log(observed_ratio) - np.log(predicted_ratio))
    
    is_consistent = scale_error < scale_threshold
    
    return is_consistent, scale_error


def compute_reprojection_error(H: np.ndarray, 
                                pt1: np.ndarray, 
                                pt2: np.ndarray) -> float:
    """
    Compute the reprojection error for a point match.
    
    Args:
        H: 3x3 homography matrix
        pt1: Point from first image (x, y)
        pt2: Point from second image (x, y)
        
    Returns:
        error: Euclidean distance between H*pt1 and pt2
    """
    # Convert to homogeneous coordinates
    pt1_h = np.array([pt1[0], pt1[1], 1.0])
    
    # Transform
    pt1_transformed = H @ pt1_h
    
    # Convert back to Cartesian
    if abs(pt1_transformed[2]) < 1e-10:
        return float('inf')
    
    pt1_cart = pt1_transformed[:2] / pt1_transformed[2]
    
    # Euclidean distance
    error = np.sqrt((pt1_cart[0] - pt2[0])**2 + (pt1_cart[1] - pt2[1])**2)
    
    return error


def ransac_with_scale_verification(
    keypoints1: List[cv2.KeyPoint],
    keypoints2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    num_iterations: int = 1000,
    position_threshold: float = 5.0,
    scale_threshold: float = 0.5,
    min_inliers: int = 10
) -> Tuple[Optional[np.ndarray], np.ndarray, dict]:
    """
    RANSAC with scale-ratio verification for robust homography estimation.
    
    This implementation extends standard RANSAC by adding a scale consistency
    check: for each candidate inlier, we verify not only that the reprojection
    error is small, but also that the SIFT scale ratio matches the local
    scaling predicted by the homography.
    
    Args:
        keypoints1: Keypoints from first image
        keypoints2: Keypoints from second image
        matches: Feature matches between the two images
        num_iterations: Number of RANSAC iterations
        position_threshold: Maximum reprojection error for position inlier
        scale_threshold: Maximum log-scale error for scale consistency
        min_inliers: Minimum number of inliers required
        
    Returns:
        best_H: Best homography matrix (or None if not found)
        inlier_mask: Boolean mask indicating inliers
        stats: Dictionary with statistics about the RANSAC process
    """
    if len(matches) < 4:
        return None, np.zeros(len(matches), dtype=bool), {
            'total_matches': len(matches),
            'best_inliers': 0,
            'position_only_inliers': 0,
            'rejected_by_scale': 0
        }
    
    # Extract matched points
    src_pts = np.array([keypoints1[m.queryIdx].pt for m in matches])
    dst_pts = np.array([keypoints2[m.trainIdx].pt for m in matches])
    
    best_H = None
    best_inlier_count = 0
    best_inlier_mask = np.zeros(len(matches), dtype=bool)
    
    # Statistics
    total_position_inliers = 0
    total_scale_rejected = 0
    
    np.random.seed(42)  # For reproducibility
    
    for iteration in range(num_iterations):
        indices = np.random.choice(len(matches), 4, replace=False)
        
        src_sample = src_pts[indices]
        dst_sample = dst_pts[indices]
        
        H, _ = cv2.findHomography(src_sample, dst_sample, method=0)
        
        if H is None:
            continue
        
        position_inlier_mask = np.zeros(len(matches), dtype=bool)
        full_inlier_mask = np.zeros(len(matches), dtype=bool)
        
        for i, match in enumerate(matches):
            kp1 = keypoints1[match.queryIdx]
            kp2 = keypoints2[match.trainIdx]
            
            pt1 = np.array(kp1.pt)
            pt2 = np.array(kp2.pt)
            
            reproj_error = compute_reprojection_error(H, pt1, pt2)
            
            if reproj_error < position_threshold:
                position_inlier_mask[i] = True
                
                is_scale_consistent, _ = verify_scale_consistency(H, kp1, kp2, scale_threshold)
                
                if is_scale_consistent:
                    full_inlier_mask[i] = True
        
        # Count statistics for this iteration
        pos_inliers = np.sum(position_inlier_mask)
        full_inliers = np.sum(full_inlier_mask)
        scale_rejected = pos_inliers - full_inliers
        
        if full_inliers > best_inlier_count:
            best_inlier_count = full_inliers
            best_inlier_mask = full_inlier_mask.copy()
            best_H = H
            total_position_inliers = pos_inliers
            total_scale_rejected = scale_rejected
    
    # Step 4: Refine homography using all inliers
    if best_inlier_count >= min_inliers:
        inlier_src = src_pts[best_inlier_mask]
        inlier_dst = dst_pts[best_inlier_mask]
        
        best_H, _ = cv2.findHomography(
            inlier_src.reshape(-1, 1, 2),
            inlier_dst.reshape(-1, 1, 2),
            method=0
        )
    
    stats = {
        'total_matches': len(matches),
        'best_inliers': int(best_inlier_count),
        'position_only_inliers': int(total_position_inliers),
        'rejected_by_scale': int(total_scale_rejected)
    }
    
    return best_H, best_inlier_mask, stats


# ============================================================================
# Test functions - DO NOT MODIFY
# ============================================================================

def test_compute_jacobian_scale():
    """Test the Jacobian scale computation."""
    # Test 1: Identity homography should give scale = 1
    H_identity = np.eye(3)
    scale = compute_jacobian_scale(H_identity, np.array([100, 100]))
    assert abs(scale - 1.0) < 0.01, f"Identity should give scale 1, got {scale}"
    
    # Test 2: Pure scaling homography
    H_scale = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=float)
    scale = compute_jacobian_scale(H_scale, np.array([50, 50]))
    assert abs(scale - 2.0) < 0.01, f"2x scaling should give scale 2, got {scale}"
    
    # Test 3: Different x and y scaling
    H_scale_xy = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 1]], dtype=float)
    scale = compute_jacobian_scale(H_scale_xy, np.array([50, 50]))
    expected = np.sqrt(6)  # sqrt(2*3)
    assert abs(scale - expected) < 0.01, f"Expected {expected}, got {scale}"
    
    print("test_compute_jacobian_scale: PASSED")
    return True


def test_verify_scale_consistency():
    """Test scale consistency verification."""
    # Create a simple scaling homography
    H = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=float)
    
    # Create mock keypoints with consistent scale ratio
    kp1 = cv2.KeyPoint(x=100, y=100, size=10)
    kp2_good = cv2.KeyPoint(x=200, y=200, size=20)  # 2x scale, consistent
    kp2_bad = cv2.KeyPoint(x=200, y=200, size=10)   # same scale, inconsistent
    
    # Test consistent case
    is_consistent, error = verify_scale_consistency(H, kp1, kp2_good, 0.5)
    assert is_consistent, f"Should be consistent, error={error}"
    
    # Test inconsistent case
    is_consistent, error = verify_scale_consistency(H, kp1, kp2_bad, 0.5)
    assert not is_consistent, f"Should not be consistent, error={error}"
    
    print("test_verify_scale_consistency: PASSED")
    return True


def test_ransac_with_scale_verification():
    """Test the full RANSAC with scale verification."""
    # Use the student's own implementation as the reference
    ref_jacobian = compute_jacobian_scale
    
    np.random.seed(42)
    
    n_points = 50
    src_points = np.random.rand(n_points, 2) * 200 + 50
    
    H_true = np.array([
        [1.2, 0.1, 10],
        [-0.1, 1.3, 20],
        [0.0001, 0.0001, 1]
    ])
    
    src_h = np.column_stack([src_points, np.ones(n_points)])
    dst_h = (H_true @ src_h.T).T
    dst_points = dst_h[:, :2] / dst_h[:, 2:3]
    
    keypoints1 = []
    keypoints2 = []
    matches = []
    
    for i in range(n_points):
        scale1 = 10.0
        local_scale = ref_jacobian(H_true, src_points[i])
        scale2 = scale1 * local_scale
        
        kp1 = cv2.KeyPoint(x=float(src_points[i, 0]), 
                           y=float(src_points[i, 1]), 
                           size=float(scale1))
        kp2 = cv2.KeyPoint(x=float(dst_points[i, 0]), 
                           y=float(dst_points[i, 1]), 
                           size=float(scale2))
        
        keypoints1.append(kp1)
        keypoints2.append(kp2)
        matches.append(cv2.DMatch(i, i, 0))
    
    n_outliers = 10
    for i in range(n_outliers):
        idx = n_points + i
        kp1 = cv2.KeyPoint(x=float(np.random.rand() * 200 + 50),
                           y=float(np.random.rand() * 200 + 50),
                           size=10.0)
        kp2 = cv2.KeyPoint(x=float(np.random.rand() * 200 + 50),
                           y=float(np.random.rand() * 200 + 50),
                           size=10.0)
        keypoints1.append(kp1)
        keypoints2.append(kp2)
        matches.append(cv2.DMatch(idx, idx, 0))
    
    H_est, inlier_mask, stats = ransac_with_scale_verification(
        keypoints1, keypoints2, matches,
        num_iterations=500,
        position_threshold=5.0,
        scale_threshold=0.5
    )
    
    assert H_est is not None, "Should find a homography"
    assert stats['best_inliers'] >= 40, f"Should have many inliers, got {stats['best_inliers']}"
    
    print("test_ransac_with_scale_verification: PASSED")
    return True


if __name__ == "__main__":
    print("Running tests...")
    test_compute_jacobian_scale()
    test_verify_scale_consistency()
    test_ransac_with_scale_verification()
    print("\nAll tests passed!")
