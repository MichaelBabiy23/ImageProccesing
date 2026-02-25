"""
Orientation-Consistent RANSAC - Student Starter Code
=====================================================

This module implements a RANSAC variant that uses SIFT orientation
information to improve homography estimation. In addition to the
standard reprojection error check, we verify that the relative
orientation between matched keypoints is consistent with the rotation
component of the homography.

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
        keypoints: List of detected keypoints (with .angle attribute in degrees)
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


def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to the range [-π, π].
    
    This is essential for comparing angles correctly, as angles
    wrap around (e.g., 359° and 1° are close, not 358° apart).
    
    Args:
        angle: Angle in radians (can be any value)
        
    Returns:
        normalized: Angle in range [-π, π]
    """
    # Hint: A robust way is to use atan2(sin(angle), cos(angle))
    # This naturally handles all edge cases and wraparound
    
    normalized = np.arctan2(np.sin(angle), np.cos(angle))
    
    return normalized


def extract_rotation_from_homography(H: np.ndarray) -> float:
    """
    Extract the approximate rotation angle from a homography matrix.
    
    For a homography that is close to a similarity transformation
    (rotation + uniform scaling + translation), the rotation angle
    can be approximated from the upper-left 2×2 submatrix using atan2.
    
    For a pure rotation matrix:
    R = [cos(θ)  -sin(θ)]
        [sin(θ)   cos(θ)]
    
    So θ = atan2(R[1,0], R[0,0]) = atan2(sin(θ), cos(θ))
    
    Args:
        H: 3x3 homography matrix
        
    Returns:
        rotation_angle: Estimated rotation angle in radians, range [-π, π]
    """
    # Hint: Check for division by zero
    if abs(H[2, 2]) > 1e-10:
        H_normalized = H / H[2, 2]
    else:
        H_normalized = H
    
    h11 = H_normalized[0, 0] #x
    h21 = H_normalized[1, 0] # y
    
    # Hint: np.arctan2(y, x) returns angle in [-π, π]
    rotation_angle = np.arctan2(h21, h11)
    
    return rotation_angle


def compute_relative_orientation(kp1: cv2.KeyPoint, kp2: cv2.KeyPoint) -> float:
    """
    Compute the relative orientation between two matched keypoints.
    
    The relative orientation is how much kp2's orientation differs from kp1's.
    This should correspond to the rotation in the transformation.
    
    Note: OpenCV SIFT stores angles in degrees (0-360).
    
    Args:
        kp1: Source keypoint
        kp2: Destination keypoint
        
    Returns:
        relative_angle: kp2.angle - kp1.angle, normalized to [-π, π], in radians
    """
    # Hint: Use np.deg2rad() for conversion
    angle1_rad = np.deg2rad(kp1.angle)
    angle2_rad = np.deg2rad(kp2.angle)
    
    relative_angle = angle2_rad - angle1_rad
    
    relative_angle = normalize_angle(relative_angle)
    
    return relative_angle


def verify_orientation_consistency(H: np.ndarray, 
                                    kp1: cv2.KeyPoint, 
                                    kp2: cv2.KeyPoint,
                                    angle_threshold: float) -> Tuple[bool, float]:
    """
    Verify that the relative orientation between matched keypoints is
    consistent with the rotation component of homography H.
    
    Args:
        H: 3x3 homography matrix
        kp1: Source keypoint
        kp2: Destination keypoint
        angle_threshold: Maximum allowed orientation error in radians
        
    Returns:
        is_consistent: True if orientation is consistent
        angle_error: The computed orientation error in radians
    """
    relative_orientation = compute_relative_orientation(kp1, kp2)
    
    predicted_rotation = extract_rotation_from_homography(H)
    
    # Hint: The error should be the absolute value of the normalized difference
    angle_error = abs(normalize_angle(relative_orientation - predicted_rotation))
    
    is_consistent = angle_error < angle_threshold
    
    return is_consistent, angle_error


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


def check_sample_orientation_consistency(keypoints1: List[cv2.KeyPoint],
                                          keypoints2: List[cv2.KeyPoint],
                                          sample_matches: List[cv2.DMatch],
                                          angle_threshold: float) -> bool:
    """
    Check if the sample matches have mutually consistent orientations.
    
    If the 4 sample points don't agree on relative orientation,
    the homography is likely wrong. This allows early rejection.
    
    Args:
        keypoints1, keypoints2: Keypoint lists
        sample_matches: The 4 matches used for homography estimation
        angle_threshold: Maximum allowed deviation between samples
        
    Returns:
        is_consistent: True if all samples have similar relative orientations
    """
    if len(sample_matches) < 2:
        return True
    
    relative_orientations = []
    for match in sample_matches:
        kp1 = keypoints1[match.queryIdx] # First image
        kp2 = keypoints2[match.trainIdx] # Second image
        rel_orient = compute_relative_orientation(kp1, kp2)
        relative_orientations.append(rel_orient)
    
    # Use the first one as reference, check others against it
    reference = relative_orientations[0]
    for orient in relative_orientations[1:]:
        diff = abs(normalize_angle(orient - reference))
        if diff > angle_threshold:
            return False
    
    return True  # Change this once you implement the checks


def ransac_with_orientation_verification(
    keypoints1: List[cv2.KeyPoint],
    keypoints2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    num_iterations: int = 1000,
    position_threshold: float = 5.0,
    angle_threshold: float = 0.35,  # ~20 degrees in radians
    min_inliers: int = 10,
    use_early_rejection: bool = True
) -> Tuple[Optional[np.ndarray], np.ndarray, dict]:
    """
    RANSAC with orientation-consistency verification for robust homography estimation.
    
    This implementation extends standard RANSAC by:
    1. Checking orientation consistency of sample points (early rejection)
    2. Verifying orientation for all candidate inliers
    
    Args:
        keypoints1: Keypoints from first image
        keypoints2: Keypoints from second image
        matches: Feature matches between the two images
        num_iterations: Number of RANSAC iterations
        position_threshold: Maximum reprojection error for position inlier
        angle_threshold: Maximum orientation error in radians
        min_inliers: Minimum number of inliers required
        use_early_rejection: If True, reject hypotheses early based on sample consistency
        
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
            'rejected_by_orientation': 0,
            'early_rejections': 0,
            'iterations_saved': 0
        }
    
    # Extract matched points
    src_pts = np.array([keypoints1[m.queryIdx].pt for m in matches])
    dst_pts = np.array([keypoints2[m.trainIdx].pt for m in matches])
    
    best_H = None
    best_inlier_count = 0
    best_inlier_mask = np.zeros(len(matches), dtype=bool)
    
    # Statistics
    total_position_inliers = 0
    total_orientation_rejected = 0
    early_rejections = 0
    
    np.random.seed(42)  # For reproducibility
    
    for iteration in range(num_iterations):
        indices = np.random.choice(len(matches), 4, replace=False)
        sample_matches = [matches[i] for i in indices]
        
        if use_early_rejection:
          if not check_sample_orientation_consistency(keypoints1, keypoints2, sample_matches, angle_threshold):
            early_rejections += 1
            continue
        
        src_sample = src_pts[indices]
        dst_sample = dst_pts[indices]
        
        H, _ = cv2.findHomography(src_sample, dst_sample, method=0)
        
        if H is None:
            continue
        
        # Count inliers with both position AND orientation verification
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
                
                is_orient_consistent, _ = verify_orientation_consistency(H, kp1, kp2, angle_threshold)
                
                if is_orient_consistent:
                    full_inlier_mask[i] = True
        
        # Count statistics
        pos_inliers = np.sum(position_inlier_mask)
        full_inliers = np.sum(full_inlier_mask)
        orient_rejected = pos_inliers - full_inliers
        
        if full_inliers > best_inlier_count:
            best_inlier_count = full_inliers
            best_inlier_mask = full_inlier_mask.copy()  # Don't forget .copy()!
            best_H = H
            total_position_inliers = pos_inliers
            total_orientation_rejected = orient_rejected
    
    # Refine homography using all inliers
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
        'rejected_by_orientation': int(total_orientation_rejected),
        'early_rejections': int(early_rejections),
        'iterations_saved': int(early_rejections)
    }
    
    return best_H, best_inlier_mask, stats


# ============================================================================
# Test functions - DO NOT MODIFY
# ============================================================================

def test_normalize_angle():
    """Test angle normalization."""
    assert abs(normalize_angle(0.5) - 0.5) < 0.001, "0.5 should stay 0.5"
    result = normalize_angle(3 * np.pi)
    assert abs(abs(result) - np.pi) < 0.001, f"3π should become ±π, got {result}"
    result = normalize_angle(-3 * np.pi)
    assert abs(abs(result) - np.pi) < 0.001, f"-3π should become ±π, got {result}"
    result = normalize_angle(np.pi)
    assert abs(abs(result) - np.pi) < 0.001, f"π should stay ±π, got {result}"
    assert abs(normalize_angle(10 * np.pi + 0.5) - 0.5) < 0.001, "10π + 0.5 should become 0.5"
    
    print("test_normalize_angle: PASSED")
    return True


def test_extract_rotation_from_homography():
    """Test rotation extraction from homography."""
    H_identity = np.eye(3)
    rot = extract_rotation_from_homography(H_identity)
    assert abs(rot) < 0.01, f"Identity should give 0 rotation, got {rot}"
    
    theta = np.pi / 4
    H_rot45 = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    rot = extract_rotation_from_homography(H_rot45)
    assert abs(rot - theta) < 0.01, f"45° rotation should give π/4, got {rot}"
    
    theta = np.pi / 2
    H_rot90 = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    rot = extract_rotation_from_homography(H_rot90)
    assert abs(rot - theta) < 0.01, f"90° rotation should give π/2, got {rot}"
    
    print("test_extract_rotation_from_homography: PASSED")
    return True


def test_verify_orientation_consistency():
    """Test orientation consistency verification."""
    theta = np.pi / 6
    H = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    kp1_good = cv2.KeyPoint(x=100, y=100, size=10, angle=0)
    kp2_good = cv2.KeyPoint(x=100, y=100, size=10, angle=30)
    
    is_consistent, error = verify_orientation_consistency(H, kp1_good, kp2_good, 0.2)
    assert is_consistent, f"Should be consistent, error={error}"
    
    kp1_bad = cv2.KeyPoint(x=100, y=100, size=10, angle=0)
    kp2_bad = cv2.KeyPoint(x=100, y=100, size=10, angle=90)
    
    is_consistent, error = verify_orientation_consistency(H, kp1_bad, kp2_bad, 0.2)
    assert not is_consistent, f"Should not be consistent, error={error}"
    
    print("test_verify_orientation_consistency: PASSED")
    return True


def test_ransac_with_orientation_verification():
    """Test the full RANSAC with orientation verification."""
    np.random.seed(42)
    
    theta = np.pi / 8
    H_true = np.array([
        [np.cos(theta), -np.sin(theta), 50],
        [np.sin(theta), np.cos(theta), 30],
        [0, 0, 1]
    ])
    
    n_points = 50
    src_points = np.random.rand(n_points, 2) * 200 + 50
    
    src_h = np.column_stack([src_points, np.ones(n_points)])
    dst_h = (H_true @ src_h.T).T
    dst_points = dst_h[:, :2] / dst_h[:, 2:3]
    
    theta_deg = np.rad2deg(theta)
    keypoints1 = []
    keypoints2 = []
    matches = []
    
    for i in range(n_points):
        base_angle = np.random.rand() * 360
        kp1 = cv2.KeyPoint(x=float(src_points[i, 0]), 
                           y=float(src_points[i, 1]), 
                           size=10.0,
                           angle=float(base_angle))
        kp2 = cv2.KeyPoint(x=float(dst_points[i, 0]), 
                           y=float(dst_points[i, 1]), 
                           size=10.0,
                           angle=float((base_angle + theta_deg) % 360))
        
        keypoints1.append(kp1)
        keypoints2.append(kp2)
        matches.append(cv2.DMatch(i, i, 0))
    
    n_outliers = 10
    for i in range(n_outliers):
        idx = n_points + i
        kp1 = cv2.KeyPoint(x=float(np.random.rand() * 200 + 50),
                           y=float(np.random.rand() * 200 + 50),
                           size=10.0,
                           angle=float(np.random.rand() * 360))
        kp2 = cv2.KeyPoint(x=float(np.random.rand() * 200 + 50),
                           y=float(np.random.rand() * 200 + 50),
                           size=10.0,
                           angle=float(np.random.rand() * 360))
        keypoints1.append(kp1)
        keypoints2.append(kp2)
        matches.append(cv2.DMatch(idx, idx, 0))
    
    H_est, inlier_mask, stats = ransac_with_orientation_verification(
        keypoints1, keypoints2, matches,
        num_iterations=500,
        position_threshold=5.0,
        angle_threshold=0.35,
        use_early_rejection=True
    )
    
    assert H_est is not None, "Should find a homography"
    assert stats['best_inliers'] >= 40, f"Should have many inliers, got {stats['best_inliers']}"
    
    print("test_ransac_with_orientation_verification: PASSED")
    return True


if __name__ == "__main__":
    print("Running tests...")
    test_normalize_angle()
    test_extract_rotation_from_homography()
    test_verify_orientation_consistency()
    test_ransac_with_orientation_verification()
    print("\nAll tests passed!")
