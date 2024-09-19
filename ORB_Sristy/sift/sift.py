import cv2
import numpy as np

def register_images_sift(im1_path, im2_path, scale_factor=0.5):
    # Load the images
    img1 = cv2.imread(im1_path)  # Image to be registered
    if img1 is None:
        print("Image 1 not loaded properly")
        return None

    img2 = cv2.imread(im2_path)  # Reference Image
    if img2 is None:
        print("Image 2 not loaded properly")
        return None

    # Resize the images based on scale_factor
    img1 = cv2.resize(img1, (0, 0), fx=scale_factor, fy=scale_factor)
    img2 = cv2.resize(img2, (0, 0), fx=scale_factor, fy=scale_factor)

    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # Match the descriptors
    matcher = cv2.BFMatcher()
    matches = matcher.match(des1, des2, None)
    matches = sorted(matches, key=lambda x: x.distance)

    # Create empty arrays to store matched keypoints
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    # Extract the matched keypoints
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find homography matrix using RANSAC
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Warp img1 to align with img2
    height, width = img2_gray.shape
    img1_reg = cv2.warpPerspective(img1, h, (width, height))

    # Draw matches
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:200], None)

    # Display results
    if img1_reg is not None:
        cv2.imshow("Registered Image", img1_reg)
        cv2.imshow("Matchpoints Image", img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img1_reg, img3
