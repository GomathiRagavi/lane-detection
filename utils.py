import cv2
import numpy as np

# -----------------------------
# 1️⃣ Perspective Transform
# -----------------------------
def perspective_transform(img):
    h, w = img.shape[:2]

    src = np.float32([
        [w * 0.45, h * 0.63],
        [w * 0.55, h * 0.63],
        [w * 0.1, h],
        [w * 0.95, h]
    ])

    dst = np.float32([
        [w * 0.25, 0],
        [w * 0.75, 0],
        [w * 0.25, h],
        [w * 0.75, h]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, (w, h))
    return warped, Minv


# -----------------------------
# 2️⃣ ROI Mask
# -----------------------------
def region_of_interest(img):
    h, w = img.shape
    mask = np.zeros_like(img)

    polygon = np.array([[
        (0, h),
        (w, h),
        (w, int(h * 0.6)),
        (0, int(h * 0.6))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 1)
    return img * mask


# -----------------------------
# 3️⃣ Improved Thresholding
# -----------------------------
def threshold_pipeline(img):

    blur = cv2.GaussianBlur(img, (5, 5), 0)

    hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
    l = hls[:, :, 1]
    s = hls[:, :, 2]

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)

    if np.max(abs_sobelx) != 0:
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    else:
        scaled_sobel = np.zeros_like(abs_sobelx)

    binary = np.zeros_like(scaled_sobel)

    binary[(scaled_sobel > 30)] = 1
    binary[(s > 100)] = 1

    binary = region_of_interest(binary)

    return binary


# -----------------------------
# 4️⃣ Sliding Window
# -----------------------------
def find_lane_pixels(binary_warped):

    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    midpoint = histogram.shape[0] // 2

    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = binary_warped.shape[0] // nwindows

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    margin = 100
    minpix = 50

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):

        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin

        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))

        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    return nonzerox[left_lane_inds], nonzeroy[left_lane_inds], \
           nonzerox[right_lane_inds], nonzeroy[right_lane_inds]


# -----------------------------
# 5️⃣ Frame Processing
# -----------------------------
def process_frame(frame):

    warped, Minv = perspective_transform(frame)
    binary = threshold_pipeline(warped)

    leftx, lefty, rightx, righty = find_lane_pixels(binary)

    # If not enough lane pixels detected
    if len(leftx) < 500 or len(rightx) < 500:
        return frame, None

    # Fit polynomial
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, frame.shape[0] - 1, frame.shape[0])

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    warp_zero = np.zeros_like(binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))

    newwarp = cv2.warpPerspective(color_warp, Minv,
                                  (frame.shape[1], frame.shape[0]))

    result = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)

    # 🔥 IMPORTANT PART — Get bottom x positions
    bottom_y = frame.shape[0] - 1

    left_bottom_x = int(left_fit[0]*bottom_y**2 +
                        left_fit[1]*bottom_y +
                        left_fit[2])

    right_bottom_x = int(right_fit[0]*bottom_y**2 +
                         right_fit[1]*bottom_y +
                         right_fit[2])

    lane_positions = (left_bottom_x, right_bottom_x)

    return result, lane_positions