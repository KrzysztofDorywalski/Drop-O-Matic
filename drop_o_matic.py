# -*- coding: utf-8 -*-
"""
Drop-O-Matic v1.0
Automated Contact Angle Measurement Tool

Created by: Krzysztof Dorywalski
License: MIT
Repository: https://github.com/KrzysztofDorywalski/Drop-O-Matic

A lightweight Python tool for precise droplet contour detection and 
contact angle calculation using ellipse/circle fitting and differential geometry.
"""

import cv2
import numpy as np
import pandas as pd
import os

# =========================
# CONFIGURATION
# =========================
# Default path for the repository sample data
VIDEO_PATH = "data/sample_droplet.avi"

video_dir = os.path.dirname(VIDEO_PATH)
video_filename = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_DIR = os.path.join(video_dir, video_filename)

if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

# Subsampling step for auto-detected contours to prevent overfitting
POINTS_SKIP = 10
# Initial UI scale multiplier
SCALE = 0.8  

# Colors in BGR format
C_GREEN = (0, 255, 0)
C_BLUE = (255, 100, 0)
C_RED = (0, 0, 255)
C_YELLOW = (0, 255, 255)

def get_tangent_at_y(model_data, target_y, mode='ellipse'):
    """
    Calculates the tangent(s) at a specific Y-coordinate (baseline) for a given model.
    Uses partial derivatives of the general quadratic curve equation.
    
    Args:
        model_data: Output from cv2.fitEllipse or fit_circle.
        target_y (float): The Y coordinate of the baseline.
        mode (str): 'ellipse' or 'circle'.
        
    Returns:
        list of dicts: Coordinates and slopes of the intersections.
    """
    if mode == 'ellipse':
        (xc, yc), (d1, d2), angle_deg = model_data
        a, b = d1 / 2, d2 / 2
        theta = np.radians(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        # General quadratic form coefficients for the rotated ellipse
        A = (cos_t**2 / a**2) + (sin_t**2 / b**2)
        B = 2 * cos_t * sin_t * (1/a**2 - 1/b**2)
        C = (sin_t**2 / a**2) + (cos_t**2 / b**2)
        D = -2*A*xc - B*yc
        E = -B*xc - 2*C*yc
        F = A*xc**2 + B*xc*yc + C*yc**2 - 1
    else: 
        # Mode: circle
        (xc, yc), r = model_data
        A, B, C = 1, 0, 1
        D, E = -2*xc, -2*yc
        F = xc**2 + yc**2 - r**2

    # Substitute target_y into the general equation to solve for x: a_q*x^2 + b_q*x + c_q = 0
    a_q = A
    b_q = B * target_y + D
    c_q = C * target_y**2 + E * target_y + F
    
    delta = b_q**2 - 4 * a_q * c_q
    if delta < 0: 
        return [] # No intersection with the baseline

    results = []
    for sign in [-1, 1]:
        x = (-b_q + sign * np.sqrt(delta)) / (2 * a_q)
        den = (B * x + 2 * C * target_y + E)
        
        # Calculate slope (dy/dx) using implicit differentiation
        slope = -(2 * A * x + B * target_y + D) / den if abs(den) > 1e-9 else 1e9
        results.append({'x': x, 'y': target_y, 'slope': slope})
        
    return results

def fit_circle(pts):
    """
    Fits a circle to a set of 2D points using the linear least squares method.
    """
    pts = np.array(pts, dtype=np.float32)
    x, y = pts[:, 0], pts[:, 1]
    
    A = np.c_[x, y, np.ones(len(x))]
    b = x**2 + y**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    xc, yc = c[0]/2, c[1]/2
    r = np.sqrt(c[2] + xc**2 + yc**2)
    
    return (xc, yc), r

def auto_detect_contour(frame, roi_rect, base_y=None):
    """
    Detects the largest contour within the specified Region of Interest (ROI).
    Assumes a dark droplet on a lighter background (standard backlighting).
    """
    r_x, r_y, r_w, r_h = roi_rect
    roi = frame[r_y:r_y+r_h, r_x:r_x+r_w]
    
    # Preprocessing: Grayscale, blur, and Otsu's thresholding
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
        
    # Isolate the main droplet body
    largest_contour = max(contours, key=cv2.contourArea)
    
    new_points = []
    # Subsample points to optimize the fitting algorithm
    for pt in largest_contour[::POINTS_SKIP]:
        cx, cy = pt[0]
        
        # Translate coordinates back to the global frame
        global_x = r_x + cx
        global_y = r_y + cy
        
        # Exclude points that fall below or directly on the baseline (with a 3px margin)
        # This prevents reflections from corrupting the fit
        if base_y is not None and global_y >= base_y - 3:
            continue
            
        new_points.append((global_x, global_y))
        
    return new_points

def print_menu():
    print("\n" + "="*55)
    print("               💧 DROP-O-MATIC - CONTROLS")
    print("="*55)
    print(" [Left Drag]  - Select Region of Interest (ROI)")
    print(" [W]          - Auto-detect droplet contour in ROI")
    print(" [Left Click] - Add/Remove contour point (Manual mode)")
    print(" [Right Drag] - Draw baseline (Hold SHIFT for horizontal)")
    print(" [E] - Fit ELLIPSE | [O] - Fit CIRCLE")
    print(" [Z] - Undo point  | [R] - Reset all points")
    print(" [X] - Cancel ROI / Zoom out to full frame")
    print(" [G] - Toggle Grayscale view")
    print(" [A/D] - Prev/Next Frame | [+/-] - Adjust Manual Zoom")
    print(" [SPACE] - Log current angles and save to memory")
    print(" [S] - Save PNG screenshot | [C] - Export data to CSV")
    print("="*55 + "\n")

def main():
    global SCALE
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): 
        print(f"Error: Cannot open video at {VIDEO_PATH}")
        return
    
    print_menu()
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load all frames into memory for fast scrubbing (suitable for short clips)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    current_idx, points, base_pts = 0, [], []
    is_drawing_base, show_gray, fitted_model, mode = False, False, None, 'ellipse'
    results_list = []

    # UI Interaction variables
    roi_rect = None
    left_down_pos = None
    left_drag_pos = None
    is_left_dragging = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal points, base_pts, is_drawing_base, roi_rect, left_down_pos, left_drag_pos, is_left_dragging
        global SCALE
        
        # Coordinate mapping based on current view scale and ROI
        if roi_rect is not None:
            roi_x, roi_y, _, _ = roi_rect
            rx = int(x / SCALE) + roi_x
            ry = int(y / SCALE) + roi_y
        else:
            rx = int(x / SCALE)
            ry = int(y / SCALE)

        # --- LEFT CLICK: ROI Drag or Manual Point Placement ---
        if event == cv2.EVENT_LBUTTONDOWN:
            left_down_pos = (rx, ry)
            is_left_dragging = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_LBUTTON and left_down_pos is not None:
                dx = rx - left_down_pos[0]
                dy = ry - left_down_pos[1]
                if dx**2 + dy**2 > 25: 
                    is_left_dragging = True
                    left_drag_pos = (rx, ry)
            
            if is_drawing_base:
                if flags & cv2.EVENT_FLAG_SHIFTKEY:
                    base_pts[1] = (rx, base_pts[0][1]) # Snap to horizontal
                else:
                    base_pts[1] = (rx, ry)

        elif event == cv2.EVENT_LBUTTONUP:
            if is_left_dragging and left_down_pos and left_drag_pos:
                # Finalize ROI selection
                x1, y1 = left_down_pos
                x2, y2 = rx, ry
                r_x, r_y = min(x1, x2), min(y1, y2)
                r_w, r_h = abs(x2 - x1), abs(y2 - y1)
                
                if r_w > 20 and r_h > 20:
                    roi_rect = [r_x, r_y, r_w, r_h]
                    SCALE = max(0.5, 600.0 / r_h) # Auto-adjust scale for ROI
            else:
                # Logic for adding or deleting single points
                clicked_near_point = False
                threshold = 10 # Pixel radius for point deletion
                
                for i, p in enumerate(points):
                    dist_sq = (p[0] - rx)**2 + (p[1] - ry)**2
                    if dist_sq < threshold**2:
                        points.pop(i)
                        clicked_near_point = True
                        break 
                
                if not clicked_near_point:
                    points.append((rx, ry))
                
            left_down_pos = None
            is_left_dragging = False

        # --- RIGHT CLICK: Baseline Placement ---
        elif event == cv2.EVENT_RBUTTONDOWN:
            base_pts = [(rx, ry), (rx, ry)]
            is_drawing_base = True
            
        elif event == cv2.EVENT_RBUTTONUP:
            if is_drawing_base:
                if flags & cv2.EVENT_FLAG_SHIFTKEY:
                    base_pts[1] = (rx, base_pts[0][1])
                else:
                    base_pts[1] = (rx, ry)
                is_drawing_base = False

    cv2.namedWindow("Drop-O-Matic")
    cv2.setMouseCallback("Drop-O-Matic", mouse_callback)

    while True:
        display = frames[current_idx].copy()
        if show_gray: 
            display = cv2.cvtColor(cv2.cvtColor(display, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        
        # Render points and baseline
        for p in points: 
            cv2.circle(display, p, 3, C_GREEN, -1)
        if len(base_pts) == 2: 
            cv2.line(display, base_pts[0], base_pts[1], C_GREEN, 1)

        # Render ROI drag box
        if is_left_dragging and left_down_pos and left_drag_pos:
            cv2.rectangle(display, left_down_pos, left_drag_pos, (255, 0, 255), 2)

        angL, angR = 0.0, 0.0
        
        # Model Fitting and Angle Calculation
        if fitted_model and len(base_pts) == 2:
            if mode == 'ellipse': 
                cv2.ellipse(display, fitted_model, C_BLUE, 2)
            else: 
                cv2.circle(display, (int(fitted_model[0][0]), int(fitted_model[0][1])), int(fitted_model[1]), C_YELLOW, 2)
            
            base_y = (base_pts[0][1] + base_pts[1][1]) / 2
            intersections = sorted(get_tangent_at_y(fitted_model, base_y, mode), key=lambda i: i['x'])
            
            if len(intersections) >= 2:
                for idx, side in enumerate(["L", "R"]):
                    item = intersections[idx]
                    cx, cy = int(item['x']), int(item['y'])
                    dx, dy = 1.0, item['slope']
                    mag = np.sqrt(dx**2 + dy**2)
                    dx, dy = dx/mag, dy/mag
                    
                    if dy > 0: dx, dy = -dx, -dy
                    
                    raw_angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
                    if side == "L":
                        angle_deg = raw_angle if dx > 0 else 180 - raw_angle
                        angL = angle_deg
                        pos_x = max(20, cx - 280) 
                    else:
                        angle_deg = raw_angle if dx < 0 else 180 - raw_angle
                        angR = angle_deg
                        pos_x = min(display.shape[1] - 350, cx + 50) 

                    # Draw tangents and text
                    cv2.line(display, (cx, cy), (int(cx + 150*dx), int(cy + 150*dy)), C_RED, 3)
                    cv2.putText(display, f"{side}: {angle_deg:.1f}", (pos_x, int(base_y)-40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, C_RED, 4)

        # Handle Viewport rendering (Full screen vs ROI)
        if roi_rect is not None:
            r_x, r_y, r_w, r_h = roi_rect
            H, W = display.shape[:2]
            r_x, r_y = max(0, r_x), max(0, r_y)
            r_w, r_h = min(r_w, W - r_x), min(r_h, H - r_y)
            
            roi_display = display[r_y:r_y+r_h, r_x:r_x+r_w]
            view = cv2.resize(roi_display, None, fx=SCALE, fy=SCALE)
        else:
            view = cv2.resize(display, None, fx=SCALE, fy=SCALE)

        # UI Overlay
        cv2.putText(view, f"Frame: {current_idx+1}/{total_frames}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_GREEN, 2)
        
        if roi_rect is not None:
            cv2.putText(view, "ROI Mode [X - Exit, W - Auto-detect]", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # Render average angle at the bottom
        if angL > 0 or angR > 0:
            avg_angle = (angL + angR) / 2
            v_H, v_W = view.shape[:2]
            text_avg = f"Avg: {avg_angle:.1f}"
            
            text_size = cv2.getTextSize(text_avg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = (v_W - text_size[0]) // 2
            text_y = v_H - 20 
            
            cv2.putText(view, text_avg, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_YELLOW, 2)

        cv2.imshow("Drop-O-Matic", view)
        
        # Keyboard listener
        key = cv2.waitKey(20) & 0xFF
        if key == 27: 
            break # ESC to exit
        elif key in [ord('w'), ord('W')]: 
            if roi_rect is not None:
                current_base_y = None
                if len(base_pts) == 2:
                    current_base_y = (base_pts[0][1] + base_pts[1][1]) / 2

                auto_pts = auto_detect_contour(frames[current_idx], roi_rect, current_base_y)
                if auto_pts:
                    points.extend(auto_pts)
                    print(f"[{current_idx}] Auto-detect: added {len(auto_pts)} points.")
                else:
                    print("Droplet not found or all points fall below the baseline.")
            else:
                print("Please select an ROI first using Left Mouse Drag!")
                
        elif key in [ord('e'), ord('E')]: 
            if len(points) >= 5: fitted_model, mode = cv2.fitEllipse(np.array(points)), 'ellipse'
        elif key in [ord('o'), ord('O')]: 
            if len(points) >= 3: fitted_model, mode = fit_circle(points), 'circle'
        elif key in [ord('z'), ord('Z')]:  
            if points: points.pop()
            fitted_model = None 
        elif key in [ord('r'), ord('R')]: 
            points, fitted_model = [], None
        elif key in [ord('x'), ord('X')]:
            roi_rect = None
            SCALE = 0.8 
        elif key == ord(' '): 
            avg = (angL + angR) / 2
            print(f"[{mode.upper()}] Frame {current_idx}: L={angL:.1f} R={angR:.1f} Avg={avg:.1f}")
            results_list.append({
                "Frame": current_idx, 
                "Model": mode, 
                "Left_Angle": angL, 
                "Right_Angle": angR, 
                "Average_Angle": avg
            })
        elif key in [ord('s'), ord('S')]: 
            fname = os.path.join(OUTPUT_DIR, f"frame_{current_idx}_{mode}.png")
            cv2.imwrite(fname, display) 
            print(f"Saved screenshot: {fname}")
        elif key in [ord('c'), ord('C')]:
            if results_list:
                fname = os.path.join(OUTPUT_DIR, f"{video_filename}_results.csv")
                pd.DataFrame(results_list).to_csv(fname, index=False)
                print(f"Saved CSV report: {fname}")
                
        elif key in [ord('d'), ord('D'), ord('a'), ord('A')]:
            current_idx = min(current_idx + 1, total_frames-1) if key in [ord('d'), ord('D')] else max(current_idx - 1, 0)
            points, fitted_model = [], None
            
        elif key in [ord('='), ord('+')]: 
            SCALE += 0.1
        elif key in [ord('-'), ord('_')]: 
            SCALE = max(0.1, SCALE - 0.1)
        elif key in [ord('g'), ord('G')]: 
            show_gray = not show_gray

    cv2.destroyAllWindows()

if __name__ == "__main__": 
    main()