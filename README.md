[![DOI](https://zenodo.org/badge/1204802582.svg)](https://doi.org/10.5281/zenodo.19470984)

# 💧 Drop-O-Matic

**Drop-O-Matic** is a lightweight, easy-to-use, open-source Python tool designed for the determination of dynamic contact angles from video measurements using **Sessile Drop** and **Captive Bubble** methods.

Developed for researchers who require a transparent, customizable, ease-to-use and efficient alternative to commercial "black-box" software, Drop-O-Matic excels in processing video sequences where wetting kinetics (advancing and receding angles) are critical.

---

## 💡 Design Philosophy

A primary priority was to develop a tool with a **minimalist user interface**, stripping away burdensome and distracting GUI elements. Instead, the focus is on **high-speed efficiency**, allowing the user to operate the software seamlessly through intuitive mouse interactions and a robust set of keyboard shortcuts. This makes it ideal for processing long video sequences where speed and precision are paramount.

---

## 🚀 Key Features

* **Dual Analysis Modes:** Full support for both Sessile Drop and Captive Bubble experimental setups.
* **Dynamic Kinetics:** Optimized for processing video sequences to extract time-dependent contact angle data.
* **Robust Auto-Detection:** Powered by OpenCV, the tool features intelligent contour detection that handles noisy backgrounds and substrate reflections.
* **Mathematical Precision:** Choice between **Ellipse** and **Circle** fitting models based on the physical profile of the droplet.
* **One-Click Export:** Instantly export analyzed data to `.csv` and save high-resolution `.png` frames with measurement overlays.

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/KrzysztofDorywalski/Drop-O-Matic.git](https://github.com/KrzysztofDorywalski/Drop-O-Matic.git)
    cd Drop-O-Matic
    ```

2.  **Install dependencies:**
    *(Requires Python 3.8+)*
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python drop_o_matic.py
    ```

---

## ⌨️ Control Guide

The interface is designed for high-speed manual and semi-automatic processing using keyboard shortcuts:

| Key | Action |
| :--- | :--- |
| **Left Drag** | Select Region of Interest (ROI) |
| **W** | Auto-detect droplet contour within the selected ROI |
| **Left Click** | Manually add/remove individual contour points |
| **Right Drag** | Draw baseline (Hold **SHIFT** for a perfect horizontal line) |
| **E / O** | Fit **E**llipse or **O**rcle (Circle) model to points |
| **Space** | Log current angles to internal memory |
| **S** | Save a high-res screenshot (`.png`) with overlays |
| **C** | Export all logged data to a `.csv` spreadsheet |
| **A / D** | Navigate through video frames (Previous / Next) |
| **R / Z** | **R**eset all points / **Z**-undo last manual point |
| **X** | Exit ROI mode / Reset zoom to full frame |
| **G** | Toggle Grayscale view |

---

## 📊 Statement of Need

In surface science, commercial goniometer software is often proprietary, expensive, and difficult to adapt for non-standard experimental conditions 

**Drop-O-Matic** was created to provide the scientific community with a transparent tool where the underlying mathematics and image processing steps are fully accessible. It is particularly useful for analyzing wetting dynamics on heterogeneous surfaces where rapid changes in contact angles require frame-by-frame precision.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request.
