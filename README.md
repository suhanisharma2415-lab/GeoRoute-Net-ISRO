# GeoRoute-Net-ISRO
# GeoRoute-Net 🛰️📍
> **Trust-Aware Road Network Extraction & Vulnerability Mapping Platform**  

---

## 👁️ Platform Overview
Standard satellite-based road extraction workflows suffer from severe **spectral blindness**. Interferences like tree canopies, building shadows, and cloud cover fragment continuous roadways into disconnected pixel artifacts. For institutional pipelines like ISRO’s National Natural Resources Management System (NNRMS) and MeitY e-governance systems, these gaps break vector topology, rendering data useless for automated routing, traffic simulations, or emergency response tracking.

**GeoRoute-Net** eliminates this bottleneck. By integrating an **Attention-Guided Vision Engine** with an explainable, **Multiplicative Graph-Healing Framework**, the platform automatically reconstructs occluded road networks. Instead of relying on a black-box automated guess that introduces dangerous layout errors (like blind proximity stitching), it implements a strict **Human-in-the-Loop validation pipeline** that keeps government data inventories trustworthy, continuous, and actionable.

---

## 🗂️ Repository Directory Structure & Assets

This repository is organized into distinct modular layers to support automated processing and institutional verification:
*   `📁 docs/`: Contains comprehensive engineering design papers, system logs, and public sector deployment roadmaps.
*   `📁 architecture/`: Holds structural block flowcharts tracking data ingestion down to the decision-support layers.
*   `📁 prototype/`: Interactive Figma UI wireframe mockups showcasing the Streamlit dashboard design layout.
*   `📁 notebooks/`: Verified executable scripts detailing dataset prep, joint-loss vision testing, and graph centrality benchmarking.
*   `📁 sample_data/`: Pre-tiled satellite imagery sandbox matrices (`input/`) and generated `.GeoJSON` vector layers (`output/`).
*   `📁 src/`: Production-ready full-stack software repository broken down by microservice engine layers (Vision, Trust, Graph, Dashboard).

---

## 📐 The Core Mathematical Innovation
To prevent unfeasible geometric tracking anomalies (such as an AI model blindly connecting perpendicular roads across blocks), GeoRoute-Net enforces a rigid **Multiplicative Gating Framework** to evaluate missing connectivity gaps:

$$\mathbf{R = C \times A \times f(D)}$$

*   **$C$ (Pixel Confidence Threshold):** Softmax probability score extracted directly from the output layer of the Attention U-Net.
*   **$A$ (Angular Alignment Multiplier):** Cosine similarity ($\cos\theta$) of heading vectors between road segments. If trajectories are perpendicular ($\theta \to 90^\circ$), $A \to 0$, causing the entire score to collapse and block false intersection mapping.
*   **$f(D)$ (Distance Decay Function):** Exponential spatial decay function ($e^{-\lambda d}$) that penalizes massive geographical gaps.
