# DBSCAN & Spectral Clustering: Step-by-Step, Math Theory & Real-World Cases

An interactive bilingual (Spanish / English) web application built with Python and Streamlit that visually explains **DBSCAN** (*Density-Based Spatial Clustering of Applications with Noise*) and **Spectral Clustering** step-by-step.

Rather than treating clustering algorithms as black-box functions, this application breaks down the mathematical intuition behind non-convex geometries, $\varepsilon$-neighborhoods, Core/Border/Noise classification, $k$-distance elbow plots, similarity graphs ($W$), the Graph Laplacian ($L = D - W$, $L_{\text{sym}}$, $L_{\text{rw}}$), the Fiedler vector ($v_2$), eigengaps, spectral embeddings, and real-world applications.

## Author

**Daniel Edgardo Rodríguez Rivera**
* Location: El Salvador
* LinkedIn: [https://www.linkedin.com/in/daniel-rodriguez-sv/](https://www.linkedin.com/in/daniel-rodriguez-sv/)
* GitHub: [https://github.com/danielrodriguezrivera/](https://github.com/danielrodriguezrivera/)

## Features

* **Bilingual Interface (Spanish & English):** Instant language toggle in the sidebar (`Español` / `English`).
* **1. Why K-Means Fails (Non-Convex Shapes):** Compare K-Means Voronoi boundaries against DBSCAN and Spectral Clustering on interlocking moons, concentric rings, double spirals, and elongated clouds with outliers.
* **2. DBSCAN Step-by-Step (Core, Border, Noise):** Inspect any individual point's $\varepsilon$-neighborhood circle $N_\varepsilon(p)$, visualize density-reachability chains between Core Points, explore Core vs. Border vs. Noise roles, and determine the optimal $\varepsilon$ using the sorted $k$-distance elbow plot.
* **3. Spectral Clustering Step-by-Step (Graph Laplacian):** Walk through all 4 mathematical stages:
  1. Similarity Graph construction (Gaussian RBF kernel vs. $k$-Nearest Neighbors).
  2. Affinity Matrix ($W$), Degree Matrix ($D$), and Graph Laplacian ($L$, $L_{\text{sym}}$, $L_{\text{rw}}$) heatmaps.
  3. Eigenvalue spectrum, Eigengap heuristic, and the Fiedler Vector ($v_2$) step function.
  4. Spectral Embedding space ($v_2$ vs. $v_3$) where non-convex manifolds become linearly separable.
* **4. Real-World Case Studies:**
  * **Seismic Faults & GPS Hotspots (DBSCAN):** Discovering curved tectonic fault lines while filtering isolated background tremors.
  * **Credit Card Fraud & Anomaly Detection (DBSCAN):** Flagging isolated transactions (`label = -1`) as potential fraud alerts with Recall/Precision metrics.
  * **Social Network Community Detection (Spectral Clustering):** Partitioning a communication network via Normalized Cuts ($\text{NCut}$) without Euclidean coordinates.
  * **Computer Vision Image Segmentation (Spectral Clustering):** Segmenting noisy images using spatial + brightness affinity graphs (Shi & Malik's Normalized Cuts).
* **5. High-Dimensional Datasets & Custom CSV:** Benchmark DBSCAN, Spectral Clustering, and K-Means on `Iris`, `Wine`, `Breast Cancer`, `Handwritten Digits`, or upload your own CSV with ARI, NMI, and Silhouette metrics.
* **6. Stress Tests & Master Comparison:** Interactive experiments showing DBSCAN's behavior under varying cluster densities, Spectral Clustering's sensitivity to noisy bridges, and a comprehensive comparison table.
* **Silhouette Coefficient Explainer (Modules 1 & 5):** Intuitive "party tables" analogy for $a(i)$, $b(i)$ and $s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$, an interactive point inspector that draws the distances used for $a(i)$ and $b(i)$, silhouette knife plots, per-point silhouette heat maps, the "silhouette paradox" (K-Means scoring higher while cutting curved shapes in half), and silhouette-vs-$k$ / silhouette-vs-$\varepsilon$ curves for choosing hyperparameters without labels.

## Tech Stack

* **Python:** Core logic and linear algebra operations.
* **Streamlit:** Frontend UI and interactive state management.
* **NumPy, SciPy & Pandas:** Pairwise distances, Graph Laplacian eigendecomposition (`eigh`), and data manipulation.
* **Plotly:** Interactive 2D, contour, network graph, and matrix heatmap visualizations.
* **Scikit-Learn:** DBSCAN, Spectral Clustering, K-Means, NearestNeighbors, and clustering evaluation metrics.

## How to Run Locally

1. Navigate to the project directory:
   ```bash
   cd dbscan-spectral-visualizer
   ```
2. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. Open your browser to the local URL provided in the terminal (usually `http://localhost:8501`).
