import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.linalg import eigh
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_blobs,
    make_circles,
    make_moons,
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
st.set_page_config(
    page_title="DBSCAN & Spectral Clustering: Math & Visualization",
    layout="wide",
)

# --- Sidebar: Author Info ---
with st.sidebar:
    st.header("Author")
    st.markdown("Daniel Edgardo Rodríguez Rivera")
    st.markdown("El Salvador")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.link_button("LinkedIn", "https://www.linkedin.com/in/daniel-rodriguez-sv/")
    with col2:
        st.link_button("GitHub", "https://github.com/danielrodriguezrivera/")
    st.markdown("---")

# --- Language Selector ---
lang_choice = st.sidebar.radio(
    "Language / Idioma:",
    ["Español", "English"],
    horizontal=True,
)
ES = lang_choice == "Español"
st.sidebar.markdown("---")


def tr(en_text: str, es_text: str) -> str:
    """Return Spanish or English text based on selected language."""
    return es_text if ES else en_text


st.title(
    tr(
        "DBSCAN & Spectral Clustering: Step-by-Step, Math Theory & Real-World Cases",
        "DBSCAN y Clustering Espectral: Paso a Paso, Teoría Matemática y Casos Reales",
    )
)
st.markdown(
    tr(
        "Explore how modern clustering algorithms discover arbitrary shapes, nonlinear manifolds, communities, and anomalies where traditional K-Means fails. Use the sidebar to navigate through the interactive modules.",
        "Explora cómo los algoritmos modernos de agrupamiento descubren formas arbitrarias, estructuras no lineales, comunidades y anomalías donde el clásico K-Means falla. Usa la barra lateral para navegar por los módulos interactivos.",
    )
)

with st.expander(
    tr(
        "Read the Mathematical Theory & Analogies (DBSCAN vs. Spectral Clustering)",
        "Leer la Teoría Matemática y Analogías (DBSCAN vs. Clustering Espectral)",
    ),
    expanded=False,
):
    if not ES:
        st.markdown(
            r"""
            ### 1. DBSCAN: The "Archipelago of Dense Islands" Analogy
            **DBSCAN** (*Density-Based Spatial Clustering of Applications with Noise*, Ester et al., 1996) defines clusters as **continuous regions of high point density** separated by oceans of low density (noise).
            Imagine walking from person to person in a crowd with a hula-hoop of radius $\varepsilon$ (`eps`):
            - **$\varepsilon$-Neighborhood ($N_\varepsilon(p)$):** All points within distance $\varepsilon$ of point $p$:
              $$N_\varepsilon(p) = \{q \in \mathcal{D} \mid \text{dist}(p, q) \le \varepsilon\}$$
            - **Core Point (*Punto Núcleo*):** A point in a crowded area that has at least $\text{MinPts}$ neighbors inside its hula-hoop (including itself):
              $$|N_\varepsilon(p)| \ge \text{MinPts}$$
            - **Border Point (*Punto Frontera*):** Has fewer than $\text{MinPts}$ neighbors in its own hoop ($|N_\varepsilon(p)| < \text{MinPts}$), but sits inside the hoop of a Core Point (the shore of the island).
            - **Noise Point / Outlier (*Ruido / Anomalía*):** Any isolated point that is neither a Core Point nor a Border Point (assigned label `-1`).
            - **Density-Reachability:** Clusters grow like a chain reaction: if $p$ is a Core Point, every point inside $N_\varepsilon(p)$ joins its cluster, and any new Core Point in that hoop expands the chain further.

            ---

            ### 2. Spectral Clustering: The "Bridges & Vibrating Drumhead" Analogy
            **Spectral Clustering** (Ng, Jordan & Weiss, 2001; Shi & Malik, 2000) treats data points as **nodes in a network graph** connected by weighted bridges, and uses the **eigenvalues and eigenvectors ("spectrum") of the Graph Laplacian** to cut the weakest bridges:
            1. **Similarity / Affinity Matrix ($W \in \mathbb{R}^{n \times n}$):** Measures how strongly point $i$ and point $j$ are connected, typically via a Gaussian RBF Kernel or $k$-Nearest Neighbors:
               $$W_{ij} = \exp\left(-\gamma \|x_i - x_j\|^2\right) = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)$$
            2. **Degree Matrix ($D$) & Graph Laplacian ($L$):** The degree $D_{ii} = \sum_j W_{ij}$ measures the total connection weight of node $i$. The unnormalized **Graph Laplacian** is $L = D - W$, which satisfies the fundamental quadratic identity:
               $$f^T L f = \frac{1}{2} \sum_{i,j=1}^{n} W_{ij} (f_i - f_j)^2$$
               To find $k$ clusters without cutting strong edges, we want vectors $f$ that make $f^T L f$ as small as possible (constant within each cluster, jumping only across weak bridges).
            3. **Normalized Graph Laplacians:**
               $$L_{\text{sym}} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} W D^{-1/2}, \qquad L_{\text{rw}} = D^{-1} L = I - D^{-1} W$$
            4. **Spectral Embedding & K-Means:** Compute the bottom $k$ eigenvectors $U = [v_1, v_2, \dots, v_k] \in \mathbb{R}^{n \times k}$ corresponding to the **smallest eigenvalues** $0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_k$ (where $v_2$ is the famous **Fiedler Vector**). In this new $k$-dimensional coordinates table, tangled rings and moons become simple, separate blobs that standard K-Means separates effortlessly.
            """
        )
    else:
        st.markdown(
            r"""
            ### 1. DBSCAN: La Analogía del "Archipiélago de Islas Densas"
            **DBSCAN** (*Density-Based Spatial Clustering of Applications with Noise*, Ester et al., 1996) define los grupos como **regiones continuas de alta densidad de puntos** separadas por océanos vacíos (ruido).
            Imagina caminar entre personas usando un aro (hula-hula) de radio $\varepsilon$ (`eps`):
            - **Vecindario-$\varepsilon$ ($N_\varepsilon(p)$):** Todos los puntos dentro de un radio $\varepsilon$ alrededor del punto $p$:
              $$N_\varepsilon(p) = \{q \in \mathcal{D} \mid \text{dist}(p, q) \le \varepsilon\}$$
            - **Punto Núcleo (*Core Point*):** Un punto en una zona densa que contiene al menos $\text{MinPts}$ puntos dentro de su aro (incluyéndose a sí mismo):
              $$|N_\varepsilon(p)| \ge \text{MinPts}$$
            - **Punto Frontera (*Border Point*):** Tiene menos de $\text{MinPts}$ vecinos en su propio aro ($|N_\varepsilon(p)| < \text{MinPts}$), pero cae dentro del aro de algún Punto Núcleo (la orilla de la isla).
            - **Punto de Ruido / Anomalía (*Noise / Outlier*):** Cualquier punto aislado que no es ni Núcleo ni Frontera (recibe la etiqueta `-1`).
            - **Alcanzabilidad por Densidad:** Los clusters crecen en cadena: si $p$ es un Punto Núcleo, todos sus vecinos en $N_\varepsilon(p)$ se unen a su isla, y cualquier otro Punto Núcleo alcanzado expande el grupo sucesivamente.

            ---

            ### 2. Clustering Espectral: La Analogía de los "Puentes y el Tambor Vibrante"
            El **Clustering Espectral** (Ng, Jordan & Weiss, 2001; Shi & Malik, 2000) convierte los datos en una **red (grafo) de nodos conectados por puentes**, y utiliza los **autovalores y autovectores ("espectro") del Laplaciano del Grafo** para cortar únicamente los puentes más débiles:
            1. **Matriz de Similitud o Afinidad ($W \in \mathbb{R}^{n \times n}$):** Mide qué tan cerca están los puntos $i$ y $j$, usando un Kernel Gaussiano RBF o los $k$ Vecinos Más Cercanos ($k$-NN):
               $$W_{ij} = \exp\left(-\gamma \|x_i - x_j\|^2\right) = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)$$
            2. **Matriz de Grados ($D$) y Laplaciano del Grafo ($L$):** El grado $D_{ii} = \sum_j W_{ij}$ suma todas las conexiones del nodo $i$. El **Laplaciano no normalizado** es $L = D - W$, el cual cumple una identidad cuadrática fundamental:
               $$f^T L f = \frac{1}{2} \sum_{i,j=1}^{n} W_{ij} (f_i - f_j)^2$$
               Para separar grupos cortando la menor cantidad posible de puentes fuertes, buscamos vectores $f$ que hagan $f^T L f$ mínimo (valores constantes dentro de cada isla, cambiando solo entre islas distintas).
            3. **Laplacianos Normalizados:**
               $$L_{\text{sym}} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} W D^{-1/2}, \qquad L_{\text{rw}} = D^{-1} L = I - D^{-1} W$$
            4. **Incrustación Espectral (*Spectral Embedding*) y K-Means:** Calculamos los primeros $k$ autovectores $U = [v_1, v_2, \dots, v_k] \in \mathbb{R}^{n \times k}$ asociados a los **autovalores más pequeños** $0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_k$ (donde $v_2$ es el famoso **Vector de Fiedler**). En este nuevo espacio espectral, anillos y medias lunas se transforman en grupos compactos que K-Means separa sin esfuerzo.
            """
        )

# --- Sidebar Navigation ---
st.sidebar.header(tr("1. Choose a Topic", "1. Elige un Módulo"))
MODULES = [
    tr("1. Why K-Means Fails (Non-Convex Shapes)", "1. Por Qué Falla K-Means (Formas No Convexas)"),
    tr("2. DBSCAN Step-by-Step (Core, Border, Noise)", "2. DBSCAN Paso a Paso (Núcleo, Frontera y Ruido)"),
    tr("3. Spectral Clustering Step-by-Step (Laplacian)", "3. Clustering Espectral Paso a Paso (Laplaciano)"),
    tr("4. Real-World Cases (GPS, Fraud, Networks, Images)", "4. Casos del Mundo Real (GPS, Fraude, Redes, Imágenes)"),
    tr("5. High-D Datasets & Custom CSV", "5. Datos de Alta Dimensión y Tu Propio CSV"),
    tr("6. Stress Tests & Master Comparison", "6. Pruebas de Estrés y Tabla Comparativa"),
]
mode_idx = st.sidebar.radio(
    tr("Explore Step-by-Step:", "Explora Paso a Paso:"),
    range(6),
    format_func=lambda i: MODULES[i],
)


# ==============================================================================
# HELPER FUNCTIONS FOR DATASETS & SPECTRAL DECOMPOSITION
# ==============================================================================
@st.cache_data
def generate_toy_dataset(
    pattern_idx: int, n_samples: int, noise: float, n_outliers: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate 2D non-convex datasets with optional uniform background noise."""
    rng = np.random.default_rng(int(seed))
    if pattern_idx == 0:
        # Interlocking Crescent Moons
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=int(seed))
    elif pattern_idx == 1:
        # Concentric Bullseye Rings
        X, y = make_circles(n_samples=n_samples, factor=0.4, noise=noise, random_state=int(seed))
    elif pattern_idx == 2:
        # Double Spirals
        n_half = n_samples // 2
        theta = np.sqrt(rng.uniform(0.15, 1.0, n_half)) * 2.6 * np.pi
        r = theta / (2.6 * np.pi) * 2.2
        x1 = np.column_stack([r * np.cos(theta), r * np.sin(theta)]) + rng.normal(0, noise, (n_half, 2))
        x2 = np.column_stack([-r * np.cos(theta), -r * np.sin(theta)]) + rng.normal(0, noise, (n_half, 2))
        X = np.vstack([x1, x2])
        y = np.hstack([np.zeros(n_half, dtype=int), np.ones(n_half, dtype=int)])
    else:
        # Anisotropic (stretched) diagonal cigars + 1 round cluster
        X_b, y = make_blobs(
            n_samples=n_samples,
            centers=[[-1.5, -0.5], [0.8, 0.5], [2.2, -1.8]],
            cluster_std=[0.55, 0.55, 0.35],
            random_state=int(seed),
        )
        transformation = np.array([[0.6, -0.65], [-0.4, 0.85]])
        mask_stretch = y < 2
        X = X_b.copy()
        X[mask_stretch] = X_b[mask_stretch] @ transformation
        X += rng.normal(0, noise * 0.6, X.shape)

    X = StandardScaler().fit_transform(X)

    if n_outliers > 0:
        mins = X.min(axis=0) - 0.5
        maxs = X.max(axis=0) + 0.5
        outliers = rng.uniform(mins, maxs, size=(n_outliers, 2))
        X = np.vstack([X, outliers])
        y = np.hstack([y, np.full(n_outliers, -1, dtype=int)])

    return X, y


def classify_dbscan_points(X: np.ndarray, eps: float, min_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run DBSCAN and return labels, point roles ('Core', 'Border', 'Noise'), and neighbor counts."""
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    core_mask = np.zeros(len(X), dtype=bool)
    core_mask[db.core_sample_indices_] = True

    nn = NearestNeighbors(radius=eps)
    nn.fit(X)
    neighbors_list = nn.radius_neighbors(X, return_distance=False)
    n_neighbors = np.array([len(nb) for nb in neighbors_list])

    roles = np.where(core_mask, "Core", np.where(labels != -1, "Border", "Noise"))
    return labels, roles, n_neighbors


@st.cache_data
def compute_spectral_step_by_step(
    X: np.ndarray,
    affinity_type: str,
    gamma: float,
    n_neighbors: int,
    laplacian_type: str,
    n_clusters: int,
    seed: int = 42,
):
    """Explicitly compute Affinity Matrix W, Degree Matrix D, Laplacian L, Eigenvalues/Vectors, and Clusters."""
    n = len(X)
    sq_dists = squareform(pdist(X, metric="sqeuclidean"))

    if affinity_type == "rbf":
        W = np.exp(-gamma * sq_dists)
        np.fill_diagonal(W, 0.0)
    else:
        k = min(int(n_neighbors), n - 1)
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X)
        dists, indices = nn.kneighbors(X)
        W = np.zeros((n, n), dtype=float)
        rows = np.repeat(np.arange(n), k)
        cols = indices[:, 1:].ravel()
        # Weight k-NN edges by local RBF similarity for smooth spectrum
        med_sq = np.median(sq_dists) + 1e-8
        W[rows, cols] = np.exp(-sq_dists[rows, cols] / (2.0 * med_sq))
        W = 0.5 * (W + W.T)

    degrees = np.sum(W, axis=1)
    D = np.diag(degrees)
    L_unnorm = D - W

    if laplacian_type == "unnormalized":
        L = L_unnorm
    elif laplacian_type == "sym":
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-10))
        L = np.eye(n) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])
    else:  # rw (Random Walk D^-1 L, solved via generalized eigenproblem L v = lambda D v)
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-10))
        L_sym = np.eye(n) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])
        L = np.eye(n) - (W / np.maximum(degrees[:, None], 1e-10))

    # Compute bottom eigenvalues and eigenvectors using symmetric solver
    if laplacian_type == "rw":
        eigvals, eigvecs_sym = eigh(L_sym)
        eigvecs = d_inv_sqrt[:, None] * eigvecs_sym
        norms = np.linalg.norm(eigvecs, axis=0, keepdims=True) + 1e-12
        eigvecs = eigvecs / norms
    else:
        eigvals, eigvecs = eigh(L)

    eigvals = np.maximum(eigvals, 0.0)
    U = eigvecs[:, :n_clusters].copy()

    if laplacian_type == "sym":
        row_norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-10
        U_norm = U / row_norms
    else:
        U_norm = U

    km = KMeans(n_clusters=int(n_clusters), n_init=10, random_state=int(seed))
    labels = km.fit_predict(U_norm)

    return W, degrees, L, eigvals, eigvecs, U_norm, labels


def format_cluster_labels(labels: np.ndarray) -> list[str]:
    """Convert integer cluster labels (-1, 0, 1, ...) to human-readable strings."""
    out = []
    for c in labels:
        if c == -1:
            out.append(tr("Noise (-1)", "Ruido (-1)"))
        else:
            out.append(f"Cluster {c + 1}")
    return out


# ==============================================================================
# SILHOUETTE COEFFICIENT: REUSABLE INTUITIVE EXPLAINER
# ==============================================================================
def compute_silhouette_values(X: np.ndarray, labels: np.ndarray):
    """Per-point silhouette s(i). Noise points (-1) are excluded (NaN). Returns None if undefined."""
    valid = labels != -1
    if len(set(labels[valid])) < 2 or np.sum(valid) < 5:
        return None
    s = np.full(len(X), np.nan)
    s[valid] = silhouette_samples(X[valid], labels[valid])
    return s


def explain_point_silhouette(X: np.ndarray, labels: np.ndarray, i: int) -> dict:
    """Compute a(i), b(i) and the nearest neighboring cluster for a single point."""
    d = np.linalg.norm(X - X[i], axis=1)
    idx = np.arange(len(X))
    own = labels[i]
    own_mask = (labels == own) & (idx != i)
    a_i = float(d[own_mask].mean()) if np.any(own_mask) else 0.0
    other_means = {c: float(d[labels == c].mean()) for c in sorted(set(labels) - {own, -1})}
    nearest = min(other_means, key=other_means.get)
    b_i = other_means[nearest]
    s_i = 0.0 if not np.any(own_mask) else (b_i - a_i) / max(a_i, b_i, 1e-12)
    return {
        "a": a_i,
        "b": b_i,
        "s": s_i,
        "own": own,
        "nearest": nearest,
        "own_mask": own_mask,
        "near_mask": labels == nearest,
        "other_means": other_means,
    }


def render_silhouette_explainer(
    X: np.ndarray,
    X_plot: np.ndarray,
    algo_labels: dict,
    key_prefix: str,
    axis_labels: tuple[str, str] = ("$x_1$", "$x_2$"),
    high_dim_note: bool = False,
) -> None:
    """Interactive, intuitive section explaining the Silhouette Coefficient for any clustering result."""
    st.markdown("---")
    st.markdown(
        tr(
            "### 📏 Silhouette Coefficient: How \"Comfortable\" Is Each Point in Its Cluster?",
            "### 📏 Coeficiente de Silhouette: ¿Qué Tan \"Cómodo\" Está Cada Punto en su Grupo?",
        )
    )

    with st.expander(
        tr(
            "Read the Intuition & Math Behind the Silhouette (The 'Party Tables' Analogy)",
            "Leer la Intuición y Matemática del Silhouette (La Analogía de las 'Mesas de la Fiesta')",
        ),
        expanded=False,
    ):
        if not ES:
            st.markdown(
                r"""
                #### The "Party Tables" Analogy
                Imagine a wedding party where guests were assigned to tables (clusters). For **each guest $i$** we ask two questions:
                1. **$a(i)$ — "How far am I from my own table?"** The average distance from $i$ to everyone else **at its own cluster**. *Small $a(i)$ = I feel at home.*
                2. **$b(i)$ — "How far is the closest *other* table?"** For every other cluster, compute the average distance from $i$ to its members, and keep the **smallest** one (the "neighboring table"). *Large $b(i)$ = no other table tempts me.*

                The **silhouette** of point $i$ compares both answers on a scale from $-1$ to $+1$:
                $$s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}}$$

                | Value of $s(i)$ | Meaning | Party Analogy |
                | :--- | :--- | :--- |
                | **$\approx +1$** | $a(i) \ll b(i)$: well inside its cluster, far from others | Sitting happily with close friends; the other tables are far away |
                | **$\approx 0$** | $a(i) \approx b(i)$: on the border between two clusters | Standing in the aisle, could sit at either table |
                | **$< 0$** | $a(i) > b(i)$: closer to *another* cluster than to its own | Seated at the wrong table! Probably misassigned |

                **Global Silhouette Score:** the average of $s(i)$ over all points, $\bar{s} = \frac{1}{n}\sum_i s(i)$. Rule of thumb (Kaufman & Rousseeuw, 1990): $> 0.70$ strong structure, $0.50$–$0.70$ reasonable, $0.25$–$0.50$ weak, $< 0.25$ no substantial structure.

                > ⚠️ **Two important caveats for DBSCAN & Spectral Clustering:**
                > - **It is biased toward round, compact (convex) clusters.** $a(i)$ uses the *average straight-line distance* to the whole cluster, so a long curved moon or a big ring has a large $a(i)$ even when it is perfectly clustered. K-Means can therefore get a **higher silhouette while being wrong**!
                > - **Noise points (`-1`) from DBSCAN are excluded** here, because noise is not a cluster. Always look at the silhouette *together with* the % of points discarded as noise.
                """
            )
        else:
            st.markdown(
                r"""
                #### La Analogía de las "Mesas de la Fiesta"
                Imagina una boda donde a cada invitado se le asignó una mesa (cluster). Para **cada invitado $i$** hacemos dos preguntas:
                1. **$a(i)$ — "¿Qué tan lejos estoy de mi propia mesa?"** Es la distancia promedio de $i$ a todos los demás **miembros de su propio grupo**. *$a(i)$ pequeño = me siento en casa.*
                2. **$b(i)$ — "¿Qué tan lejos está la *otra* mesa más cercana?"** Para cada uno de los otros grupos calculamos la distancia promedio de $i$ a sus miembros y nos quedamos con la **menor** (la "mesa vecina"). *$b(i)$ grande = ninguna otra mesa me tienta.*

                El **silhouette** del punto $i$ compara ambas respuestas en una escala de $-1$ a $+1$:
                $$s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}}$$

                | Valor de $s(i)$ | Significado | Analogía de la Fiesta |
                | :--- | :--- | :--- |
                | **$\approx +1$** | $a(i) \ll b(i)$: bien adentro de su grupo y lejos de los demás | Sentado feliz con sus amigos; las otras mesas están lejos |
                | **$\approx 0$** | $a(i) \approx b(i)$: en la frontera entre dos grupos | Parado en el pasillo, podría sentarse en cualquiera de las dos mesas |
                | **$< 0$** | $a(i) > b(i)$: está más cerca de *otro* grupo que del suyo | ¡Lo sentaron en la mesa equivocada! Probablemente mal asignado |

                **Silhouette Global:** el promedio de $s(i)$ sobre todos los puntos, $\bar{s} = \frac{1}{n}\sum_i s(i)$. Regla práctica (Kaufman y Rousseeuw, 1990): $> 0.70$ estructura fuerte, $0.50$–$0.70$ razonable, $0.25$–$0.50$ débil, $< 0.25$ sin estructura clara.

                > ⚠️ **Dos advertencias importantes para DBSCAN y Clustering Espectral:**
                > - **Favorece a los grupos redondos y compactos (convexos).** $a(i)$ usa la *distancia promedio en línea recta* a todo el grupo, así que una media luna larga o un anillo grande tiene un $a(i)$ alto aunque esté perfectamente agrupado. ¡Por eso K-Means puede obtener **mejor silhouette estando equivocado**!
                > - **Los puntos de ruido (`-1`) de DBSCAN se excluyen**, porque el ruido no es un grupo. Siempre mira el silhouette *junto con* el % de puntos descartados como ruido.
                """
            )

    algo_names = list(algo_labels.keys())
    algo = st.radio(
        tr("Analyze the silhouette of:", "Analizar el silhouette de:"),
        algo_names,
        horizontal=True,
        key=f"{key_prefix}_sil_algo",
    )
    labels = np.asarray(algo_labels[algo])
    s_vals = compute_silhouette_values(X, labels)
    if s_vals is None:
        st.warning(
            tr(
                "The silhouette needs at least 2 clusters (excluding noise). Adjust the parameters to obtain 2 or more clusters.",
                "El silhouette necesita al menos 2 grupos (sin contar el ruido). Ajusta los parámetros para obtener 2 o más grupos.",
            )
        )
        return

    valid_idx = np.where(~np.isnan(s_vals))[0]
    s_valid = s_vals[valid_idx]
    mean_s = float(np.mean(s_valid))
    n_noise_excl = int(np.sum(labels == -1))
    pct_neg = float(np.mean(s_valid < 0) * 100)

    if mean_s > 0.70:
        verdict = tr("Strong structure", "Estructura fuerte")
    elif mean_s > 0.50:
        verdict = tr("Reasonable structure", "Estructura razonable")
    elif mean_s > 0.25:
        verdict = tr("Weak structure", "Estructura débil")
    else:
        verdict = tr("No substantial structure", "Sin estructura clara")

    sm1, sm2, sm3 = st.columns(3)
    sm1.metric(tr("Global Silhouette (mean s(i))", "Silhouette Global (promedio s(i))"), f"{mean_s:.3f}", verdict)
    sm2.metric(
        tr("Points with s(i) < 0 (likely misassigned)", "Puntos con s(i) < 0 (posible mala asignación)"),
        f"{pct_neg:.1f}%",
    )
    sm3.metric(
        tr("Noise points excluded (-1)", "Puntos de ruido excluidos (-1)"),
        f"{n_noise_excl}",
        f"{n_noise_excl / len(labels) * 100:.1f}%",
    )

    sil_tabs = st.tabs(
        [
            tr("1. Point Inspector: a(i) vs. b(i)", "1. Inspector de Puntos: a(i) vs. b(i)"),
            tr("2. Silhouette Plot (Knife Plot)", "2. Gráfico de Silhouette (Gráfico de Cuchillos)"),
            tr("3. Silhouette Heat Map", "3. Mapa de Calor del Silhouette"),
        ]
    )

    # ---- Tab 1: point inspector ----
    with sil_tabs[0]:
        ci1, ci2 = st.columns([1, 2])
        with ci1:
            pick_mode = st.radio(
                tr("Choose a point:", "Elige un punto:"),
                ["manual", "worst", "border", "best"],
                format_func=lambda m: {
                    "manual": tr("Manual (slider)", "Manual (slider)"),
                    "worst": tr("Worst classified (min s)", "Peor clasificado (s mínimo)"),
                    "border": tr("On the border (s ≈ 0)", "En la frontera (s ≈ 0)"),
                    "best": tr("Best classified (max s)", "Mejor clasificado (s máximo)"),
                }[m],
                key=f"{key_prefix}_sil_mode",
            )
            if pick_mode == "manual":
                pos = st.slider(
                    tr("Point # (among clustered points)", "Punto # (entre puntos agrupados)"),
                    0,
                    len(valid_idx) - 1,
                    0,
                    key=f"{key_prefix}_sil_pt",
                )
                i_pt = int(valid_idx[pos])
            elif pick_mode == "worst":
                i_pt = int(valid_idx[np.argmin(s_valid)])
            elif pick_mode == "best":
                i_pt = int(valid_idx[np.argmax(s_valid)])
            else:
                i_pt = int(valid_idx[np.argmin(np.abs(s_valid))])

            info = explain_point_silhouette(X[valid_idx], labels[valid_idx], int(np.where(valid_idx == i_pt)[0][0]))
            a_i, b_i, s_i = info["a"], info["b"], info["s"]
            own_name = f"Cluster {info['own'] + 1}"
            near_name = f"Cluster {info['nearest'] + 1}"

            st.markdown(f"#### {tr('Point', 'Punto')} `#{i_pt}` → **{own_name}**")
            st.markdown(
                f"- 🔵 **a(i)** = {tr('avg. distance to its own', 'distancia promedio a su propio')} **{own_name}** = `{a_i:.3f}`\n"
                f"- 🔴 **b(i)** = {tr('avg. distance to the nearest other cluster', 'distancia promedio al grupo vecino más cercano')} **{near_name}** = `{b_i:.3f}`"
            )
            st.latex(
                rf"s(i) = \frac{{b(i) - a(i)}}{{\max\{{a(i), b(i)\}}}} = \frac{{{b_i:.3f} - {a_i:.3f}}}{{{max(a_i, b_i):.3f}}} = \mathbf{{{s_i:.3f}}}"
            )
            if s_i > 0.5:
                st.success(
                    tr(
                        f"**Comfortable point:** its own cluster is much closer ({a_i:.2f}) than the neighboring cluster ({b_i:.2f}). It clearly belongs where it is.",
                        f"**Punto cómodo:** su propio grupo está mucho más cerca ({a_i:.2f}) que el grupo vecino ({b_i:.2f}). Claramente pertenece donde está.",
                    )
                )
            elif s_i >= 0:
                st.warning(
                    tr(
                        f"**Border point:** a(i) and b(i) are similar, so it sits between {own_name} and {near_name}. A small change in parameters could move it.",
                        f"**Punto fronterizo:** a(i) y b(i) son parecidos; está entre {own_name} y {near_name}. Un pequeño cambio en los parámetros podría moverlo de grupo.",
                    )
                )
            else:
                st.error(
                    tr(
                        f"**Negative silhouette:** on average, this point is closer to {near_name} ({b_i:.2f}) than to its own {own_name} ({a_i:.2f}). Either it was misassigned, or its cluster is long/curved (see the convexity caveat above).",
                        f"**Silhouette negativo:** en promedio este punto está más cerca de {near_name} ({b_i:.2f}) que de su propio {own_name} ({a_i:.2f}). O fue mal asignado, o su grupo es largo/curvo (ver la advertencia de convexidad arriba).",
                    )
                )
            if high_dim_note:
                st.caption(
                    tr(
                        "Distances a(i) and b(i) are computed in the original high-dimensional (standardized) space; the plot only shows a 2D PCA projection.",
                        "Las distancias a(i) y b(i) se calculan en el espacio original de alta dimensión (estandarizado); el gráfico solo muestra una proyección PCA 2D.",
                    )
                )

        with ci2:
            fig_pt = go.Figure()
            sub = valid_idx
            fig_pt.add_trace(
                go.Scatter(
                    x=X_plot[sub, 0],
                    y=X_plot[sub, 1],
                    mode="markers",
                    name=tr("Other points", "Otros puntos"),
                    marker=dict(size=6, color="rgba(170,170,170,0.45)"),
                    hoverinfo="skip",
                )
            )
            own_global = valid_idx[info["own_mask"]]
            near_global = valid_idx[info["near_mask"]]
            rng_lines = np.random.default_rng(0)
            for members, color, name in [
                (own_global, "rgba(99,110,250,0.25)", tr("Lines used for a(i)", "Líneas usadas para a(i)")),
                (near_global, "rgba(239,85,59,0.22)", tr("Lines used for b(i)", "Líneas usadas para b(i)")),
            ]:
                shown = members if len(members) <= 70 else rng_lines.choice(members, 70, replace=False)
                lx, ly = [], []
                for j in shown:
                    lx.extend([X_plot[i_pt, 0], X_plot[j, 0], None])
                    ly.extend([X_plot[i_pt, 1], X_plot[j, 1], None])
                fig_pt.add_trace(go.Scatter(x=lx, y=ly, mode="lines", name=name, line=dict(color=color, width=1), hoverinfo="skip"))
            fig_pt.add_trace(
                go.Scatter(
                    x=X_plot[own_global, 0],
                    y=X_plot[own_global, 1],
                    mode="markers",
                    name=tr(f"Own cluster ({own_name})", f"Grupo propio ({own_name})"),
                    marker=dict(size=7, color="#636EFA"),
                )
            )
            fig_pt.add_trace(
                go.Scatter(
                    x=X_plot[near_global, 0],
                    y=X_plot[near_global, 1],
                    mode="markers",
                    name=tr(f"Nearest other cluster ({near_name})", f"Grupo vecino más cercano ({near_name})"),
                    marker=dict(size=7, color="#EF553B"),
                )
            )
            fig_pt.add_trace(
                go.Scatter(
                    x=[X_plot[i_pt, 0]],
                    y=[X_plot[i_pt, 1]],
                    mode="markers",
                    name=tr(f"Point #{i_pt} (s={s_i:.2f})", f"Punto #{i_pt} (s={s_i:.2f})"),
                    marker=dict(size=17, symbol="star", color="gold", line=dict(width=1.5, color="black")),
                )
            )
            fig_pt.update_layout(
                title=tr(
                    f"Point #{i_pt}: a(i)={a_i:.2f} (blue lines) vs. b(i)={b_i:.2f} (red lines)",
                    f"Punto #{i_pt}: a(i)={a_i:.2f} (líneas azules) vs. b(i)={b_i:.2f} (líneas rojas)",
                ),
                xaxis_title=axis_labels[0],
                yaxis_title=axis_labels[1],
                yaxis=dict(scaleanchor="x", scaleratio=1),
                height=500,
                template="plotly_white",
                legend=dict(orientation="h", y=-0.18),
            )
            st.plotly_chart(fig_pt, width="stretch", key=f"{key_prefix}_sil_ptfig")

    # ---- Tab 2: knife plot ----
    with sil_tabs[1]:
        kc1, kc2 = st.columns([1, 2])
        with kc1:
            st.info(
                tr(
                    "**How to read the knife plot:**\n\n"
                    "* Each horizontal bar is **one point**, grouped by cluster and sorted from largest to smallest $s(i)$.\n"
                    "* **Wide, uniform 'knives'** that cross the red dashed line (the mean) = a healthy, well-separated cluster.\n"
                    "* **Thin knives or bars extending to the left (negative)** = a cluster with many border or misassigned points.\n"
                    "* **Very unequal knife thicknesses** = clusters of very different sizes (not necessarily bad).",
                    "**Cómo leer el gráfico de cuchillos:**\n\n"
                    "* Cada barra horizontal es **un punto**, agrupado por cluster y ordenado de mayor a menor $s(i)$.\n"
                    "* **'Cuchillos' anchos y uniformes** que superan la línea roja punteada (el promedio) = un grupo sano y bien separado.\n"
                    "* **Cuchillos delgados o barras hacia la izquierda (negativas)** = un grupo con muchos puntos fronterizos o mal asignados.\n"
                    "* **Grosores muy distintos** = grupos de tamaños muy diferentes (no necesariamente es malo).",
                )
            )
            per_cluster = (
                pd.DataFrame({"cluster": labels[valid_idx], "s": s_valid})
                .groupby("cluster")["s"]
                .agg(["count", "mean", "min"])
                .rename(
                    index=lambda c: f"Cluster {c + 1}",
                    columns={
                        "count": tr("Points", "Puntos"),
                        "mean": tr("Mean s(i)", "s(i) Promedio"),
                        "min": tr("Min s(i)", "s(i) Mínimo"),
                    },
                )
            )
            st.dataframe(per_cluster.style.format({per_cluster.columns[1]: "{:.3f}", per_cluster.columns[2]: "{:.3f}"}))
        with kc2:
            fig_knife = go.Figure()
            palette = px.colors.qualitative.Plotly
            y_lower = 0
            for c in sorted(set(labels[valid_idx])):
                s_c = np.sort(s_valid[labels[valid_idx] == c])[::-1]
                y_pos = np.arange(y_lower, y_lower + len(s_c))
                fig_knife.add_trace(
                    go.Bar(
                        x=s_c,
                        y=y_pos,
                        orientation="h",
                        name=f"Cluster {c + 1}",
                        marker=dict(color=palette[c % len(palette)], line=dict(width=0)),
                        width=1.0,
                        hovertemplate="s(i)=%{x:.3f}<extra>Cluster " + str(c + 1) + "</extra>",
                    )
                )
                y_lower += len(s_c) + max(4, len(s_valid) // 40)
            fig_knife.add_vline(
                x=mean_s,
                line_dash="dash",
                line_color="red",
                annotation_text=tr(f"Mean = {mean_s:.3f}", f"Promedio = {mean_s:.3f}"),
            )
            fig_knife.add_vline(x=0, line_color="black", line_width=1)
            fig_knife.update_layout(
                title=tr(f"Silhouette Plot — {algo}", f"Gráfico de Silhouette — {algo}"),
                xaxis_title=tr("Silhouette coefficient s(i)", "Coeficiente de silhouette s(i)"),
                yaxis=dict(showticklabels=False, title=tr("Points grouped by cluster", "Puntos agrupados por cluster")),
                xaxis=dict(range=[min(-0.3, float(s_valid.min()) - 0.05), 1.0]),
                bargap=0,
                height=500,
                template="plotly_white",
            )
            st.plotly_chart(fig_knife, width="stretch", key=f"{key_prefix}_sil_knife")

    # ---- Tab 3: heat map of s(i) ----
    with sil_tabs[2]:
        hc1, hc2 = st.columns([1, 2])
        with hc1:
            st.info(
                tr(
                    "**Where are the 'uncomfortable' points?**\n\n"
                    "Each point is colored by its own $s(i)$: **green** = comfortable, **yellow** ≈ border, **red** = closer to another cluster.\n\n"
                    "Notice that low values usually concentrate **where two clusters meet**, and—for curved shapes—**at the far ends of long clusters**, which are far from the rest of their own group in straight-line distance.",
                    "**¿Dónde están los puntos 'incómodos'?**\n\n"
                    "Cada punto se colorea según su propio $s(i)$: **verde** = cómodo, **amarillo** ≈ frontera, **rojo** = más cerca de otro grupo.\n\n"
                    "Observa que los valores bajos suelen concentrarse **donde dos grupos se tocan** y —en formas curvas— **en los extremos de los grupos alargados**, que quedan lejos (en línea recta) del resto de su propio grupo.",
                )
            )
        with hc2:
            fig_heat_s = go.Figure()
            if n_noise_excl > 0:
                nm = labels == -1
                fig_heat_s.add_trace(
                    go.Scatter(
                        x=X_plot[nm, 0],
                        y=X_plot[nm, 1],
                        mode="markers",
                        name=tr("Noise (excluded)", "Ruido (excluido)"),
                        marker=dict(size=6, symbol="x", color="#999999"),
                    )
                )
            fig_heat_s.add_trace(
                go.Scatter(
                    x=X_plot[valid_idx, 0],
                    y=X_plot[valid_idx, 1],
                    mode="markers",
                    name="s(i)",
                    text=[f"#{j} | {f'Cluster {labels[j] + 1}'} | s={s_vals[j]:.3f}" for j in valid_idx],
                    hoverinfo="text",
                    marker=dict(
                        size=8,
                        color=s_valid,
                        colorscale="RdYlGn",
                        cmin=-0.5,
                        cmax=1.0,
                        colorbar=dict(title="s(i)"),
                        line=dict(width=0.5, color="black"),
                    ),
                )
            )
            fig_heat_s.update_layout(
                title=tr(f"Points Colored by Their Silhouette — {algo}", f"Puntos Coloreados por su Silhouette — {algo}"),
                xaxis_title=axis_labels[0],
                yaxis_title=axis_labels[1],
                yaxis=dict(scaleanchor="x", scaleratio=1),
                height=500,
                template="plotly_white",
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_heat_s, width="stretch", key=f"{key_prefix}_sil_heat")


# ==============================================================================
# MODULE 1: WHY K-MEANS FAILS ON NON-CONVEX SHAPES
# ==============================================================================
if mode_idx == 0:
    st.subheader(
        tr(
            "Module 1: Why Traditional K-Means Fails on Non-Convex Geometries",
            "Módulo 1: Por Qué Falla el K-Means Tradicional en Formas No Convexas",
        )
    )
    st.info(
        tr(
            "**Interactive tip:** Change the **Geometric Pattern** and **Add Background Noise Outliers** in the sidebar. Notice how **K-Means** always slices space with straight lines (Voronoi cells around centroids), cutting curved shapes in half, while **DBSCAN** and **Spectral Clustering** follow the natural shape of the data.",
            "**Sugerencia interactiva:** Cambia el **Patrón Geométrico** y **Añadir Puntos de Ruido** en la barra lateral. Observa cómo **K-Means** siempre corta el espacio con líneas rectas (celdas de Voronoi alrededor de centroides), partiendo a la mitad las formas curvas, mientras **DBSCAN** y **Clustering Espectral** siguen la geometría natural de los datos.",
        )
    )

    st.sidebar.header(tr("Dataset & Algorithm Controls", "Controles de Datos y Algoritmos"))
    PATTERN_NAMES = [
        tr("Interlocking Crescent Moons", "Medias Lunas Entrelazadas"),
        tr("Concentric Bullseye Rings", "Anillos Concéntricos (Diana)"),
        tr("Double Intertwined Spirals", "Espirales Dobles Entrelazadas"),
        tr("Elongated Cigars + Round Blob", "Nubes Alargadas + Grupo Circular"),
    ]
    pat_idx = st.sidebar.selectbox(
        tr("Geometric Pattern:", "Patrón Geométrico:"),
        range(4),
        format_func=lambda i: PATTERN_NAMES[i],
    )
    n_pts_m1 = st.sidebar.slider(tr("Number of points", "Número de puntos"), 150, 600, 320, step=20)
    noise_m1 = st.sidebar.slider(
        tr("Scatter / Width Noise", "Dispersión / Grosor de Ruido"),
        0.02,
        0.20,
        0.06 if pat_idx < 3 else 0.10,
        step=0.01,
    )
    n_outliers_m1 = st.sidebar.slider(
        tr("Add Background Noise Outliers", "Añadir Puntos de Ruido Aislados"),
        0,
        50,
        10,
        step=5,
    )

    default_k = 3 if pat_idx == 3 else 2
    k_val = st.sidebar.slider(
        tr("Number of Clusters ($k$) for K-Means & Spectral", "Número de Clusters ($k$) para K-Means y Espectral"),
        2,
        6,
        default_k,
    )
    default_eps = [0.46, 0.44, 0.34, 0.16][pat_idx]
    eps_m1 = st.sidebar.slider(
        tr("DBSCAN Radius ($\\varepsilon$)", "Radio DBSCAN ($\\varepsilon$)"),
        0.08,
        0.80,
        default_eps,
        step=0.02,
    )
    min_pts_m1 = st.sidebar.slider(
        tr("DBSCAN MinPts", "DBSCAN MinPts (Vecinos Mínimos)"),
        3,
        20,
        6,
    )
    gamma_m1 = st.sidebar.slider(
        tr("Spectral RBF $\\gamma$ (Locality)", "RBF $\\gamma$ Espectral (Localidad)"),
        1.0,
        30.0,
        12.0 if pat_idx != 2 else 30.0,
        step=1.0,
    )

    X_m1, y_true_m1 = generate_toy_dataset(pat_idx, n_pts_m1, noise_m1, n_outliers_m1)

    # 1. K-Means
    km_model = KMeans(n_clusters=k_val, n_init=10, random_state=42)
    y_km = km_model.fit_predict(X_m1)
    centroids = km_model.cluster_centers_

    # 2. DBSCAN
    db_model = DBSCAN(eps=eps_m1, min_samples=min_pts_m1)
    y_db = db_model.fit_predict(X_m1)

    # 3. Spectral Clustering
    spec_model = SpectralClustering(
        n_clusters=k_val,
        affinity="rbf",
        gamma=gamma_m1,
        random_state=42,
    )
    y_spec = spec_model.fit_predict(X_m1)

    # Compute ARI on non-outlier ground truth points
    valid_mask = y_true_m1 != -1
    ari_km = adjusted_rand_score(y_true_m1[valid_mask], y_km[valid_mask])
    ari_db = adjusted_rand_score(y_true_m1[valid_mask], y_db[valid_mask])
    ari_spec = adjusted_rand_score(y_true_m1[valid_mask], y_spec[valid_mask])
    n_db_clusters = len(set(y_db) - {-1})
    n_db_noise = int(np.sum(y_db == -1))

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric(
        tr("K-Means Accuracy (ARI vs True Shape)", "Precisión K-Means (ARI vs Forma Real)"),
        f"{ari_km * 100:.1f}%",
        tr("Uses straight Euclidean lines", "Usa fronteras rectas euclidianas"),
    )
    mc2.metric(
        tr("DBSCAN Accuracy (ARI vs True Shape)", "Precisión DBSCAN (ARI vs Forma Real)"),
        f"{ari_db * 100:.1f}%",
        tr(f"Found {n_db_clusters} clusters & {n_db_noise} noise pts", f"Halló {n_db_clusters} grupos y {n_db_noise} pts de ruido"),
    )
    mc3.metric(
        tr("Spectral Accuracy (ARI vs True Shape)", "Precisión Espectral (ARI vs Forma Real)"),
        f"{ari_spec * 100:.1f}%",
        tr("Follows graph connectivity", "Sigue la conectividad del grafo"),
    )

    col_p1, col_p2, col_p3 = st.columns(3)

    # Grid for K-Means Voronoi regions
    gx = np.linspace(X_m1[:, 0].min() - 0.3, X_m1[:, 0].max() + 0.3, 120)
    gy = np.linspace(X_m1[:, 1].min() - 0.3, X_m1[:, 1].max() + 0.3, 120)
    GX, GY = np.meshgrid(gx, gy)
    grid_preds = km_model.predict(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)

    with col_p1:
        fig_km = go.Figure()
        fig_km.add_trace(
            go.Contour(
                x=gx,
                y=gy,
                z=grid_preds,
                showscale=False,
                opacity=0.18,
                colorscale="Portland",
                contours=dict(coloring="fill", showlines=True),
                hoverinfo="skip",
            )
        )
        for c_id in range(k_val):
            mask = y_km == c_id
            fig_km.add_trace(
                go.Scatter(
                    x=X_m1[mask, 0],
                    y=X_m1[mask, 1],
                    mode="markers",
                    name=f"Cluster {c_id + 1}",
                    marker=dict(size=7, opacity=0.85),
                )
            )
        fig_km.add_trace(
            go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode="markers",
                name=tr("Centroids", "Centroides"),
                marker=dict(size=14, symbol="x", color="black", line=dict(width=2, color="white")),
            )
        )
        fig_km.update_layout(
            title=tr(
                f"1. K-Means (k={k_val}) — Straight Voronoi Cuts",
                f"1. K-Means (k={k_val}) — Cortes Rectos de Voronoi",
            ),
            xaxis_title="$x_1$",
            yaxis_title="$x_2$",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            height=460,
            template="plotly_white",
            legend=dict(orientation="h", y=-0.18),
        )
        st.plotly_chart(fig_km, width="stretch")

    with col_p2:
        fig_db = px.scatter(
            x=X_m1[:, 0],
            y=X_m1[:, 1],
            color=format_cluster_labels(y_db),
            symbol=[tr("Noise", "Ruido") if c == -1 else tr("Cluster", "Cluster") for c in y_db],
            color_discrete_map={tr("Noise (-1)", "Ruido (-1)"): "#7F7F7F"},
            labels={"x": "$x_1$", "y": "$x_2$", "color": tr("Assignment", "Asignación")},
            title=tr(
                f"2. DBSCAN (ε={eps_m1}, MinPts={min_pts_m1})",
                f"2. DBSCAN (ε={eps_m1}, MinPts={min_pts_m1})",
            ),
        )
        fig_db.update_traces(marker=dict(size=7, opacity=0.88))
        fig_db.update_yaxes(scaleanchor="x", scaleratio=1)
        fig_db.update_layout(height=460, template="plotly_white", legend=dict(orientation="h", y=-0.18))
        st.plotly_chart(fig_db, width="stretch")

    with col_p3:
        fig_sp = px.scatter(
            x=X_m1[:, 0],
            y=X_m1[:, 1],
            color=format_cluster_labels(y_spec),
            labels={"x": "$x_1$", "y": "$x_2$", "color": tr("Assignment", "Asignación")},
            title=tr(
                f"3. Spectral Clustering (k={k_val}, γ={gamma_m1})",
                f"3. Clustering Espectral (k={k_val}, γ={gamma_m1})",
            ),
        )
        fig_sp.update_traces(marker=dict(size=7, opacity=0.88))
        fig_sp.update_yaxes(scaleanchor="x", scaleratio=1)
        fig_sp.update_layout(height=460, template="plotly_white", legend=dict(orientation="h", y=-0.18))
        st.plotly_chart(fig_sp, width="stretch")

    if not ES:
        st.markdown(
            r"""
            ### Mathematical Reason Why K-Means Fails Here
            - **K-Means Objective (Inertia):** K-Means minimizes the sum of squared Euclidean distances to $k$ centroids $\mu_c$:
              $$\min_{\{\mu_c\}} \sum_{c=1}^{k} \sum_{x_i \in C_c} \|x_i - \mu_c\|^2$$
              Because every point is assigned to its closest centroid in straight-line Euclidean distance, the decision boundaries between clusters are **always flat hyperplanes** (perpendicular bisectors between centroids). A flat line can never separate two concentric circles or interlocking moons!
            - **How DBSCAN Solves It:** Ignores centroids completely. Instead, it connects points that lie within distance $\varepsilon$ of dense **Core Points**, allowing clusters to snake around any curve while automatically rejecting isolated outliers (`-1`).
            - **How Spectral Clustering Solves It:** Builds a local neighborhood graph $W$ and computes the eigenvectors of the Graph Laplacian $L = D - W$, mapping connected chains onto the same coordinates regardless of how curved they are in 2D.
            """
        )
    else:
        st.markdown(
            r"""
            ### Razón Matemática por la que K-Means Falla Aquí
            - **Función Objetivo de K-Means (Inercia):** K-Means minimiza la suma de distancias euclidianas al cuadrado hacia $k$ centroides $\mu_c$:
              $$\min_{\{\mu_c\}} \sum_{c=1}^{k} \sum_{x_i \in C_c} \|x_i - \mu_c\|^2$$
              Como cada punto se asigna al centroide más cercano en línea recta, las fronteras entre grupos son **siempre hiperplanos rectos** (mediatrices entre centroides). ¡Una línea recta jamás puede separar dos anillos concéntricos ni dos medias lunas entrelazadas!
            - **Cómo lo Resuelve DBSCAN:** Ignora por completo los centroides. Conecta puntos que están a una distancia $\varepsilon$ de **Puntos Núcleo** densos, permitiendo seguir cualquier curva y descartando automáticamente el ruido aislado (`-1`).
            - **Cómo lo Resuelve el Clustering Espectral:** Construye un grafo de vecindad local $W$ y calcula los autovectores del Laplaciano $L = D - W$, proyectando cada cadena conectada hacia una misma región sin importar su curvatura original.
            """
        )

    # --- Silhouette paradox: internal metric vs. true shape ---
    def _sil_str(lbls: np.ndarray) -> str:
        s = compute_silhouette_values(X_m1, lbls)
        return "N/A" if s is None else f"{np.nanmean(s):.3f}"

    sil_km_m1, sil_db_m1, sil_sp_m1 = _sil_str(y_km), _sil_str(y_db), _sil_str(y_spec)
    st.markdown(
        tr(
            "### 🤔 The Silhouette Paradox: A Higher Score Does Not Always Mean a Better Clustering",
            "### 🤔 La Paradoja del Silhouette: Un Puntaje Más Alto No Siempre Significa Mejor Agrupamiento",
        )
    )
    par_df = pd.DataFrame(
        {
            tr("Algorithm", "Algoritmo"): ["K-Means", "DBSCAN", tr("Spectral", "Espectral")],
            tr("ARI vs. True Shape (needs labels)", "ARI vs. Forma Real (requiere etiquetas)"): [
                f"{ari_km * 100:.1f}%",
                f"{ari_db * 100:.1f}%",
                f"{ari_spec * 100:.1f}%",
            ],
            tr("Silhouette (no labels needed)", "Silhouette (no requiere etiquetas)"): [sil_km_m1, sil_db_m1, sil_sp_m1],
        }
    )
    pc1, pc2 = st.columns([1, 1])
    with pc1:
        st.dataframe(par_df, hide_index=True, width="stretch")
    with pc2:
        st.info(
            tr(
                "**Compare both columns on the Moons, Rings and Spirals:** K-Means often gets a **similar or even higher silhouette** than DBSCAN/Spectral while cutting the shapes in half (low ARI). "
                "Why? The silhouette rewards **compact, round** groups—exactly what K-Means builds—and penalizes long curved clusters whose far ends are distant from the rest of their own group. "
                "**Lesson:** the silhouette is a great tool when you have no labels, but it measures *compactness & separation*, not *correct shape*.",
                "**Compara ambas columnas en Lunas, Anillos y Espirales:** K-Means suele obtener un **silhouette parecido o incluso mayor** que DBSCAN/Espectral aunque parta las formas a la mitad (ARI bajo). "
                "¿Por qué? El silhouette premia los grupos **compactos y redondos** —justo lo que construye K-Means— y castiga los grupos curvos y alargados, cuyos extremos quedan lejos del resto de su propio grupo. "
                "**Lección:** el silhouette es excelente cuando no tienes etiquetas, pero mide *compacidad y separación*, no *la forma correcta*.",
            )
        )

    render_silhouette_explainer(
        X_m1,
        X_m1,
        {"K-Means": y_km, "DBSCAN": y_db, tr("Spectral", "Espectral"): y_spec},
        key_prefix="m1",
    )


# ==============================================================================
# MODULE 2: DBSCAN STEP-BY-STEP (CORE, BORDER & NOISE + K-DISTANCE ELBOW)
# ==============================================================================
elif mode_idx == 1:
    st.subheader(
        tr(
            "Module 2: DBSCAN Step-by-Step — Core Points, Border Points, Noise & Choosing ε",
            "Módulo 2: DBSCAN Paso a Paso — Puntos Núcleo, Frontera, Ruido y Elección de ε",
        )
    )

    st.sidebar.header(tr("DBSCAN Laboratory Controls", "Controles del Laboratorio DBSCAN"))
    db_pat_idx = st.sidebar.selectbox(
        tr("Select Dataset Shape:", "Selecciona Forma de Datos:"),
        range(4),
        format_func=lambda i: [
            tr("Interlocking Crescent Moons", "Medias Lunas Entrelazadas"),
            tr("Concentric Rings", "Anillos Concéntricos"),
            tr("Double Spirals", "Espirales Dobles"),
            tr("Elongated Blobs", "Nubes Alargadas"),
        ][i],
    )
    n_pts_db = st.sidebar.slider(tr("Number of points", "Número de puntos"), 80, 400, 180, step=20)
    noise_db = st.sidebar.slider(tr("Pattern Scatter Noise", "Ruido de Dispersión"), 0.02, 0.22, 0.08, step=0.01)
    outliers_db = st.sidebar.slider(tr("Background Outliers", "Puntos Aislados de Ruido"), 0, 40, 14, step=2)

    eps_val = st.sidebar.slider(
        tr("Radius $\\varepsilon$ (eps)", "Radio $\\varepsilon$ (eps)"),
        0.08,
        0.75,
        0.46,
        step=0.02,
    )
    min_pts_val = st.sidebar.slider(
        tr("MinPts (Minimum Neighbors)", "MinPts (Vecinos Mínimos)"),
        3,
        18,
        5,
    )

    X_db, _ = generate_toy_dataset(db_pat_idx, n_pts_db, noise_db, outliers_db, seed=42)
    labels_db, roles_db, n_nb_db = classify_dbscan_points(X_db, eps_val, min_pts_val)

    n_core = int(np.sum(roles_db == "Core"))
    n_border = int(np.sum(roles_db == "Border"))
    n_noise = int(np.sum(roles_db == "Noise"))
    n_clusters_found = len(set(labels_db) - {-1})

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        tr("Clusters Discovered", "Clusters Descubiertos"),
        f"{n_clusters_found}",
        tr("No need to specify k!", "¡Sin necesidad de pedir k!"),
    )
    k2.metric(
        tr("Core Points (Dense Interior)", "Puntos Núcleo (Interior Denso)"),
        f"{n_core}",
        f"|N_ε(p)| ≥ {min_pts_val}",
    )
    k3.metric(
        tr("Border Points (Cluster Edge)", "Puntos Frontera (Orilla del Grupo)"),
        f"{n_border}",
        f"|N_ε(p)| < {min_pts_val}",
    )
    k4.metric(
        tr("Noise / Outliers (Label -1)", "Ruido / Anomalías (Etiqueta -1)"),
        f"{n_noise}",
        f"{n_noise / len(X_db) * 100:.1f}%",
    )

    db_tabs = st.tabs(
        [
            tr("1. Interactive ε-Neighborhood Inspector", "1. Inspector Interactivo de Vecindarios-ε"),
            tr("2. Core vs. Border vs. Noise Anatomy", "2. Anatomía: Núcleo vs. Frontera vs. Ruido"),
            tr("3. How to Choose ε Automatically (k-Distance Elbow)", "3. Cómo Elegir ε Automáticamente (Gráfico k-Distancia)"),
        ]
    )

    with db_tabs[0]:
        col_ins_ctrl, col_ins_plot = st.columns([1, 2])
        with col_ins_ctrl:
            pt_idx = st.slider(
                tr("Inspect Point # ($p$):", "Inspeccionar Punto # ($p$):"),
                0,
                len(X_db) - 1,
                0,
            )
            show_all_core_edges = st.checkbox(
                tr("Show density-reachability graph (links between Core Points)", "Mostrar red de alcanzabilidad (enlaces entre Puntos Núcleo)"),
                value=True,
            )
            show_core_circles = st.checkbox(
                tr("Draw faint ε-circles around Core Points", "Dibujar círculos-ε tenues en los Puntos Núcleo"),
                value=False,
            )

            p_coord = X_db[pt_idx]
            dists_to_p = np.linalg.norm(X_db - p_coord, axis=1)
            in_ball_mask = dists_to_p <= eps_val
            count_in_ball = int(np.sum(in_ball_mask))
            role_p = roles_db[pt_idx]
            cluster_p = labels_db[pt_idx]

            role_badge = {
                "Core": tr("🟢 CORE POINT (Punto Núcleo)", "🟢 PUNTO NÚCLEO (Core Point)"),
                "Border": tr("🟡 BORDER POINT (Punto Frontera)", "🟡 PUNTO FRONTERA (Border Point)"),
                "Noise": tr("🔴 NOISE / OUTLIER (Ruido)", "🔴 PUNTO DE RUIDO (Noise / Outlier)"),
            }[role_p]

            st.markdown(f"#### {tr('Point', 'Punto')} `p = #{pt_idx}`")
            st.markdown(f"**{tr('Classification', 'Clasificación')}:** {role_badge}")
            st.markdown(
                f"- **{tr('Points inside ε-circle', 'Puntos dentro del círculo-ε')} ($|N_\\varepsilon(p)|$):** `{count_in_ball}`\n"
                f"- **{tr('Required MinPts', 'MinPts requerido')}:** `{min_pts_val}`\n"
                f"- **{tr('Assigned Cluster', 'Cluster Asignado')}:** `{tr('Noise (-1)', 'Ruido (-1)') if cluster_p == -1 else f'Cluster {cluster_p + 1}'}`"
            )
            if role_p == "Core":
                st.success(
                    tr(
                        f"Because $|N_\\varepsilon(p)| = {count_in_ball} \\ge {min_pts_val}$, point #{pt_idx} is a **Core Point** and expands its cluster to all {count_in_ball} points inside its red circle!",
                        f"Como $|N_\\varepsilon(p)| = {count_in_ball} \\ge {min_pts_val}$, ¡el punto #{pt_idx} es un **Punto Núcleo** y expande su cluster a todos los {count_in_ball} puntos dentro de su círculo rojo!",
                    )
                )
            elif role_p == "Border":
                st.warning(
                    tr(
                        f"Point #{pt_idx} only has `{count_in_ball} < {min_pts_val}` points in its circle, so it cannot start or expand a cluster—but at least one neighbor inside its circle IS a Core Point, making it a **Border Point**.",
                        f"El punto #{pt_idx} solo tiene `{count_in_ball} < {min_pts_val}` puntos en su círculo, por lo que no puede expandir el grupo; pero al menos uno de sus vecinos SÍ es un Punto Núcleo, convirtiéndolo en **Punto Frontera**.",
                    )
                )
            else:
                st.error(
                    tr(
                        f"Point #{pt_idx} has `{count_in_ball} < {min_pts_val}` neighbors and no Core Point reaches it. DBSCAN flags it as **Noise (`-1`)**.",
                        f"El punto #{pt_idx} tiene `{count_in_ball} < {min_pts_val}` vecinos y ningún Punto Núcleo lo alcanza. DBSCAN lo marca como **Ruido (`-1`)**.",
                    )
                )

        with col_ins_plot:
            fig_ins = go.Figure()

            # Optional faint circles around core points
            if show_core_circles:
                theta_c = np.linspace(0, 2 * np.pi, 36)
                cx_list, cy_list = [], []
                core_indices = np.where(roles_db == "Core")[0][:80]
                for ci in core_indices:
                    cx_list.extend(list(X_db[ci, 0] + eps_val * np.cos(theta_c)) + [None])
                    cy_list.extend(list(X_db[ci, 1] + eps_val * np.sin(theta_c)) + [None])
                fig_ins.add_trace(
                    go.Scatter(
                        x=cx_list,
                        y=cy_list,
                        mode="lines",
                        name=tr("Core ε-Balls", "Bolas-ε de Núcleos"),
                        line=dict(color="rgba(100, 180, 255, 0.25)", width=1),
                        hoverinfo="skip",
                    )
                )

            # Optional density-reachability edges between Core points
            if show_all_core_edges:
                core_idx = np.where(roles_db == "Core")[0]
                if len(core_idx) > 1:
                    X_core = X_db[core_idx]
                    c_dists = squareform(pdist(X_core))
                    ei, ej = np.nonzero(np.triu(c_dists <= eps_val, k=1))
                    edge_x, edge_y = [], []
                    for u, v in zip(ei, ej):
                        edge_x.extend([X_core[u, 0], X_core[v, 0], None])
                        edge_y.extend([X_core[u, 1], X_core[v, 1], None])
                    fig_ins.add_trace(
                        go.Scatter(
                            x=edge_x,
                            y=edge_y,
                            mode="lines",
                            name=tr("Core-to-Core Bridges (≤ ε)", "Puentes Núcleo-a-Núcleo (≤ ε)"),
                            line=dict(color="rgba(80, 80, 80, 0.35)", width=1.2),
                            hoverinfo="skip",
                        )
                    )

            # Plot points by cluster
            unique_lbls = sorted(set(labels_db))
            palette = px.colors.qualitative.Bold
            for lbl in unique_lbls:
                m_lbl = labels_db == lbl
                lbl_name = tr("Noise (-1)", "Ruido (-1)") if lbl == -1 else f"Cluster {lbl + 1}"
                color_val = "#888888" if lbl == -1 else palette[lbl % len(palette)]
                symbol_val = "x" if lbl == -1 else "circle"
                fig_ins.add_trace(
                    go.Scatter(
                        x=X_db[m_lbl, 0],
                        y=X_db[m_lbl, 1],
                        mode="markers",
                        name=lbl_name,
                        marker=dict(size=7 if lbl != -1 else 6, color=color_val, symbol=symbol_val, opacity=0.8),
                    )
                )

            # Highlight inspected point's epsilon circle
            theta_ring = np.linspace(0, 2 * np.pi, 100)
            fig_ins.add_trace(
                go.Scatter(
                    x=p_coord[0] + eps_val * np.cos(theta_ring),
                    y=p_coord[1] + eps_val * np.sin(theta_ring),
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(239, 85, 59, 0.15)",
                    line=dict(color="#EF553B", width=2.5, dash="dash"),
                    name=tr(f"N_ε(p=#{pt_idx}) Circle (r={eps_val})", f"Círculo N_ε(p=#{pt_idx}) (r={eps_val})"),
                )
            )

            # Highlight neighbors inside the ball
            fig_ins.add_trace(
                go.Scatter(
                    x=X_db[in_ball_mask, 0],
                    y=X_db[in_ball_mask, 1],
                    mode="markers",
                    name=tr(f"Neighbors in N_ε ({count_in_ball} pts)", f"Vecinos en N_ε ({count_in_ball} pts)"),
                    marker=dict(size=11, color="rgba(0,0,0,0)", line=dict(color="#EF553B", width=2.5)),
                )
            )

            # Highlight center point p
            fig_ins.add_trace(
                go.Scatter(
                    x=[p_coord[0]],
                    y=[p_coord[1]],
                    mode="markers",
                    name=tr(f"Inspected Point #{pt_idx}", f"Punto Inspeccionado #{pt_idx}"),
                    marker=dict(size=14, color="#EF553B", symbol="star", line=dict(color="black", width=1.5)),
                )
            )

            fig_ins.update_layout(
                title=tr(
                    f"Inspecting Point #{pt_idx}: {count_in_ball} points inside radius ε={eps_val} (MinPts={min_pts_val})",
                    f"Inspeccionando Punto #{pt_idx}: {count_in_ball} puntos dentro del radio ε={eps_val} (MinPts={min_pts_val})",
                ),
                xaxis_title="$x_1$",
                yaxis_title="$x_2$",
                yaxis=dict(scaleanchor="x", scaleratio=1),
                height=520,
                template="plotly_white",
            )
            st.plotly_chart(fig_ins, width="stretch")

    with db_tabs[1]:
        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            if not ES:
                st.markdown(
                    f"""
                    #### Anatomy of the 3 Point Types in DBSCAN
                    Every point in your dataset is classified into one of three roles based on $\\varepsilon = {eps_val}$ and $\\text{{MinPts}} = {min_pts_val}$:

                    1. **🟢 Core Points ({n_core} pts):**
                       - Have $\\ge {min_pts_val}$ points within distance $\\varepsilon$.
                       - Form the dense **skeleton** of each cluster.
                    2. **🟡 Border Points ({n_border} pts):**
                       - Have $< {min_pts_val}$ points within $\\varepsilon$, so they live on the sparser outskirts.
                       - However, they are within distance $\\varepsilon$ of at least one Core Point, so they get absorbed into its cluster.
                    3. **🔴 Noise Points ({n_noise} pts):**
                       - Have $< {min_pts_val}$ neighbors and are not reachable from any Core Point.
                    """
                )
            else:
                st.markdown(
                    f"""
                    #### Anatomía de los 3 Tipos de Puntos en DBSCAN
                    Cada punto del dataset se clasifica en uno de tres roles según $\\varepsilon = {eps_val}$ y $\\text{{MinPts}} = {min_pts_val}$:

                    1. **🟢 Puntos Núcleo / Core ({n_core} pts):**
                       - Tienen $\\ge {min_pts_val}$ puntos dentro del radio $\\varepsilon$.
                       - Forman el **esqueleto denso** de cada cluster.
                    2. **🟡 Puntos Frontera / Border ({n_border} pts):**
                       - Tienen $< {min_pts_val}$ puntos dentro de $\\varepsilon$ (están en la orilla).
                       - Pero caen dentro del aro $\\varepsilon$ de al menos un Punto Núcleo, por lo que son absorbidos por ese grupo.
                    3. **🔴 Puntos de Ruido / Noise ({n_noise} pts):**
                       - Tienen $< {min_pts_val}$ vecinos y ningún Punto Núcleo los alcanza.
                    """
                )
        with col_r2:
            fig_roles = go.Figure()
            role_configs = [
                ("Core", tr("Core Point (≥ MinPts)", "Punto Núcleo (≥ MinPts)"), "#00CC96", "circle", 9),
                ("Border", tr("Border Point (< MinPts, near Core)", "Punto Frontera (< MinPts, junto a Núcleo)"), "#FFA15A", "diamond", 9),
                ("Noise", tr("Noise / Outlier (-1)", "Ruido / Anomalía (-1)"), "#EF553B", "x", 8),
            ]
            for r_key, r_label, r_col, r_sym, r_sz in role_configs:
                m_r = roles_db == r_key
                fig_roles.add_trace(
                    go.Scatter(
                        x=X_db[m_r, 0],
                        y=X_db[m_r, 1],
                        mode="markers",
                        name=r_label,
                        text=[f"Point #{i} (|N_ε|={n_nb_db[i]})" for i in np.where(m_r)[0]],
                        marker=dict(size=r_sz, color=r_col, symbol=r_sym, line=dict(width=1, color="black")),
                    )
                )
            fig_roles.update_layout(
                title=tr(
                    "DBSCAN Point Roles: Core (Green Circles), Border (Orange Diamonds) & Noise (Red X)",
                    "Roles en DBSCAN: Núcleo (Círculos Verdes), Frontera (Rombos Naranjas) y Ruido (X Rojas)",
                ),
                xaxis_title="$x_1$",
                yaxis_title="$x_2$",
                yaxis=dict(scaleanchor="x", scaleratio=1),
                height=500,
                template="plotly_white",
            )
            st.plotly_chart(fig_roles, width="stretch")

    with db_tabs[2]:
        k_nn = max(1, min_pts_val - 1)
        nn_k = NearestNeighbors(n_neighbors=k_nn + 1)
        nn_k.fit(X_db)
        k_dists, _ = nn_k.kneighbors(X_db)
        sorted_k_dists = np.sort(k_dists[:, -1])

        # Estimate elbow via maximum curvature (second difference)
        diffs1 = np.gradient(sorted_k_dists)
        diffs2 = np.gradient(diffs1)
        elbow_idx = int(np.argmax(diffs2[int(len(sorted_k_dists) * 0.5) :]) + int(len(sorted_k_dists) * 0.5))
        suggested_eps = float(sorted_k_dists[elbow_idx])

        c_el1, c_el2 = st.columns([1, 2])
        with c_el1:
            if not ES:
                st.markdown(
                    f"""
                    #### The $k$-Distance Elbow Method
                    **Question:** How do we choose $\\varepsilon$ in practice if our dataset has 10 dimensions and we can't eyeball circles?

                    **Step-by-step recipe (Ester et al., 1996):**
                    1. Pick $\\text{{MinPts}}$ (rule of thumb: $\\text{{MinPts}} \\ge D + 1$, currently **{min_pts_val}**).
                    2. For every point, measure the distance to its **$k$-th nearest neighbor** ($k = \\text{{MinPts}} - 1 = {k_nn}$).
                    3. Sort all points from smallest $k$-distance (dense core points) to largest $k$-distance (isolated noise points).
                    4. Set $\\varepsilon$ right at the **"Elbow" (sharp bend)** where the curve shoots steeply upward!
                    - **Suggested Elbow $\\varepsilon$:** `{suggested_eps:.2f}`
                    - **Your Current $\\varepsilon$:** `{eps_val:.2f}`
                    """
                )
            else:
                st.markdown(
                    f"""
                    #### El Método del Codo de $k$-Distancia
                    **Pregunta:** ¿Cómo elegimos $\\varepsilon$ en la práctica cuando nuestros datos tienen 10 dimensiones y no podemos ver círculos a simple vista?

                    **Receta paso a paso (Ester et al., 1996):**
                    1. Elige $\\text{{MinPts}}$ (regla práctica: $\\text{{MinPts}} \\ge D + 1$, actualmente **{min_pts_val}**).
                    2. Para cada punto, calcula la distancia a su **$k$-ésimo vecino más cercano** ($k = \\text{{MinPts}} - 1 = {k_nn}$).
                    3. Ordena todos los puntos de menor a mayor distancia (de puntos densos a puntos aislados).
                    4. Coloca $\\varepsilon$ justo en el **"Codo" (quiebre brusco)** donde la curva se dispara hacia arriba:
                    - **$\\varepsilon$ Sugerido en el Codo:** `{suggested_eps:.2f}`
                    - **Tu $\\varepsilon$ Actual:** `{eps_val:.2f}`
                    """
                )
        with c_el2:
            fig_elbow = go.Figure()
            fig_elbow.add_trace(
                go.Scatter(
                    x=list(range(len(sorted_k_dists))),
                    y=sorted_k_dists,
                    mode="lines+markers",
                    name=tr(f"{k_nn}-th Neighbor Distance", f"Distancia al Vecino #{k_nn}"),
                    line=dict(color="#636EFA", width=3),
                    marker=dict(size=4),
                )
            )
            fig_elbow.add_hline(
                y=eps_val,
                line_dash="dash",
                line_color="#EF553B",
                annotation_text=tr(
                    f"Current ε = {eps_val:.2f} (Below line = Core Points)",
                    f"ε Actual = {eps_val:.2f} (Debajo de la línea = Puntos Núcleo)",
                ),
                annotation_position="top left",
            )
            fig_elbow.add_hline(
                y=suggested_eps,
                line_dash="dot",
                line_color="#00CC96",
                annotation_text=tr(
                    f"Suggested Elbow ≈ {suggested_eps:.2f}",
                    f"Codo Sugerido ≈ {suggested_eps:.2f}",
                ),
                annotation_position="bottom right",
            )
            fig_elbow.update_layout(
                title=tr(
                    f"Sorted k-Distance Plot (k = MinPts - 1 = {k_nn}) for Choosing ε",
                    f"Gráfico de k-Distancia Ordenada (k = MinPts - 1 = {k_nn}) para Elegir ε",
                ),
                xaxis_title=tr("Points Sorted by Density (Dense → Sparse)", "Puntos Ordenados por Densidad (Denso → Aislado)"),
                yaxis_title=tr(f"Distance to {k_nn}-th Nearest Neighbor", f"Distancia al {k_nn}.º Vecino Más Cercano"),
                height=460,
                template="plotly_white",
            )
            st.plotly_chart(fig_elbow, width="stretch")


# ==============================================================================
# MODULE 3: SPECTRAL CLUSTERING STEP-BY-STEP (LAPLACIAN & FIEDLER VECTOR)
# ==============================================================================
elif mode_idx == 2:
    st.subheader(
        tr(
            "Module 3: Spectral Clustering Step-by-Step — From Similarity Graphs to the Fiedler Vector",
            "Módulo 3: Clustering Espectral Paso a Paso — De Grafos de Similitud al Vector de Fiedler",
        )
    )

    st.sidebar.header(tr("Spectral Graph Parameters", "Parámetros del Grafo Espectral"))
    sp_pat_idx = st.sidebar.selectbox(
        tr("Select Dataset Pattern:", "Selecciona Patrón de Datos:"),
        range(4),
        format_func=lambda i: [
            tr("Concentric Rings (2 rings)", "Anillos Concéntricos (2 anillos)"),
            tr("Interlocking Moons (2 moons)", "Medias Lunas Entrelazadas (2 lunas)"),
            tr("3 Separated Blobs (3 islands)", "3 Nubes Separadas (3 islas)"),
            tr("Double Spirals (2 spirals)", "Espirales Dobles (2 espirales)"),
        ][i],
    )
    n_pts_sp = st.sidebar.slider(
        tr("Number of nodes ($N$)", "Número de nodos ($N$)"),
        80,
        260,
        140,
        step=20,
        help=tr("Kept moderate so you can inspect the full N×N Laplacian matrix clearly.", "Moderado para visualizar con claridad la matriz Laplaciana N×N completa."),
    )
    noise_sp = st.sidebar.slider(tr("Pattern Noise", "Ruido del Patrón"), 0.02, 0.16, 0.05, step=0.01)

    aff_type = st.sidebar.radio(
        tr("How to Build the Graph ($W$):", "Cómo Construir el Grafo ($W$):"),
        ["rbf", "knn"],
        format_func=lambda a: (
            tr("Gaussian RBF Kernel (exp(-γ d²))", "Kernel Gaussiano RBF (exp(-γ d²))")
            if a == "rbf"
            else tr("Mutual k-Nearest Neighbors (k-NN)", "k-Vecinos Más Cercanos (k-NN)")
        ),
    )
    gamma_sp = (
        st.sidebar.slider(
            tr("RBF Kernel $\\gamma$ (Falloff speed)", "Parámetro RBF $\\gamma$ (Caída con la distancia)"),
            1.0,
            60.0,
            30.0 if sp_pat_idx == 3 else 16.0,
            step=1.0,
        )
        if aff_type == "rbf"
        else 15.0
    )
    knn_sp = (
        st.sidebar.slider(
            tr("Number of Neighbors ($k_{\\text{NN}}$)", "Cantidad de Vecinos ($k_{\\text{NN}}$)"),
            4,
            25,
            10,
        )
        if aff_type == "knn"
        else 10
    )
    lap_type = st.sidebar.selectbox(
        tr("Graph Laplacian Formula ($L$):", "Fórmula del Laplaciano ($L$):"),
        ["sym", "unnormalized", "rw"],
        format_func=lambda l: {
            "sym": tr("Normalized Symmetric: L_sym = I - D^(-1/2) W D^(-1/2)", "Simétrico Normalizado: L_sym = I - D^(-1/2) W D^(-1/2)"),
            "unnormalized": tr("Unnormalized: L = D - W", "No Normalizado: L = D - W"),
            "rw": tr("Random Walk: L_rw = I - D^(-1) W", "Caminata Aleatoria: L_rw = I - D^(-1) W"),
        }[l],
    )
    k_clusters_sp = st.sidebar.slider(
        tr("Number of Clusters ($k$)", "Número de Clusters ($k$)"),
        2,
        5,
        3 if sp_pat_idx == 2 else 2,
    )

    # Generate selected pattern
    if sp_pat_idx == 0:
        X_sp, y_sp_true = generate_toy_dataset(1, n_pts_sp, noise_sp, 0)
    elif sp_pat_idx == 1:
        X_sp, y_sp_true = generate_toy_dataset(0, n_pts_sp, noise_sp, 0)
    elif sp_pat_idx == 2:
        X_sp_raw, y_sp_true = make_blobs(n_samples=n_pts_sp, centers=3, cluster_std=0.65, random_state=42)
        X_sp = StandardScaler().fit_transform(X_sp_raw)
    else:
        X_sp, y_sp_true = generate_toy_dataset(2, n_pts_sp, noise_sp, 0)

    # Order points by ground-truth cluster so the block-diagonal structure of W and L is visually crystal-clear
    order_idx = np.argsort(y_sp_true)
    X_sp = X_sp[order_idx]
    y_sp_true = y_sp_true[order_idx]

    W_sp, deg_sp, L_sp, eigvals_sp, eigvecs_sp, U_sp, y_sp_pred = compute_spectral_step_by_step(
        X_sp, aff_type, gamma_sp, knn_sp, lap_type, k_clusters_sp
    )

    eigengap = float(eigvals_sp[k_clusters_sp] - eigvals_sp[k_clusters_sp - 1])
    ari_sp_step = adjusted_rand_score(y_sp_true, y_sp_pred)

    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric(
        tr("Smallest Eigenvalue (λ₁)", "Primer Autovalor (λ₁)"),
        f"{eigvals_sp[0]:.4f}",
        tr("Always 0 (connected graph)", "Siempre 0 (grafo conexo)"),
    )
    sm2.metric(
        tr("Fiedler Value (λ₂)", "Valor de Fiedler (λ₂)"),
        f"{eigvals_sp[1]:.4f}",
        tr("Algebraic connectivity", "Conectividad algebraica"),
    )
    sm3.metric(
        tr(f"Eigengap (λ_{k_clusters_sp+1} − λ_{k_clusters_sp})", f"Salto Espectral (λ_{k_clusters_sp+1} − λ_{k_clusters_sp})"),
        f"{eigengap:.4f}",
        tr("Larger gap = cleaner cut", "Mayor salto = corte más limpio"),
    )
    sm4.metric(
        tr("Clustering Accuracy (ARI)", "Precisión del Agrupamiento (ARI)"),
        f"{ari_sp_step * 100:.1f}%",
    )

    sp_tabs = st.tabs(
        [
            tr("Step 1: Similarity Graph (W)", "Paso 1: Grafo de Similitud (W)"),
            tr("Step 2: Matrices W, D & Laplacian L", "Paso 2: Matrices W, D y Laplaciano L"),
            tr("Step 3: Eigenvalues & The Fiedler Vector (v₂)", "Paso 3: Autovalores y Vector de Fiedler (v₂)"),
            tr("Step 4: Spectral Embedding Space (U)", "Paso 4: Espacio de Incrustación Espectral (U)"),
        ]
    )

    with sp_tabs[0]:
        c_g1, c_g2 = st.columns([1, 2])
        with c_g1:
            edge_thresh = st.slider(
                tr("Minimum Edge Weight Shown ($W_{ij} \\ge$)", "Peso Mínimo de Arista Mostrado ($W_{ij} \\ge$)"),
                0.05,
                0.70,
                0.20,
                step=0.05,
            )
            st.info(
                tr(
                    "**Step 1: Turning Points into a Network**\n\n"
                    "Before doing any linear algebra, Spectral Clustering connects nearby points with weighted edges $W_{ij}$:\n\n"
                    "* **Strong edges (short distances):** Keep neighbors within the same ring or moon tightly bound together.\n"
                    "* **Weak/Zero edges (long distances):** Points on different rings barely connect.\n\n"
                    "Try lowering **RBF $\\gamma$** in the sidebar: if $\\gamma$ is too small, bridges form *between* different rings and the clusters merge!",
                    "**Paso 1: Convirtiendo Puntos en una Red**\n\n"
                    "Antes de aplicar álgebra lineal, el Clustering Espectral une los puntos cercanos mediante aristas con peso $W_{ij}$:\n\n"
                    "* **Aristas fuertes (distancias cortas):** Mantienen unidos a los vecinos dentro del mismo anillo o media luna.\n"
                    "* **Aristas débiles/cero (distancias largas):** Puntos en distintos anillos casi no tienen conexión.\n\n"
                    "Prueba bajar **RBF $\\gamma$** en la barra lateral: ¡si $\\gamma$ es muy pequeño, se crean puentes *entre* anillos distintos y los grupos se mezclan!",
                )
            )
        with c_g2:
            fig_graph = go.Figure()
            ei, ej = np.nonzero(np.triu(W_sp >= edge_thresh, k=1))
            edge_x, edge_y = [], []
            for u, v in zip(ei, ej):
                edge_x.extend([X_sp[u, 0], X_sp[v, 0], None])
                edge_y.extend([X_sp[u, 1], X_sp[v, 1], None])

            fig_graph.add_trace(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    name=tr(f"Graph Edges (W_ij ≥ {edge_thresh})", f"Aristas del Grafo (W_ij ≥ {edge_thresh})"),
                    line=dict(color="rgba(100, 110, 250, 0.35)", width=1.3),
                    hoverinfo="skip",
                )
            )
            for c_id in range(k_clusters_sp):
                m_c = y_sp_pred == c_id
                fig_graph.add_trace(
                    go.Scatter(
                        x=X_sp[m_c, 0],
                        y=X_sp[m_c, 1],
                        mode="markers",
                        name=f"Cluster {c_id + 1}",
                        marker=dict(size=8, line=dict(width=1, color="black")),
                    )
                )
            fig_graph.update_layout(
                title=tr(
                    f"Similarity Graph W ({len(ei)} active edges with W_ij ≥ {edge_thresh})",
                    f"Grafo de Similitud W ({len(ei)} aristas activas con W_ij ≥ {edge_thresh})",
                ),
                xaxis_title="$x_1$",
                yaxis_title="$x_2$",
                yaxis=dict(scaleanchor="x", scaleratio=1),
                height=500,
                template="plotly_white",
            )
            st.plotly_chart(fig_graph, width="stretch")

    with sp_tabs[1]:
        st.markdown(
            tr(
                """
                #### Visualizing the Block-Diagonal Structure of $W$ and $L$
                Notice the **bright square blocks** along the diagonal of the Affinity Matrix $W$:
                - Each block is an island of points that strongly connect to each other, while the dark off-diagonal areas show almost zero connection between different islands.
                - The **Graph Laplacian** $L = D - W$ (or its normalized form $L_{\\text{sym}} = I - D^{-1/2}WD^{-1/2}$) has positive degrees on the diagonal and negative weights off the diagonal, acting as a **discrete Laplace operator** on the graph.
                """,
                """
                #### Visualizando la Estructura de Bloques Diagonales de $W$ y $L$
                Observa los **bloques cuadrados brillantes** sobre la diagonal de la Matriz de Afinidad $W$:
                - Cada bloque corresponde a una isla de puntos fuertemente conectados entre sí, mientras que las zonas oscuras fuera de la diagonal muestran que casi no hay puentes entre islas distintas.
                - El **Laplaciano del Grafo** $L = D - W$ (o su versión normalizada $L_{\\text{sym}} = I - D^{-1/2}WD^{-1/2}$) coloca los grados positivos en la diagonal y los pesos negativos fuera de ella.
                """,
            )
        )
        cm1, cm2 = st.columns(2)
        with cm1:
            fig_w = px.imshow(
                W_sp,
                color_continuous_scale="Viridis",
                title=tr("1. Affinity Matrix W (Similarity Between Nodes)", "1. Matriz de Afinidad W (Similitud Entre Nodos)"),
                labels={"x": tr("Node j", "Nodo j"), "y": tr("Node i", "Nodo i")},
            )
            fig_w.update_layout(height=440)
            st.plotly_chart(fig_w, width="stretch")
        with cm2:
            fig_l = px.imshow(
                L_sp,
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0.0,
                title=tr(f"2. Graph Laplacian Matrix L ({lap_type})", f"2. Matriz Laplaciana del Grafo L ({lap_type})"),
                labels={"x": tr("Node j", "Nodo j"), "y": tr("Node i", "Nodo i")},
            )
            fig_l.update_layout(height=440)
            st.plotly_chart(fig_l, width="stretch")

    with sp_tabs[2]:
        if not ES:
            st.markdown(
                r"""
                #### Why Do the Smallest Eigenvalues Reveal the Clusters?
                Recall the quadratic form of the Graph Laplacian:
                $$f^T L f = \frac{1}{2} \sum_{i,j} W_{ij} (f_i - f_j)^2$$
                - **Eigenvalue $\lambda_1 = 0$:** Achieved by the constant vector $v_1 = [1, 1, \dots, 1]^T$.
                - **If the graph has $k$ completely disconnected islands:** Then $\lambda_1 = \lambda_2 = \dots = \lambda_k = 0$, and their eigenvectors are indicator vectors for each island!
                - **The Fiedler Vector ($v_2$, for $\lambda_2$):** Assigned positive values ($+$) to one cluster and negative values ($-$) to the other cluster, cutting only the weakest bridges! Look at how sorting $v_2$ below creates a clean **step function**.
                """
            )
        else:
            st.markdown(
                r"""
                #### ¿Por Qué los Autovalores Más Pequeños Revelan los Grupos?
                Recordemos la forma cuadrática del Laplaciano del Grafo:
                $$f^T L f = \frac{1}{2} \sum_{i,j} W_{ij} (f_i - f_j)^2$$
                - **Autovalor $\lambda_1 = 0$:** Se obtiene con el vector constante $v_1 = [1, 1, \dots, 1]^T$.
                - **Si el grafo tiene $k$ islas desconectadas:** Entonces los primeros $k$ autovalores son casi cero ($\lambda_1 \approx \dots \approx \lambda_k \approx 0$), seguidos por un salto brusco (**Eigengap**) en $\lambda_{k+1}$.
                - **El Vector de Fiedler ($v_2$, asociado a $\lambda_2$):** Asigna valores positivos ($+$) a una isla y negativos ($-$) a la otra. ¡Observa abajo cómo al ordenar $v_2$ aparece un **escalón perfecto** que separa los grupos!
                """
            )

        ce1, ce2, ce3 = st.columns(3)
        with ce1:
            n_show_eig = min(12, len(eigvals_sp))
            bar_colors = ["#00CC96" if i < k_clusters_sp else "#636EFA" for i in range(n_show_eig)]
            fig_eig = go.Figure(
                go.Bar(
                    x=[f"λ_{i+1}" for i in range(n_show_eig)],
                    y=eigvals_sp[:n_show_eig],
                    marker_color=bar_colors,
                    text=[f"{v:.3f}" for v in eigvals_sp[:n_show_eig]],
                    textposition="outside",
                )
            )
            fig_eig.add_vline(
                x=k_clusters_sp - 0.5,
                line_dash="dash",
                line_color="#EF553B",
                annotation_text=tr(f"Eigengap (k={k_clusters_sp})", f"Salto Espectral (k={k_clusters_sp})"),
            )
            fig_eig.update_layout(
                title=tr("1. Smallest Eigenvalues of L", "1. Menores Autovalores de L"),
                xaxis_title=tr("Eigenvalue Index", "Índice del Autovalor"),
                yaxis_title=tr("Eigenvalue (λ)", "Autovalor (λ)"),
                height=430,
                template="plotly_white",
            )
            st.plotly_chart(fig_eig, width="stretch")

        with ce2:
            fiedler_vec = eigvecs_sp[:, 1]
            fig_f2d = px.scatter(
                x=X_sp[:, 0],
                y=X_sp[:, 1],
                color=fiedler_vec,
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0.0,
                labels={"x": "$x_1$", "y": "$x_2$", "color": "v₂ (Fiedler)"},
                title=tr("2. Points Colored by Fiedler Vector (v₂)", "2. Puntos Coloreados por el Vector de Fiedler (v₂)"),
            )
            fig_f2d.update_traces(marker=dict(size=8, line=dict(width=1, color="black")))
            fig_f2d.update_yaxes(scaleanchor="x", scaleratio=1)
            fig_f2d.update_layout(height=430, template="plotly_white")
            st.plotly_chart(fig_f2d, width="stretch")

        with ce3:
            sorted_v2 = np.sort(fiedler_vec)
            fig_step = go.Figure()
            fig_step.add_trace(
                go.Scatter(
                    x=list(range(len(sorted_v2))),
                    y=sorted_v2,
                    mode="lines+markers",
                    line=dict(color="#EF553B", width=2.5),
                    marker=dict(size=5),
                    name="Sorted v₂",
                )
            )
            fig_step.add_hline(y=0.0, line_dash="dot", line_color="black")
            fig_step.update_layout(
                title=tr("3. Sorted Fiedler Vector v₂ (Step Function)", "3. Vector de Fiedler v₂ Ordenado (Escalón)"),
                xaxis_title=tr("Sorted Node Index", "Índice de Nodo Ordenado"),
                yaxis_title=tr("Fiedler Value v₂(i)", "Valor de Fiedler v₂(i)"),
                height=430,
                template="plotly_white",
            )
            st.plotly_chart(fig_step, width="stretch")

    with sp_tabs[3]:
        st.markdown(
            tr(
                """
                #### Step 4: Why spectral clustering Works Effortlessly Inside the Spectral Embedding ($U$)
                Look at the comparison below:
                - **Left (Original 2D Space):** The rings or moons are curved and non-convex, so Euclidean K-Means fails.
                - **Right (Spectral Embedding Coordinates $v_2$ vs. $v_3$):** When we plot each point using the eigenvectors of the Graph Laplacian as its new coordinates, **every connected island collapses into a tight, linearly separable cluster**! Standard K-Means now separates them with $100\\%$ accuracy.
                """,
                """
                #### Paso 4: Por Qué el clustering espectral Funciona Sin Esfuerzo en el Espacio Espectral ($U$)
                Observa la comparación de abajo:
                - **Izquierda (Espacio 2D Original):** Los anillos o medias lunas son curvos y no convexos, por lo que K-Means falla.
                - **Derecha (Coordenadas Espectrales $v_2$ vs. $v_3$):** Cuando graficamos cada punto usando los autovectores del Laplaciano como sus nuevas coordenadas, **¡cada isla conectada se agrupa en un rincón perfectamente separado!** Ahora K-Means los separa sin ningún problema.
                """,
            )
        )
        cs1, cs2 = st.columns(2)
        with cs1:
            fig_orig_sp = px.scatter(
                x=X_sp[:, 0],
                y=X_sp[:, 1],
                color=format_cluster_labels(y_sp_pred),
                labels={"x": "$x_1$", "y": "$x_2$", "color": tr("Spectral Cluster", "Cluster Espectral")},
                title=tr("Original 2D Data Colored by Final Spectral Clusters", "Datos 2D Originales con Clusters Espectrales Finales"),
            )
            fig_orig_sp.update_traces(marker=dict(size=8))
            fig_orig_sp.update_yaxes(scaleanchor="x", scaleratio=1)
            fig_orig_sp.update_layout(height=450, template="plotly_white")
            st.plotly_chart(fig_orig_sp, width="stretch")

        with cs2:
            v_x = eigvecs_sp[:, 1]
            v_y = eigvecs_sp[:, 2] if eigvecs_sp.shape[1] > 2 else eigvecs_sp[:, 0]
            fig_emb_sp = px.scatter(
                x=v_x,
                y=v_y,
                color=format_cluster_labels(y_sp_pred),
                labels={
                    "x": tr("2nd Eigenvector v₂ (Fiedler)", "2.º Autovector v₂ (Fiedler)"),
                    "y": tr("3rd Eigenvector v₃", "3.º Autovector v₃"),
                    "color": tr("Spectral Cluster", "Cluster Espectral"),
                },
                title=tr(
                    "Spectral Embedding Space (v₂ vs. v₃) — Linearly Separable!",
                    "Espacio de Incrustación Espectral (v₂ vs. v₃) — ¡Linealmente Separable!",
                ),
            )
            fig_emb_sp.update_traces(marker=dict(size=9, opacity=0.85))
            fig_emb_sp.update_layout(height=450, template="plotly_white")
            st.plotly_chart(fig_emb_sp, width="stretch")


# ==============================================================================
# MODULE 4: REAL-WORLD CASE STUDIES (GPS, FRAUD, NETWORKS, IMAGE SEGMENTATION)
# ==============================================================================
elif mode_idx == 3:
    st.subheader(
        tr(
            "Module 4: Real-World Applications of DBSCAN & Spectral Clustering",
            "Módulo 4: Aplicaciones del Mundo Real de DBSCAN y Clustering Espectral",
        )
    )

    case_tabs = st.tabs(
        [
            tr("1. 🌎 Seismic Faults & GPS Hotspots (DBSCAN)", "1. 🌎 Fallas Sísmicas y Hotspots GPS (DBSCAN)"),
            tr("2. 💳 Credit Card Fraud & Anomaly Detection (DBSCAN)", "2. 💳 Detección de Fraude Bancario y Anomalías (DBSCAN)"),
            tr("3. 🕸️ Social Network Community Detection (Spectral)", "3. 🕸️ Detección de Comunidades en Redes (Espectral)"),
            tr("4. 🖼️ Image Segmentation via Normalized Cuts (Spectral)", "4. 🖼️ Segmentación de Imágenes con Normalized Cuts (Espectral)"),
        ]
    )

    # --- CASE 1: SEISMIC FAULTS & GPS HOTSPOTS ---
    with case_tabs[0]:
        st.markdown(
            tr(
                """
                ### Case Study 1: Mapping Tectonic Fault Lines from Earthquake Epicenters (DBSCAN)
                **Real-World Problem:** Seismologists and ride-sharing platforms (like Uber/Lyft) collect thousands of **GPS coordinates (Latitude, Longitude)**:
                - Earthquakes occur along **curved tectonic fault lines** (like the Central American Volcanic Arc or San Andreas Fault), plus isolated background tremors scattered everywhere.
                - **Why K-Means Fails:** K-Means forces circular zones and includes random background tremors, distorting the fault lines.
                - **Why DBSCAN Excels:** Discovers the exact curved geometry of each fault line and classifies isolated tremors as background noise (`-1`).
                """,
                """
                ### Caso Real 1: Mapeo de Fallas Tectónicas a partir de Epicentros Sísmicos (DBSCAN)
                **Problema Real:** Sismólogos y plataformas de movilidad (como Uber o pedidos a domicilio) registran miles de **coordenadas GPS (Latitud, Longitud)**:
                - Los sismos se concentran a lo largo de **fallas tectónicas curvas** (como el Arco Volcánico Centroamericano o la Fosa Mesoamericana), acompañados de temblores aislados de fondo.
                - **Por qué falla K-Means:** Obliga a partir el mapa en zonas redondas e incluye el ruido de fondo, deformando las fallas.
                - **Por qué brilla DBSCAN:** Sigue la forma alargada y curva de cada falla sísmica y descarta automáticamente los sismos aislados como ruido (`-1`).
                """,
            )
        )
        cg1, cg2 = st.columns([1, 2])
        with cg1:
            eps_geo = st.slider(
                tr("Search Radius $\\varepsilon$ (degrees lat/lon)", "Radio de Búsqueda $\\varepsilon$ (grados lat/lon)"),
                0.05,
                0.35,
                0.13,
                step=0.01,
                key="eps_geo",
            )
            min_geo = st.slider(
                tr("Minimum Epicenters ($\\text{MinPts}$)", "Mínimo de Epicentros ($\\text{MinPts}$)"),
                4,
                20,
                7,
                key="min_geo",
            )

            rng_geo = np.random.default_rng(101)
            # Fault 1: Curved subduction trench off the coast
            lon1 = np.linspace(-90.2, -87.6, 130)
            lat1 = 13.05 - 0.35 * np.sin((lon1 + 90.2) * 1.1) + rng_geo.normal(0, 0.045, 130)
            lon1 += rng_geo.normal(0, 0.045, 130)
            # Fault 2: Volcanic chain inland
            lon2 = np.linspace(-89.8, -88.0, 110)
            lat2 = 13.72 - 0.22 * (lon2 + 89.8) + rng_geo.normal(0, 0.04, 110)
            lon2 += rng_geo.normal(0, 0.04, 110)
            # Fault 3: Localized seismic swarm
            lon3 = rng_geo.normal(-88.3, 0.07, 60)
            lat3 = rng_geo.normal(13.95, 0.06, 60)
            # Background isolated tremors
            lon_bg = rng_geo.uniform(-90.4, -87.4, 45)
            lat_bg = rng_geo.uniform(12.4, 14.3, 45)

            X_geo = np.column_stack(
                [
                    np.concatenate([lon1, lon2, lon3, lon_bg]),
                    np.concatenate([lat1, lat2, lat3, lat_bg]),
                ]
            )
            db_geo = DBSCAN(eps=eps_geo, min_samples=min_geo).fit(X_geo)
            lbl_geo = db_geo.labels_
            n_faults = len(set(lbl_geo) - {-1})
            n_bg_found = int(np.sum(lbl_geo == -1))

            st.metric(tr("Active Fault Zones Identified", "Zonas de Falla Identificadas"), f"{n_faults}")
            st.metric(tr("Isolated Tremors Filtered (Noise)", "Sismos Aislados Filtrados (Ruido)"), f"{n_bg_found} / 45")

        with cg2:
            fig_geo = px.scatter(
                x=X_geo[:, 0],
                y=X_geo[:, 1],
                color=format_cluster_labels(lbl_geo),
                color_discrete_map={tr("Noise (-1)", "Ruido (-1)"): "#888888"},
                labels={
                    "x": tr("Longitude (°W)", "Longitud (°O)"),
                    "y": tr("Latitude (°N)", "Latitud (°N)"),
                    "color": tr("Seismic Zone", "Zona Sísmica"),
                },
                title=tr(
                    f"DBSCAN Seismic Fault Discovery (ε={eps_geo}°, MinPts={min_geo})",
                    f"Descubrimiento de Fallas Sísmicas con DBSCAN (ε={eps_geo}°, MinPts={min_geo})",
                ),
            )
            fig_geo.update_traces(marker=dict(size=8, opacity=0.85))
            fig_geo.update_yaxes(scaleanchor="x", scaleratio=1)
            fig_geo.update_layout(height=470, template="plotly_white")
            st.plotly_chart(fig_geo, width="stretch")

    # --- CASE 2: CREDIT CARD FRAUD / ANOMALY DETECTION ---
    with case_tabs[1]:
        st.markdown(
            tr(
                """
                ### Case Study 2: Credit Card Fraud & Anomaly Detection (DBSCAN)
                **Real-World Problem:** Banks monitor millions of transactions described by features like **Transaction Amount** and **Distance from Home / Time-of-Day Anomaly Score**:
                - Legitimate customer behavior forms dense clusters (e.g., **daily groceries/coffee** and **monthly bill/payroll purchases**).
                - Fraudulent transactions appear as **isolated points in low-density regions** (e.g., sudden high-amount international purchases at 3 AM).
                - **How DBSCAN Solves It:** Any point assigned `label = -1` (Noise) is automatically flagged as an **anomaly for fraud review**.
                """,
                """
                ### Caso Real 2: Detección de Fraude con Tarjetas de Crédito y Anomalías (DBSCAN)
                **Problema Real:** Los bancos monitorean millones de transacciones descritas por variables como **Monto de Compra** y **Distancia al Hogar / Puntaje de Horario Inusual**:
                - Las compras legítimas forman grupos densos (por ejemplo, **compras diarias pequeñas** y **pagos mensuales o supermercado**).
                - Los fraudes aparecen como **puntos aislados en zonas de baja densidad** (compras repentinas de alto monto o en ubicaciones atípicas).
                - **Cómo lo Resuelve DBSCAN:** Todo punto etiquetado como `label = -1` (Ruido) se marca automáticamente como **alerta de fraude / anomalía**.
                """,
            )
        )
        cf1, cf2 = st.columns([1, 2])
        with cf1:
            eps_fr = st.slider(
                tr("DBSCAN Radius $\\varepsilon$ (standardized)", "Radio DBSCAN $\\varepsilon$ (estandarizado)"),
                0.15,
                0.80,
                0.36,
                step=0.02,
                key="eps_fr",
            )
            min_fr = st.slider(
                tr("MinPts (Normal Pattern Threshold)", "MinPts (Umbral de Comportamiento Normal)"),
                4,
                20,
                8,
                key="min_fr",
            )

            rng_fr = np.random.default_rng(77)
            # Cluster 1: Everyday small purchases near home
            c1_amt = rng_fr.normal(25, 10, 140)
            c1_dist = rng_fr.normal(4, 2.0, 140)
            # Cluster 2: Weekend supermarket / mall shopping
            c2_amt = rng_fr.normal(140, 22, 100)
            c2_dist = rng_fr.normal(12, 3.5, 100)
            # Fraudulent anomalies
            fr_amt = rng_fr.uniform(220, 480, 18)
            fr_dist = rng_fr.uniform(35, 95, 18)

            X_fr_raw = np.column_stack(
                [
                    np.concatenate([c1_amt, c2_amt, fr_amt]),
                    np.concatenate([c1_dist, c2_dist, fr_dist]),
                ]
            )
            X_fr_raw = np.clip(X_fr_raw, 1.0, None)
            is_true_fraud = np.array([False] * 240 + [True] * 18)

            X_fr_scaled = StandardScaler().fit_transform(X_fr_raw)
            lbl_fr = DBSCAN(eps=eps_fr, min_samples=min_fr).fit_predict(X_fr_scaled)
            pred_fraud = lbl_fr == -1

            tp = int(np.sum(pred_fraud & is_true_fraud))
            fp = int(np.sum(pred_fraud & ~is_true_fraud))
            fn = int(np.sum(~pred_fraud & is_true_fraud))
            recall_fr = tp / max(1, (tp + fn)) * 100.0
            prec_fr = tp / max(1, (tp + fp)) * 100.0

            st.metric(tr("Fraud Recall (True Frauds Caught)", "Fraudes Detectados (Recall)"), f"{recall_fr:.1f}% ({tp}/18)")
            st.metric(tr("Alert Precision", "Precisión de Alertas"), f"{prec_fr:.1f}%", tr(f"{fp} false alarms", f"{fp} falsas alarmas"))

        with cf2:
            status_labels = []
            for i in range(len(X_fr_raw)):
                if pred_fraud[i] and is_true_fraud[i]:
                    status_labels.append(tr("🚨 True Fraud Caught (-1)", "🚨 Fraude Real Detectado (-1)"))
                elif pred_fraud[i] and not is_true_fraud[i]:
                    status_labels.append(tr("⚠️ False Alarm (-1)", "⚠️ Falsa Alarma (-1)"))
                elif not pred_fraud[i] and is_true_fraud[i]:
                    status_labels.append(tr("❌ Missed Fraud", "❌ Fraude No Detectado"))
                else:
                    status_labels.append(tr(f"✅ Legitimate (Cluster {lbl_fr[i]+1})", f"✅ Legítima (Cluster {lbl_fr[i]+1})"))

            fig_fr = px.scatter(
                x=X_fr_raw[:, 0],
                y=X_fr_raw[:, 1],
                color=status_labels,
                labels={
                    "x": tr("Transaction Amount ($ USD)", "Monto de la Transacción ($ USD)"),
                    "y": tr("Distance from Cardholder Home (km)", "Distancia al Domicilio del Cliente (km)"),
                    "color": tr("Detection Status", "Estado de Detección"),
                },
                title=tr(
                    "Credit Card Anomaly Detection via DBSCAN Noise Classification",
                    "Detección de Anomalías Bancarias mediante Clasificación de Ruido en DBSCAN",
                ),
            )
            fig_fr.update_traces(marker=dict(size=9, opacity=0.88))
            fig_fr.update_layout(height=470, template="plotly_white")
            st.plotly_chart(fig_fr, width="stretch")

    # --- CASE 3: SOCIAL / COMMUNICATION NETWORK COMMUNITIES ---
    with case_tabs[2]:
        st.markdown(
            tr(
                r"""
                ### Case Study 3: Community Detection in Social & Organizational Networks (Spectral Clustering)
                **Real-World Problem:** In social networks, scientific citation graphs, or company communication networks, there are **no $(x, y)$ coordinates**—only a graph of who interacts with whom ($W_{ij}$).
                - **Why K-Means & DBSCAN Cannot Run Directly:** You only have an adjacency matrix $W$, not Euclidean feature vectors.
                - **Why Spectral Clustering Is the Gold Standard:** Minimizing the Normalized Cut ($\text{NCut}$) via the eigenvectors of $L_{\text{sym}} = I - D^{-1/2}WD^{-1/2}$ partitions the network into tightly knit communities while cutting the minimum number of cross-team bridges!
                """,
                r"""
                ### Caso Real 3: Detección de Comunidades en Redes Sociales y Organizacionales (Clustering Espectral)
                **Problema Real:** En redes sociales, grafos de citas científicas o comunicaciones dentro de una empresa, **no existen coordenadas $(x, y)$**, solo una tabla de quién interactúa con quién ($W_{ij}$).
                - **Por qué K-Means y DBSCAN no aplican directamente:** Solo tenemos una matriz de adyacencia $W$, no vectores en el espacio euclidiano.
                - **Por qué el Clustering Espectral es el Estándar de Oro:** Minimizar el Corte Normalizado ($\text{NCut}$) usando los autovectores de $L_{\text{sym}} = I - D^{-1/2}WD^{-1/2}$ divide la red en comunidades densas cortando el mínimo número de puentes entre departamentos.
                """,
            )
        )
        cn1, cn2 = st.columns([1, 2])
        with cn1:
            n_comm = st.slider(
                tr("Number of Communities ($k$) to Partition", "Número de Comunidades ($k$) a Detectar"),
                2,
                4,
                3,
                key="k_comm",
            )
            p_bridge = st.slider(
                tr("Cross-Community Bridge Probability", "Probabilidad de Puentes Entre Comunidades"),
                0.02,
                0.25,
                0.06,
                step=0.02,
                key="p_bridge",
            )

            # Stochastic Block Model (3 departments of 16 people each = 48 nodes)
            rng_net = np.random.default_rng(42)
            sizes = [16, 16, 16]
            n_nodes = sum(sizes)
            true_dept = np.repeat([0, 1, 2], sizes)
            W_net = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    prob = 0.55 if true_dept[i] == true_dept[j] else p_bridge
                    if rng_net.random() < prob:
                        W_net[i, j] = W_net[j, i] = 1.0

            deg_net = np.sum(W_net, axis=1)
            d_inv_s = 1.0 / np.sqrt(np.maximum(deg_net, 1e-8))
            L_net = np.eye(n_nodes) - (d_inv_s[:, None] * W_net * d_inv_s[None, :])
            evals_net, evecs_net = eigh(L_net)
            U_net = evecs_net[:, :n_comm]
            U_net_norm = U_net / (np.linalg.norm(U_net, axis=1, keepdims=True) + 1e-8)
            comm_labels = KMeans(n_clusters=n_comm, n_init=10, random_state=42).fit_predict(U_net_norm)

            # Count internal vs cut edges
            ei_n, ej_n = np.nonzero(np.triu(W_net > 0, k=1))
            cut_edges = int(np.sum(comm_labels[ei_n] != comm_labels[ej_n]))
            total_edges = len(ei_n)

            st.metric(tr("Total Network Connections", "Conexiones Totales en la Red"), f"{total_edges}")
            st.metric(
                tr("Bridges Cut Between Communities", "Puentes Cortados Entre Comunidades"),
                f"{cut_edges} ({cut_edges / max(1, total_edges) * 100:.1f}%)",
            )

        with cn2:
            # Position nodes using Spectral layout (v2, v3) + slight jitter for clear node visibility
            rng_pos = np.random.default_rng(7)
            pos_x = evecs_net[:, 1] + rng_pos.normal(0, 0.025, n_nodes)
            pos_y = evecs_net[:, 2] + rng_pos.normal(0, 0.025, n_nodes)

            in_x, in_y, cut_x, cut_y = [], [], [], []
            for u, v in zip(ei_n, ej_n):
                if comm_labels[u] == comm_labels[v]:
                    in_x.extend([pos_x[u], pos_x[v], None])
                    in_y.extend([pos_y[u], pos_y[v], None])
                else:
                    cut_x.extend([pos_x[u], pos_x[v], None])
                    cut_y.extend([pos_y[u], pos_y[v], None])

            fig_net = go.Figure()
            fig_net.add_trace(
                go.Scatter(
                    x=in_x,
                    y=in_y,
                    mode="lines",
                    name=tr("Internal Community Links", "Enlaces Internos de Comunidad"),
                    line=dict(color="rgba(100, 120, 220, 0.45)", width=1.4),
                    hoverinfo="skip",
                )
            )
            fig_net.add_trace(
                go.Scatter(
                    x=cut_x,
                    y=cut_y,
                    mode="lines",
                    name=tr("Weak Bridges Cut (NCut)", "Puentes Débiles Cortados (NCut)"),
                    line=dict(color="rgba(239, 85, 59, 0.55)", width=1.2, dash="dot"),
                    hoverinfo="skip",
                )
            )
            for c_id in range(n_comm):
                m_c = comm_labels == c_id
                fig_net.add_trace(
                    go.Scatter(
                        x=pos_x[m_c],
                        y=pos_y[m_c],
                        mode="markers",
                        name=tr(f"Community {c_id + 1}", f"Comunidad {c_id + 1}"),
                        marker=dict(size=12, line=dict(width=1.5, color="black")),
                    )
                )
            fig_net.update_layout(
                title=tr(
                    "Spectral Community Detection in a Social Network (Positioned by Laplacian Eigenvectors v₂, v₃)",
                    "Detección Espectral de Comunidades en una Red Social (Ubicados según Autovectores v₂, v₃)",
                ),
                xaxis_title=tr("Fiedler Vector Coordinate (v₂)", "Coordenada Vector de Fiedler (v₂)"),
                yaxis_title=tr("3rd Laplacian Eigenvector (v₃)", "3.º Autovector del Laplaciano (v₃)"),
                height=470,
                template="plotly_white",
            )
            st.plotly_chart(fig_net, width="stretch")

    # --- CASE 4: IMAGE SEGMENTATION VIA NORMALIZED CUTS ---
    with case_tabs[3]:
        st.markdown(
            tr(
                r"""
                ### Case Study 4: Computer Vision Image Segmentation via Normalized Cuts (Shi & Malik, 2000)
                **Real-World Problem:** How does a computer segment an image into distinct objects (foreground vs. background) when pixels have both **brightness ($I_i$)** and **spatial position ($x_i$)**?
                - Shi & Malik (2000) revolutionized computer vision by defining pixel-to-pixel graph similarity combining **brightness similarity** and **spatial proximity**:
                  $$W_{ij} = \exp\left(-\frac{\|I_i - I_j\|^2}{\sigma_I^2}\right) \cdot \exp\left(-\frac{\|x_i - x_j\|^2}{\sigma_X^2}\right)$$
                - Computing the top eigenvectors of the Normalized Laplacian $L_{\text{sym}}$ segments contiguous visual regions cleanly!
                """,
                r"""
                ### Caso Real 4: Segmentación de Imágenes en Visión por Computadora con Normalized Cuts (Shi & Malik, 2000)
                **Problema Real:** ¿Cómo separa una computadora los objetos de una imagen (figura vs. fondo) considerando tanto el **brillo de cada píxel ($I_i$)** como su **ubicación espacial ($x_i$)**?
                - Shi y Malik (2000) revolucionaron la visión artificial definiendo el peso entre dos píxeles como el producto de su **similitud de intensidad** y su **cercanía espacial**:
                  $$W_{ij} = \exp\left(-\frac{\|I_i - I_j\|^2}{\sigma_I^2}\right) \cdot \exp\left(-\frac{\|x_i - x_j\|^2}{\sigma_X^2}\right)$$
                - Los primeros autovectores del Laplaciano Normalizado $L_{\text{sym}}$ permiten segmentar regiones contiguas limpiamente.
                """,
            )
        )
        ci1, ci2, ci3 = st.columns([1, 1, 1])
        with ci1:
            k_seg = st.slider(
                tr("Number of Image Segments ($k$)", "Número de Regiones a Segmentar ($k$)"),
                2,
                4,
                3,
                key="k_seg",
            )
        with ci2:
            sigma_i = st.slider(
                tr("Brightness Sensitivity ($\\sigma_I$)", "Sensibilidad al Brillo ($\\sigma_I$)"),
                0.08,
                0.50,
                0.18,
                step=0.02,
                key="sig_i",
            )
        with ci3:
            img_noise_seg = st.slider(
                tr("Camera Sensor Noise", "Ruido del Sensor de Cámara"),
                0.01,
                0.18,
                0.06,
                step=0.01,
                key="noise_seg",
            )

        # Create 22x22 synthetic scene (sun/circle + ground horizon + sky) = 484 pixels
        grid_sz = 22
        rng_im = np.random.default_rng(42)
        gy_im, gx_im = np.mgrid[0:grid_sz, 0:grid_sz] / float(grid_sz)
        img_clean = np.full((grid_sz, grid_sz), 0.15)  # Sky
        # Bright circular object in upper-left/center
        sun_mask = (gx_im - 0.38) ** 2 + (gy_im - 0.35) ** 2 <= 0.055
        img_clean[sun_mask] = 0.90
        # Medium-intensity foreground hill at the bottom
        hill_mask = gy_im > (0.68 + 0.08 * np.sin(gx_im * 5.0))
        img_clean[hill_mask] = 0.52
        img_noisy = np.clip(img_clean + rng_im.normal(0, img_noise_seg, (grid_sz, grid_sz)), 0.0, 1.0)

        coords_px = np.column_stack([gx_im.ravel(), gy_im.ravel()])
        intens_px = img_noisy.ravel()[:, None]

        dist_spatial_sq = cdist(coords_px, coords_px, metric="sqeuclidean")
        dist_intens_sq = cdist(intens_px, intens_px, metric="sqeuclidean")

        W_img = np.exp(-dist_intens_sq / (sigma_i**2)) * np.exp(-dist_spatial_sq / (0.22**2))
        W_img[dist_spatial_sq > 0.12] = 0.0
        np.fill_diagonal(W_img, 0.0)

        deg_img = np.sum(W_img, axis=1)
        d_inv_img = 1.0 / np.sqrt(np.maximum(deg_img, 1e-8))
        L_img = np.eye(len(intens_px)) - (d_inv_img[:, None] * W_img * d_inv_img[None, :])
        _, evecs_img = eigh(L_img)

        U_img = evecs_img[:, :k_seg]
        U_img_norm = U_img / (np.linalg.norm(U_img, axis=1, keepdims=True) + 1e-8)
        seg_labels = KMeans(n_clusters=k_seg, n_init=10, random_state=42).fit_predict(U_img_norm)

        p_im1, p_im2, p_im3 = st.columns(3)
        with p_im1:
            fig_im1 = px.imshow(
                img_noisy,
                color_continuous_scale="gray",
                title=tr("1. Input Noisy Image (22×22 Pixels)", "1. Imagen de Entrada con Ruido (22×22 Píxeles)"),
            )
            fig_im1.update_layout(height=360, coloraxis_showscale=False)
            st.plotly_chart(fig_im1, width="stretch")
        with p_im2:
            fig_im2 = px.imshow(
                evecs_img[:, 1].reshape(grid_sz, grid_sz),
                color_continuous_scale="RdBu_r",
                title=tr("2. Fiedler Eigenvector Map (v₂)", "2. Mapa del Autovector de Fiedler (v₂)"),
            )
            fig_im2.update_layout(height=360, coloraxis_showscale=False)
            st.plotly_chart(fig_im2, width="stretch")
        with p_im3:
            fig_im3 = px.imshow(
                seg_labels.reshape(grid_sz, grid_sz),
                color_continuous_scale="Viridis",
                title=tr(f"3. Spectral Segmented Regions (k={k_seg})", f"3. Regiones Segmentadas Espectralmente (k={k_seg})"),
            )
            fig_im3.update_layout(height=360, coloraxis_showscale=False)
            st.plotly_chart(fig_im3, width="stretch")


# ==============================================================================
# MODULE 5: HIGH-DIMENSIONAL BENCHMARK DATASETS & CUSTOM CSV
# ==============================================================================
elif mode_idx == 4:
    st.subheader(
        tr(
            "Module 5: Exploring High-Dimensional Datasets & Your Own CSV",
            "Módulo 5: Explorando Conjuntos de Datos de Alta Dimensión y Tu Propio CSV",
        )
    )

    DS_LIST = [
        tr("Iris Flowers (4 features, 3 species)", "Flores Iris (4 variables, 3 especies)"),
        tr("Wine Chemistry (13 features, 3 cultivars)", "Química del Vino (13 variables, 3 cepas)"),
        tr("Breast Cancer Cells (30 features, 2 diagnoses)", "Células de Cáncer de Mama (30 variables, 2 diagnósticos)"),
        tr("Handwritten Digits 0–4 (64 features, 5 classes)", "Dígitos Manuscritos 0–4 (64 variables, 5 clases)"),
        tr("Upload Your Own CSV", "Subir tu Propio Archivo CSV"),
    ]
    ds_choice = st.sidebar.selectbox(
        tr("Select Dataset:", "Selecciona un Conjunto de Datos:"),
        range(5),
        format_func=lambda i: DS_LIST[i],
    )

    if ds_choice < 4:
        if ds_choice == 0:
            raw_d = load_iris()
            df_hd = pd.DataFrame(raw_d.data, columns=raw_d.feature_names)
            y_hd_true = raw_d.target
            y_hd_names = pd.Series([raw_d.target_names[i] for i in raw_d.target], name=tr("True Class", "Clase Real"))
            def_eps, def_k = 0.85, 3
        elif ds_choice == 1:
            raw_d = load_wine()
            df_hd = pd.DataFrame(raw_d.data, columns=raw_d.feature_names)
            y_hd_true = raw_d.target
            y_hd_names = pd.Series([raw_d.target_names[i] for i in raw_d.target], name=tr("True Class", "Clase Real"))
            def_eps, def_k = 2.35, 3
        elif ds_choice == 2:
            raw_d = load_breast_cancer()
            df_hd = pd.DataFrame(raw_d.data, columns=raw_d.feature_names)
            y_hd_true = raw_d.target
            y_hd_names = pd.Series([raw_d.target_names[i] for i in raw_d.target], name=tr("True Class", "Clase Real"))
            def_eps, def_k = 3.20, 2
        else:
            raw_d = load_digits(n_class=5)
            df_hd = pd.DataFrame(raw_d.data[:400], columns=[f"pixel_{i}" for i in range(64)])
            y_hd_true = raw_d.target[:400]
            y_hd_names = pd.Series([f"Digit {d}" for d in y_hd_true], name=tr("True Digit", "Dígito Real"))
            def_eps, def_k = 4.20, 5
        has_ground_truth = True
    else:
        up_csv = st.sidebar.file_uploader(tr("Upload CSV file", "Subir archivo CSV"), type=["csv"])
        if up_csv is None:
            st.info(
                tr(
                    "Please upload a CSV file in the sidebar, or choose a built-in benchmark dataset.",
                    "Por favor sube un archivo CSV en la barra lateral o elige un conjunto de datos integrado.",
                )
            )
            st.stop()
        raw_csv = pd.read_csv(up_csv)
        num_cols = raw_csv.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            st.error(tr("CSV must have at least 2 numeric columns.", "El CSV debe tener al menos 2 columnas numéricas."))
            st.stop()
        sel_cols = st.sidebar.multiselect(tr("Select Features:", "Selecciona Variables:"), num_cols, default=num_cols)
        df_hd = raw_csv[sel_cols].dropna()
        y_hd_true = np.zeros(len(df_hd), dtype=int)
        y_hd_names = pd.Series(["Sample"] * len(df_hd), name="Group")
        def_eps, def_k = 1.50, 3
        has_ground_truth = False

    st.sidebar.header(tr("Clustering Hyperparameters", "Hiperparámetros de Clustering"))
    std_hd = st.sidebar.checkbox(
        tr("Standardize features (Z-score)", "Estandarizar variables (Z-score)"),
        value=True,
    )
    eps_hd = st.sidebar.slider(
        tr("DBSCAN Radius ($\\varepsilon$)", "Radio DBSCAN ($\\varepsilon$)"),
        0.20,
        6.00,
        def_eps,
        step=0.05,
    )
    min_hd = st.sidebar.slider(
        tr("DBSCAN MinPts", "DBSCAN MinPts"),
        3,
        25,
        6,
    )
    k_hd = st.sidebar.slider(
        tr("Spectral & K-Means Clusters ($k$)", "Clusters Espectral y K-Means ($k$)"),
        2,
        10,
        def_k,
    )
    nn_hd = st.sidebar.slider(
        tr("Spectral k-NN Graph Neighbors", "Vecinos del Grafo k-NN Espectral"),
        5,
        30,
        12,
    )

    X_hd = StandardScaler().fit_transform(df_hd) if std_hd else df_hd.to_numpy()
    X_hd_pca = PCA(n_components=2, random_state=42).fit_transform(X_hd)

    lbl_hd_db = DBSCAN(eps=eps_hd, min_samples=min_hd).fit_predict(X_hd)
    lbl_hd_sp = SpectralClustering(
        n_clusters=k_hd,
        affinity="nearest_neighbors",
        n_neighbors=min(nn_hd, len(X_hd) - 1),
        random_state=42,
    ).fit_predict(X_hd)
    lbl_hd_km = KMeans(n_clusters=k_hd, n_init=10, random_state=42).fit_predict(X_hd)

    def safe_silhouette(X_arr: np.ndarray, lbls: np.ndarray) -> str:
        valid = lbls != -1
        if len(set(lbls[valid])) < 2 or np.sum(valid) < 5:
            return "N/A"
        return f"{silhouette_score(X_arr[valid], lbls[valid]):.3f}"

    hm1, hm2, hm3 = st.columns(3)
    if has_ground_truth:
        hm1.metric(
            tr("DBSCAN ARI / NMI vs True Classes", "DBSCAN ARI / NMI vs Clases Reales"),
            f"ARI: {adjusted_rand_score(y_hd_true, lbl_hd_db):.2f} | NMI: {normalized_mutual_info_score(y_hd_true, lbl_hd_db):.2f}",
            f"Silhouette: {safe_silhouette(X_hd, lbl_hd_db)}",
        )
        hm2.metric(
            tr("Spectral ARI / NMI vs True Classes", "Espectral ARI / NMI vs Clases Reales"),
            f"ARI: {adjusted_rand_score(y_hd_true, lbl_hd_sp):.2f} | NMI: {normalized_mutual_info_score(y_hd_true, lbl_hd_sp):.2f}",
            f"Silhouette: {safe_silhouette(X_hd, lbl_hd_sp)}",
        )
        hm3.metric(
            tr("K-Means ARI / NMI vs True Classes", "K-Means ARI / NMI vs Clases Reales"),
            f"ARI: {adjusted_rand_score(y_hd_true, lbl_hd_km):.2f} | NMI: {normalized_mutual_info_score(y_hd_true, lbl_hd_km):.2f}",
            f"Silhouette: {safe_silhouette(X_hd, lbl_hd_km)}",
        )
    else:
        hm1.metric("DBSCAN Silhouette", safe_silhouette(X_hd, lbl_hd_db))
        hm2.metric("Spectral Silhouette", safe_silhouette(X_hd, lbl_hd_sp))
        hm3.metric("K-Means Silhouette", safe_silhouette(X_hd, lbl_hd_km))

    hd_c1, hd_c2, hd_c3 = st.columns(3)
    with hd_c1:
        fig_h1 = px.scatter(
            x=X_hd_pca[:, 0],
            y=X_hd_pca[:, 1],
            color=y_hd_names,
            labels={"x": "PC1", "y": "PC2", "color": y_hd_names.name},
            title=tr("1. Ground Truth Classes (on 2D PCA Map)", "1. Clases Reales (en Mapa PCA 2D)"),
        )
        fig_h1.update_layout(height=430, template="plotly_white", legend=dict(orientation="h", y=-0.22))
        st.plotly_chart(fig_h1, width="stretch")

    with hd_c2:
        fig_h2 = px.scatter(
            x=X_hd_pca[:, 0],
            y=X_hd_pca[:, 1],
            color=format_cluster_labels(lbl_hd_db),
            color_discrete_map={tr("Noise (-1)", "Ruido (-1)"): "#888888"},
            labels={"x": "PC1", "y": "PC2", "color": "DBSCAN"},
            title=tr(f"2. DBSCAN Clusters (ε={eps_hd}, MinPts={min_hd})", f"2. Clusters DBSCAN (ε={eps_hd}, MinPts={min_hd})"),
        )
        fig_h2.update_layout(height=430, template="plotly_white", legend=dict(orientation="h", y=-0.22))
        st.plotly_chart(fig_h2, width="stretch")

    with hd_c3:
        fig_h3 = px.scatter(
            x=X_hd_pca[:, 0],
            y=X_hd_pca[:, 1],
            color=format_cluster_labels(lbl_hd_sp),
            labels={"x": "PC1", "y": "PC2", "color": "Spectral"},
            title=tr(f"3. Spectral Clustering (k={k_hd}, k-NN={nn_hd})", f"3. Clustering Espectral (k={k_hd}, k-NN={nn_hd})"),
        )
        fig_h3.update_layout(height=430, template="plotly_white", legend=dict(orientation="h", y=-0.22))
        st.plotly_chart(fig_h3, width="stretch")

    render_silhouette_explainer(
        X_hd,
        X_hd_pca,
        {"DBSCAN": lbl_hd_db, tr("Spectral", "Espectral"): lbl_hd_sp, "K-Means": lbl_hd_km},
        key_prefix="m5",
        axis_labels=("PC1", "PC2"),
        high_dim_note=True,
    )

    # --- Using the silhouette to choose k ---
    st.markdown(
        tr(
            "#### 🎯 Using the Silhouette to Choose the Number of Clusters ($k$)",
            "#### 🎯 Usar el Silhouette para Elegir el Número de Clusters ($k$)",
        )
    )
    st.markdown(
        tr(
            "When you have **no labels** (like with your own CSV), a common strategy is to try several values of $k$ and keep the one with the **highest average silhouette**. "
            "For DBSCAN, $k$ is not a parameter, so we sweep $\\varepsilon$ instead (and show the % of noise, since discarding many points can artificially inflate the score).",
            "Cuando **no tienes etiquetas** (como con tu propio CSV), una estrategia común es probar varios valores de $k$ y quedarte con el de **mayor silhouette promedio**. "
            "En DBSCAN $k$ no es un parámetro, así que barremos $\\varepsilon$ (y mostramos el % de ruido, porque descartar muchos puntos puede inflar artificialmente el puntaje).",
        )
    )
    if st.checkbox(
        tr("Compute silhouette curves (takes a few seconds)", "Calcular curvas de silhouette (tarda unos segundos)"),
        value=False,
        key="m5_sil_curve",
    ):

        @st.cache_data
        def silhouette_k_curves(X_arr: np.ndarray, n_nb: int, eps_center: float, min_pts: int):
            ks = list(range(2, 9))
            km_s, sp_s = [], []
            for k_ in ks:
                km_s.append(silhouette_score(X_arr, KMeans(n_clusters=k_, n_init=10, random_state=42).fit_predict(X_arr)))
                lb = SpectralClustering(
                    n_clusters=k_, affinity="nearest_neighbors", n_neighbors=min(n_nb, len(X_arr) - 1), random_state=42
                ).fit_predict(X_arr)
                sp_s.append(silhouette_score(X_arr, lb) if len(set(lb)) > 1 else np.nan)
            eps_grid = np.round(np.linspace(eps_center * 0.5, eps_center * 1.6, 12), 3)
            db_s, db_noise, db_k = [], [], []
            for e in eps_grid:
                lb = DBSCAN(eps=float(e), min_samples=min_pts).fit_predict(X_arr)
                s_ = compute_silhouette_values(X_arr, lb)
                db_s.append(np.nan if s_ is None else float(np.nanmean(s_)))
                db_noise.append(float(np.mean(lb == -1) * 100))
                db_k.append(len(set(lb) - {-1}))
            return ks, km_s, sp_s, eps_grid, db_s, db_noise, db_k

        ks, km_s, sp_s, eps_grid, db_s, db_noise, db_k = silhouette_k_curves(X_hd, nn_hd, float(eps_hd), min_hd)
        kc1, kc2 = st.columns(2)
        with kc1:
            fig_k = go.Figure()
            fig_k.add_trace(go.Scatter(x=ks, y=km_s, mode="lines+markers", name="K-Means", line=dict(width=3)))
            fig_k.add_trace(go.Scatter(x=ks, y=sp_s, mode="lines+markers", name=tr("Spectral", "Espectral"), line=dict(width=3)))
            best_k_km = ks[int(np.nanargmax(km_s))]
            fig_k.add_vline(
                x=best_k_km,
                line_dash="dot",
                line_color="gray",
                annotation_text=tr(f"Best k (K-Means) = {best_k_km}", f"Mejor k (K-Means) = {best_k_km}"),
            )
            fig_k.update_layout(
                title=tr("Average Silhouette vs. Number of Clusters k", "Silhouette Promedio vs. Número de Clusters k"),
                xaxis_title="k",
                yaxis_title=tr("Average silhouette", "Silhouette promedio"),
                height=420,
                template="plotly_white",
            )
            st.plotly_chart(fig_k, width="stretch")
        with kc2:
            fig_e = go.Figure()
            fig_e.add_trace(
                go.Scatter(
                    x=eps_grid,
                    y=db_s,
                    mode="lines+markers",
                    name=tr("DBSCAN silhouette", "Silhouette DBSCAN"),
                    text=[tr(f"{k_} clusters", f"{k_} clusters") for k_ in db_k],
                    line=dict(width=3, color="#00CC96"),
                )
            )
            fig_e.add_trace(
                go.Bar(
                    x=eps_grid,
                    y=db_noise,
                    name=tr("% noise", "% ruido"),
                    yaxis="y2",
                    opacity=0.3,
                    marker_color="#888888",
                )
            )
            fig_e.update_layout(
                title=tr("DBSCAN: Silhouette and % Noise vs. ε", "DBSCAN: Silhouette y % de Ruido vs. ε"),
                xaxis_title="ε",
                yaxis=dict(title=tr("Silhouette (noise excluded)", "Silhouette (sin ruido)")),
                yaxis2=dict(title=tr("% noise", "% ruido"), overlaying="y", side="right", range=[0, 100]),
                height=420,
                template="plotly_white",
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_e, width="stretch")
        if has_ground_truth:
            st.caption(
                tr(
                    f"This dataset truly has {len(set(y_hd_true))} classes. Does the silhouette peak agree? Real classes are not always the most compact groups in feature space.",
                    f"Este dataset realmente tiene {len(set(y_hd_true))} clases. ¿Coincide el pico del silhouette? Las clases reales no siempre son los grupos más compactos en el espacio de variables.",
                )
            )


# ==============================================================================
# MODULE 6: STRESS TESTS, LIMITATIONS & MASTER COMPARISON TABLE
# ==============================================================================
elif mode_idx == 5:
    st.subheader(
        tr(
            "Module 6: Stress Tests — When Do DBSCAN and Spectral Clustering Struggle?",
            "Módulo 6: Pruebas de Estrés — ¿Cuándo Sufren DBSCAN y el Clustering Espectral?",
        )
    )
    st.markdown(
        tr(
            """
            No clustering algorithm is magic (`No Free Lunch Theorem`). Understanding **when an algorithm breaks** is just as important as knowing how it works. Interact with the two classic failure modes below!
            """,
            """
            Ningún algoritmo de agrupamiento es mágico (*Teorema de No Free Lunch*). Comprender **cuándo falla un algoritmo** es tan importante como saber cómo funciona. ¡Interactúa abajo con los dos modos de fallo clásicos!
            """,
        )
    )

    st_tabs = st.tabs(
        [
            tr("1. DBSCAN's Achilles' Heel: Varying Densities", "1. El Talón de Aquiles de DBSCAN: Densidades Variables"),
            tr("2. Spectral's Achilles' Heel: Noisy Bridges Between Clusters", "2. El Talón de Aquiles Espectral: Puentes de Ruido"),
            tr("3. Master Comparison Table (K-Means vs. DBSCAN vs. Spectral)", "3. Tabla Comparativa Maestra (K-Means vs. DBSCAN vs. Espectral)"),
        ]
    )

    with st_tabs[0]:
        c_vd1, c_vd2 = st.columns([1, 2])
        with c_vd1:
            eps_vd = st.slider(
                tr("Single Global Radius $\\varepsilon$", "Radio Global Único $\\varepsilon$"),
                0.10,
                1.20,
                0.35,
                step=0.05,
                key="eps_vd",
            )
            min_vd = st.slider(tr("MinPts", "MinPts"), 3, 15, 6, key="min_vd")
            st.warning(
                tr(
                    "**Try to find a single $\\varepsilon$ that separates all 3 clusters:**\n\n"
                    "- **If $\\varepsilon$ is small (`0.20–0.40`):** DBSCAN separates the two tight clusters on the left, but shatters the sparse cluster on the right into **Noise (-1)**.\n"
                    "- **If $\\varepsilon$ is large (`0.80–1.00`):** It captures the sparse right cluster, but **merges the two tight left clusters into one**!\n\n"
                    "*(Solution in practice: **OPTICS** or **HDBSCAN**, or **Spectral Clustering with k-NN**, which adapts locally to density!)*",
                    "**Intenta encontrar un único $\\varepsilon$ que separe los 3 grupos:**\n\n"
                    "- **Si $\\varepsilon$ es pequeño (`0.20–0.40`):** DBSCAN separa bien los dos grupos compactos de la izquierda, pero destruye el grupo disperso de la derecha marcándolo como **Ruido (-1)**.\n"
                    "- **Si $\\varepsilon$ es grande (`0.80–1.00`):** Logra capturar el grupo disperso de la derecha, ¡pero **fusiona los dos grupos densos de la izquierda en uno solo**!\n\n"
                    "*(Solución en la práctica: **OPTICS**, **HDBSCAN** o **Clustering Espectral con k-NN**, que se adapta localmente a la densidad).* ",
                )
            )
        with c_vd2:
            X_vd, y_vd = make_blobs(
                n_samples=[110, 110, 90],
                centers=[[-2.2, 0.8], [-2.2, -0.8], [2.5, 0.0]],
                cluster_std=[0.22, 0.22, 1.15],
                random_state=42,
            )
            lbl_vd_db = DBSCAN(eps=eps_vd, min_samples=min_vd).fit_predict(X_vd)
            lbl_vd_sp = SpectralClustering(n_clusters=3, affinity="nearest_neighbors", n_neighbors=12, random_state=42).fit_predict(X_vd)

            sub1, sub2 = st.columns(2)
            with sub1:
                fig_vd1 = px.scatter(
                    x=X_vd[:, 0],
                    y=X_vd[:, 1],
                    color=format_cluster_labels(lbl_vd_db),
                    color_discrete_map={tr("Noise (-1)", "Ruido (-1)"): "#888888"},
                    title=tr(f"DBSCAN (Fixed ε={eps_vd})", f"DBSCAN (ε fijo = {eps_vd})"),
                    labels={"x": "$x_1$", "y": "$x_2$", "color": "DBSCAN"},
                )
                fig_vd1.update_yaxes(scaleanchor="x", scaleratio=1)
                fig_vd1.update_layout(height=420, template="plotly_white", legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig_vd1, width="stretch")
            with sub2:
                fig_vd2 = px.scatter(
                    x=X_vd[:, 0],
                    y=X_vd[:, 1],
                    color=format_cluster_labels(lbl_vd_sp),
                    title=tr("Spectral k-NN (Adapts to Local Density!)", "Espectral k-NN (¡Se Adapta a la Densidad Local!)"),
                    labels={"x": "$x_1$", "y": "$x_2$", "color": "Spectral"},
                )
                fig_vd2.update_yaxes(scaleanchor="x", scaleratio=1)
                fig_vd2.update_layout(height=420, template="plotly_white", legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig_vd2, width="stretch")

    with st_tabs[1]:
        c_br1, c_br2 = st.columns([1, 2])
        with c_br1:
            n_bridge_pts = st.slider(
                tr("Number of Noisy Bridge Points Between Moons", "Cantidad de Puntos Puente Entre las Lunas"),
                0,
                35,
                18,
                step=1,
            )
            min_pts_br = st.slider(
                tr("DBSCAN MinPts (Higher breaks thin bridges)", "DBSCAN MinPts (Más alto rompe puentes delgados)"),
                3,
                18,
                10,
            )
            st.info(
                tr(
                    "**The 'Bridge Effect' (Chaining Problem):**\n\n"
                    "When a thin chain of noisy points connects two clusters:\n"
                    "- **Spectral Clustering** treats every point as a graph node (it has no concept of 'Noise `-1`'), which can cause the graph cut to shift or merge across the bridge.\n"
                    "- **DBSCAN with high `MinPts`** requires a **dense** neighborhood to propagate, so a thin 1-point-wide bridge does not reach `MinPts` and gets discarded as Noise (`-1`)!",
                    "**El 'Efecto Puente' (*Chaining Problem*):**\n\n"
                    "Cuando una fila delgada de puntos de ruido conecta dos grupos:\n"
                    "- **Clustering Espectral** intenta agrupar absolutamente todos los puntos (no tiene etiqueta `-1` de ruido), por lo que un puente denso puede alterar el corte del grafo.\n"
                    "- **DBSCAN con un `MinPts` alto** exige una vecindad **densa** para propagarse; así, un puente delgado no alcanza `MinPts` y se descarta como Ruido (`-1`).",
                )
            )
        with c_br2:
            X_moons_b, _ = make_moons(n_samples=240, noise=0.06, random_state=42)
            if n_bridge_pts > 0:
                rng_br = np.random.default_rng(19)
                bx = np.linspace(0.05, 0.95, n_bridge_pts) + rng_br.normal(0, 0.03, n_bridge_pts)
                by = np.linspace(0.45, -0.05, n_bridge_pts) + rng_br.normal(0, 0.03, n_bridge_pts)
                X_br = np.vstack([X_moons_b, np.column_stack([bx, by])])
            else:
                X_br = X_moons_b
            X_br = StandardScaler().fit_transform(X_br)

            lbl_br_db = DBSCAN(eps=0.25, min_samples=min_pts_br).fit_predict(X_br)
            lbl_br_sp = SpectralClustering(n_clusters=2, affinity="rbf", gamma=15.0, random_state=42).fit_predict(X_br)

            sb1, sb2 = st.columns(2)
            with sb1:
                fig_br1 = px.scatter(
                    x=X_br[:, 0],
                    y=X_br[:, 1],
                    color=format_cluster_labels(lbl_br_db),
                    color_discrete_map={tr("Noise (-1)", "Ruido (-1)"): "#888888"},
                    title=tr(f"DBSCAN (eps=0.25, MinPts={min_pts_br})", f"DBSCAN (eps=0.25, MinPts={min_pts_br})"),
                )
                fig_br1.update_yaxes(scaleanchor="x", scaleratio=1)
                fig_br1.update_layout(height=420, template="plotly_white", legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig_br1, width="stretch")
            with sb2:
                fig_br2 = px.scatter(
                    x=X_br[:, 0],
                    y=X_br[:, 1],
                    color=format_cluster_labels(lbl_br_sp),
                    title=tr("Spectral Clustering (Forced to assign bridge)", "Clustering Espectral (Obligado a asignar el puente)"),
                )
                fig_br2.update_yaxes(scaleanchor="x", scaleratio=1)
                fig_br2.update_layout(height=420, template="plotly_white", legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig_br2, width="stretch")

    with st_tabs[2]:
        if not ES:
            st.markdown(
                r"""
                ### Master Comparison Table: K-Means vs. DBSCAN vs. Spectral Clustering

                | Feature / Criterion | **K-Means** | **DBSCAN** | **Spectral Clustering** |
                | :--- | :--- | :--- | :--- |
                | **Core Mathematical Principle** | Minimizes Euclidean variance (inertia) around $k$ centroids | Density-reachability via $\varepsilon$-balls and Core Points | Eigen-decomposition of the Graph Laplacian ($L = D - W$) |
                | **Cluster Geometry** | **Convex / Spherical only** (Voronoi cells) | **Arbitrary non-convex shapes** (rings, moons, fault lines) | **Arbitrary non-convex shapes & graph manifolds** |
                | **Must You Specify $k$ (# of clusters)?** | **Yes** ($k$ required) | **No** (discovers $k$ automatically from density) | **Yes** ($k$ required, though **Eigengap** helps pick $k$) |
                | **Handles Noise & Outliers?** | **No** (outliers pull centroids and distort clusters) | **Excellent** (explicitly classifies outliers as `-1`) | **Weak** (assigns every node to a cluster unless pre-filtered) |
                | **Is the Silhouette a Fair Judge?** | **Yes** (both favor compact, round clusters) | **Partially** (excludes noise `-1`; penalizes long/curved clusters) | **Partially** (penalizes rings & moons even when perfectly separated) |
                | **Handles Varying Cluster Densities?** | **Moderate** | **Weak** (single global $\varepsilon$ struggles; use HDBSCAN/OPTICS) | **Good** (especially with $k$-NN affinity graph + $L_{\text{sym}}$) |
                | **Works on Pure Graphs (No Coordinates)?** | **No** (needs Euclidean coordinates) | **Yes** (with precomputed distance matrix) | **Yes — Native** (works directly on any adjacency matrix $W$) |
                | **Key Hyperparameters** | `n_clusters` ($k$) | `eps` ($\varepsilon$), `min_samples` ($\text{MinPts}$) | `n_clusters` ($k$), `gamma` ($\gamma$) or `n_neighbors` |
                | **Computational Complexity** | **Very Fast:** $O(N \cdot k \cdot d \cdot i)$ | **Fast with spatial index:** $O(N \log N)$ (worst $O(N^2)$) | **Heavy on large $N$:** $O(N^3)$ (or $O(N^2)$ with sparse Lanczos) |
                | **Best Real-World Use Cases** | Customer segmentation, vector quantization, fast baseline | **GPS hotspots, seismic faults, fraud/anomaly detection, LiDAR** | **Social network communities, image segmentation (NCut), manifold data** |
                """
            )
        else:
            st.markdown(
                r"""
                ### Tabla Comparativa Maestra: K-Means vs. DBSCAN vs. Clustering Espectral

                | Criterio / Característica | **K-Means** | **DBSCAN** | **Clustering Espectral** |
                | :--- | :--- | :--- | :--- |
                | **Principio Matemático** | Minimiza la inercia (varianza euclidiana) hacia $k$ centroides | Alcanzabilidad por densidad mediante bolas-$\varepsilon$ y Puntos Núcleo | Autovalores y autovectores del Laplaciano del Grafo ($L = D - W$) |
                | **Geometría de los Grupos** | **Solo formas convexas / esféricas** (celdas de Voronoi) | **Formas arbitrarias no convexas** (anillos, lunas, fallas) | **Formas arbitrarias no convexas y variedades en grafos** |
                | **¿Requiere pedir el número de grupos ($k$)?** | **Sí** ($k$ obligatorio) | **No** (descubre el número de grupos automáticamente) | **Sí** ($k$ requerido, aunque el **Salto Espectral** ayuda a elegirlo) |
                | **¿Maneja Ruido y Anomalías?** | **No** (los outliers arrastran los centroides) | **Excelente** (aísla automáticamente el ruido con etiqueta `-1`) | **Bajo** (obliga a asignar cada nodo a algún grupo) |
                | **¿Es el Silhouette un Juez Justo?** | **Sí** (ambos favorecen grupos compactos y redondos) | **Parcialmente** (excluye el ruido `-1`; castiga grupos alargados/curvos) | **Parcialmente** (castiga anillos y lunas aunque estén perfectamente separados) |
                | **¿Maneja Densidades Muy Diferentes?** | **Moderado** | **Difícil** (un único $\varepsilon$ global sufre; usar HDBSCAN/OPTICS) | **Bueno** (especialmente usando grafo $k$-NN + $L_{\text{sym}}$) |
                | **¿Funciona en Redes Puras (Sin Coordenadas)?** | **No** (requiere vectores euclidianos) | **Sí** (con matriz de distancias precalculada) | **Sí — Nativo** (opera directamente sobre cualquier matriz de afinidad $W$) |
                | **Hiperparámetros Clave** | `n_clusters` ($k$) | `eps` ($\varepsilon$), `min_samples` ($\text{MinPts}$) | `n_clusters` ($k$), `gamma` ($\gamma$) o `n_neighbors` |
                | **Complejidad Computacional** | **Muy Rápido:** $O(N \cdot k \cdot d \cdot i)$ | **Rápido con índice espacial:** $O(N \log N)$ (peor caso $O(N^2)$) | **Costoso en $N$ grande:** $O(N^3)$ (u $O(N^2)$ con matrices dispersas) |
                | **Casos de Uso Ideales en el Mundo Real** | Segmentación general de clientes, compresión vectorial rápida | **Hotspots GPS, fallas sísmicas, detección de fraude/anomalías, LiDAR** | **Comunidades en redes sociales, segmentación de imágenes (NCut), grafos** |
                """
            )
