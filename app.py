import os
import io
import re
import base64
from itertools import product
from collections import Counter
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

app = Flask(__name__)
CORS(app)

# ─── Parsing ──────────────────────────────────────────────────────────────────

def parse_plain_multiline(text):
    """One-edge-per-line: header 'N M', then 'u v' per line."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        raise ValueError("File must contain at least two lines.")
    n_declared = int(lines[0].split()[0])
    G = nx.Graph()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 2:
            G.add_edge(int(parts[0]), int(parts[1]))
    return n_declared, [G]


def parse_graph_text(text):
    """All-edges-on-one-line or parenthesis (u,v) format."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        raise ValueError("File must contain at least two lines.")
    n_declared = int(lines[0].split()[0])
    graphs = []
    for line in lines[1:]:
        edges = re.findall(r'\((\d+)\s*,\s*(\d+)\)', line)
        if not edges:
            nums = re.findall(r'\d+', line)
            edges = [(nums[i], nums[i+1]) for i in range(0, len(nums)-1, 2)]
        if edges:
            G = nx.Graph()
            for u, v in edges:
                G.add_edge(int(u), int(v))
            graphs.append(G)
    return n_declared, graphs


def smart_parse(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        raise ValueError("File must contain at least two lines.")
    sample = lines[1]
    nums = re.findall(r'\d+', sample)
    parens = re.findall(r'\((\d+)\s*,\s*(\d+)\)', sample)
    if parens or len(nums) > 2:
        return parse_graph_text(text)
    else:
        return parse_plain_multiline(text)


# ─── k-WL (exact from notebook) ───────────────────────────────────────────────

def k_wl_refinement(G, k):
    nodes = list(G.nodes())
    tuples = list(product(nodes, repeat=k))
    colors = {t: 0 for t in tuples}
    history = [colors.copy()]

    while True:
        new_colors = {}
        signatures = {}
        color_map = {}
        next_color = 0

        for t in tuples:
            neighbor_multiset = []
            for i in range(k):
                node = t[i]
                for v in G.neighbors(node):
                    new_tuple = list(t)
                    new_tuple[i] = v
                    new_tuple = tuple(new_tuple)
                    if new_tuple in colors:
                        neighbor_multiset.append((colors[new_tuple], 1))
            neighbor_multiset.sort()
            degrees = tuple(G.degree(node) for node in t)
            signature = (colors[t], tuple(neighbor_multiset), degrees)
            signatures[t] = signature

        for t in tuples:
            sig = signatures[t]
            if sig not in color_map:
                color_map[sig] = next_color
                next_color += 1
            new_colors[t] = color_map[sig]

        history.append(new_colors.copy())
        if new_colors == colors:
            break
        colors = new_colors

    return colors, history


def graph_signature(final_colors):
    freq = {}
    for c in final_colors.values():
        freq[c] = freq.get(c, 0) + 1
    return tuple(sorted(freq.items()))


# ─── Non-isomorphism analysis ─────────────────────────────────────────────────

def analyse_non_isomorphism(G1, G2, histories):
    reasons = []

    if G1.number_of_nodes() != G2.number_of_nodes():
        reasons.append({
            'type': 'node_count',
            'title': 'Different number of nodes',
            'detail': (f'Graph 1 has {G1.number_of_nodes()} nodes; '
                       f'Graph 2 has {G2.number_of_nodes()} nodes.')
        })

    if G1.number_of_edges() != G2.number_of_edges():
        reasons.append({
            'type': 'edge_count',
            'title': 'Different number of edges',
            'detail': (f'Graph 1 has {G1.number_of_edges()} edges; '
                       f'Graph 2 has {G2.number_of_edges()} edges.')
        })

    deg1 = sorted([d for _, d in G1.degree()], reverse=True)
    deg2 = sorted([d for _, d in G2.degree()], reverse=True)
    if deg1 != deg2:
        diff_idx = next(
            (i for i, (a, b) in enumerate(zip(deg1, deg2)) if a != b), None
        )
        reasons.append({
            'type': 'degree_sequence',
            'title': 'Degree sequences differ',
            'detail': (
                f'The sorted degree sequences are not equal.\n'
                f'Graph 1: {deg1}\n'
                f'Graph 2: {deg2}\n'
                + (f'First mismatch at position {diff_idx}: '
                   f'G1 has degree {deg1[diff_idx]}, G2 has degree {deg2[diff_idx]}.'
                   if diff_idx is not None else '')
            ),
            'deg1': deg1,
            'deg2': deg2
        })

    dist1 = dict(sorted(Counter(deg1).items()))
    dist2 = dict(sorted(Counter(deg2).items()))
    if dist1 != dist2:
        all_degs = sorted(set(dist1) | set(dist2))
        diffs = [
            f'  Degree {d}: Graph 1 → {dist1.get(d, 0)} node(s), Graph 2 → {dist2.get(d, 0)} node(s)'
            for d in all_degs if dist1.get(d, 0) != dist2.get(d, 0)
        ]
        reasons.append({
            'type': 'degree_distribution',
            'title': 'Degree distribution mismatch',
            'detail': 'Number of nodes per degree value differs:\n' + '\n'.join(diffs),
            'dist1': dist1,
            'dist2': dist2
        })

    unmatched = []
    for node, deg in sorted(G1.degree()):
        if deg not in dist2:
            unmatched.append(
                f'Node {node} in Graph 1 has degree {deg} — '
                f'no node with this degree exists in Graph 2.'
            )
    for node, deg in sorted(G2.degree()):
        if deg not in dist1:
            unmatched.append(
                f'Node {node} in Graph 2 has degree {deg} — '
                f'no node with this degree exists in Graph 1.'
            )
    if unmatched:
        reasons.append({
            'type': 'unmatched_degrees',
            'title': 'Nodes with unmatched degree values',
            'detail': '\n'.join(unmatched[:12]) + ('\n…' if len(unmatched) > 12 else '')
        })

    for iter_idx, (c1, c2) in enumerate(zip(histories[0], histories[1])):
        hist1 = Counter(c1.values())
        hist2 = Counter(c2.values())
        freq1 = tuple(sorted(hist1.values(), reverse=True))
        freq2 = tuple(sorted(hist2.values(), reverse=True))
        if freq1 != freq2:
            reasons.append({
                'type': 'wl_divergence',
                'title': f'k-WL colour refinement diverges at iteration {iter_idx}',
                'detail': (
                    f'At WL iteration {iter_idx}, the colour frequency multisets first differ.\n'
                    f'Graph 1 colour profile: {sorted(hist1.values(), reverse=True)}\n'
                    f'Graph 2 colour profile: {sorted(hist2.values(), reverse=True)}\n\n'
                    f'Interpretation: After {iter_idx} rounds of neighbourhood aggregation, '
                    f'the two graphs produce different colour distributions, proving they '
                    f'have structurally distinct {iter_idx}-hop neighbourhoods.'
                ),
                'iteration': iter_idx
            })
            break

    cc1 = nx.number_connected_components(G1)
    cc2 = nx.number_connected_components(G2)
    if cc1 != cc2:
        reasons.append({
            'type': 'connected_components',
            'title': 'Different number of connected components',
            'detail': (f'Graph 1: {cc1} component(s); Graph 2: {cc2} component(s).\n'
                       f'Isomorphic graphs must have the same component structure.')
        })

    tri1 = sum(nx.triangles(G1).values()) // 3
    tri2 = sum(nx.triangles(G2).values()) // 3
    if tri1 != tri2:
        reasons.append({
            'type': 'triangles',
            'title': 'Different triangle counts',
            'detail': (f'Graph 1 contains {tri1} triangle(s); '
                       f'Graph 2 contains {tri2} triangle(s).\n'
                       f'Triangle count is preserved under isomorphism.')
        })

    avg_c1 = round(nx.average_clustering(G1), 4)
    avg_c2 = round(nx.average_clustering(G2), 4)
    if avg_c1 != avg_c2:
        reasons.append({
            'type': 'clustering',
            'title': 'Average clustering coefficients differ',
            'detail': (f'Graph 1: {avg_c1}\nGraph 2: {avg_c2}\n'
                       f'Different clustering reflects different local neighbourhood densities.')
        })

    if not reasons:
        reasons.append({
            'type': 'wl_only',
            'title': 'Subtle non-isomorphism (k-WL only)',
            'detail': (
                'All basic structural invariants match (degree sequence, triangle count, etc.), '
                'but k-WL refinement still distinguishes the graphs. '
                'This is a case where the non-isomorphism only becomes visible through '
                'higher-order neighbourhood structure.'
            )
        })

    return reasons


# ─── Rendering ────────────────────────────────────────────────────────────────

def node_vals_for_iter(G, colors_at_iter):
    """Map k-tuple colours back to per-node average values for colouring."""
    nodes = list(G.nodes())
    node_color_map = {n: [] for n in nodes}
    for tup, color in colors_at_iter.items():
        for n in tup:
            node_color_map[n].append(color)
    return [
        sum(node_color_map[n]) / len(node_color_map[n]) if node_color_map[n] else 0
        for n in nodes
    ]


def render_single_iteration(G, colors_at_iter, pos, k, iter_idx, num_iters):
    """Render one graph at one iteration. Returns base64 PNG."""
    nodes = list(G.nodes())
    node_vals = node_vals_for_iter(G, colors_at_iter)

    fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0d0d0d')
    ax.set_facecolor('#1a1a2e')
    nx.draw(
        G, pos, with_labels=True,
        node_color=node_vals,
        cmap=plt.cm.plasma,
        node_size=500, font_size=9,
        font_color='white',
        edge_color='#8888aa',
        ax=ax
    )
    converged = (iter_idx == num_iters - 1)
    title = f'Iteration {iter_idx}' + (' (converged)' if converged else '')
    ax.set_title(title, color='#aaaacc', fontsize=10)
    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def render_per_iteration_pairs(graphs, histories, k):
    """
    Returns a list of dicts, one per iteration step:
      { 'iteration': i, 'images': [b64_g1, b64_g2] }   (compare mode)
      { 'iteration': i, 'images': [b64_g1] }             (single mode)

    Both graphs are rendered at the same iteration side by side.
    The number of steps is the max of the two histories.
    """
    # Pre-compute fixed spring layouts per graph so positions don't jump
    positions = [
        nx.spring_layout(G, seed=42, k=0.5)
        for G in graphs
    ]

    num_iters = max(len(h) for h in histories)
    steps = []

    for i in range(num_iters):
        step_images = []
        for g_idx, (G, history, pos) in enumerate(zip(graphs, histories, positions)):
            # If this graph converged earlier, keep showing its final state
            iter_colors = history[min(i, len(history) - 1)]
            b64 = render_single_iteration(
                G, iter_colors, pos, k, i, num_iters
            )
            step_images.append(b64)
        steps.append({'iteration': i, 'images': step_images})

    return steps


def render_iterations_to_base64(graphs, histories, k, title_prefix="Graph"):
    """
    Legacy single-image-per-graph renderer — kept for single-graph visualise mode
    so each iteration is still shown in a neat grid for one graph.
    """
    images = []
    for g_index, (G, history) in enumerate(zip(graphs, histories)):
        nodes = list(G.nodes())
        pos = nx.spring_layout(G, seed=42, k=0.5)

        num_iters = len(history)
        cols = min(3, num_iters)
        rows = (num_iters + cols - 1) // cols

        fig = plt.figure(figsize=(5 * cols, 4 * rows), facecolor='#0d0d0d')
        fig.suptitle(
            f"{title_prefix} {g_index+1} | k={k} WL Iterations",
            fontsize=14, color='white', fontweight='bold'
        )

        for i, colors in enumerate(history):
            node_vals = node_vals_for_iter(G, colors)
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_facecolor('#1a1a2e')
            nx.draw(
                G, pos, with_labels=True,
                node_color=node_vals,
                cmap=plt.cm.plasma,
                node_size=500, font_size=9,
                font_color='white',
                edge_color='#8888aa',
                ax=ax
            )
            ax.set_title(f"Iteration {i}", color='#aaaacc', fontsize=10)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
        buf.seek(0)
        images.append(base64.b64encode(buf.read()).decode('utf-8'))
        plt.close(fig)

    return images


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/api/visualize', methods=['POST'])
def visualize():
    try:
        k = int(request.form.get('k', 2))
        file = request.files.get('file1')
        if not file:
            return jsonify({'error': 'No file provided.'}), 400

        text = file.read().decode('utf-8')
        n_declared, graphs = smart_parse(text)
        if not graphs:
            return jsonify({'error': 'No valid edges found in file.'}), 400

        G = graphs[0]
        final_colors, history = k_wl_refinement(G, k)

        # Single graph: use legacy grid view (all iterations at once)
        images = render_iterations_to_base64([G], [history], k, title_prefix="Single Graph")

        return jsonify({
            'mode': 'visualize',
            'k': k,
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'num_iterations': len(history) - 1,
            'images': images
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/isomorphism', methods=['POST'])
def isomorphism():
    try:
        k = int(request.form.get('k', 2))
        file1 = request.files.get('file1')
        file2 = request.files.get('file2')

        if not file1:
            return jsonify({'error': 'At least one file is required.'}), 400

        text1 = file1.read().decode('utf-8')
        n1, g_list1 = smart_parse(text1)
        graphs_to_compare = [g_list1[0]]

        if file2:
            text2 = file2.read().decode('utf-8')
            n2, g_list2 = smart_parse(text2)
            graphs_to_compare.append(g_list2[0])
        else:
            if len(g_list1) >= 2:
                graphs_to_compare.append(g_list1[1])
            else:
                return jsonify({
                    'error': 'Only one graph found. Provide a second file or include two edge-rows.'
                }), 400

        histories = []
        sigs = []
        for G in graphs_to_compare:
            final_colors, history = k_wl_refinement(G, k)
            histories.append(history)
            sigs.append(graph_signature(final_colors))

        wl_isomorphic = (sigs[0] == sigs[1])
        exact_isomorphic = nx.is_isomorphic(graphs_to_compare[0], graphs_to_compare[1])

        non_iso_analysis = None
        if not wl_isomorphic or not exact_isomorphic:
            non_iso_analysis = analyse_non_isomorphism(
                graphs_to_compare[0], graphs_to_compare[1], histories
            )

        # Compare mode: side-by-side per-iteration steps
        iteration_steps = render_per_iteration_pairs(graphs_to_compare, histories, k)

        return jsonify({
            'mode': 'isomorphism',
            'k': k,
            'wl_isomorphic': wl_isomorphic,
            'exact_isomorphic': bool(exact_isomorphic),
            'graph1': {
                'num_nodes': graphs_to_compare[0].number_of_nodes(),
                'num_edges': graphs_to_compare[0].number_of_edges(),
                'num_iterations': len(histories[0]) - 1
            },
            'graph2': {
                'num_nodes': graphs_to_compare[1].number_of_nodes(),
                'num_edges': graphs_to_compare[1].number_of_edges(),
                'num_iterations': len(histories[1]) - 1
            },
            'non_iso_analysis': non_iso_analysis,
            'iteration_steps': iteration_steps   # NEW — replaces flat 'images' list
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
