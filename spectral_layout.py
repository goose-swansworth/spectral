import numpy as np
import matplotlib.pyplot as plt
import matplotx
from numpy.linalg import eig, norm, qr
from random import randint, uniform
from itertools import combinations
from spectral import spectral
from spectralbarycenter import spectral_barycenter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.style.use(matplotx.styles.ayu["light"])

def turing_machine_to_undirected_graph(instructions):
    nodes = set()
    # add all states into node set
    for i, line in enumerate(instructions.splitlines()):
        if i == 0:
            nodes.update(line.split(" "))
        else:
            tail, _, _, _, head = line.split(" ")
            nodes.update({tail, head})
    # sort alphabetically into a dict
    node_index = {label: i for i, label in enumerate(sorted(nodes))}
    # create the adjacency matrix
    n = len(nodes)
    A = np.zeros((n, n))
    for line in instructions.splitlines()[1:]:
        tail, _, _, _, head = line.split(" ")
        if tail != head:
            A[node_index[tail], node_index[head]] = 1
            A[node_index[head], node_index[tail]] = 1
    return A
    
    
def has_eq_coords(v1, v2):
    for i, coord1 in enumerate(zip(v1, v2)):
        for j, coord2 in enumerate(zip(v1, v2)):
            if i != j and norm(np.array(coord1) - np.array(coord2)) < 10**-3:
                return True
    return False
    
def random_graph(n, chance):
    A = np.zeros((n, n))
    for i in range(n - 1):
         for j in range(i + 1, n):
            if uniform(0, 1) < chance:
                A[i, j] = 1
            else:
                A[i, j] = 0
    return A + np.transpose(A)

def laplacian(A):
    D = np.diag(np.sum(A, 0))
    return D - A
    
def draw_edges(axes, A, v2, v3, v4=None, color="gray", lineweight=1):
    n = len(A)
    for i in range(n):
        for j in range(n):
            if A[i, j] != 0:
                if v4 is not None:
                    axes.plot((v2[i], v2[j]), (v3[i], v3[j]), (v4[i], v4[j]), color=color, linewidth=lineweight)
                else:
                    axes.plot((v2[i], v2[j]), (v3[i], v3[j]), color=color, linewidth=lineweight)
    

def draw_graph(A, R, filename="", R3=False, both=False, node_color="blue", edge_color="gray", annonate=True, lineweight=1, show=True):
    fig = None
    axes1 = None
    axes2 = None
    fsize = 18
    if both:
        fig = plt.figure(figsize=(11, 8.5))    
        axes1 = fig.add_subplot(121, facecolor="#ffffff")
        axes1.set_aspect("equal")
        axes1.set_xlabel("$x$", fontsize=fsize)
        axes1.set_ylabel("$y$", fontsize=fsize)
        axes1.grid(True)
        axes2 = fig.add_subplot(122, projection="3d", facecolor="#ffffff")
        axes2.set_xlabel("$x$", fontsize=fsize)
        axes2.set_ylabel("$y$", fontsize=fsize)
        axes2.set_zlabel("$z$", fontsize=fsize)
    if R3 and not both:
        axes2 = plt.figure(figsize=(7, 7)).add_subplot(projection="3d", facecolor="#ffffff")
        axes2.set_xlabel("$x$", fontsize=fsize)
        axes2.set_ylabel("$y$", fontsize=fsize)
        axes2.set_zlabel("$z$", fontsize=fsize)
    if not R3:
        axes1 = plt.axes(facecolor="#ffffff")
        axes1.set_aspect("equal")
        axes1.set_xlabel("$x$", fontsize=fsize)
        axes1.set_ylabel("$y$", fontsize=fsize)
        axes1.grid(True)
    # Plot nodes
    if not R3 or both:
        draw_edges(axes1, A, R[:, 0], R[:, 1], color=edge_color, lineweight=lineweight)
        #axes1.plot(R[:, 0], R[:, 1], marker="o", color=node_color, linestyle="")
    #if R3:
        #axes2.plot(R[:, 0], R[:, 1], R[:, 2], marker="o", color=node_color, linestyle="")
    # Plot edges
    if annonate:
        if both or not R3:
            for i, (x, y) in enumerate(zip(R[:, 0], R[:, 1])):
                axes1.annotate("$v_{" + f"{i+1}" + "}$", (x + 0.015, y+0.015), fontsize=13)
    if R3:
        draw_edges(axes2, A, R[:, 0], R[:, 1], R[:, 2], edge_color)
    
    plt.savefig(f"{filename}.png", dpi=400, facecolor="#ffffff")
    if show:
        plt.show()
    

def barycenter(A, v0, P):
    V = np.zeros((len(A), 2))
    degree = np.sum(A, 0)
    free = set(range(len(A))) - set(v0)
    j = 0
    for i in v0:
        V[i] = P[j]
        j += 1
    iters = 0
    tol = 10**-4
    while iters == 0 or norm(V_last - V) < tol and iters < 5:
        V_last = V
        for v in free:
            V[v] = (1 / degree[v]) * sum(V[u] for u in range(len(A)) if A[u, v] == 1)
        iters += 1
    return V
            
def is_sym(A):
    n = len(A)
    for i in range(n):
        for j in range(n):
            if A[i, j] != A[j, i]:
                print(f"({i+1}, {j+1}) = {A[i, j]} | ({j+1}, {i+1})={A[j, i]}")
                return False
    return True

def load_adj_list():
    A = np.zeros((16, 16), dtype=int)
    with open("planar.txt") as infile:
        lines = infile.read().splitlines()
        for line in lines:
            i, j = map(int, line.split(","))
            A[i-1, j-1] = 1
            A[j-1, i-1] = 1
    return A
    
    
def adj_to_latex(A):
    out = r"\begin{bmatrix}" + "\n"
    for row in A:
        for i, a in enumerate(row):
            if i != len(row) - 1:
                out += f"{a}&"
            else:
                out += str(a)
        out += r"\\" + "\n"
    out += r"\end{bmatrix}"
    return out
            
def draw_polyhedra(R, filename="", color="blue", show=False):
    x, y, z = R[:, 0], R[:, 1], R[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    cube = np.array([[x[i], y[i], z[i]] for i in range(len(x))])
    hull = ConvexHull(cube)
    # draw the polygons of the convex hull
    for s in hull.simplices:
        tri = Poly3DCollection([cube[s]], facecolors=[color])
        #tri.set_color()
        tri.set_alpha(0.5)
        ax.add_collection3d(tri)
    # draw the vertices
    ax.scatter(cube[:, 0], cube[:, 1], cube[:, 2], marker='o', color='gray')
    draw_edges(ax, A, x, y, z)
    plt.tight_layout()
    ax.set_xlim((min(x), max(x)))
    ax.set_ylim((min(y), max(y)))
    ax.set_zlim((min(z), max(z)))
    if show:
        plt.show()
    plt.savefig(f"{filename}.png", dpi=400, facecolor="#ffffff")
        





# with open("dodecahedron.txt") as infile:
#     lines = infile.read().splitlines()
#     A = np.array([list(map(int, line)) for line in lines])
#     R = spectral_barycenter(A, 3)
#     draw_polyhedra(R, filename="D12", color="darkorange")
    
# with open("ico.txt") as infile:
#     lines = infile.read().splitlines()
#     A = np.array([list(map(int, line)) for line in lines])
#     R = spectral_barycenter(A, 2)
#     #draw_polyhedra(R, filename="ico_bary", color="tomato")
#     draw_graph(A, R, filename="icobary", node_color="tomato")
#     #print(adj_to_latex(A))

# with open("delta.txt") as infile:
#     lines = infile.read().splitlines()
#     A = np.array([list(map(int, line)) for line in lines])
#     R = spectral_barycenter(A, 3)
#     draw_polyhedra(R, filename="D10", color="royalblue")
#     #draw_graph(A, R, R3=True)

# with open("oct.txt") as infile:
#     lines = infile.read().splitlines()
#     A = np.array([list(map(int, line)) for line in lines])
#     R = spectral_barycenter(A, #     draw_polyhedra(R, filename="D8", color="indigo")
#     #draw_graph(A, R, R3=True)

# with open("cube.txt") as infile:
#     lines = infile.read().splitlines()
#     A = np.array([list(map(int, line)) for line in lines])
#     print(is_sym(A))
#     R = spectral_barycenter(A, 3)
#     draw_polyhedra(R, filename="D6", color="seagreen")
#     #draw_graph(A, R, R3=True)




# with open("doubled-higmansims.am.csv") as infile:
#     lines = infile.read().splitlines()
#     A = np.array([list(map(int, line.split(","))) for line in lines])
#     print(np.sum(A)/2)
#     R = spectral_barycenter(A, 2, epsilon=10**-7, max_iters=10000)
#     draw_graph(A, R, filename="big3", lineweight=0.1, annonate=False, node_color="teal")

    


# with open("planar.txt") as infile:
#     A = load_adj_list()
#     # P = [[-1, -1], [1, 1], [1, -1], [-1, 1]]
#     # barycenter(A, (3, 1, 2, 0), P)
#     print(adj_to_latex(A))
    
    
        

# with open("example.txt") as infile:
#     lines = infile.read().splitlines()
#     A = np.array([list(map(int, line)) for line in lines])
#     # spetral_draw(A, "example", weighted=True, node_color="steelblue")
#     R = spectral_barycenter(A, 2)
#     print(R)
#     draw_graph(A, R)

# with open("bin_palen.txt") as infile:
#     A = turing_machine_to_undirected_graph(infile.read())
# #     spectral(A, filename="spectral_bin_plaen", node_color="steelblue")
# #     spectral_barycenter(A, 2, filename="bary_bin_palen", node_color="steelblue")
    
    
# with open("succ.txt") as infile:
#     A = turing_machine_to_undirected_graph(infile.read())
#     print(eig(laplacian(A)))
#     R = spectral(A, 2)
#     print(A)
#     draw_graph(A, R, filename="succ_graph", node_color="steelblue")
    

with open("crack.txt") as infile:
    lines = infile.read().splitlines()
    m, _, _ = map(int, lines[0].split(" "))
    A = np.zeros((m, m))
    for line in lines[1:]:
        i, j = map(int, line.split(" "))
        A[i-1, j-1] = 1
        A[j-1, i-1] = 1
    R = spectral_barycenter(A, 2, max_iters=1000)
    draw_graph(A, R, annonate=False, lineweight=0.01)
