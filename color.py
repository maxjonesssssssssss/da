def is_safe(node, color, assignment, graph):
    for neighbor in graph[node]:
        if assignment.get(neighbor) == color:
            return False
    return True

def backtrack(node, num_colors, assignment, graph, nodes):
    if len(assignment) == len(nodes):
        return True

    for color in range(1, num_colors + 1):
        if is_safe(node, color, assignment, graph):
            assignment[node] = color

            next_nodes = [n for n in nodes if n not in assignment]
            if not next_nodes:
                return True

            if backtrack(next_nodes[0], num_colors, assignment, graph, nodes):
                return True

            del assignment[node]

    return False

def solve_graph_coloring(graph, num_colors, nodes):
    assignment = {}
    if backtrack(nodes[0], num_colors, assignment, graph, nodes):
        return assignment
    return None

def print_solution(assignment, num_colors):
    color_names = ["", "Red", "Green", "Blue", "Yellow",
                   "Orange", "Purple", "Pink", "Cyan"]
    print("\n Graph Coloring Solution:")
    print("-" * 30)
    for node, color in sorted(assignment.items()):
        name = color_names[color] if color < len(color_names) else f"Color-{color}"
        print(f"  Node {node:>3}  →  {name} (Color {color})")
    print("-" * 30)
    print(f"  Colors used : {num_colors}")
    print(f"  Nodes colored: {len(assignment)}\n")

if __name__ == "__main__":
    print("=" * 40)
    print("  Graph Coloring — Backtracking + B&B")
    print("=" * 40)

    while True:
        try:
            n = int(input("\n Enter number of nodes: "))
            if n < 1:
                print(" Enter a positive integer.")
                continue
            break
        except ValueError:
            print(" Invalid input.")

    while True:
        try:
            e = int(input(" Enter number of edges: "))
            if e < 0:
                print(" Edges cannot be negative.")
                continue
            break
        except ValueError:
            print(" Invalid input.")

    graph = {i: [] for i in range(1, n + 1)}
    nodes = list(range(1, n + 1))

    print(f"\n Enter each edge as: u v  (nodes are 1 to {n})")
    for i in range(e):
        while True:
            try:
                u, v = map(int, input(f"  Edge {i+1}: ").split())
                if u < 1 or v < 1 or u > n or v > n:
                    print(f"  Nodes must be between 1 and {n}.")
                    continue
                if u == v:
                    print("  Self-loops not allowed.")
                    continue
                graph[u].append(v)
                graph[v].append(u)
                break
            except ValueError:
                print("  Enter two integers separated by space.")

    while True:
        try:
            k = int(input(f"\n Enter number of colors to try: "))
            if k < 1:
                print(" At least 1 color required.")
                continue
            break
        except ValueError:
            print(" Invalid input.")

    print(f"\n Solving with {k} color(s) using Backtracking + Branch & Bound...\n")
    result = solve_graph_coloring(graph, k, nodes)

    if result:
        print(" Solution FOUND!")
        print_solution(result, k)
    else:
        print(f"\n No valid coloring found with {k} color(s).")
        print(" Try increasing the number of colors.\n")
