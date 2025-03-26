import heapq
import pandas as pd
import datetime as dt
import datetime
import time as t
import geopy.distance
import networkx as nx


def search(arr, low, high, x):
    if high >= low:
        mid = (high + low) // 2
        if arr[mid].leave_time == x:
            return mid
        elif arr[mid].leave_time > x:
            return search(arr, low, mid - 1, x)
        else:
            return search(arr, mid + 1, high, x)
    else:
        if low >= len(arr):
            return -1
        if arr[low].leave_time > x:
            return low
        else:
            return -1


def time_diff(end_time, start_time):
    return (end_time.hour * 60 + end_time.minute) - (start_time.hour * 60 + start_time.minute)


class Edge:
    def __init__(self, name, arrive_time, leave_time, start_x, start_y, end_x, end_y):
        self.name = name
        self.arrive_time = arrive_time
        self.leave_time = leave_time
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y


class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def min_time_cost(self, start_node, end_node, time):
        possible_edges = self.edges[start_node, end_node]
        edge_found_index = search(possible_edges, 0, len(possible_edges)-1, time)
        if edge_found_index != -1:
            edge_found = possible_edges[edge_found_index]
            if edge_found.leave_time >= time:
                return time_diff(edge_found.arrive_time, time), (edge_found.name, edge_found.leave_time, edge_found.arrive_time)
        return None

    def min_transfer_cost(self, start_node, end_node, time, cur_line):
        possible_edges = self.edges[start_node, end_node]
        edge_found_index = search(possible_edges, 0, len(possible_edges) - 1, time)
        if edge_found_index != -1:
            edge_found = possible_edges[edge_found_index]
            basic_cost = time_diff(edge_found.arrive_time, time)
            if edge_found.name != cur_line:
                basic_cost += 600
            return basic_cost, (edge_found.name, edge_found.leave_time, edge_found.arrive_time)
        return None


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self) -> bool:
        return not self.elements

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


def fix_invalid_time(time_str):
    """Fix invalid time values such as '24:01:00' by rolling over to '00:01:00'."""
    try:
        h, m, s = map(int, time_str.split(':'))
        if h >= 24:
            h = h % 24  # Convert "24:01:00" to "00:01:00"
        return f"{h:02}:{m:02}:{s:02}"
    except ValueError:
        return "00:00:00"  # Default for completely incorrect values

# def load_data(file):
#     df = pd.read_csv("connection_graph.csv", index_col=0, low_memory=False)
#
#     # Fix invalid times
#     df["departure_time"] = df["departure_time"].apply(fix_invalid_time)
#     df["arrival_time"] = df["arrival_time"].apply(fix_invalid_time)
#
#     # Convert to datetime.time format
#     df["departure_time"] = pd.to_datetime(df["departure_time"], format="%H:%M:%S").dt.time
#     df["arrival_time"] = pd.to_datetime(df["arrival_time"], format="%H:%M:%S").dt.time
#
#     graph = Graph()
#
#     for _, row in df.iterrows():
#         start_stop = str(row[6]).lower()
#         end_stop = str(row[7]).lower()
#
#         if start_stop in graph.nodes.keys():
#             graph.nodes[start_stop].append(end_stop)
#         else:
#             graph.nodes[start_stop] = [end_stop]
#
#         edge = Edge(
#             row["line"],
#             row["arrival_time"],
#             row["departure_time"],
#             row["start_stop_lat"],
#             row["start_stop_lon"],
#             row["end_stop_lat"],
#             row["end_stop_lon"]
#         )
#
#         if (start_stop, end_stop) in graph.edges.keys():
#             graph.edges[(start_stop, end_stop)].append(edge)
#         else:
#             graph.edges[(start_stop, end_stop)] = [edge]
#
#     for row in graph.edges.values():
#         row.sort(key=lambda x: x.leave_time)
#
#     #debug
#     print("Start stops:", df['start_stop'].unique()[:10])  # First 10 unique values
#     print("End stops:", df['end_stop'].unique()[:10])  # First 10 unique values
#
#     # print("Available nodes:", graph.nodes.keys())
#     # print("Edges:", graph.edges.keys())
#     if ("broniewskiego", "pola") in graph.edges:
#         print("Edge exists!")
#     else:
#         print("Edge not found!")
#     print("Nodes in graph:", list(graph.nodes.items())[:5])  # Print first 5 nodes
#
#     return graph


def load_data(file_path):
    # Load CSV with all columns as strings to avoid mixed types issues
    df = pd.read_csv(file_path, dtype=str)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Debug: Print available columns
    print("Columns in CSV:", df.columns.tolist())

    # Ensure the required columns exist
    required_columns = {"start_stop", "end_stop", "start_stop_lat", "start_stop_lon", "end_stop_lat", "end_stop_lon"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns in CSV: {required_columns - set(df.columns)}")

    # Create a graph with stop names as nodes
    graph = nx.DiGraph()

    # Add nodes using stop names
    for _, row in df.iterrows():
        start_stop = row["start_stop"].strip()
        end_stop = row["end_stop"].strip()

        graph.add_node(start_stop)
        graph.add_node(end_stop)

    # Add edges with weights (travel times if available)
    for _, row in df.iterrows():
        start_stop = row["start_stop"].strip()
        end_stop = row["end_stop"].strip()

        # Try to add a weight based on departure and arrival time (optional)
        try:
            travel_time = pd.to_datetime(row["arrival_time"]) - pd.to_datetime(row["departure_time"])
            travel_time_minutes = travel_time.total_seconds() / 60  # Convert to minutes
        except:
            travel_time_minutes = None  # If time parsing fails, set to None

        graph.add_edge(start_stop, end_stop, weight=travel_time_minutes)

    return graph


def get_data():
    while True:
        start_stop = input("Przystanek początkowy: ")
        end_stop = input("Przystanek końcowy: ")
        time = input("Godzina odjazdu: ")
        option = input("t - minimalizacja czasu dojazdu, p - minimalizacja liczby przesiadek")

        return start_stop, end_stop, option, datetime.datetime.strptime(time, "%H:%M").time()


def display_results(path, lines):
    if not lines:  # ✅ Prevents IndexError
        print("⚠️ No valid route found. Cannot display results.")
        return

    # Get current stop name
    cur_stop = lines[0][0]
    print("Linia: " + str(cur_stop) + " Przystanek: " + str(path[0]) +
          " Odjazd: " + str(lines[0][1]))

    i = 2
    changes = 0

    for elem in lines[2:]:
        if elem[0] != cur_stop:
            changes += 1
            cur_stop = elem[0]
            print("Przesiadka!")
            print("Linia: " + str(cur_stop) + " odjazd: " +
                  str(elem[1]) + " przystanek: " + str(path[i]))
        i += 1

    print("Dojedziesz do celu o godzinie: " +
          str(lines[-1][2]) + " przystanek: " + str(path[-1]))
    time = time_diff(lines[-1][2], lines[0][1])
    print("Długość podróży: " + str(time) + "min")
    print("Ilość przesiadek: " + str(changes))
    print("")


def heurisitic(a, b, g):
    edge = g.edges[(a, b)][0]
    return geopy.distance.distance((edge.start_x, edge.start_y), (edge.end_x, edge.end_y)).km * 15/60


def astar_search(graph, start, goal, time, opt_time=True):
    front = PriorityQueue()
    front.put(start, 0)
    came_from = {start: None}
    cost_so_far: dict[str, float] = {start: 0}
    time_so_far: dict[str, dt.time] = {start: time}
    line_so_far: dict[str, str] = {start: ""}

    while not front.empty():
        cur_stop = front.get()
        if cur_stop == goal:
            break
        try:
            graph.nodes[cur_stop]
        except KeyError:
            continue

        for neighbour in graph.nodes[cur_stop]:
            if opt_time:
                cost = graph.min_time_cost(cur_stop, neighbour, time_so_far[cur_stop])
            else:
                cost = graph.min_transfer_cost(cur_stop, neighbour, time_so_far[cur_stop], line_so_far[cur_stop])
            if cost is None:
                continue

            new_cost = cost_so_far[cur_stop] + cost[0]
            if neighbour not in cost_so_far or new_cost < cost_so_far[neighbour]:
                cost_so_far[neighbour] = new_cost
                priority = new_cost + heurisitic(cur_stop, neighbour, graph)
                front.put(neighbour, priority)
                came_from[neighbour] = cur_stop, cost[1]
                time_so_far[neighbour] = cost[1][2]
                line_so_far[neighbour] = cost[1][0]

    return came_from, cost_so_far


def dijkstra_search(graph, start, goal, time):
    front = PriorityQueue()
    front.put(start, 0)
    came_from = {start: None}
    cost_so_far: dict[str, float] = {start: 0}
    time_so_far: dict[str, dt.time] = {start: time}

    while not front.empty():
        cur_stop = front.get()
        if cur_stop == goal:
            break
        try:
            graph.nodes[cur_stop]
        except KeyError:
            continue

        for neighbour in graph.nodes[cur_stop]:
            cost = graph.min_time_cost(cur_stop, neighbour, time_so_far[cur_stop])
            if cost is None:
                continue
            new_cost = cost_so_far[cur_stop] + cost[0]
            if neighbour not in cost_so_far or new_cost < cost_so_far[neighbour]:
                cost_so_far[neighbour] = new_cost
                priority = new_cost
                front.put(neighbour, priority)
                came_from[neighbour] = cur_stop, cost[1]
                time_so_far[neighbour] = cost[1][2]

    return came_from, cost_so_far




def create_path(came_from, start, goal):
    if goal not in came_from:
        print(f"⚠️ Goal '{goal}' is not in came_from. No valid path found.")
        return [], []  # Return empty lists to avoid unpacking error

    current = goal
    path = []
    lines = []

    while current != start:
        path.append(current)
        current = came_from[current][0]
        if current != start:
            lines.append(came_from[current][1])

    path.append(start)
    path.reverse()
    lines.reverse()
    lines.append(came_from[goal][1])

    return path, lines


def astar(graph, start, goal, time, opt_time):
    start_time = t.time()
    came_from, cost = astar_search(graph, start, goal, time, opt_time)
    end_time = t.time()
    path, lines = create_path(came_from, start, goal)
    display_results(path, lines)
    print("Czas: ", (end_time - start_time))


def dijkstra(graph, start, goal, time):
    start_time = t.time()
    came_from, cost = dijkstra_search(graph, start, goal, time)
    end_time = t.time()
    path, lines = create_path(came_from, start, goal)
    display_results(path, lines)
    print("Czas: ", (end_time - start_time))


if __name__ == "__main__":
    graph = load_data('connection_graph.csv')
    # dijkstra(graph, "kwiska", "most grunwaldzki", datetime.time(7, 52))
    dijkstra(graph, "broniewskiego", "pola", datetime.time(7, 52))

    # df = pd.read_csv("connection_graph.csv", index_col=0, low_memory=False)
    # print(df.columns)  # Check actual column names
    # print(df.head())  # Check how data is structured
