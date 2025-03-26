import pandas as pd
import geopy.distance
import heapq
from datetime import datetime, timedelta


class Edge:
    def __init__(self, start, end, leave_time, arrival_time, line, start_stop_lat, start_stop_lon, end_stop_lat, end_stop_lon):
        self.start = start
        self.end = end
        self.leave_time = leave_time
        self.arrival_time = arrival_time
        self.line = line
        self.start_x = start_stop_lat
        self.start_y = start_stop_lon
        self.end_x = end_stop_lat
        self.end_y = end_stop_lon


def fix_invalid_time(time_str):
    """Naprawia błędne formaty godzin (np. 24:01:00 → 00:01:00)."""
    try:
        hours, minutes, seconds = map(int, time_str.split(':'))
        if hours >= 24:
            hours -= 24
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    except ValueError:
        return "00:00:00"  # Domyślna wartość w przypadku błędu


def parse_time(time_str):
    """Konwertuje ciąg znaków na obiekt datetime.time."""
    try:
        time_str = fix_invalid_time(time_str)
        return datetime.strptime(time_str, "%H:%M:%S").time()
    except ValueError:
        return None


class Graph:
    def __init__(self):
        self.edges = {}

    def add_edge(self, edge):
        if (edge.start, edge.end) not in self.edges:
            self.edges[(edge.start, edge.end)] = []
        self.edges[(edge.start, edge.end)].append(edge)
        self.edges[(edge.start, edge.end)].sort(key=lambda e: e.leave_time)  # Sortujemy po czasie odjazdu

    def get_neighbors(self, node):
        node = node.lower()  # Zmieniamy na małe litery, by porównanie było niezależne od wielkości liter
        neighbors = []
        for (start, end), edges in self.edges.items():
            if start.lower() == node:  # Zmieniamy na małe litery dla porównania
                neighbors.extend(edges)
        return neighbors

def load_data(filename):
    # Wczytujemy plik CSV z odpowiednim nagłówkiem i delimiterem jako przecinek
    df = pd.read_csv(filename, delimiter=',', header=0, encoding='utf-8', low_memory=False)  # Ignorujemy ostrzeżenie DtypeWarning

    # Usuwamy nadmiarowe spacje w nazwach kolumn
    df.columns = df.columns.str.strip()

    # Usuwamy kolumnę 'Unnamed: 0', która jest niepotrzebna
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    if 'departure_time' not in df.columns:
        print("Brak kolumny 'departure_time' w pliku CSV.")
        return None  # Zwróć None lub odpowiednią wartość, jeśli kolumna nie istnieje

    # Przekształcamy dane w odpowiedni format czasu
    df["departure_time"] = df["departure_time"].apply(parse_time)
    df["arrival_time"] = df["arrival_time"].apply(parse_time)

    # Usuwamy wiersze z brakującymi wartościami w kolumnach 'departure_time' i 'arrival_time'
    df.dropna(subset=["departure_time", "arrival_time"], inplace=True)

    # Tworzymy graf
    graph = Graph()
    for _, row in df.iterrows():
        edge = Edge(row["start_stop"], row["end_stop"], row["departure_time"], row["arrival_time"],
                    row["line"], row["start_stop_lat"], row["start_stop_lon"], row["end_stop_lat"], row["end_stop_lon"])
        graph.add_edge(edge)

    return graph


def heuristic(a, b, graph):
    edges = graph.edges.get((a, b), [])
    if not edges:
        return float('inf')
    avg_distance = sum(geopy.distance.distance((e.start_x, e.start_y), (e.end_x, e.end_y)).km for e in edges) / len(
        edges)
    return avg_distance * 15 / 60  # Przeliczamy na czas podróży


def astar_search(graph, start, goal, start_time):
    open_set = []
    heapq.heappush(open_set, (0, start.lower(), start_time, []))  # Zamiana startu na małe litery
    came_from = {}
    cost_so_far = {start.lower(): 0}  # Zamiana startu na małe litery

    while open_set:
        _, current, current_time, path = heapq.heappop(open_set)

        if current == goal.lower():  # Zamiana celu na małe litery
            return path

        for edge in graph.get_neighbors(current):
            if edge.leave_time >= current_time:
                new_cost = cost_so_far[current] + (
                            datetime.combine(datetime.today(), edge.arrival_time) - datetime.combine(datetime.today(),
                                                                                                     edge.leave_time)).seconds / 60
                if edge.end.lower() not in cost_so_far or new_cost < cost_so_far[edge.end.lower()]:
                    cost_so_far[edge.end.lower()] = new_cost
                    priority = new_cost + heuristic(edge.end, goal, graph)
                    heapq.heappush(open_set, (
                    priority, edge.end.lower(), edge.arrival_time, path + [(edge.line, edge.start, edge.leave_time)]))

    return []


def display_results(path):
    if not path:
        print("Brak trasy do celu.")
        return

    for line, stop, time in path:
        print(f"Linia: {line}, Przystanek: {stop}, Odjazd: {time}")

def print_graph_edges(graph):
    # Iterujemy po wszystkich krawędziach w grafie
    for (start, end), edges in graph.edges.items():
        print(f"From {start} to {end}:")
        for edge in edges:
            # Wydrukuj szczegóły każdej krawędzi
            print(f"  Line: {edge.line}, start_stop: {edge.start}, end_stop: {edge.end}, Departure: {edge.leave_time}, Arrival: {edge.arrival_time}, "
                  f"Start Coordinates: ({edge.start_x}, {edge.start_y}), "
                  f"End Coordinates: ({edge.end_x}, {edge.end_y})")
        print("----------")  # Oddzielamy różne połączenia


if __name__ == "__main__":
    graph = load_data('connection_graph.csv')
    # graph = load_data('testowy.csv')
    # print_graph_edges(graph)

    start_stop = "Kwiska"
    goal_stop = "Pl. Grunwaldzki"
    start_time = parse_time("08:00:00")
    # start_time = parse_time("20:58:00")
    path = astar_search(graph, start_stop, goal_stop, start_time)
    display_results(path)
