import heapq
import pandas as pd
import datetime as dt
import datetime
import time as t
import geopy.distance


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

        # 🔧 Konwersja arrive_time i leave_time na datetime.time
        if isinstance(arrive_time, str):
            self.arrive_time = datetime.strptime(arrive_time, "%H:%M:%S").time()
        else:
            self.arrive_time = arrive_time

        if isinstance(leave_time, str):
            self.leave_time = datetime.strptime(leave_time, "%H:%M:%S").time()
        else:
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
        edge_found_index = search(possible_edges, 0, len(possible_edges) - 1, time)

        if edge_found_index != -1:
            edge_found = possible_edges[edge_found_index]
            if edge_found.leave_time >= time:
                return time_diff(edge_found.arrive_time, time), (
                edge_found.name, edge_found.leave_time, edge_found.arrive_time)

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


def safe_convert_time(value):
    try:
        return pd.to_datetime(fix_invalid_time(value), format='%H:%M:%S').time()
    except ValueError:
        print(f"⚠️ Błąd konwersji: {value}")  # Możesz usunąć, jeśli nie chcesz widzieć błędów
        return None  # Ustawia brak wartości dla błędnych danych


def fix_invalid_time(time_str):
    """Naprawia błędne formaty godzin (np. 24:01:00 → 00:01:00)."""
    try:
        hours, minutes, seconds = map(int, time_str.split(':'))
        if hours >= 24:
            hours -= 24
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    except ValueError:
        return "00:00:00"  # Domyślna wartość w przypadku błędu


def load_data(file):
    df = pd.read_csv(
        file,
        delimiter=',',
        header=0,
        encoding='utf-8',
        low_memory=False
    )

    df['departure_time'] = df['departure_time'].apply(safe_convert_time)
    df['arrival_time'] = df['arrival_time'].apply(safe_convert_time)

    graph = Graph()

    for row in df.values[1:]:
        start_stop = str(row[5]).lower()
        end_stop = str(row[6]).lower()

        if start_stop in graph.nodes.keys():
            graph.nodes[start_stop].append(end_stop)
        else:
            graph.nodes[start_stop] = [end_stop]

        edge = Edge(row[2], row[4], row[3], row[7], row[8], row[9], row[10])

        if (start_stop, end_stop) in graph.edges.keys():
            graph.edges[(start_stop, end_stop)].append(edge)
        else:
            graph.edges[(start_stop, end_stop)] = [edge]

    for row in graph.edges.values():
        row.sort(key=lambda x: x.leave_time)

    return graph


def get_data():
    while True:
        start_stop = input("Przystanek początkowy: ")
        end_stop = input("Przystanek końcowy: ")
        time = input("Godzina odjazdu: ")
        option = input("t - minimalizacja czasu dojazdu, p - minimalizacja liczby przesiadek")

        return start_stop, end_stop, option, datetime.datetime.strptime(time, "%H:%M").time()


def display_results(path, lines):
    ## get current stop name
    cur_stop = lines[0][0]
    print("Linia: " + str(cur_stop) + " Przystanek: " + str(path[0]) +
          " Odjazd: " + str(lines[0][1]))
    i = 2
    #start with number of changes: 0
    changes = 0
    ##loop through elements in lines array
    for elem in lines[2:]:
        ##when we have to change stop increment changes by 1
        if elem[0] != cur_stop:
            changes += 1
            cur_stop = elem[0]
            #print details about change to the user
            print("Przesiadka!")
            print("Linia: " + str(cur_stop) + " odjazd: " +
                  str(elem[1]) + " przystanek: " + str(path[i]))
        i += 1
    #print final info about the route
    print("Dojedziesz do celu o godzinie: " +
          str(lines[-1][2]) + " przystanek: " + str(path[-1]))
    time = time_diff(lines[-1][2], lines[0][1])
    print("Długość podrózy: " + str(time) + "min")
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
    # print(f"📌 Węzły w came_from: {list(came_from.keys())}")
    return came_from, cost_so_far


def create_path(came_from, start, goal):
    if goal not in came_from:
        # print(f"⚠️ Nie znaleziono ścieżki do: {goal}")
        # print(f"Znalezione węzły: {list(came_from.keys())}")  # Wypisuje znalezione węzły
        return []

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
    # graph = load_data('testowy.csv')
    # dijkstra(graph, "bezpieczna", "katedra", datetime.time(10, 00))
    dijkstra(graph, "kwiska", "pl. grunwaldzki", datetime.time(10, 00))