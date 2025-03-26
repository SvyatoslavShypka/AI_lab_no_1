import pandas as pd
import heapq
import sys
from collections import defaultdict
from datetime import datetime, timedelta


def load_data(filename):
    """Load CSV file into a Pandas DataFrame with explicit data types."""
    return pd.read_csv(filename, dtype={"line": str, "company": str}, low_memory=False)

def parse_time(time_str):
    """Convert time string HH:MM:SS into minutes."""
    h, m, s = map(int, time_str.split(':'))
    return h * 60 + m  # Ignore seconds for simplicity


def return_time(minutes):
    """Convert minutes into time string HH:MM:SS"""
    h_int = minutes // 60
    m_int = minutes % 60
    h = str(h_int) if h_int >= 10 else '0' + str(h_int)
    m = str(m_int) if m_int >= 10 else '0' + str(m_int)
    return h + ':' + m + ':00'  # Ignore seconds for simplicity


def build_graph(df):
    """Build a graph representation from the DataFrame."""
    graph = defaultdict(list)
    for _, row in df.iterrows():
        start, end = row['start_stop'], row['end_stop']
        departure, arrival = parse_time(row['departure_time']), parse_time(row['arrival_time'])
        line, company = row['line'], row['company']
        graph[start].append((end, departure, arrival, line, company))
    return graph


def dijkstra(graph, start, end, arrival_time):
    """Dijkstra's algorithm to find the shortest path based on travel time."""
    start_time = datetime.now()
    pq = [(arrival_time, start, [])]  # (current_time, stop, path)
    visited = {}

    while pq:
        current_time, current_stop, path = heapq.heappop(pq)

        if current_stop == end:
            end_time = datetime.now()
            sys.stderr.write(f"Cost function value: {current_time}\n")
            sys.stderr.write(f"Computation time: {end_time - start_time}\n")
            return path  # Solution found

        if current_stop in visited and visited[current_stop] <= current_time:
            continue
        visited[current_stop] = current_time

        for neighbor, dep, arr, line, company in graph[current_stop]:
            if dep >= current_time:
                new_path = path + [(line, current_stop, dep, neighbor, arr)]
                heapq.heappush(pq, (arr, neighbor, new_path))

    return None  # No path found


def a_star(graph, start, end, arrival_time, heuristic):
    """A* algorithm to find the optimal path based on a heuristic."""
    start_time = datetime.now()
    pq = [(arrival_time, 0, start, [])]  # (cost, transfers, stop, path)
    visited = {}

    while pq:
        cost, transfers, current_stop, path = heapq.heappop(pq)

        if current_stop == end:
            end_time = datetime.now()
            sys.stderr.write(f"Cost function value: {cost}\n")
            sys.stderr.write(f"Computation time: {end_time - start_time}\n")
            return path  # Solution found

        if current_stop in visited and visited[current_stop] <= cost:
            continue
        visited[current_stop] = cost

        for neighbor, dep, arr, line, company in graph[current_stop]:
            if dep >= cost:
                new_transfers = transfers + (1 if path and path[-1][0] != line else 0)
                new_path = path + [(line, current_stop, dep, neighbor, arr)]
                priority = arr + heuristic(neighbor, end)
                heapq.heappush(pq, (priority, new_transfers, neighbor, new_path))

    return None  # No path found


def heuristic_dummy(stop, end):
    """Heuristic function (for testing purposes, returns 0)."""
    return 0


def find_route(graph, start, end, criterion, arrival_time):

    arrival_time_parsed = parse_time(arrival_time)

    if criterion == 't':
        path = a_star(graph, start, end, arrival_time_parsed, heuristic_dummy)
    elif criterion == 'p':
        path = a_star(graph, start, end, arrival_time_parsed, lambda x, y: 0)
    else:
        path = dijkstra(graph, start, end, arrival_time_parsed)

    if path:
        for line, start_stop, dep, end_stop, arr in path:
            print(f"{line}: {start_stop} ({return_time(dep)}) -> {end_stop} ({return_time(arr)})")
    else:
        print("No route found.")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print(return_time(485))
    # find_route('DANE - lista 1.csv', 'Pola', 'Berenta', 't', '08:00:00')
    df = load_data('DANE - lista 1.csv')
    graph = build_graph(df)
    find_route(graph,'Pola', 'Szczepin', 't', '08:00:00')
    # find_route(graph,'Pola', 'Szczepin', 'p', '08:00:00')
