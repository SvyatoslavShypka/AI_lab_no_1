import sys
from copy import deepcopy
import pickle
import heapq
import math
from datetime import datetime, date, time
from typing import Callable
from functools import wraps
from datetime import datetime, time
import time
import pprint
from dataclasses import dataclass
import pandas as pd


@dataclass
class TimeInformation:
    """ class for keeping info about time"""
    hour: int
    minute: int
    minAfterOO: int

    def __init__(self, str: str):
        correct_time_format(str)
        self.hour, self.minute = int(str[0:2]), int(str[3:5])
        self.minAfterOO = self.hour * 60 + self.minute

    def __str__(self):
        return str(self.hour).zfill(2) + ":" + str(self.minute).zfill(2) + ":00"

    def __lt__(self, other):
        return self.minAfterOO < other.minAfterOO

    def __le__(self, other):
        return self.minAfterOO <= other.minAfterOO

    def __gt__(self, other):
        return self.minAfterOO > other.minAfterOO

    def __ge__(self, other):
        return self.minAfterOO >= other.minAfterOO


@dataclass
class VerticleInformation:
    name: str
    latitiude: float
    longitude: float


@dataclass
class EdgeInformation:
    line: str
    departure_time: TimeInformation
    arrival_time: TimeInformation
    start_stop: str
    end_stop: str


class Node:
    parent_name: str
    f: float  # calkowity koszt g+h
    g: int  # koszt przejscia od wezla poczatkowego do danego wezla
    h: float  # heurystyka

    def __init__(self, geo_info: VerticleInformation):
        self.geo_info = geo_info
        self.edges = list()  # lista krawedzi do sasiednich wezlow
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0.0
        self.parent_name = ""

    def __str__(self):
        return f"Node {self.ride_info.line} start {self.ride_info.start_stop} {self.ride_info.departure_time}" \
               f"end {self.ride_info.end_stop} {self.ride_info.arrival_time}"

    def __lt__(self, other):
        return self if self.ride_info.departure_time < other.ride_info.departure_time else other

    def set_neighbour(self, ride: EdgeInformation):
        """ creates a list of neighbours to the node"""
        self.edges.append(ride)
        return self

    def edge_time_cost(edg: EdgeInformation):
        return edg.arrival_time.minAfterOO - edg.departure_time.minAfterOO


# Funkcja do konwersji godziny 24 na 00
def correct_time_format(time_str):
    hour = int(time_str[0:2])
    if hour > 23:
        hour -= 24
        time_str = str(hour).zfill(2) + time_str[2:]
    return time_str


def stop_search(graph, stop: str):
    stop_name = "DWORZEC GŁÓWNY"  # Nazwa przystanku, dla którego chcesz zobaczyć dane
    if stop_name in graph:
        node = graph[stop_name]
        print(f"\nDane dla przystanku: {stop_name}")
        print(f" - Nazwa: {node.geo_info.name}")
        print(f" - Szerokość geograficzna: {node.geo_info.latitiude}")
        print(f" - Długość geograficzna: {node.geo_info.longitude}")
        print(f" - Liczba krawędzi (połączeń): {len(node.edges)}")

        print("\nLista połączeń:")
        for edge in node.edges:
            print(f"   Linia: {edge.line}, Odjazd: {edge.departure_time}, Przyjazd: {edge.arrival_time}, "
                  f"Do: {edge.end_stop}")
    else:
        print(f"\nPrzystanek '{stop_name}' nie został znaleziony w grafie.")


def create_graph(filename: str):
    graph: dict[str, Node]
    graph = {}

    df = pd.read_csv(filename,
                     dtype={"id": int, "company": str, "line": str, "departure_time": str, "arrival_time": str
                         , "start_stop": str, "end_stop": str, "start_stop_lat": float, "start_stop_lon": float,
                            "end_stop_lat": float, "end_stop_lon": float})
    df.sort_values(['line', 'departure_time', 'start_stop'])
    # Zastosowanie poprawki na kolumnie
    df['departure_time'] = df['departure_time'].apply(correct_time_format)
    df["arrival_time"] = df["arrival_time"].apply(correct_time_format)

    nodes_created = 0
    nodes_count = 0
    print("Creating nodes")
    for index, row in df.iterrows():

        ride_info = EdgeInformation(
            row['line'],
            TimeInformation(row['departure_time']),
            TimeInformation(row['arrival_time']),
            row['start_stop'],
            row['end_stop']
        )

        start_info = VerticleInformation(
            row['start_stop'],
            float(row['start_stop_lat']),
            float(row['start_stop_lon'])
        )

        stop_info = VerticleInformation(
            row['end_stop'],
            float(row['end_stop_lat']),
            float(row['end_stop_lon'])
        )

        if str(row['start_stop']) not in graph:
            graph[row['start_stop']] = Node(start_info)
            nodes_created += 1
            print(".", end='')

        if str(row['end_stop']) not in graph:
            graph[row['end_stop']] = Node(stop_info)
            nodes_created += 1
            print(".", end='')

        s = graph[str(row['start_stop'])]

        graph[str(row['start_stop'])] = s.set_neighbour(ride_info)  # dodawanie krawedzi sasiadow do wezla

        if nodes_created % 100 == 0 and nodes_created > nodes_count:
            print(f"{nodes_created} nodes created")
        nodes_count = nodes_created
    print(f"{nodes_created} nodes created totally")

    #debug
    stop_name = "DWORZEC GŁÓWNY"
    stop_search(graph, stop_name)

    return graph


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds', file=sys.stderr)
        return result

    return timeit_wrapper


@timeit
def astar_search(graph, start, goal, cost_fn, start_time):
    # convert start_time to TimeInformation object
    start_time = TimeInformation(start_time)

    # start node
    start_node = graph[start]
    start_node.g = 0
    start_node.h = 0
    start_node.f = 0
    graph[start] = start_node

    # priority queue
    priority_queue = []
    heapq.heappush(priority_queue, (0, start))

    visited = set()

    # chosen edge
    chosen_edge = None

    while priority_queue:
        # node with the lowest cost from the priority queue
        cost, current_name = heapq.heappop(priority_queue)
        current_node = graph[current_name]

        if current_name in visited:
            continue

        visited.add(current_name)
        # check if the current node is the goal node
        if current_name == goal:
            return graph

        if chosen_edge is None:
            tim = start_time
            prev_line = ""
        else:
            tim = chosen_edge.arrival_time
            prev_line = chosen_edge.line

        # explore the neighbors of the current node
        for next_node in neighbours(graph, current_node, tim):
            # calculate the cost of moving to the next node
            cost_fn_res, chosen_edge = cost_fn(current_node, next_node, tim, prev_line)
            new_cost = current_node.g + cost_fn_res

            # update the cost of the next node if a shorter path is found
            if next_node.geo_info.name not in visited and new_cost < graph[next_node.geo_info.name].g:
                graph[next_node.geo_info.name].g = new_cost
                graph[next_node.geo_info.name].parent_name = current_name
                # add heuristic value to the cost for A*
                priority = new_cost + h(next_node, graph[goal])
                heapq.heappush(priority_queue, (priority, next_node.geo_info.name))

    return None


# obliczanie heurystyki, euklidesowa
def h(curr: Node, next: Node):
    return math.sqrt((curr.geo_info.latitiude - next.geo_info.latitiude) ** 2 + (
        curr.geo_info.longitude - next.geo_info.longitude) ** 2)


# krawedzi wezla krórych czas wiekszy od czasu pojawienia sie na przystanku
def neighbours(graph: dict[str, Node], start: Node, time: TimeInformation):
    n_list = list()
    for edges in start.edges:
        if graph[edges.end_stop] not in n_list:
            n_list.append(graph[edges.end_stop])

    return n_list


# koszt przejazdu z jednego wezla do drugiego
def cost_fun_for_time(curr: Node, next: Node, s_time: TimeInformation, prev_line: str):
    possible_commutes = list()
    # mozliwe polaczenia z curr do next
    for edge in curr.edges:
        if edge.end_stop == next.geo_info.name:
            possible_commutes.append(edge)

    #zakładamy 1 min na przesiadkę
    przesiadka = 3
    cost = float('inf')
    ret_edge: EdgeInformation
    ret_edge = None
    # znajduje krawedz o min koszcie
    for edge in possible_commutes:
        # Jeżeli ta sama linia, nie potrzebujemy przesiadki (=0 min)
        if edge.line == '' or edge.line == prev_line or prev_line == '':
            przesiadka = 0
        time_diff = edge.departure_time.minAfterOO - s_time.minAfterOO
        if time_diff < 0:
            time_diff += 24 * 60
        if time_diff + przesiadka < cost:
            cost = time_diff + przesiadka
            ret_edge = edge
            # print(f'Start time {edge.departure_time} To {edge.end_stop} by line {edge.line} with cost {cost}')
        przesiadka = 3

    print(f'Returned {ret_edge.end_stop} with start time {ret_edge.departure_time} and stop time {ret_edge.arrival_time} by line {ret_edge.line} with cost {cost}')
    return cost, ret_edge


def cost_fun_for_switch(curr: Node, next: Node, s_time: TimeInformation, prev_line: str):
    posible_comutes = list()
    for edge in curr.edges:
        if (edge.end_stop == next.geo_info.name):
            posible_comutes.append(edge)
    cost = float('inf')
    ret_edge: EdgeInformation
    ret_edge = None
    swich_multiplyer = 1000
    for edge in posible_comutes:
        if edge.line != prev_line:
            mult = swich_multiplyer
        else:
            mult = 0

        time_diff = edge.departure_time.minAfterOO - s_time.minAfterOO
        if time_diff < 0:
            time_diff += 24 * 60
        if time_diff < cost:
            cost = time_diff + mult
            ret_edge = edge
    return cost, ret_edge


def cost_fun_for_switch_modified(curr: Node, next: Node, s_time: TimeInformation, prev_line: str):
    possible_commutes = [edge for edge in curr.edges if edge.end_stop == next.geo_info.name]

    cost = float('inf')
    ret_edge = None
    switch_multiplier = 100

    for edge in possible_commutes:
        if edge.line != prev_line:
            # uwzględnienie odległości między przystankami jako czynnika wpływającego na koszt przesiadki
            distance_cost = calculate_distance_cost(curr, next)
            # modyfikacja kosztu przesiadki na podstawie odległości między przystankami
            mult = switch_multiplier * distance_cost
        else:
            mult = 0

        time_diff = edge.departure_time.minAfterOO - s_time.minAfterOO
        if time_diff < 0:
            time_diff += 24 * 60
        if time_diff < cost:
            cost = time_diff + mult
            ret_edge = edge
    return cost, ret_edge


def calculate_distance_cost(curr: Node, next: Node):
    distance = h(curr, next)
    return distance


@timeit
def dijkstra(graph, start, goal, cost_fn, start_time):
    # convert start_time to TimeInformation object
    # start_time = TimeInformation(start_time)
    # Start node
    start_node = graph[start]
    start_node.g = 0
    graph[start] = start_node

    # priority queue
    priority_queue = []
    heapq.heappush(priority_queue, (0, start))

    visited = set()

    # chosen edge
    chosen_edge = None

    while priority_queue:
        # node with the lowest cost from the priority queue
        cost, current_name = heapq.heappop(priority_queue)
        print(f"Przetwarzam przystanek: {current_name}, koszt: {cost}")

        current_node = graph[current_name]

        if current_name in visited:
            continue

        visited.add(current_name)
        # check if the current node is the goal node
        if current_name == goal:
            return graph #Correct exit when the route was found

        if chosen_edge is None:
            tim = start_time
            prev_line = ""
        else:
            tim = chosen_edge.arrival_time
            prev_line = chosen_edge.line
        # if chosen_edge is not None and chosen_edge.arrival_time.minAfterOO > 1440:
        #     print(chosen_edge.arrival_time.minAfterOO, chosen_edge.arrival_time, chosen_edge.start_stop)
        # explore the neighbors of the current node
        all_neighbours = neighbours(graph, current_node, tim)
        for next_node in all_neighbours:
            # calculate the cost of moving to the next node
            cost_fn_res, chosen_edge = cost_fn(current_node, next_node, tim, prev_line)
            # if cost_fn_res < 0:
            #     print(f'cost_fn_res: {cost_fn_res}')
            new_cost = current_node.g + cost_fn_res

            # update the cost of the next node if a shorter path is found
            # if next_node.geo_info.name not in visited and new_cost < graph[next_node.geo_info.name].g:
            if new_cost < graph[next_node.geo_info.name].g:
                graph[next_node.geo_info.name].g = new_cost
                # print(f'{next_node.geo_info.name} g.cost= {new_cost}')
                graph[next_node.geo_info.name].parent_name = current_name
                heapq.heappush(priority_queue, (new_cost, next_node.geo_info.name))
    return None


#tworzy sciezke przechodzac wstecz po rodzicach
def read_path(graph: dict[str, Node], end: str):
    node_list = list()
    node_end = graph[end]
    while node_end.parent_name != "":
        node_list.append(node_end)
        node_end = graph[node_end.parent_name]
    node_list.append(node_end)
    node_list.reverse()
    return node_list

#znajduje krawedz do wezla end
def find_edge_to_goal(node: Node, end: str, time: TimeInformation):
    min_time_difference = float('inf')
    selected_edge = None

    for edge in node.edges:
        if edge.end_stop == end:
            time_difference = edge.departure_time.minAfterOO - time.minAfterOO
            if time_difference < 0:
                time_difference += 24 * 60

            if time_difference < min_time_difference:
                min_time_difference = time_difference
                selected_edge = edge

    return selected_edge


#przejscie po wezlach i odczytanie czasu z krawedzi
def path_with_information(node_list: list[Node], time):
    # time = TimeInformation(tim)
    last_stop = None
    num_changes = 0
    prev_line = None
    total_cost = 0  # Zmienna przechowująca koszt całkowity

    for x in range(0, len(node_list)-1):
        edge = find_edge_to_goal(node_list[x], node_list[x+1].geo_info.name, time)
        travel_time = edge.arrival_time.minAfterOO - edge.departure_time.minAfterOO
        waiting_time = edge.departure_time.minAfterOO - time.minAfterOO
        print(f"Waiting time {waiting_time}")
        total_cost += (travel_time + waiting_time)  # Sumowanie kosztu
        print(f"From {node_list[x].geo_info.name} to {node_list[x+1].geo_info.name} at {edge.departure_time} until {edge.arrival_time} by {edge.line}. Travel time: {travel_time} minutes, total travel time: {total_cost}, koszt: {node_list[x+1].g}")
        time = edge.arrival_time
        last_stop = node_list[x+1].geo_info.name

        if prev_line != edge.line:
            num_changes += 1
        prev_line = edge.line

    time_temp = time.minAfterOO
    # total_travel_time = time.minAfterOO - TimeInformation(tim).minAfterOO
    total_travel_time = time.minAfterOO - time.minAfterOO
    if total_travel_time < 0:
        time_temp += 24 * 60  # Dodajemy 24 godziny w minutach
    # total_travel_time = time_temp - TimeInformation(tim).minAfterOO
    total_travel_time = time_temp - time.minAfterOO
    print(f"Total time: {total_travel_time} minutes")

    print(f"Total travel cost: {total_cost} minutes", file=sys.stderr)  # Wypisanie na stderr
    print(f"Number of changes: {num_changes}")


def save_graph(graph, filename):
    with open(filename, 'wb') as file:
        pickle.dump(graph, file)

def load_graph(filename):
    with open(filename, 'rb') as file:
        graph = pickle.load(file)
    return graph


def get_data():
    while True:
        start_stop = input("Przystanek początkowy: ")
        end_stop = input("Przystanek końcowy: ")
        time = input("Godzina odjazdu HH:MM:SS : ")
        time_or_stops = input("t - minimalizacja czasu dojazdu, p - minimalizacja liczby przesiadek: ")
        return start_stop, end_stop, time_or_stops, time

def main():
    graph_filename = "_graph_data.pickle"
    try:
        # Spróbuj wczytać zapisany graf
        graph = load_graph(graph_filename)
        print("Graph loaded successfully from file.")
    except FileNotFoundError:
        # Jeśli plik nie istnieje, stwórz nowy graf
        graph = create_graph("connection_graph.csv")
        # graph = create_graph("test16-18.csv")
        # Zapisz graf do pliku
        save_graph(graph, graph_filename)
        print("Graph created and saved to file.")


    print("start searching")
    # start, end, time_or_stops, start_time = get_data()
    # start, end, start_time= "KROMERA", "Solskiego", "10:14:00"
    start, end, start_time= "Renoma", "PORT LOTNICZY", TimeInformation("16:48:00")
    # start, end, start_time= "Zajezdnia Obornicka", "PORT LOTNICZY", "20:50:00"

    print("------------------------ dijkstra time")
    # path_dijkstra = dijkstra(load_graph(graph_filename), start, end, cost_fun_for_time, start_time )
    path_dijkstra = dijkstra(graph, start, end, cost_fun_for_time, start_time)
    if path_dijkstra is not None:
        list_ = read_path(path_dijkstra, end)
        path_with_information(list_, start_time)
    else:
        print("no path found")

    # print("------------------------ astar time")
    # path_astar = astar_search(load_graph(graph_filename), start, end, cost_fun_for_time, start_time )
    # if path_astar != None:
    #     list_ = read_path(path_astar, end)
    #     path_with_information(list_, start_time)
    # else:
    #     print("no path found")
    #
    # print("------------------------ astar switch")
    # path_astar = astar_search(load_graph(graph_filename), start, end , cost_fun_for_switch, start_time )
    # if path_astar is not None:
    #     list_ = read_path(path_astar, end)
    #     path_with_information(list_, start_time)
    # else:
    #     print("no path found")
    #
    # print("------------------------ astar modified")
    # path_astar = astar_search(load_graph(graph_filename), start, end , cost_fun_for_switch_modified, start_time )
    # if path_astar is not None:
    #     list_ = read_path(path_astar, end)
    #     path_with_information(list_, start_time)
    # else:
    #     print("no path found")

if __name__ == "__main__":
    main()