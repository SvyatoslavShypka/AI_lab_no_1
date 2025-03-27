import heapq
import math
from datetime import datetime, date, time
from typing import Callable

import data_pre_processing
from data_pre_processing import EdgeInformation, Node
from data_pre_processing import TimeInformation

from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper



@timeit
def astar_search(graph, start, goal, cost_fn, start_time):
    #convert start_time to TimeInformation object
    start_time = TimeInformation(start_time)

    #start node
    start_node = graph[start]
    start_node.g = 0
    start_node.h = 0
    start_node.f = 0
    graph[start] = start_node

    #priority queue
    priority_queue = []
    heapq.heappush(priority_queue, (0, start))

    visited = set()

    #chosen edge
    chosen_edge = None

    while priority_queue:
        #node with the lowest cost from the priority queue
        cost, current_name = heapq.heappop(priority_queue)
        current_node = graph[current_name]

        if current_name in visited:
            continue

        visited.add(current_name)
        #check if the current node is the goal node
        if current_name == goal:
            return graph

        if chosen_edge is None:
            tim = start_time
            prev_line = ""
        else:
            tim = chosen_edge.arrival_time
            prev_line = chosen_edge.line

        #explore the neighbors of the current node
        for next_node in neighbours(graph, current_node, tim):
            #calculate the cost of moving to the next node
            cost_fn_res, chosen_edge = cost_fn(current_node, next_node, tim, prev_line)
            new_cost = current_node.g + cost_fn_res

            #update the cost of the next node if a shorter path is found
            if next_node.geo_info.name not in visited and new_cost < graph[next_node.geo_info.name].g:
                graph[next_node.geo_info.name].g = new_cost
                graph[next_node.geo_info.name].parent_name = current_name
                #add heuristic value to the cost for A*
                priority = new_cost + h(next_node, graph[goal])
                heapq.heappush(priority_queue, (priority, next_node.geo_info.name))

    return None


#obliczanie heurystyki, euklidesowa
def h(curr: Node, next: Node):
    return math.sqrt( (curr.geo_info.latitiude - next.geo_info.latitiude)**2 + abs(curr.geo_info.longitude - next.geo_info.longitude)**2) 
    
#krawedzi wezla krórych czas wiekszy od czasu pojawienia sie na przystanku
def neighbours (graph : dict[str, Node], start: Node, time: TimeInformation ):
    n_list = list()
    for edges in start.edges:
        if graph[edges.end_stop] not in n_list:
            n_list.append(graph[edges.end_stop])
    
    return n_list



#koszt przejazdu z jednego wezla do drugiego
def cost_fun_for_time(curr: Node, next: Node, s_time: TimeInformation, prev_line: str): 
    posible_comutes = list()
    #mozliwe polaczenia z curr do next
    for edge in curr.edges:
        if(edge.end_stop == next.geo_info.name):
            posible_comutes.append(edge)

    cost = float('inf')
    ret_edge : EdgeInformation
    ret_edge = None
    #znajduje krawedz o min koszcie
    for edge in posible_comutes:
        time_diff = edge.departure_time.minAfterOO - s_time.minAfterOO;
        if time_diff < 0:
            time_diff += 24 * 60
        if time_diff < cost:
            cost = time_diff
            ret_edge = edge

    return cost, ret_edge

def cost_fun_for_switch(curr: Node, next: Node, s_time: TimeInformation, prev_line: str):
    posible_comutes = list()
    for edge in curr.edges:
        if(edge.end_stop == next.geo_info.name):
            posible_comutes.append(edge)
    cost = float('inf')
    ret_edge : EdgeInformation
    ret_edge = None
    swich_multiplyer = 1000
    for edge in posible_comutes:
        if edge.line != prev_line:
            mult = swich_multiplyer
        else:
            mult = 0

        time_diff = edge.departure_time.minAfterOO - s_time.minAfterOO;
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
            #uwzględnienie odległości między przystankami jako czynnika wpływającego na koszt przesiadki
            distance_cost = calculate_distance_cost(curr, next)
            #modyfikacja kosztu przesiadki na podstawie odległości między przystankami
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
    distance = h(curr,next)
    return distance

# def cost_fun_for_swich(curr: Node, next: Node, s_time: TimeInformation, prev_line: str):
#     posible_comutes = list()
#     for edge in curr.edges:
#         if edge.end_stop == next.geo_info.name:
#             posible_comutes.append(edge)
#     cost = float('inf')
#     ret_edge: EdgeInformation
#     ret_edge = None
#     swich_multiplyer = 10000
#     for edge in posible_comutes:
#         if edge.line != prev_line:
#             mult = swich_multiplyer
#         else:
#             mult = 0

#         if mult < cost:  # Nie dodajemy tu czasu, bo patrzymy tylko na przesiadki
#             cost = mult
#             ret_edge = edge
#     return cost, ret_edge
