import heapq

import astar

import heapq


import heapq
import data_pre_processing
from data_pre_processing import Node, TimeInformation

@astar.timeit
def dijkstra(graph, start, goal, cost_fn, start_time):
    #convert start_time to TimeInformation object
    start_time = TimeInformation(start_time)
    
    #tart node
    start_node = graph[start]
    start_node.g = 0
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
        
        if chosen_edge == None:
            tim = start_time
            prev_line = ""
        else:
            tim = chosen_edge.arrival_time
            prev_line = chosen_edge.line

        #explore the neighbors of the current node
        for next_node in astar.neighbours(graph, current_node, tim): 
            #calculate the cost of moving to the next node
            cost_fn_res, chosen_edge = cost_fn(current_node, next_node, tim, prev_line)
            new_cost = current_node.g + cost_fn_res
            
            #update the cost of the next node if a shorter path is found
            if next_node.geo_info.name not in visited and new_cost < graph[next_node.geo_info.name].g:
                graph[next_node.geo_info.name].g = new_cost
                graph[next_node.geo_info.name].parent_name = current_name
                heapq.heappush(priority_queue, (new_cost, next_node.geo_info.name))
    
    return None