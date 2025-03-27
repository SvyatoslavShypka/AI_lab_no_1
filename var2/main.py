import astar
import data_pre_processing
import dijkstra
from copy import deepcopy
import pickle

#tworzy sciezke przechodzac wstecz po rodzicach
def read_path(graph: dict[str, data_pre_processing.Node], end: str):
    node_list = list()
    node_end = graph[end]
    while node_end.parent_name != "":
        node_list.append(node_end)
        node_end = graph[node_end.parent_name]
    node_list.append(node_end)
    node_list.reverse()
    return node_list

#znajduje krawedz do wezla end
def find_edge_to_goal(node: data_pre_processing.Node, end: str, time: data_pre_processing.TimeInformation):
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
def path_with_information(node_list: list[data_pre_processing.Node], tim: str):
    time = data_pre_processing.TimeInformation(tim)
    last_stop = None
    num_changes = 0  # Licznik przesiadek
    prev_line = None
    for x in range(0, len(node_list)-1):
        edge = find_edge_to_goal(node_list[x], node_list[x+1].geo_info.name, time)
        print(f"From {node_list[x].geo_info.name} to {node_list[x+1].geo_info.name} at {edge.departure_time} by {edge.line} in {edge.arrival_time.minAfterOO - edge.departure_time.minAfterOO} minutes")
        time = edge.arrival_time
        last_stop = node_list[x+1].geo_info.name

        if prev_line != edge.line:  # Jeśli linia się zmieniła, zwiększ liczbę przesiadek
            num_changes += 1
        prev_line = edge.line
       

    print(f"Arrival time at {last_stop}: {time}")
    time_temp = time.minAfterOO
    total_travel_time = time.minAfterOO - data_pre_processing.TimeInformation(tim).minAfterOO
    if total_travel_time < 0:
        time_temp += 24 * 60  # Dodajemy 24 godziny w minutach
    total_travel_time = time_temp - data_pre_processing.TimeInformation(tim).minAfterOO
    print(f"Total time: {total_travel_time} minutes")
    print(f"Number of changes: {num_changes}")




def save_graph(graph, filename):
    with open(filename, 'wb') as file:
        pickle.dump(graph, file)

def load_graph(filename):
    with open(filename, 'rb') as file:
        graph = pickle.load(file)
    return graph

def main():

    graph_filename = "saved_graph.pickle"
    try:
        # Spróbuj wczytać zapisany graf
        graph = load_graph(graph_filename)
        print("Graph loaded successfully from file.")
    except FileNotFoundError:
        # Jeśli plik nie istnieje, stwórz nowy graf
        graph = data_pre_processing.create_graph("connection_graph.csv")
        # Zapisz graf do pliku
        save_graph(graph, graph_filename)
        print("Graph created and saved to file.")


    print("start searching")
    start, end, start_time= "KROMERA", "Solskiego", "10:14:00"
    
    print("------------------------ dijkstra time")
    path_dijkstra = dijkstra.dijkstra(load_graph(graph_filename), start, end ,astar.cost_fun_for_time, start_time )
    if path_dijkstra != None:
        list_ = read_path(path_dijkstra, end)
        path_with_information(list_, start_time)
    else:
        print("no path found")

    print("------------------------ astar time")
    path_astar = astar.astar_search(load_graph(graph_filename), start, end ,astar.cost_fun_for_time, start_time )
    if path_astar != None:
        list_ = read_path(path_astar, end)
        path_with_information(list_, start_time)
    else:
        print("no path found")
    
    print("------------------------ astar switch")
    path_astar = astar.astar_search(load_graph(graph_filename), start, end ,astar.cost_fun_for_switch, start_time )
    if path_astar != None:
        list_ = read_path(path_astar, end)
        path_with_information(list_, start_time)
    else:
        print("no path found")
 
    print("------------------------ astar modified")
    path_astar = astar.astar_search(load_graph(graph_filename), start, end ,astar.cost_fun_for_switch_modified, start_time )
    if path_astar != None:
        list_ = read_path(path_astar, end)
        path_with_information(list_, start_time)
    else:
        print("no path found")

if __name__ == "__main__":
    main()