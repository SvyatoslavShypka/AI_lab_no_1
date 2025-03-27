from datetime import datetime,time
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
        self.hour, self.minute = int(str[0:2]), int(str[3:5])
        self.minAfterOO = self.hour*60 + self.minute
    def __str__(self):
        return str(self.hour)+":"+str(self.minute)+":00"
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
    f: float #calkowity koszt g+h
    g: int #koszt przejscia od wezla poczatkowego do danego wezla
    h: float #heurystyka
    def __init__(self, geo_info: VerticleInformation):
        self.geo_info = geo_info
        self.edges = list() #lista krawedzi do sasiednich wezlow
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
    
    def edge_time_cost(edg : EdgeInformation):
        return edg.arrival_time.minAfterOO - edg.departure_time.minAfterOO




def create_graph(filename: str):
    graph :dict[str, Node]
    graph = {}

    df = pd.read_csv(filename,
                     dtype={"id": int, "company": str, "line": str, "departure_time": str, "arrival_time": str
                         , "start_stop": str, "end_stop": str, "start_stop_lat": float, "start_stop_lon": float,
                            "end_stop_lat": float, "end_stop_lon": float})
    df.sort_values(['line','departure_time','start_stop'])
    # print(df["line"],df["departure_time"],df['start_stop'])

    nodes_created = 0
    nodes_count = 0
    print("Creating nodes")
    for index, row in df.iterrows():
    
        ride_info = EdgeInformation( 
            row['line'],
            TimeInformation(row['departure_time']),
            TimeInformation(row['arrival_time'])    ,
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

        if str(row['start_stop'])  not in graph:
            graph[row['start_stop']] = Node(start_info)
            nodes_created += 1
            print(".", end='')

        if str(row['end_stop'])  not in graph:
            graph[row['end_stop']] = Node(stop_info)
            nodes_created += 1
            print(".", end='')

        s =  graph[str(row['start_stop'])]

        graph[str(row['start_stop'])] = s.set_neighbour(ride_info) #dodawanie krawedzi sasiadow do wezla

        if nodes_created%100==0 and nodes_created > nodes_count:
            print(f"{nodes_created} nodes created")
        nodes_count = nodes_created
    print(f"{nodes_created} nodes created totally")
    return graph
