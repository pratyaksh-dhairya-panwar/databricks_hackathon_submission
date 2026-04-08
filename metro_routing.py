import streamlit as st
import math
import csv
import re
import heapq
from collections import defaultdict
from geopy.geocoders import Nominatim
import time
import io

@st.cache_data(show_spinner=False)
def load_metro_data():
    """Load metro data either from Databricks Volumes or local fallback."""
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        r = w.files.download("/Volumes/hackathon/default/train_running_history/Delhi_metro.csv")
        content = r.contents.read().decode('utf-8')
        return content
    except Exception:
        # Fallback to local file
        try:
            with open("c:/dev/Databricks Hackathon/data_traffic_delhi/Delhi_metro.csv", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            try:
                with open("Delhi_metro.csv", "r", encoding="utf-8") as f:
                    return f.read()
            except FileNotFoundError:
                return None

@st.cache_data(show_spinner=False)
def get_station_coords_cache():
    content = load_metro_data()
    coords_dict = {}
    if not content:
        return coords_dict
        
    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        raw_name = row['Station Names']
        name = re.sub(r'\[.*|\(.*|Conn:.*', '', raw_name, flags=re.IGNORECASE).strip().title()
        lat = float(row['Latitude'])
        lon = float(row['Longitude'])
        coords_dict[name.lower()] = (lat, lon)
    return coords_dict

@st.cache_data(show_spinner=False)
def geocode_location(location_name):
    # 1. Direct match with metro station names
    stations_coords = get_station_coords_cache()
    loc_lower = location_name.lower()
    for st_name, coords in stations_coords.items():
        if loc_lower in st_name or st_name in loc_lower:
            return coords

    # 2. Fallback to Nominatim
    query = f"{location_name}, Delhi"
    geolocator = Nominatim(user_agent="route_optimizer_app_delhi_v3")
    
    for _ in range(3):
        try:
            location = geolocator.geocode(query, timeout=5)
            if location:
                return location.latitude, location.longitude
            return None
        except Exception:
            time.sleep(1)
    return None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@st.cache_data(show_spinner=False)
def get_metro_travel_time_and_path(origin_coords, dest_coords):
    stations = []
    lines = defaultdict(list)
    station_nodes = defaultdict(list)
    
    content = load_metro_data()
    if not content:
        return None
        
    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        raw_name = row['Station Names']
        name = re.sub(r'\[.*|\(.*|Conn:.*', '', raw_name, flags=re.IGNORECASE).strip().title()
        
        lat_str = row['Latitude']
        lon_str = row['Longitude']
        if not lat_str or not lon_str: continue
        lat = float(lat_str)
        lon = float(lon_str)
        line = row['Metro Line'].strip().title()
        station_id = int(row['ID (Station ID)'])
        
        node = (name, line)
        stations.append({
            'name': name,
            'lat': lat,
            'lon': lon,
            'line': line,
            'node': node
        })
        
        lines[line].append((station_id, node))
        if node not in station_nodes[name]:
            station_nodes[name].append(node)
            
    graph = defaultdict(list)
    
    for line, s_list in lines.items():
        s_list.sort()
        for i in range(len(s_list) - 1):
            node1 = s_list[i][1]
            node2 = s_list[i+1][1]
            graph[node1].append((node2, 3))
            graph[node2].append((node1, 3))
            
    for name, nodes in station_nodes.items():
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                node1 = nodes[i]
                node2 = nodes[j]
                graph[node1].append((node2, 8))
                graph[node2].append((node1, 8))
                
    if not stations: return None
    
    nearest_origin = min(stations, key=lambda s: haversine(origin_coords[0], origin_coords[1], s['lat'], s['lon']))
    nearest_dest = min(stations, key=lambda s: haversine(dest_coords[0], dest_coords[1], s['lat'], s['lon']))
    
    start_nodes = station_nodes[nearest_origin['name']]
    end_nodes = set(station_nodes[nearest_dest['name']])
    
    pq = []
    for sn in start_nodes:
        heapq.heappush(pq, (0, sn))
        
    distances = {sn: 0 for sn in start_nodes}
    previous_nodes = {sn: None for sn in start_nodes}
    
    shortest_time = float('inf')
    best_end_node = None
    
    while pq:
        current_dist, current_node = heapq.heappop(pq)
        
        if current_dist > distances.get(current_node, float('inf')):
            continue
            
        if current_node in end_nodes:
            if current_dist < shortest_time:
                shortest_time = current_dist
                best_end_node = current_node
                
        for neighbor, weight in graph[current_node]:
            distance = current_dist + weight
            if distance < distances.get(neighbor, float('inf')):
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))
                
    if shortest_time == float('inf') or best_end_node is None:
        return None
        
    path = []
    curr = best_end_node
    while curr is not None:
        path.append(curr)
        curr = previous_nodes.get(curr)
    path.reverse()
    
    instructions = []
    if not path:
        return None
        
    current_line = path[0][1]
    last_station = path[0][0]
    board_station = path[0][0]
    
    instructions.append({"type": "board", "station": board_station, "line": current_line})
    
    for i in range(1, len(path)):
        station, line = path[i]
        if line != current_line:
            if board_station != last_station:
                instructions.append({"type": "travel", "from": board_station, "to": last_station, "line": current_line})
            instructions.append({"type": "transfer", "station": station, "from_line": current_line, "to_line": line})
            current_line = line
            board_station = station
        last_station = station
        
    if board_station != last_station:
        instructions.append({"type": "travel", "from": board_station, "to": last_station, "line": current_line})
    instructions.append({"type": "alight", "station": last_station, "line": current_line})
    
    return {
        'time_mins': shortest_time,
        'origin_station': nearest_origin['name'],
        'dest_station': nearest_dest['name'],
        'raw_path': path,
        'instructions': instructions
    }
