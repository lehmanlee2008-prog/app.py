import sqlite3
import uuid
import heapq
import networkx as nx
import osmnx as ox
from datetime import datetime
from collections import deque
from math import radians, sin, cos, sqrt, atan2

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


# ================================================
# PAGE CONFIGURATION
# ================================================

st.set_page_config(
    page_title="SILENT SIGNAL",
    page_icon="🚨",
    layout="wide"
)


# ================================================
# CUSTOM CSS
# ================================================

st.markdown("""
<style>
.stApp {
    background-color: #0B1120;
    color: white;
}
h1, h2, h3 {
    color: #F8FAFC;
}
.main-title {
    font-size: 45px;
    font-weight: 800;
    color: #38BDF8;
}
.subtitle {
    color: #94A3B8;
    font-size: 17px;
}
.card, .metric-card {
    background-color: #172238;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #263750;
    margin-bottom: 14px;
}
.metric-card {
    text-align: center;
}
.stButton button {
    background-color: #0284C7;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}
.stButton button:hover {
    background-color: #0369A1;
    color: white;
}
div[data-testid="stForm"] {
    background-color: #172238;
    padding: 20px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ================================================
# DATABASE SETUP
# ================================================

conn = sqlite3.connect("complaints.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS complaints (
    id TEXT PRIMARY KEY,
    name TEXT,
    category TEXT,
    location TEXT,
    severity TEXT,
    description TEXT,
    date TEXT,
    status TEXT,
    latitude REAL,
    longitude REAL
)
""")

# Add coordinate columns to older databases, if they are missing.
cursor.execute("PRAGMA table_info(complaints)")
existing_columns = [column[1] for column in cursor.fetchall()]

if "latitude" not in existing_columns:
    cursor.execute("ALTER TABLE complaints ADD COLUMN latitude REAL")

if "longitude" not in existing_columns:
    cursor.execute("ALTER TABLE complaints ADD COLUMN longitude REAL")

conn.commit()

# Store team assignments so they remain available after app refreshes.
cursor.execute("""
CREATE TABLE IF NOT EXISTS team_assignments (
    complaint_id TEXT PRIMARY KEY,
    team_name TEXT NOT NULL,
    assigned_at TEXT NOT NULL
)
""")
conn.commit()


# ================================================
# HELPER FUNCTIONS
# ================================================

def calculate_distance_km(lat1, lon1, lat2, lon2):
    """Calculate the distance between two coordinates using Haversine."""
    earth_radius_km = 6371.0

    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))
    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))

    difference_lat = lat2 - lat1
    difference_lon = lon2 - lon1

    a = (
        sin(difference_lat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(difference_lon / 2) ** 2
    )
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return earth_radius_km * c


def find_bfs_clusters(items, distance_limit_km=1.0):
    """
    Group complaints using BFS.
    Two complaints are connected when they are within the distance limit.
    """
    count = len(items)
    visited = [False] * count
    clusters = []

    for start in range(count):
        if visited[start]:
            continue

        queue = deque([start])
        visited[start] = True
        current_cluster = []

        while queue:
            current = queue.popleft()
            current_cluster.append(items[current])

            for neighbour in range(count):
                if visited[neighbour]:
                    continue

                distance = calculate_distance_km(
                    items[current][8],
                    items[current][9],
                    items[neighbour][8],
                    items[neighbour][9]
                )

                if distance <= distance_limit_km:
                    visited[neighbour] = True
                    queue.append(neighbour)

        clusters.append(current_cluster)

    return clusters


def get_marker_color(complaint):
    if complaint[7] == "Resolved":
        return "green"
    if complaint[4] == "Critical":
        return "red"
    if complaint[4] == "High":
        return "orange"
    if complaint[4] == "Medium":
        return "beige"
    return "blue"


def predict_incident_risk(complaint):
    """
    Transparent prototype score estimating urgency from submitted severity,
    category and description keywords. This is a rule-based estimate, not
    a trained machine-learning prediction.
    """
    severity_points = {
        "Low": 10,
        "Medium": 30,
        "High": 60,
        "Critical": 80
    }

    category_points = {
        "Road Damage": 10,
        "Garbage Accumulation": 5,
        "Water Leakage": 15,
        "Broken Streetlight": 8,
        "Drainage Problem": 15,
        "Other": 5
    }

    score = severity_points.get(complaint[4], 10)
    score += category_points.get(complaint[2], 5)

    description = (complaint[5] or "").lower()
    urgent_words = [
        "accident", "injury", "danger", "flood", "overflow",
        "collapse", "open manhole", "live wire", "blocking"
    ]

    matched_words = [word for word in urgent_words if word in description]
    score += min(len(matched_words) * 5, 15)
    score = min(score, 100)

    if score >= 80:
        level = "Urgent"
    elif score >= 60:
        level = "High"
    elif score >= 35:
        level = "Moderate"
    else:
        level = "Low"

    return score, level, matched_words


def a_star_route(points, start_index, goal_index, maximum_edge_km):
    """
    A* search on a simplified waypoint graph.
    Each point is (latitude, longitude). Nodes are connected when they are
    within maximum_edge_km. This is not a real street/road network.
    """
    open_heap = []
    heapq.heappush(open_heap, (0, start_index))

    came_from = {}
    cost_so_far = {start_index: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal_index:
            route = [current]
            while current in came_from:
                current = came_from[current]
                route.append(current)
            route.reverse()
            return route, cost_so_far[goal_index]

        for neighbour in range(len(points)):
            if neighbour == current:
                continue

            edge_distance = calculate_distance_km(
                points[current][0], points[current][1],
                points[neighbour][0], points[neighbour][1]
            )

            if edge_distance > maximum_edge_km:
                continue

            new_cost = cost_so_far[current] + edge_distance

            if neighbour not in cost_so_far or new_cost < cost_so_far[neighbour]:
                cost_so_far[neighbour] = new_cost
                came_from[neighbour] = current

                heuristic = calculate_distance_km(
                    points[neighbour][0], points[neighbour][1],
                    points[goal_index][0], points[goal_index][1]
                )
                priority = new_cost + heuristic
                heapq.heappush(open_heap, (priority, neighbour))

    return None, None



def priority_aware_all_stops_route(complaints, depot, maximum_edge_km):
    """Plan a depot-to-all-complaints route using A* for each leg.

    Stop order uses a distance/priority score: urgent complaints get a
    stronger preference, while nearby stops are still favoured. This is a
    practical heuristic, not a guaranteed globally optimal TSP solution.
    """
    points = [depot] + [(float(c[8]), float(c[9])) for c in complaints]
    remaining = set(range(1, len(points)))
    current = 0
    full_path = [0]
    total_distance = 0.0
    ordered_complaints = []
    priority_weights = {"Critical": 5.0, "High": 3.5, "Medium": 2.0, "Low": 1.0}

    while remaining:
        candidates = []
        for target in remaining:
            path, distance = a_star_route(points, current, target, maximum_edge_km)
            if path is None:
                continue
            complaint = complaints[target - 1]
            weight = priority_weights.get(complaint[4], 1.0)
            score = distance / weight
            candidates.append((score, distance, target, path))

        if not candidates:
            return None

        _, leg_distance, target, leg_path = min(candidates, key=lambda item: item[0])
        full_path.extend(leg_path[1:])
        total_distance += leg_distance
        ordered_complaints.append(complaints[target - 1])
        remaining.remove(target)
        current = target

    # Remove consecutive duplicate coordinates where legs meet.
    route_coordinates = [points[index] for index in full_path]
    return {
        "coordinates": route_coordinates,
        "distance": total_distance,
        "depot": depot,
        "complaints": [
            (c[0], c[2], c[3], c[4], (float(c[8]), float(c[9])))
            for c in ordered_complaints
        ],
    }

def road_astar_route(graph, start_coordinate, end_coordinate):
    """Find a drivable street route using A* on an OpenStreetMap road graph."""
    start_node = ox.distance.nearest_nodes(
        graph, X=float(start_coordinate[1]), Y=float(start_coordinate[0])
    )
    end_node = ox.distance.nearest_nodes(
        graph, X=float(end_coordinate[1]), Y=float(end_coordinate[0])
    )

    def heuristic(node_a, node_b):
        a = graph.nodes[node_a]
        b = graph.nodes[node_b]
        return calculate_distance_km(a["y"], a["x"], b["y"], b["x"]) * 1000

    node_path = nx.astar_path(
        graph, start_node, end_node, heuristic=heuristic, weight="length"
    )
    distance_m = 0.0
    for first_node, second_node in zip(node_path[:-1], node_path[1:]):
        edge_data = graph.get_edge_data(first_node, second_node)
        distance_m += min(data.get("length", 0.0) for data in edge_data.values())
    coordinates = [(graph.nodes[n]["y"], graph.nodes[n]["x"]) for n in node_path]
    return coordinates, distance_m / 1000


def build_road_graph(coordinates):
    """Download the driveable OSM road network around all stops."""
    centre_lat = sum(point[0] for point in coordinates) / len(coordinates)
    centre_lon = sum(point[1] for point in coordinates) / len(coordinates)
    radius_m = max(
        calculate_distance_km(centre_lat, centre_lon, point[0], point[1]) * 1000
        for point in coordinates
    ) + 5000
    radius_m = max(5000, min(radius_m, 30000))
    graph = ox.graph_from_point(
        (centre_lat, centre_lon), dist=radius_m, network_type="drive", simplify=True
    )
    return graph


def priority_road_route(complaints, depot):
    """Greedy priority-aware multi-stop route; A* follows actual mapped roads."""
    coordinates = [depot] + [(float(c[8]), float(c[9])) for c in complaints]
    graph = build_road_graph(coordinates)
    remaining = list(range(len(complaints)))
    current = depot
    full_coordinates = []
    ordered = []
    total_km = 0.0
    weights = {"Critical": 8.0, "High": 4.0, "Medium": 2.0, "Low": 1.0}

    while remaining:
        options = []
        for index in remaining:
            target = coordinates[index + 1]
            try:
                road_coords, distance = road_astar_route(graph, current, target)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            priority_weight = weights.get(complaints[index][4], 1.0)
            options.append((distance / priority_weight, distance, index, road_coords))

        if not options:
            raise ValueError("No connected road route found for one or more complaint locations.")

        _, distance, selected, leg = min(options, key=lambda item: item[0])
        if full_coordinates:
            full_coordinates.extend(leg[1:])
        else:
            full_coordinates.extend(leg)
        total_km += distance
        ordered.append(complaints[selected])
        current = coordinates[selected + 1]
        remaining.remove(selected)

    return {"coordinates": full_coordinates, "distance": total_km, "depot": depot,
            "complaints": [(c[0], c[2], c[3], c[4], (float(c[8]), float(c[9]))) for c in ordered]}


def allocate_teams_with_csp(tasks, team_definitions):
    """
    Backtracking CSP allocator.
    Constraints: a team must have the required skill and must not exceed
    its maximum workload. Critical/high-priority complaints are considered first.
    """
    assignments = {team["name"]: [] for team in team_definitions}
    workloads = {team["name"]: 0 for team in team_definitions}
    ordered_tasks = sorted(
        tasks,
        key=lambda task: (
            {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}.get(task[4], 4)
        )
    )

    def backtrack(task_index):
        if task_index == len(ordered_tasks):
            return True

        task = ordered_tasks[task_index]
        category = task[2]

        suitable_teams = [
            team for team in team_definitions
            if category in team["skills"]
            and workloads[team["name"]] < team["capacity"]
        ]

        # Try the team with the smallest current workload first.
        suitable_teams.sort(key=lambda team: workloads[team["name"]])

        for team in suitable_teams:
            team_name = team["name"]
            assignments[team_name].append(task)
            workloads[team_name] += 1

            if backtrack(task_index + 1):
                return True

            assignments[team_name].pop()
            workloads[team_name] -= 1

        return False

    all_assigned = backtrack(0)

    # If not every task can be allocated, return a clear partial allocation.
    if not all_assigned:
        assignments = {team["name"]: [] for team in team_definitions}
        workloads = {team["name"]: 0 for team in team_definitions}
        unassigned = []

        for task in ordered_tasks:
            suitable_teams = [
                team for team in team_definitions
                if task[2] in team["skills"]
                and workloads[team["name"]] < team["capacity"]
            ]
            suitable_teams.sort(key=lambda team: workloads[team["name"]])

            if suitable_teams:
                selected_team = suitable_teams[0]["name"]
                assignments[selected_team].append(task)
                workloads[selected_team] += 1
            else:
                unassigned.append(task)
    else:
        unassigned = []

    return assignments, unassigned


# ================================================
# SESSION STATE
# ================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"


# ================================================
# HEADER
# ================================================

st.markdown(
    '<div class="main-title">🚨 SILENT SIGNAL</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">AI-POWERED INFRASTRUCTURE INCIDENT MANAGEMENT SYSTEM</div>',
    unsafe_allow_html=True
)
st.divider()


# ================================================
# NAVIGATION
# ================================================

if st.session_state.page != "Home":
    if st.button("← Back to Home"):
        st.session_state.page = "Home"
        st.rerun()


# ================================================
# HOME PAGE
# ================================================

if st.session_state.page == "Home":

    st.markdown("## Welcome to Silent Signal")
    st.write(
        "A centralized platform to report, monitor and manage "
        "infrastructure emergencies."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
            <h2>👤 Citizen Portal</h2>
            <p>Report potholes, water leakage, garbage accumulation,
            drainage problems and other infrastructure issues.</p>
            <h4 style="color:#38BDF8;">REPORT AN INCIDENT</h4>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Enter Citizen Portal", use_container_width=True):
            st.session_state.page = "Citizen"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="card">
            <h2>🛡️ Administrator Portal</h2>
            <p>Monitor incoming complaints, analyse incidents,
            prioritize emergencies and manage their resolution.</p>
            <h4 style="color:#F97316;">MANAGE INCIDENTS</h4>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Enter Administrator Portal", use_container_width=True):
            st.session_state.page = "Admin"
            st.rerun()

    st.divider()
    st.markdown("### How Silent Signal Works")

    first, second, third = st.columns(3)

    with first:
        st.markdown("""
        <div class="metric-card">
            <h2>📝</h2>
            <h4>Report</h4>
            <p>Citizens submit infrastructure complaints.</p>
        </div>
        """, unsafe_allow_html=True)

    with second:
        st.markdown("""
        <div class="metric-card">
            <h2>🧠</h2>
            <h4>Analyse</h4>
            <p>Algorithms identify nearby incidents and priorities.</p>
        </div>
        """, unsafe_allow_html=True)

    with third:
        st.markdown("""
        <div class="metric-card">
            <h2>🚀</h2>
            <h4>Resolve</h4>
            <p>Administrators coordinate maintenance operations.</p>
        </div>
        """, unsafe_allow_html=True)


# ================================================
# CITIZEN PORTAL
# ================================================

elif st.session_state.page == "Citizen":

    st.header("📝 Report an Infrastructure Problem")
    st.write(
        "Provide accurate information to help authorities "
        "identify and resolve the issue."
    )

    with st.form("complaint_form"):

        name = st.text_input("Your Name")

        category = st.selectbox(
            "Problem Category",
            [
                "Road Damage",
                "Garbage Accumulation",
                "Water Leakage",
                "Broken Streetlight",
                "Drainage Problem",
                "Other"
            ]
        )

        location = st.text_input(
            "Incident Location",
            placeholder="Enter area, street or landmark"
        )

        st.write("📍 Incident Coordinates")
        coordinate_col1, coordinate_col2 = st.columns(2)

        with coordinate_col1:
            latitude = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=13.0827,
                format="%.6f"
            )

        with coordinate_col2:
            longitude = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=80.2707,
                format="%.6f"
            )

        st.caption(
            "The default coordinates are for Chennai. "
            "Enter the actual coordinates of the incident for accurate mapping."
        )

        severity = st.select_slider(
            "Severity Level",
            options=["Low", "Medium", "High", "Critical"],
            value="Medium"
        )

        description = st.text_area(
            "Describe the Problem",
            placeholder="Explain what happened and provide details..."
        )

        submitted = st.form_submit_button(
            "🚨 Submit Complaint",
            use_container_width=True
        )

    if submitted:
        if name.strip() and location.strip() and description.strip():

            complaint_id = str(uuid.uuid4())[:8].upper()
            complaint_date = datetime.now().strftime("%d-%m-%Y %H:%M")

            cursor.execute("""
                INSERT INTO complaints (
                    id, name, category, location, severity,
                    description, date, status, latitude, longitude
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                complaint_id,
                name.strip(),
                category,
                location.strip(),
                severity,
                description.strip(),
                complaint_date,
                "Pending",
                latitude,
                longitude
            ))

            conn.commit()

            st.success("Complaint submitted successfully!")
            st.markdown(
                f"""
                <div class="card">
                    <h3 style="color:#38BDF8;">Complaint Registered</h3>
                    <h2>{complaint_id}</h2>
                    <p>Please save this ID to track your complaint.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.error("Please fill in your name, location and description.")


# ================================================
# ADMINISTRATOR PORTAL
# ================================================

elif st.session_state.page == "Admin":

    st.header("🛡️ Administrator Dashboard")

    cursor.execute("""
        SELECT id, name, category, location, severity,
               description, date, status, latitude, longitude
        FROM complaints
        ORDER BY rowid DESC
    """)
    complaints = cursor.fetchall()

    # Only use complaints with valid coordinates for the map and BFS.
    map_complaints = [
        complaint for complaint in complaints
        if complaint[8] is not None
        and complaint[9] is not None
        and -90 <= complaint[8] <= 90
        and -180 <= complaint[9] <= 180
    ]

    # --------------------------------------------
    # GEOGRAPHIC INCIDENT MONITORING
    # --------------------------------------------

    st.divider()
    st.subheader("🗺️ Geographic Incident Monitoring")

    if map_complaints:

        average_latitude = sum(c[8] for c in map_complaints) / len(map_complaints)
        average_longitude = sum(c[9] for c in map_complaints) / len(map_complaints)

        incident_map = folium.Map(
            location=[average_latitude, average_longitude],
            zoom_start=12
        )

        for complaint in map_complaints:

            popup_text = f"""
            <b>Complaint ID:</b> {complaint[0]}<br>
            <b>Category:</b> {complaint[2]}<br>
            <b>Location:</b> {complaint[3]}<br>
            <b>Severity:</b> {complaint[4]}<br>
            <b>Status:</b> {complaint[7]}
            """

            folium.Marker(
                location=[complaint[8], complaint[9]],
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"{complaint[2]} - {complaint[4]}",
                icon=folium.Icon(
                    color=get_marker_color(complaint),
                    icon="info-sign"
                )
            ).add_to(incident_map)

        st_folium(
            incident_map,
            height=500,
            use_container_width=True
        )

        st.caption(
            "Red: Critical | Orange: High | Beige: Medium | "
            "Blue: Low | Green: Resolved"
        )

    else:
        st.info(
            "No complaints with coordinates are available yet. "
            "Submit a complaint to populate the map."
        )

    # --------------------------------------------
    # BFS COMPLAINT CLUSTERING
    # --------------------------------------------

    st.divider()
    st.subheader("🧭 Nearby Complaint Clusters (BFS)")

    distance_limit = st.slider(
        "Maximum distance between connected complaints (km)",
        min_value=0.2,
        max_value=5.0,
        value=1.0,
        step=0.2
    )

    if map_complaints:
        clusters = find_bfs_clusters(map_complaints, distance_limit)

        st.write(
            f"BFS identified **{len(clusters)} cluster(s)** "
            f"from **{len(map_complaints)} complaint(s)** with coordinates."
        )

        for cluster_number, cluster in enumerate(clusters, start=1):
            with st.expander(
                f"Cluster {cluster_number} — {len(cluster)} complaint(s)"
            ):
                for complaint in cluster:
                    st.write(
                        f"**{complaint[2]}** | {complaint[3]} | "
                        f"{complaint[4]} priority | Status: {complaint[7]}"
                    )
    else:
        st.info("BFS clusters will appear when complaints have coordinates.")

    # --------------------------------------------
    # ROAD-NETWORK A* ROUTING + SAVED TEAM ASSIGNMENTS
    # --------------------------------------------

    st.divider()
    st.subheader("🧭 Road-Network A* Maintenance Routing")
    active_geo_complaints = [c for c in map_complaints if c[7] != "Resolved"]

    route_col1, route_col2 = st.columns(2)
    with route_col1:
        depot_lat = st.number_input("Maintenance depot latitude", min_value=-90.0,
            max_value=90.0, value=13.0827, format="%.6f", key="depot_lat")
    with route_col2:
        depot_lon = st.number_input("Maintenance depot longitude", min_value=-180.0,
            max_value=180.0, value=80.2707, format="%.6f", key="depot_lon")

    st.caption("Uses OpenStreetMap driveable roads. A* follows mapped street connections; stop order balances road distance and complaint priority.")
    if active_geo_complaints and st.button("Plan road route to all active locations", key="road_astar_button"):
        try:
            with st.spinner("Loading road network and calculating A* routes..."):
                st.session_state["road_astar_result"] = priority_road_route(
                    active_geo_complaints, (depot_lat, depot_lon))
                st.session_state["road_astar_error"] = None
        except Exception as error:
            st.session_state["road_astar_result"] = None
            st.session_state["road_astar_error"] = str(error)

    if st.session_state.get("road_astar_error"):
        st.error("Could not build the road route: " + st.session_state["road_astar_error"])
        st.info("Check that the coordinates are valid and that internet access is available. OpenStreetMap data is downloaded when planning a route.")

    road_result = st.session_state.get("road_astar_result")
    if road_result:
        st.success(f"Road route covers {len(road_result['complaints'])} complaints. Estimated road distance: {road_result['distance']:.2f} km.")
        for number, item in enumerate(road_result["complaints"], start=1):
            st.write(f"{number}. **{item[1]}** — {item[2]} ({item[3]} priority)")
        route_map = folium.Map(location=list(road_result["depot"]), zoom_start=13)
        folium.Marker(list(road_result["depot"]), tooltip="Maintenance Depot",
            icon=folium.Icon(color="green", icon="home")).add_to(route_map)
        for number, item in enumerate(road_result["complaints"], start=1):
            folium.Marker(list(item[4]), tooltip=f"Stop {number}: {item[1]}",
                popup=f"{item[1]} | {item[2]} | {item[3]} priority",
                icon=folium.Icon(color="red" if item[3] in ["Critical", "High"] else "orange", icon="wrench")).add_to(route_map)
        folium.PolyLine(road_result["coordinates"], color="blue", weight=5, opacity=0.85).add_to(route_map)
        st_folium(route_map, height=480, use_container_width=True, key="road_astar_map")
    elif not active_geo_complaints:
        st.info("No unresolved complaints with coordinates are available for routing.")

    # --------------------------------------------
    # CSP TEAM ALLOCATION - SAVE TO DATABASE
    # --------------------------------------------

    st.divider()
    st.subheader("👷 Automatic Maintenance Team Allocation (CSP)")
    team_capacity = st.slider("Maximum active complaints per team", min_value=1,
        max_value=10, value=3, key="team_capacity")
    team_definitions = [
        {"name": "Road Repair Team", "skills": {"Road Damage"}, "capacity": team_capacity},
        {"name": "Sanitation Team", "skills": {"Garbage Accumulation"}, "capacity": team_capacity},
        {"name": "Water & Drainage Team", "skills": {"Water Leakage", "Drainage Problem"}, "capacity": team_capacity},
        {"name": "Electrical Team", "skills": {"Broken Streetlight"}, "capacity": team_capacity},
        {"name": "General Maintenance Team", "skills": {"Other"}, "capacity": team_capacity}
    ]
    tasks_for_allocation = [c for c in complaints if c[7] != "Resolved"]
    assignments, unassigned_tasks = allocate_teams_with_csp(tasks_for_allocation, team_definitions)
    if tasks_for_allocation:
        assigned_count = sum(len(items) for items in assignments.values())
        st.write(f"CSP has allocated **{assigned_count} of {len(tasks_for_allocation)} active complaints**.")
        if st.button("💾 Save / Update Team Assignments", key="save_team_assignments"):
            cursor.execute("DELETE FROM team_assignments")
            saved_time = datetime.now().isoformat(timespec="seconds")
            for team_name, team_tasks in assignments.items():
                for task in team_tasks:
                    cursor.execute("INSERT OR REPLACE INTO team_assignments (complaint_id, team_name, assigned_at) VALUES (?, ?, ?)",
                        (task[0], team_name, saved_time))
            conn.commit()
            st.success("Team assignments saved to the database.")
            st.rerun()

        cursor.execute("SELECT complaint_id, team_name, assigned_at FROM team_assignments")
        saved_rows = cursor.fetchall()
        saved_map = {row[0]: row[1] for row in saved_rows}
        if saved_rows:
            st.caption("Saved assignments (persist after refresh/restart):")
        for team in team_definitions:
            team_tasks = assignments[team["name"]]
            with st.expander(f"{team['name']} — {len(team_tasks)}/{team['capacity']} tasks"):
                st.write("Skills: " + ", ".join(sorted(team["skills"])))
                if team_tasks:
                    for task in team_tasks:
                        saved_label = "✓ Saved" if saved_map.get(task[0]) == team["name"] else "Not saved yet"
                        st.write(f"{task[0]} | {task[2]} | {task[3]} | {task[4]} | {saved_label}")
                else:
                    st.write("No tasks assigned.")
        if unassigned_tasks:
            st.warning("Some complaints could not be allocated:")
            for task in unassigned_tasks:
                st.write(f"{task[0]} | {task[2]} | {task[3]}")
        else:
            st.success("All active complaints were assigned to suitable teams.")

        # Generate a separate road route for each saved team's workload.
        if saved_rows and st.button("🗺️ Build routes for saved teams", key="build_team_routes"):
            st.session_state["team_road_routes"] = {}
            for team_name, team_tasks in assignments.items():
                # Build routes from the saved team-to-complaint mapping.
                saved_team_tasks = [task for task in tasks_for_allocation
                                    if saved_map.get(task[0]) == team_name]
                geo_tasks = [task for task in saved_team_tasks if task[8] is not None and task[9] is not None]
                if geo_tasks:
                    try:
                        st.session_state["team_road_routes"][team_name] = priority_road_route(geo_tasks, (depot_lat, depot_lon))
                    except Exception as error:
                        st.session_state["team_road_routes"][team_name] = {"error": str(error)}
            st.rerun()

        for team_name, result in st.session_state.get("team_road_routes", {}).items():
            with st.expander(f"{team_name} road route"):
                if "error" in result:
                    st.warning(result["error"])
                else:
                    st.write(f"Distance: {result['distance']:.2f} km")
                    team_map = folium.Map(location=list(result["depot"]), zoom_start=13)
                    folium.Marker(list(result["depot"]), tooltip="Depot", icon=folium.Icon(color="green")).add_to(team_map)
                    for number, item in enumerate(result["complaints"], start=1):
                        folium.Marker(list(item[4]), tooltip=f"Stop {number}: {item[1]}").add_to(team_map)
                    folium.PolyLine(result["coordinates"], color="blue", weight=5).add_to(team_map)
                    st_folium(team_map, height=350, use_container_width=True, key="team_map_" + team_name.replace(" ", "_"))
    else:
        st.info("No active complaints need team allocation.")

    # --------------------------------------------
    # INCIDENT RISK PREDICTION (PROTOTYPE)
    # --------------------------------------------

    st.divider()
    st.subheader("🔮 Incident Risk & Urgency Prediction")
    st.caption(
        "Rule-based prototype estimate, not a trained machine-learning model. "
        "It uses submitted severity, category and risk keywords."
    )

    if complaints:
        prediction_rows = []
        for complaint in complaints:
            score, level, matched_words = predict_incident_risk(complaint)
            prediction_rows.append({
                "Complaint ID": complaint[0],
                "Category": complaint[2],
                "Location": complaint[3],
                "Reported Severity": complaint[4],
                "Estimated Risk Score": score,
                "Predicted Urgency": level,
                "Risk Keywords": ", ".join(matched_words) if matched_words else "None",
                "Status": complaint[7]
            })

        prediction_df = pd.DataFrame(prediction_rows)
        st.dataframe(prediction_df, use_container_width=True, hide_index=True)
        st.write("**Top 5 highest estimated-risk incidents**")
        st.dataframe(
            prediction_df.sort_values(
                "Estimated Risk Score", ascending=False
            ).head(5),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Prediction results will appear after complaints are submitted.")

    # --------------------------------------------
    # DASHBOARD METRICS
    # --------------------------------------------

    total = len(complaints)
    critical = sum(1 for c in complaints if c[4] == "Critical")
    pending = sum(1 for c in complaints if c[7] == "Pending")
    in_progress = sum(1 for c in complaints if c[7] == "In Progress")
    resolved = sum(1 for c in complaints if c[7] == "Resolved")

    st.divider()
    st.markdown("### Incident Overview")

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric("Total Reports", total)

    with metric2:
        st.metric("Critical", critical)

    with metric3:
        st.metric("Pending", pending)

    with metric4:
        st.metric("Resolved", resolved)

    # --------------------------------------------
    # COMPLAINT MANAGEMENT
    # --------------------------------------------

    st.divider()
    st.subheader("📋 Complaint Management")

    filter1, filter2 = st.columns(2)

    with filter1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "Pending", "In Progress", "Resolved"]
        )

    with filter2:
        severity_filter = st.selectbox(
            "Filter by Severity",
            ["All", "Critical", "High", "Medium", "Low"]
        )

    filtered_complaints = complaints

    if status_filter != "All":
        filtered_complaints = [
            c for c in filtered_complaints if c[7] == status_filter
        ]

    if severity_filter != "All":
        filtered_complaints = [
            c for c in filtered_complaints if c[4] == severity_filter
        ]

    if filtered_complaints:

        for complaint in filtered_complaints:

            with st.expander(
                f"🚨 {complaint[2]} | {complaint[3]} | "
                f"{complaint[4]} Priority"
            ):

                st.write("Complaint ID:", complaint[0])
                st.write("Citizen:", complaint[1])
                st.write("Category:", complaint[2])
                st.write("Location:", complaint[3])
                st.write("Severity:", complaint[4])
                st.write("Description:", complaint[5])
                st.write("Date:", complaint[6])
                st.write("Current Status:", complaint[7])

                status_options = ["Pending", "In Progress", "Resolved"]
                current_status_index = (
                    status_options.index(complaint[7])
                    if complaint[7] in status_options else 0
                )

                new_status = st.selectbox(
                    "Update Complaint Status",
                    status_options,
                    index=current_status_index,
                    key=f"status_{complaint[0]}"
                )

                if st.button(
                    "Update Status",
                    key=f"update_{complaint[0]}"
                ):
                    cursor.execute(
                        "UPDATE complaints SET status = ? WHERE id = ?",
                        (new_status, complaint[0])
                    )
                    conn.commit()
                    st.success("Status updated!")
                    st.rerun()

                # Delete requires a second, explicit confirmation.
                if st.button(
                    "🗑️ Delete Complaint",
                    key=f"delete_{complaint[0]}"
                ):
                    st.session_state.confirm_delete_id = complaint[0]
                    st.rerun()

                if st.session_state.get("confirm_delete_id") == complaint[0]:
                    st.warning(
                        "Are you sure you want to permanently delete this complaint? "
                        "This action cannot be undone."
                    )
                    confirm_col, cancel_col = st.columns(2)

                    with confirm_col:
                        if st.button(
                            "Yes, confirm deletion",
                            key=f"confirm_delete_{complaint[0]}",
                            type="primary"
                        ):
                            cursor.execute(
                                "DELETE FROM complaints WHERE id = ?",
                                (complaint[0],)
                            )
                            conn.commit()
                            del st.session_state.confirm_delete_id
                            st.success("Complaint deleted successfully.")
                            st.rerun()

                    with cancel_col:
                        if st.button(
                            "Cancel",
                            key=f"cancel_delete_{complaint[0]}"
                        ):
                            del st.session_state.confirm_delete_id
                            st.rerun()

    else:
        st.info("No complaints match your selected filters.")

    # --------------------------------------------
    # COMPLAINT ANALYTICS
    # --------------------------------------------

    st.divider()
    st.subheader("📊 Complaint Analytics")

    if complaints:

        dataframe = pd.DataFrame(
            complaints,
            columns=[
                "ID",
                "Name",
                "Category",
                "Location",
                "Severity",
                "Description",
                "Date",
                "Status",
                "Latitude",
                "Longitude"
            ]
        )

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.write("Reports by Category")
            st.bar_chart(dataframe["Category"].value_counts())

        with chart_col2:
            st.write("Reports by Severity")
            st.bar_chart(dataframe["Severity"].value_counts())

    else:
        st.info("Analytics will appear when complaints are submitted.")


# Close the database connection when the script finishes.
conn.close()