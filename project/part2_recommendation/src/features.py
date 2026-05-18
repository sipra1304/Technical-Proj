import numpy as np
from collections import deque


def compute_user_distances(graph, user_id):
    """
    Single-source BFS from a user node.
    Returns dict: node -> shortest distance.
    """
    source = f"u_{user_id}"
    distances = {}
    if source not in graph.adj:
        return distances

    queue = deque()
    distances[source] = 0
    queue.append(source)

    while queue:
        node = queue.popleft()
        for neighbor in graph.adj[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)

    return distances


def extract_features(user_id, movie_id, user_distances, liked_movies, k_hop=4):
    """
    Extract 5 structural graph features for a user-movie pair.

    Features:
    1. direct_distance: shortest path distance from user to candidate movie
    2. min_distance_to_liked: minimum distance from candidate movie to any
       movie previously liked by the user
    3. k_hop_reachable: binary indicator if movie is within k hops of user
    4. inverse_distance: 1/(1+distance), higher for closer movies
    5. avg_distance_to_liked: average distance from candidate to all liked movies

    Args:
        user_id: user identifier
        movie_id: movie identifier
        user_distances: precomputed BFS distances from this user (dict)
        liked_movies: set of movie IDs that user has liked
        k_hop: threshold for k-hop reachability feature
    """
    movie_node = f"m_{movie_id}"
    MAX_DIST = 100

    # Feature 1: Direct structural distance from user to candidate movie
    direct_dist = user_distances.get(movie_node, MAX_DIST)

    # Feature 2: Minimum distance from candidate movie to user's liked movies
    # (approximated via user's BFS tree since all liked movies are at distance 2 from user)
    min_dist_to_liked = MAX_DIST
    if liked_movies:
        for lm in liked_movies:
            lm_node = f"m_{lm}"
            lm_dist = user_distances.get(lm_node, MAX_DIST)
            # Distance between two movies through user's BFS tree:
            # |dist(user, movie) - dist(user, liked_movie)| <= actual dist <= sum
            # For bipartite: movies connected through shared users at distance 2
            # Use triangle inequality lower bound as approximation
            pair_dist = abs(direct_dist - lm_dist)
            if pair_dist < min_dist_to_liked:
                min_dist_to_liked = pair_dist

    # Feature 3: k-hop reachability (is movie within k hops?)
    k_hop_reachable = 1 if direct_dist <= k_hop else 0

    # Feature 4: Inverse distance score (higher = closer)
    inverse_dist = 1.0 / (1.0 + direct_dist)

    # Feature 5: Average distance from candidate to all liked movies
    avg_dist_to_liked = MAX_DIST
    if liked_movies:
        total = 0.0
        count = 0
        for lm in liked_movies:
            lm_node = f"m_{lm}"
            lm_dist = user_distances.get(lm_node, MAX_DIST)
            total += abs(direct_dist - lm_dist)
            count += 1
        avg_dist_to_liked = total / count if count > 0 else MAX_DIST

    return [
        direct_dist,
        min_dist_to_liked,
        k_hop_reachable,
        inverse_dist,
        avg_dist_to_liked,
    ]


FEATURE_NAMES = [
    "direct_distance",
    "min_distance_to_liked",
    "k_hop_reachable",
    "inverse_distance",
    "avg_distance_to_liked",
]
