import numpy as np
import osmnx as ox
import networkx as nx
from shapely.geometry import LineString

def extract_topology_from_mask(binary_mask, spatial_transform, crs="EPSG:4326"):
    """
    Transforms binary segmentation grids into routable NetworkX directed graphs.
    """
    from skimage.morphology import skeletonize
    
    # 1. Compress mask width down to a single-pixel skeleton matrix
    skeleton = skeletonize(binary_mask > 0).astype(np.uint8)
    points = np.argwhere(skeleton == 1)
    
    G = nx.DiGraph(crs=crs)
    node_counter = 0
    
    # 2. Build local coordinate dictionary lists
    for pt in points:
        y_pixel, x_pixel = pt[0], pt[1]
        lon, lat = spatial_transform * (x_pixel, y_pixel)
        
        G.add_node(node_counter, x=lon, y=lat)
        node_counter += 1
        
    # 3. Simple spatial grid distance check to snap adjacent node points
    nodes = list(G.nodes(data=True))
    for i, (n1, d1) in enumerate(nodes):
        for j, (n2, d2) in enumerate(nodes):
            if i >= j:
                continue
            dist = np.sqrt((d1['x'] - d2['x'])**2 + (d1['y'] - d2['y'])**2)
            if dist < 0.0001:  # Pixel proximity boundary condition thresholds
                G.add_edge(n1, n2, weight=dist, geometry=LineString([(d1['x'], d1['y']), (d2['x'], d2['y'])]))
                
    return G
