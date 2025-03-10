import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
from PIL import Image
import time
import random

# For ArangoDB Connection (if available)
try:
    from arango import ArangoClient
    HAS_ARANGO = True
except ImportError:
    HAS_ARANGO = False

# For GPU acceleration (if available)
try:
    import cudf
    import cugraph
    import cupy
    HAS_CUGRAPH = True
except ImportError:
    HAS_CUGRAPH = False

st.set_page_config(
    page_title="GraphRAG - Amazon Product Network Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global variables
G = None
node_metrics_df = None
communities = None
arangodb_client = None
arangodb_db = None

# Function to load sample data if no graph is provided
@st.cache_data
def load_sample_data():
    # Create a sample metrics DataFrame
    sample_metrics = pd.DataFrame({
        'product_id': list(range(1, 101)),
        'in_degree': np.random.randint(1, 50, 100),
        'out_degree': np.random.randint(1, 40, 100),
        'total_degree': np.random.randint(5, 80, 100),
        'pagerank': np.random.uniform(0.0001, 0.002, 100),
    })
    
    # Add percentile ranks
    sample_metrics['pagerank_percentile'] = sample_metrics['pagerank'].rank(pct=True) * 100
    sample_metrics['in_degree_percentile'] = sample_metrics['in_degree'].rank(pct=True) * 100
    sample_metrics['out_degree_percentile'] = sample_metrics['out_degree'].rank(pct=True) * 100
    
    # Assign influence tiers
    def assign_influence_tier(percentile):
        if percentile < 50:
            return 'Low'
        elif percentile < 90:
            return 'Medium'
        elif percentile < 99:
            return 'High'
        else:
            return 'Very High'
    
    sample_metrics['influence_tier'] = sample_metrics['pagerank_percentile'].apply(assign_influence_tier)
    
    # Create a sample graph
    G_sample = nx.DiGraph()
    for i in range(1, 101):
        G_sample.add_node(i)
    
    # Add some edges
    for i in range(1, 95):
        # Each node connects to several random nodes
        connections = np.random.choice(range(i+1, 101), np.random.randint(1, 5), replace=False)
        for j in connections:
            G_sample.add_edge(i, j)
    
    # Create sample communities
    sample_communities = []
    remaining_nodes = set(range(1, 101))
    
    # Create 5 communities
    for i in range(5):
        size = len(remaining_nodes) // (5 - i)
        nodes = random.sample(list(remaining_nodes), size)
        remaining_nodes -= set(nodes)
        
        # Create a community dictionary
        community = {
            "id": i,
            "size": size,
            "nodes": nodes,
            "density": np.random.uniform(0.01, 0.03),
            "avg_clustering": np.random.uniform(0.3, 0.6),
            "central_nodes": random.sample(nodes, min(3, len(nodes)))
        }
        sample_communities.append(community)
    
    return G_sample, sample_metrics, sample_communities

# Function to create a network visualization
def create_network_visualization(G, node_metrics_df, max_nodes=150):
    # Select influential nodes based on PageRank
    influential_nodes = node_metrics_df.nlargest(max_nodes // 3, 'pagerank')['product_id'].tolist()
    
    # Add some of their neighbors
    neighbors = set()
    for node in influential_nodes:
        if len(neighbors) + len(influential_nodes) >= max_nodes:
            break
        try:
            # Add successors (products bought together)
            successors = list(G.successors(node))
            neighbors.update(successors[:min(5, len(successors))])
            
            # Add predecessors (products that lead to this one)
            predecessors = list(G.predecessors(node))
            neighbors.update(predecessors[:min(5, len(predecessors))])
        except:
            continue
    
    # Combine influential nodes and neighbors
    sample_nodes = set(influential_nodes)
    remaining_slots = max_nodes - len(sample_nodes)
    neighbors = list(neighbors - sample_nodes)
    sample_nodes.update(neighbors[:remaining_slots])
    
    # Create the subgraph
    try:
        subgraph = G.subgraph(sample_nodes)
    except:
        st.error("Error creating subgraph. Using sample nodes directly.")
        subgraph = G
    
    # Prepare node data
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    # Use force-directed layout
    pos = nx.spring_layout(subgraph, seed=42)
    
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Node metrics
        try:
            in_deg = subgraph.in_degree(node)
            row = node_metrics_df[node_metrics_df['product_id'] == node]
            if not row.empty:
                pagerank = row['pagerank'].values[0]
            else:
                pagerank = 0.0001
        except:
            in_deg = 0
            pagerank = 0.0001
        
        # Text information
        node_text.append(f"Product: {node}<br>In-Degree: {in_deg}<br>PageRank: {pagerank:.6f}")
        
        # Node size based on PageRank
        size = 10 + (pagerank * 5000)
        node_size.append(size)
        
        # Node color based on whether it's influential
        if node in influential_nodes:
            node_color.append('rgba(255, 0, 0, 0.8)')  # Red for influential
        else:
            node_color.append('rgba(0, 116, 217, 0.7)')  # Blue for others
    
    # Create edges
    edge_x = []
    edge_y = []
    
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            color=node_color,
            size=node_size,
            line=dict(width=2, color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f'Amazon Product Network Sample ({len(subgraph.nodes())} products)',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=600
        )
    )
    
    return fig

# Function to create community visualization
def create_community_visualization(communities):
    if not communities or len(communities) <= 1:
        st.warning("Not enough communities to visualize")
        return None, None, None
    
    # Limit to top 8 communities for better visualization
    top_communities = communities[:8]
    
    # 1. Community sizes visualization
    community_sizes = [comm["size"] for comm in top_communities]
    total_size = sum(community_sizes)
    size_percentages = [size/total_size*100 for size in community_sizes]
    
    if "degree_range" in top_communities[0]:
        # For degree-based communities, use degree range as label
        community_ids = [f"Degree {comm.get('degree_range', comm['id'])}" for comm in top_communities]
    else:
        community_ids = [f"Comm {comm['id']}" for comm in top_communities]
    
    fig1 = go.Figure(data=[go.Pie(
        labels=community_ids,
        values=community_sizes,
        hole=0.3,
        textinfo='label+percent',
        insidetextorientation='radial',
        marker=dict(
            colors=px.colors.qualitative.Plotly[:len(top_communities)],
            line=dict(color='white', width=2)
        )
    )])
    
    fig1.update_layout(
        title_text='Product Communities by Size',
        annotations=[dict(text='Product Groups', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    
    # 2. Community metrics visualization
    fig2 = make_subplots(rows=1, cols=2, 
                         subplot_titles=("Community Density", "Community Clustering Coefficient"))
    
    fig2.add_trace(
        go.Bar(
            x=community_ids,
            y=[comm["density"] for comm in top_communities],
            marker_color='lightseagreen'
        ),
        row=1, col=1
    )
    
    fig2.add_trace(
        go.Bar(
            x=community_ids,
            y=[comm["avg_clustering"] for comm in top_communities],
            marker_color='darkorange'
        ),
        row=1, col=2
    )
    
    fig2.update_layout(
        title_text='Community Cohesion Metrics',
        showlegend=False,
        height=400,
        width=800
    )
    
    # 3. Central products in top communities visualization
    central_products_data = []
    
    for comm in top_communities[:3]:  # Limit to top 3 communities
        central_nodes = comm["central_nodes"]
        comm_id = comm.get('degree_range', str(comm['id']))
        
        for node in central_nodes:
            try:
                # Find metrics for this node
                metrics = node_metrics_df[node_metrics_df['product_id'] == node]
                if not metrics.empty:
                    central_products_data.append({
                        'product_id': node,
                        'community': f"Comm {comm_id}",
                        'pagerank': metrics['pagerank'].iloc[0],
                        'in_degree': metrics['in_degree'].iloc[0]
                    })
            except:
                pass
    
    if central_products_data:
        central_df = pd.DataFrame(central_products_data)
        
        fig3 = px.scatter(
            central_df, 
            x='in_degree', 
            y='pagerank',
            color='community',
            size='in_degree',
            size_max=25,
            hover_name='product_id',
            log_x=True,
            title="Central Products in Top Communities"
        )
        
        fig3.update_layout(
            height=400,
            width=800
        )
    else:
        fig3 = None
    
    return fig1, fig2, fig3

# Function to create metrics visualizations
def create_metrics_visualization(G, node_metrics_df, sample_size=1000):
    # Sample the dataframe for faster visualization
    if len(node_metrics_df) > sample_size:
        sampled_df = node_metrics_df.sample(sample_size)
    else:
        sampled_df = node_metrics_df
    
    # Create a figure with multiple subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Degree Distribution', 
            'PageRank Distribution (Sample)',
            'Top Products by In-Degree', 
            'Top Products by PageRank'
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # 1. Simplified Degree Distribution (histogram instead of scatter)
    in_degrees = dict(G.in_degree())
    degree_values = list(in_degrees.values())
    
    # Create histogram bins
    bins = list(range(0, min(50, max(degree_values) + 1), 2))
    hist, bin_edges = np.histogram(degree_values, bins=bins)
    
    fig.add_trace(go.Bar(
        x=[f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges)-1)],
        y=hist,
        name='In-Degree'
    ), row=1, col=1)
    
    fig.update_xaxes(title='Degree Range', row=1, col=1)
    fig.update_yaxes(title='Frequency', type='log', row=1, col=1)
    
    # 2. PageRank Distribution (using sampled data)
    pagerank_values = sorted(sampled_df['pagerank'].values, reverse=True)
    
    fig.add_trace(go.Scatter(
        x=list(range(len(pagerank_values))),
        y=pagerank_values,
        mode='lines',
        name='PageRank',
        line=dict(width=2, color='green')
    ), row=1, col=2)
    
    fig.update_xaxes(title='Product Rank', row=1, col=2)
    fig.update_yaxes(title='PageRank Value', row=1, col=2)
    
    # 3. Top Products by In-Degree
    top_by_in_degree = node_metrics_df.nlargest(10, 'in_degree')
    
    fig.add_trace(go.Bar(
        x=top_by_in_degree['product_id'].astype(str),
        y=top_by_in_degree['in_degree'],
        name='In-Degree',
        marker_color='purple'
    ), row=2, col=1)
    
    fig.update_xaxes(title='Product ID', row=2, col=1)
    fig.update_yaxes(title='In-Degree', row=2, col=1)
    
    # 4. Top Products by PageRank
    top_by_pagerank = node_metrics_df.nlargest(10, 'pagerank')
    
    fig.add_trace(go.Bar(
        x=top_by_pagerank['product_id'].astype(str),
        y=top_by_pagerank['pagerank'],
        name='PageRank',
        marker_color='orange'
    ), row=2, col=2)
    
    fig.update_xaxes(title='Product ID', row=2, col=2)
    fig.update_yaxes(title='PageRank', row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title='Amazon Product Network Metrics',
        height=700,
        width=900,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

# Function to process natural language queries
def process_query(query):
    import re
    q_lower = query.lower()
    
    try:
        # 1. Top influential products query
        if re.search(r'\b(top|influential|pagerank)\b', q_lower):
            num_match = re.search(r'top\s+(\d+)', q_lower)
            limit = int(num_match.group(1)) if num_match else 5
            
            # Get top products by PageRank
            top_products = node_metrics_df.nlargest(limit, 'pagerank')
            
            response = f"Here are the top {limit} most influential products in the network based on PageRank:\n\n"
            for i, row in top_products.iterrows():
                response += f"{i+1}. Product {int(row['product_id'])} - PageRank: {row['pagerank']:.6f}, In-Degree: {row['in_degree']}\n"
            
            response += ("\nThese products represent key nodes in your product network. "
                        "Products with high PageRank scores are frequently co-purchased with other popular products, "
                        "making them important connection points in your catalog.")
            return response
        
        # 2. Community detection query
        elif re.search(r'\b(communities|community|groups)\b', q_lower):
            response = "Product Communities in the Network:\n\n"
            
            for i, community in enumerate(communities[:5]):  # Top 5 communities
                response += f"Community {community['id']}:\n"
                response += f"- Size: {community['size']} products\n"
                response += f"- Density: {community['density']:.4f}\n"
                response += f"- Clustering: {community['avg_clustering']:.4f}\n"
                response += f"- Central Products: {', '.join(str(p) for p in community['central_nodes'][:3])}\n\n"
            
            response += ("These product communities represent natural product categories or complementary product sets.")
            return response
        
        # 3. Similar products query
        elif re.search(r'\bsimilar\b', q_lower):
            similar_match = re.search(r'product\s+(\d+)', q_lower)
            if not similar_match:
                return "Please specify a product ID to find similar products."
            
            product_id = int(similar_match.group(1))
            
            # Simple similarity based on common connections
            if product_id not in G.nodes():
                return f"Product {product_id} not found in the graph."
            
            # Get products connected to this product
            try:
                successors = set(G.successors(product_id))
                predecessors = set(G.predecessors(product_id))
                connections = successors.union(predecessors)
                
                # For each connected product, calculate similarity
                similarities = []
                for node in G.nodes():
                    if node == product_id or node not in node_metrics_df['product_id'].values:
                        continue
                    
                    # Get its connections
                    try:
                        node_successors = set(G.successors(node))
                        node_predecessors = set(G.predecessors(node))
                        node_connections = node_successors.union(node_predecessors)
                        
                        # Calculate Jaccard similarity
                        intersection = len(connections.intersection(node_connections))
                        union = len(connections.union(node_connections))
                        similarity = intersection / union if union > 0 else 0
                        
                        if similarity > 0:
                            # Get metrics
                            metrics = node_metrics_df[node_metrics_df['product_id'] == node]
                            pagerank = metrics['pagerank'].iloc[0] if not metrics.empty else 0
                            
                            similarities.append({
                                'product_id': node,
                                'similarity': similarity,
                                'pagerank': pagerank
                            })
                    except:
                        continue
                
                # Sort by similarity
                similarities.sort(key=lambda x: x['similarity'], reverse=True)
                top_similar = similarities[:10]
                
                response = f"Products similar to Product {product_id}:\n\n"
                for i, prod in enumerate(top_similar):
                    response += f"{i+1}. Product {prod['product_id']} - Similarity: {prod['similarity']:.4f}, PageRank: {prod['pagerank']:.6f}\n"
                
                response += (f"\nThese products share similar co-purchasing patterns with Product {product_id}, "
                             "making them excellent candidates for 'Customers Also Bought' recommendations.")
                return response
            except:
                return f"Error calculating similarities for Product {product_id}."
        
        # 4. Complementary products query
        elif re.search(r'\b(complementary|co-purchased)\b', q_lower):
            product_match = re.search(r'product\s+(\d+)', q_lower)
            if not product_match:
                return "Please specify a product ID to get complementary product recommendations."
            
            product_id = int(product_match.group(1))
            
            # Get directly connected products (successors)
            if product_id not in G.nodes():
                return f"Product {product_id} not found in the graph."
            
            try:
                complementary = list(G.successors(product_id))
                if not complementary:
                    return f"No complementary products found for Product {product_id}."
                
                # Limit to top 10
                complementary = complementary[:10]
                
                # Get metrics for these products
                results = []
                for prod in complementary:
                    metrics = node_metrics_df[node_metrics_df['product_id'] == prod]
                    if not metrics.empty:
                        results.append({
                            'product_id': prod,
                            'pagerank': metrics['pagerank'].iloc[0],
                            'in_degree': metrics['in_degree'].iloc[0]
                        })
                
                response = f"Complementary Products for Product {product_id}:\n\n"
                for i, product in enumerate(results):
                    response += (f"{i+1}. Product {product['product_id']} - "
                                 f"PageRank: {product['pagerank']:.6f}, "
                                 f"In-Degree: {product['in_degree']}\n")
                
                response += ("\nThese products are frequently co-purchased with Product "
                             f"{product_id}. They make excellent candidates for cross-selling, bundling promotions, "
                             "and 'Frequently Bought Together' recommendations.")
                return response
            except:
                return f"Error finding complementary products for Product {product_id}."
        
        # 5. Shortest path query
        elif re.search(r'\b(shortest path|path between)\b', q_lower):
            product_matches = re.findall(r'product\s+(\d+)', q_lower)
            if len(product_matches) < 2:
                return "Please specify two product IDs to find the path between them."
            
            source_id = int(product_matches[0])
            target_id = int(product_matches[1])
            
            if source_id not in G.nodes():
                return f"Source product {source_id} not found in the graph."
            if target_id not in G.nodes():
                return f"Target product {target_id} not found in the graph."
            
            try:
                path = nx.shortest_path(G, source=source_id, target=target_id)
                
                response = f"Shortest Path from Product {source_id} to Product {target_id}:\n\n"
                response += " → ".join(str(v) for v in path) + "\n\n"
                response += f"Path Length: {len(path) - 1} steps\n\n"
                response += "This path represents a chain of co-purchasing relationships that connect the two products."
                return response
            except nx.NetworkXNoPath:
                return f"No path found between Product {source_id} and Product {target_id}."
            except:
                return f"Error finding path between Product {source_id} and Product {target_id}."
        
        # 6. Cross-selling strategy query
        elif re.search(r'\b(cross-selling|cross-sell)\b', q_lower):
            product_match = re.search(r'product\s+(\d+)', q_lower)
            if not product_match:
                return "Please specify a product ID to develop a cross-selling strategy."
            
            product_id = int(product_match.group(1))
            
            if product_id not in G.nodes():
                return f"Product {product_id} not found in the graph."
            
            try:
                # Get product metrics
                metrics = node_metrics_df[node_metrics_df['product_id'] == product_id]
                if metrics.empty:
                    return f"No metrics found for Product {product_id}."
                
                # Get co-purchased products
                copurchases = list(G.successors(product_id))
                if not copurchases:
                    return f"No co-purchased products found for Product {product_id}."
                
                # Get metrics for co-purchased products
                copurchase_data = []
                for prod in copurchases[:10]:  # Limit to top 10
                    prod_metrics = node_metrics_df[node_metrics_df['product_id'] == prod]
                    if not prod_metrics.empty:
                        copurchase_data.append({
                            'product_id': prod,
                            'pagerank': prod_metrics['pagerank'].iloc[0],
                            'in_degree': prod_metrics['in_degree'].iloc[0],
                            'influence_tier': prod_metrics['influence_tier'].iloc[0]
                        })
                
                # Categorize co-purchased products by influence tier
                high_influence = [p for p in copurchase_data if p['influence_tier'] in ['High', 'Very High']]
                medium_influence = [p for p in copurchase_data if p['influence_tier'] == 'Medium']
                low_influence = [p for p in copurchase_data if p['influence_tier'] == 'Low']
                
                response = f"Cross-Selling Strategy for Product {product_id}:\n\n"
                response += "PRODUCT PROFILE:\n"
                response += f"- Influence Tier: {metrics['influence_tier'].iloc[0]}\n"
                response += f"- PageRank: {metrics['pagerank'].iloc[0]:.6f}\n"
                response += f"- In-Degree: {metrics['in_degree'].iloc[0]}\n"
                response += f"- Out-Degree: {metrics['out_degree'].iloc[0]}\n\n"
                
                response += "RECOMMENDED CROSS-SELLING STRATEGIES:\n\n"
                response += "1. Premium Bundle Strategy:\n"
                response += "   Create premium product bundles including:\n"
                if high_influence:
                    for product in high_influence[:3]:
                        response += f"   - Product {product['product_id']} (High Influence)\n"
                else:
                    response += "   - No high-influence products available for bundling\n"
                response += "\n"
                
                response += "2. Complementary Products Strategy:\n"
                response += "   Position these as 'Frequently Bought Together':\n"
                if medium_influence:
                    for product in medium_influence[:3]:
                        response += f"   - Product {product['product_id']}\n"
                else:
                    for product in copurchase_data[:3]:
                        response += f"   - Product {product['product_id']}\n"
                response += "\n"
                
                response += "3. In-Cart Recommendation Strategy:\n"
                response += "   Show these items during checkout:\n"
                for product in (low_influence + medium_influence)[:3]:
                    response += f"   - Product {product['product_id']}\n"
                response += "\n"
                
                response += "IMPLEMENTATION RECOMMENDATIONS:\n"
                response += "1. Highlight the most influential complementary products on product detail pages\n"
                response += "2. Create bundled discount offers for items with high co-purchase frequency\n"
                response += "3. Implement 'Complete Your Collection' campaigns for product groups\n"
                
                return response
            except:
                return f"Error generating cross-selling strategy for Product {product_id}."
        
        # 7. Comprehensive product analysis
        elif re.search(r'\banalyze\b', q_lower) and re.search(r'product\s+(\d+)', q_lower):
            product_id = int(re.search(r'product\s+(\d+)', q_lower).group(1))
            
            if product_id not in G.nodes():
                return f"Product {product_id} not found in the graph."
            
            try:
                # Get product metrics
                metrics = node_metrics_df[node_metrics_df['product_id'] == product_id]
                if metrics.empty:
                    return f"No metrics found for Product {product_id}."
                
                # Get similar products (simplified)
                successors = set(G.successors(product_id))
                predecessors = set(G.predecessors(product_id))
                connections = successors.union(predecessors)
                
                # Simple similarity for demonstration
                similar_products = []
                for node in list(connections)[:20]:
                    if node == product_id:
                        continue
                    
                    # Get node connections
                    try:
                        node_successors = set(G.successors(node))
                        node_predecessors = set(G.predecessors(node))
                        node_connections = node_successors.union(node_predecessors)
                        
                        # Calculate Jaccard similarity
                        intersection = len(connections.intersection(node_connections))
                        union = len(connections.union(node_connections))
                        similarity = intersection / union if union > 0 else 0
                        
                        if similarity > 0:
                            similar_products.append({
                                'product_id': node,
                                'similarity': similarity
                            })
                    except:
                        continue
                
                # Sort and take top 5
                similar_products.sort(key=lambda x: x['similarity'], reverse=True)
                similar_products = similar_products[:5]
                
                # Get directly co-purchased products
                copurchased = list(G.successors(product_id))[:5]
                
                # Format response
                response = f"Comprehensive Analysis of Product {product_id}:\n\n"
                
                response += "INFLUENCE METRICS:\n"
                response += f"- PageRank: {metrics['pagerank'].iloc[0]:.6f} (Percentile: {metrics['pagerank_percentile'].iloc[0]:.1f}%)\n"
                response += f"- In-Degree: {metrics['in_degree'].iloc[0]} (Percentile: {metrics['in_degree_percentile'].iloc[0]:.1f}%)\n"
                response += f"- Out-Degree: {metrics['out_degree'].iloc[0]}\n"
                response += f"- Influence Tier: {metrics['influence_tier'].iloc[0]}\n\n"
                
                response += "SIMILAR PRODUCTS:\n"
                for i, product in enumerate(similar_products):
                    response += f"{i+1}. Product {product['product_id']} - Similarity: {product['similarity']:.4f}\n"
                response += "\n"
                
                response += "FREQUENTLY CO-PURCHASED PRODUCTS:\n"
                for i, prod_id in enumerate(copurchased):
                    response += f"{i+1}. Product {prod_id}\n"
                response += "\n"
                
                # Add business insights
                response += "BUSINESS INSIGHTS:\n"
                tier = metrics['influence_tier'].iloc[0]
                
                if tier in ['High', 'Very High']:
                    response += "- This is a high-influence product that plays a key role in your product network\n"
                    response += "- Feature prominently in marketing and store placement\n"
                    response += "- Ensure consistent stock availability\n"
                elif metrics['in_degree'].iloc[0] > metrics['out_degree'].iloc[0] * 2:
                    response += "- This product is frequently purchased after viewing other products\n"
                    response += "- Excellent candidate for upselling and cross-promotion\n"
                elif metrics['out_degree'].iloc[0] > metrics['in_degree'].iloc[0] * 2:
                    response += "- This product leads to many subsequent purchases\n"
                    response += "- Consider as an entry point product or loss leader\n"
                else:
                    response += "- This product has balanced co-purchasing relationships\n"
                    response += "- Good candidate for bundling with complementary products\n"
                
                return response
            except Exception as e:
                return f"Error analyzing Product {product_id}: {str(e)}"
        
        # Default response for unknown queries
        return "I couldn't understand that query. Try asking about influential products, similar products, communities, or product paths."
    
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Sidebar with app controls
st.sidebar.title("GraphRAG")
st.sidebar.subheader("Amazon Product Network Analysis")

# Option to connect to ArangoDB
if HAS_ARANGO:
    st.sidebar.subheader("ArangoDB Connection")
    use_arango = st.sidebar.checkbox("Connect to ArangoDB", value=False)
    
    if use_arango:
        arango_url = st.sidebar.text_input("ArangoDB URL", "http://localhost:8529")
        arango_username = st.sidebar.text_input("Username", "root")
        arango_password = st.sidebar.text_input("Password", type="password")
        arango_db_name = st.sidebar.text_input("Database Name", "amazon_graph")
        
        if st.sidebar.button("Connect"):
            try:
                # Connect to ArangoDB
                client = ArangoClient(hosts=arango_url)
                db = client.db(
                    arango_db_name, 
                    username=arango_username, 
                    password=arango_password,
                    verify=True
                )
                
                # Test connection
                db.properties()
                
                st.sidebar.success("Connected to ArangoDB successfully!")
                arangodb_client = client
                arangodb_db = db
                
                # Get sample data from ArangoDB
                # This would need to be implemented based on your ArangoDB structure
                # For now, we'll use sample data
                G, node_metrics_df, communities = load_sample_data()
                
            except Exception as e:
                st.sidebar.error(f"Error connecting to ArangoDB: {str(e)}")

# Option to upload NetworkX graph data
st.sidebar.subheader("Graph Data")
graph_option = st.sidebar.radio(
    "Graph Data Source",
    ["Use Sample Data", "Upload Graph Data"]
)

if graph_option == "Upload Graph Data":
    uploaded_file = st.sidebar.file_uploader("Upload NetworkX Graph (pickle)", type=["pkl", "pickle"])
    
    if uploaded_file is not None:
        try:
            # Load the graph from pickle
            import pickle
            graph_data = pickle.load(uploaded_file)
            
            # Check if it's a proper NetworkX graph
            if isinstance(graph_data, nx.Graph):
                G = graph_data
                st.sidebar.success(f"Graph loaded with {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges")
                
                # Create simple metrics DataFrame
                data = []
                for node in G.nodes():
                    data.append({
                        'product_id': node,
                        'in_degree': G.in_degree(node),
                        'out_degree': G.out_degree(node),
                        'total_degree': G.in_degree(node) + G.out_degree(node),
                        'pagerank': 0.0001,  # Placeholder
                    })
                
                node_metrics_df = pd.DataFrame(data)
                
                # Calculate PageRank
                with st.sidebar.spinner("Computing PageRank..."):
                    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
                    for node, score in pagerank.items():
                        idx = node_metrics_df.index[node_metrics_df['product_id'] == node].tolist()
                        if idx:
                            node_metrics_df.at[idx[0], 'pagerank'] = score
                
                # Add percentile ranks
                node_metrics_df['pagerank_percentile'] = node_metrics_df['pagerank'].rank(pct=True) * 100
                node_metrics_df['in_degree_percentile'] = node_metrics_df['in_degree'].rank(pct=True) * 100
                node_metrics_df['out_degree_percentile'] = node_metrics_df['out_degree'].rank(pct=True) * 100
                
                # Assign influence tiers
                def assign_influence_tier(percentile):
                    if percentile < 50:
                        return 'Low'
                    elif percentile < 90:
                        return 'Medium'
                    elif percentile < 99:
                        return 'High'
                    else:
                        return 'Very High'
                
                node_metrics_df['influence_tier'] = node_metrics_df['pagerank_percentile'].apply(assign_influence_tier)
                
                # Detect communities
                with st.sidebar.spinner("Detecting communities..."):
                    G_undirected = G.to_undirected()
                    try:
                        # Try to use Louvain if available
                        from community import community_louvain
                        partition = community_louvain.best_partition(G_undirected)
                        
                        # Group nodes by community
                        community_map = {}
                        for node, comm_id in partition.items():
                            if comm_id not in community_map:
                                community_map[comm_id] = []
                            community_map[comm_id].append(node)
                        
                        # Convert to sorted list of communities
                        sorted_communities = sorted(community_map.items(), key=lambda x: len(x[1]), reverse=True)
                        
                        communities = []
                        for i, (comm_id, nodes) in enumerate(sorted_communities[:10]):
                            # Calculate metrics for this community
                            comm_subgraph = G_undirected.subgraph(nodes[:500])  # Limit for calculation
                            density = nx.density(comm_subgraph)
                            avg_clustering = nx.average_clustering(comm_subgraph)
                            
                            # Find central nodes
                            pagerank = nx.pagerank(G.subgraph(nodes[:500]), alpha=0.85, max_iter=100)
                            central_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
                            
                            communities.append({
                                "id": comm_id,
                                "size": len(nodes),
                                "nodes": nodes[:500],  # Limit for memory
                                "density": density,
                                "avg_clustering": avg_clustering,
                                "central_nodes": [node for node, score in central_nodes]
                            })
                    except:
                        # Simple connected components
                        components = list(nx.connected_components(G_undirected))
                        components_sorted = sorted(components, key=len, reverse=True)
                        
                        communities = []
                        for i, component in enumerate(components_sorted[:10]):
                            nodes = list(component)
                            # Calculate metrics
                            comm_subgraph = G_undirected.subgraph(nodes[:500])  # Limit for calculation
                            density = nx.density(comm_subgraph)
                            avg_clustering = nx.average_clustering(comm_subgraph)
                            
                            communities.append({
                                "id": i,
                                "size": len(nodes),
                                "nodes": nodes[:500],  # Limit for memory
                                "density": density,
                                "avg_clustering": avg_clustering,
                                "central_nodes": nodes[:5]  # Simple approach
                            })
                
                st.sidebar.success("Graph analysis complete!")
                
            else:
                st.sidebar.error("Uploaded file is not a valid NetworkX graph.")
        except Exception as e:
            st.sidebar.error(f"Error loading graph data: {str(e)}")
else:
    # Load sample data
    G, node_metrics_df, communities = load_sample_data()

# Main content area
st.title("GraphRAG - Amazon Product Network Analysis")
st.write("A Graph-based Retrieval Augmented Generation system for e-commerce insights")

# Organize content using tabs
tab1, tab2, tab3, tab4 = st.tabs(["Network Visualization", "Graph Metrics", "Communities", "Natural Language Interface"])

with tab1:
    st.header("Network Visualization")
    st.write("This visualization shows a sample of the product network, highlighting influential products (red) and their connections.")
    
    if G is not None:
        network_fig = create_network_visualization(G, node_metrics_df, max_nodes=150)
        st.plotly_chart(network_fig, use_container_width=True)
    else:
        st.warning("No graph data available. Please upload a graph or use sample data.")

with tab2:
    st.header("Graph Metrics")
    st.write("Key metrics about the product network structure and properties.")
    
    if G is not None and node_metrics_df is not None:
        # Create metrics visualization
        metrics_fig = create_metrics_visualization(G, node_metrics_df)
        st.plotly_chart(metrics_fig, use_container_width=True)
        
        # Show basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Products", f"{G.number_of_nodes():,}")
        with col2:
            st.metric("Total Relationships", f"{G.number_of_edges():,}")
        with col3:
            density = nx.density(G)
            st.metric("Graph Density", f"{density:.6f}")
    else:
        st.warning("No graph data available. Please upload a graph or use sample data.")

with tab3:
    st.header("Product Communities")
    st.write("Communities represent natural groupings of products based on co-purchasing patterns.")
    
    if communities is not None:
        # Create community visualizations
        fig1, fig2, fig3 = create_community_visualization(communities)
        
        if fig1 is not None:
            st.subheader("Community Sizes")
            st.plotly_chart(fig1, use_container_width=True)
        
        if fig2 is not None:
            st.subheader("Community Cohesion Metrics")
            st.plotly_chart(fig2, use_container_width=True)
        
        if fig3 is not None:
            st.subheader("Central Products in Top Communities")
            st.plotly_chart(fig3, use_container_width=True)
        
        # Show community details
        st.subheader("Community Details")
        for i, community in enumerate(communities[:5]):
            with st.expander(f"Community {community['id']} ({community['size']} products)"):
                st.write(f"**Density:** {community['density']:.4f}")
                st.write(f"**Clustering Coefficient:** {community['avg_clustering']:.4f}")
                st.write("**Central Products:**")
                for node in community['central_nodes'][:5]:
                    st.write(f"- Product {node}")
    else:
        st.warning("No community data available. Please upload a graph or use sample data.")

with tab4:
    st.header("Natural Language Interface")
    st.write("Ask questions about the product network using natural language.")
    
    # Example queries for user guidance
    st.subheader("Example Queries")
    example_queries = [
        "What are the top 5 most influential products in the network?",
        "Find products similar to product 1",
        "Which product communities exist in the network?",
        "Recommend complementary products for product 10",
        "What is the shortest path between product 1 and product 100?",
        "Suggest a cross-selling strategy for product 5",
        "Analyze product 10 comprehensively",
    ]
    
    selected_example = st.selectbox("Try an example query:", [""] + example_queries)
    
    # Query input
    query = st.text_area("Or enter your own query:", value=selected_example, height=100)
    
    if st.button("Submit Query"):
        if query:
            with st.spinner('Processing query...'):
                # Add small delay to simulate processing
                time.sleep(1)
                
                # Process the query
                if G is not None and node_metrics_df is not None:
                    response = process_query(query)
                    st.success("Query processed!")
                    st.markdown(response)
                else:
                    st.error("No graph data available. Please upload a graph or use sample data.")
        else:
            st.warning("Please enter a query.")

# Footer
st.markdown("---")
st.markdown("GraphRAG - Amazon Product Network Analysis © 2025")
st.markdown(
    "This application combines graph analytics with generative AI to provide business insights from e-commerce data."
)
st.markdown("GPU acceleration is " + ("enabled" if HAS_CUGRAPH else "not available") + " on this system.")