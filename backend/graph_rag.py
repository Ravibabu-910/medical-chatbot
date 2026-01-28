import os
import pickle
import logging
import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict

from backend.config import Config

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Advanced Knowledge Graph with community detection and embeddings"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.embeddings = {}
        self.communities = {}
        self.community_summaries = {}
    
    def build_from_sheets(self, sheets: Dict, embeddings_model):
        """Build comprehensive knowledge graph from Excel sheets"""
        logger.info("Building knowledge graph from sheets...")
        
        # Track entities and relationships
        entity_counts = defaultdict(int)
        relationship_counts = defaultdict(int)
        
        for sheet_name, df in sheets.items():
            logger.info(f"Processing sheet: {sheet_name}")
            
            for idx, row in df.iterrows():
                # Extract all non-null values as potential entities
                entities = {}
                
                for col in df.columns:
                    value = row[col]
                    if pd.notna(value) and str(value).strip():
                        entity_key = f"{col}_{str(value).strip()}"
                        entities[col] = {
                            "value": str(value).strip(),
                            "key": entity_key,
                            "type": col.lower()
                        }
                        
                        # Add node with metadata
                        self.graph.add_node(
                            entity_key,
                            label=str(value).strip(),
                            type=col.lower(),
                            sheet=sheet_name,
                            row_id=idx
                        )
                        entity_counts[entity_key] += 1
                
                # Create edges between related entities in the same row
                entity_list = list(entities.values())
                for i in range(len(entity_list)):
                    for j in range(i + 1, len(entity_list)):
                        entity_a = entity_list[i]
                        entity_b = entity_list[j]
                        
                        # Determine relationship type
                        rel_type = f"{entity_a['type']}_to_{entity_b['type']}"
                        
                        # Add edge
                        if self.graph.has_edge(entity_a['key'], entity_b['key']):
                            self.graph[entity_a['key']][entity_b['key']]['weight'] += 1
                        else:
                            self.graph.add_edge(
                                entity_a['key'],
                                entity_b['key'],
                                relation=rel_type,
                                weight=1,
                                sheet=sheet_name
                            )
                        
                        relationship_counts[rel_type] += 1
        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, "
                   f"{self.graph.number_of_edges()} edges")
        logger.info(f"Top entities: {sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")
        logger.info(f"Relationships: {dict(relationship_counts)}")
        
        # Generate embeddings for nodes
        self._generate_node_embeddings(embeddings_model)
        
        # Detect communities
        if Config.COMMUNITY_DETECTION_ENABLED:
            self._detect_communities()
        
        return self.graph
    
    def _generate_node_embeddings(self, embeddings_model):
        """Generate embeddings for all nodes"""
        logger.info("Generating node embeddings...")
        
        texts = []
        node_ids = []
        
        for node, data in self.graph.nodes(data=True):
            # Create rich text representation
            text = f"{data.get('label', node)} {data.get('type', '')}"
            texts.append(text)
            node_ids.append(node)
        
        # Generate embeddings in batches
        batch_size = Config.BATCH_SIZE
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = embeddings_model.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        
        # Store embeddings
        self.embeddings = dict(zip(node_ids, all_embeddings))
        logger.info(f"Generated embeddings for {len(self.embeddings)} nodes")
    
    def _detect_communities(self):
        """Detect communities in the graph"""
        logger.info("Detecting communities...")
        
        if self.graph.number_of_nodes() < 2:
            logger.warning("Graph too small for community detection")
            return
        
        try:
            # Use Louvain algorithm for community detection
            import community as community_louvain
            self.communities = community_louvain.best_partition(self.graph)
            
            num_communities = len(set(self.communities.values()))
            logger.info(f"Detected {num_communities} communities")
            
            # Generate community summaries
            self._generate_community_summaries()
            
        except ImportError:
            logger.warning("python-louvain not installed, using simple connected components")
            # Fallback to connected components
            components = nx.connected_components(self.graph)
            self.communities = {}
            for idx, component in enumerate(components):
                for node in component:
                    self.communities[node] = idx
    
    def _generate_community_summaries(self):
        """Generate summaries for each community"""
        logger.info("Generating community summaries...")
        
        community_nodes = defaultdict(list)
        for node, comm_id in self.communities.items():
            community_nodes[comm_id].append(node)
        
        for comm_id, nodes in community_nodes.items():
            # Get node types and labels
            types = defaultdict(int)
            labels = []
            
            for node in nodes[:20]:  # Limit to avoid too large summaries
                data = self.graph.nodes[node]
                types[data.get('type', 'unknown')] += 1
                labels.append(data.get('label', node))
            
            summary = {
                "size": len(nodes),
                "types": dict(types),
                "sample_entities": labels[:10],
                "description": f"Community with {len(nodes)} entities"
            }
            
            self.community_summaries[comm_id] = summary
    
    def get_relevant_subgraph(self, query_embedding: np.ndarray, top_k: int = 5, max_hops: int = 2) -> nx.Graph:
        """Retrieve relevant subgraph based on query embedding"""
        if not self.embeddings:
            return nx.Graph()
        
        # Find most similar nodes
        similarities = {}
        for node, node_emb in self.embeddings.items():
            similarity = np.dot(query_embedding, node_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(node_emb)
            )
            similarities[node] = similarity
        
        # Get top-k nodes
        top_nodes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        selected_nodes = [node for node, _ in top_nodes]
        
        # Expand to neighbors
        expanded_nodes = set(selected_nodes)
        for node in selected_nodes:
            # Get neighbors within max_hops
            neighbors = nx.single_source_shortest_path_length(
                self.graph, node, cutoff=max_hops
            )
            expanded_nodes.update(neighbors.keys())
        
        # Create subgraph
        subgraph = self.graph.subgraph(expanded_nodes).copy()
        
        logger.info(f"Retrieved subgraph: {subgraph.number_of_nodes()} nodes, "
                   f"{subgraph.number_of_edges()} edges")
        
        return subgraph
    
    def get_context_from_subgraph(self, subgraph: nx.Graph) -> str:
        """Extract context text from subgraph"""
        context_parts = []
        
        # Get node information
        for node, data in subgraph.nodes(data=True):
            label = data.get('label', node)
            node_type = data.get('type', 'entity')
            context_parts.append(f"{node_type.title()}: {label}")
        
        # Get relationship information
        for u, v, data in subgraph.edges(data=True):
            u_label = subgraph.nodes[u].get('label', u)
            v_label = subgraph.nodes[v].get('label', v)
            rel = data.get('relation', 'related_to')
            weight = data.get('weight', 1)
            
            if weight > 1:
                context_parts.append(f"{u_label} {rel} {v_label} (strength: {weight})")
            else:
                context_parts.append(f"{u_label} {rel} {v_label}")
        
        return "\n".join(context_parts)
    
    def save(self):
        """Save graph and embeddings"""
        # Save graph
        nx.write_gml(self.graph, Config.GRAPH_PATH)
        logger.info(f"Graph saved to {Config.GRAPH_PATH}")
        
        # Save embeddings
        with open(Config.GRAPH_EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(self.embeddings, f)
        logger.info(f"Embeddings saved to {Config.GRAPH_EMBEDDINGS_PATH}")
        
        # Save community summaries
        if self.community_summaries:
            with open(Config.COMMUNITY_SUMMARY_PATH, 'wb') as f:
                pickle.dump({
                    'communities': self.communities,
                    'summaries': self.community_summaries
                }, f)
            logger.info(f"Community data saved to {Config.COMMUNITY_SUMMARY_PATH}")
    
    def load(self):
        """Load graph and embeddings"""
        try:
            # Load graph
            self.graph = nx.read_gml(Config.GRAPH_PATH)
            logger.info(f"Graph loaded: {self.graph.number_of_nodes()} nodes")
            
            # Load embeddings
            if os.path.exists(Config.GRAPH_EMBEDDINGS_PATH):
                with open(Config.GRAPH_EMBEDDINGS_PATH, 'rb') as f:
                    self.embeddings = pickle.load(f)
                logger.info(f"Embeddings loaded: {len(self.embeddings)} nodes")
            
            # Load community data
            if os.path.exists(Config.COMMUNITY_SUMMARY_PATH):
                with open(Config.COMMUNITY_SUMMARY_PATH, 'rb') as f:
                    community_data = pickle.load(f)
                    self.communities = community_data['communities']
                    self.community_summaries = community_data['summaries']
                logger.info(f"Community data loaded: {len(self.community_summaries)} communities")
            
            return True
        except Exception as e:
            logger.error(f"Error loading graph: {e}")
            return False


# For community detection
try:
    import pandas as pd
except ImportError:
    pass