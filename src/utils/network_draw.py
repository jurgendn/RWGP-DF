from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx


class NetworkVisualizer:
    def __init__(self):
        pass

    def make_layout(
        self, graph: nx.Graph, communities: Dict[int, int]
    ) -> Dict[int, Dict[str, float]]:
        """
        Create a circular layout for each community, arranging community centers circularly.
        Each node is assigned a position based on its community.
        Args:
            graph (nx.Graph): The input graph.
            communities (Dict[int, int]): Mapping from node to community.
        Returns:
            Dict[int, Dict[str, float]]: Node positions as {node: {"x": x, "y": y}}
        """
        layout = {}

        # Create a hypergraph of communities
        community_ids = set(communities.values())
        hypergraph = nx.Graph()
        for cid in community_ids:
            hypergraph.add_node(cid)
        hypergraph_pos = nx.circular_layout(hypergraph, scale=1.0)
        hypergraph_pos = {
            cid: {"x": pos[0], "y": pos[1]} for cid, pos in hypergraph_pos.items()
        }

        # Arrange nodes within each community
        for cid in community_ids:
            community_nodes = [
                node for node, comm in communities.items() if comm == cid
            ]
            subgraph = graph.subgraph(community_nodes)
            node_layout = nx.spiral_layout(subgraph, scale=0.3, dim=2)
            for node, pos in node_layout.items():
                layout[node] = {
                    "x": hypergraph_pos[cid]["x"] + pos[0],
                    "y": hypergraph_pos[cid]["y"] + pos[1],
                }
        return layout

    def make_communities_colors(self, communities: Dict[int, int]) -> Dict[int, str]:
        """
        Create a color map for the communities in the graph.
        Args:
            communities (Dict[int, int]): Mapping from node to community.
        Returns:
            Dict[int, str]: Mapping from node to hex color string.
        """
        cmap = plt.cm.get_cmap("tab10")
        node_colors = {}
        for node, comm in communities.items():
            color = cmap(comm % 10)
            # Convert RGBA to hex (ignore alpha, set alpha to 0x20)
            hex_color = "#{:02x}{:02x}{:02x}20".format(
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            )
            node_colors[node] = hex_color
        return node_colors

    def draw(
        self, graph: nx.Graph, df_results: Dict[int, int], gp_df_results: Dict[int, int]
    ):
        """
        Assigns community colors and layouts to nodes for two sets of results.
        Args:
            graph (nx.Graph): The input graph.
            df_results (Dict[int, int]): First community mapping.
            gp_df_results (Dict[int, int]): Second community mapping.
        """
        df_colors = self.make_communities_colors(df_results)
        gp_df_colors = self.make_communities_colors(gp_df_results)
        nx.set_node_attributes(graph, df_colors, "df-community-color")
        nx.set_node_attributes(graph, gp_df_colors, "gp-df-community-color")
        df_layout = self.make_layout(graph, df_results)
        gp_df_layout = self.make_layout(graph, gp_df_results)
        # Optionally, you can return layouts if needed
        return df_layout, gp_df_layout
