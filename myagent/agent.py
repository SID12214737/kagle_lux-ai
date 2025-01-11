from lux.utils import direction_to
import numpy as np

class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg
        
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_targets = {}

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        unit_mask = np.array(obs["units_mask"][self.team_id])  # shape (max_units,)
        unit_positions = np.array(obs["units"]["position"][self.team_id])  # shape (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id]).flatten()  # shape (max_units,)
        observed_relic_node_positions = np.array(obs["relic_nodes"])  # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"])  # shape (max_relic_nodes,)
        
        available_unit_ids = np.where(unit_mask)[0]
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
        
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        # Save discovered relic nodes
        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(observed_relic_node_positions[id])

        # Unit decision-making
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energys[unit_id]  # Flattened energy array
            
            # Prioritize nearest relic node
            if self.relic_node_positions:
                nearest_relic_node = min(
                    self.relic_node_positions,
                    key=lambda node: abs(unit_pos[0] - node[0]) + abs(unit_pos[1] - node[1])
                )
                distance_to_relic = abs(unit_pos[0] - nearest_relic_node[0]) + abs(unit_pos[1] - nearest_relic_node[1])
                
                if distance_to_relic <= 4:
                    # Structured hover near relic node
                    hover_direction = np.random.choice([1, 2, 3, 4])
                    actions[unit_id] = [hover_direction, 0, 0]
                else:
                    # Move towards relic node
                    move_direction = direction_to(unit_pos, nearest_relic_node)
                    actions[unit_id] = [move_direction, 0, 0]
            else:
                # Explore map for resources or relics
                if unit_id not in self.unit_explore_targets or step % 20 == 0:
                    rand_x = np.random.randint(0, self.env_cfg["map_width"])
                    rand_y = np.random.randint(0, self.env_cfg["map_height"])
                    self.unit_explore_targets[unit_id] = (rand_x, rand_y)
                
                explore_target = self.unit_explore_targets[unit_id]
                move_direction = direction_to(unit_pos, explore_target)
                actions[unit_id] = [move_direction, 0, 0]

        return actions
