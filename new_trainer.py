from luxai_s3.wrappers import LuxAIS3GymEnv
from megaagent.agent import Agent
from myagent.agent import Agent as ExampleAgent  # Your example agent

def evaluate_agents(learning_agent_cls, example_agent_cls, seed=None, training=True, games_to_play=3):
    env = LuxAIS3GymEnv(numpy_output=True)
    obs, info = env.reset(seed=seed)
    
    env_cfg = info["params"]  

    # Initialize agents
    player_0 = learning_agent_cls("player_0", info["params"], training=training)  # Learning agent
    player_1 = example_agent_cls("player_1", info["params"], training=False)  # Fixed example agent

    for i in range(games_to_play):
        obs, info = env.reset(seed=seed)
        game_done = False
        step = 0
        last_obs = None
        last_actions = None
        print(f"Game {i+1}/{games_to_play}")
        
        while not game_done:
            actions = {}
            
            # Store current observation for learning
            if training:
                last_obs = {
                    "player_0": obs["player_0"].copy(),
                    "player_1": obs["player_1"].copy()
                }

            # Get actions
            actions["player_0"] = player_0.act(step=step, obs=obs["player_0"])
            actions["player_1"] = player_1.act(step=step, obs=obs["player_1"])  # Example agent action

            if training:
                last_actions = actions.copy()

            # Environment step
            obs, rewards, terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] | truncated[k] for k in terminated}
            rewards = {
                "player_0": obs["player_0"]["team_points"][player_0.team_id],
                "player_1": obs["player_1"]["team_points"][player_1.team_id]
            }
            
            # Store experiences and learn
            if training and last_obs is not None:
                for unit_id in range(env_cfg["max_units"]):
                    if obs["player_0"]["units_mask"][player_0.team_id][unit_id]:
                        # Learning agent's experience
                        current_state = player_0._state_representation(
                            last_obs["player_0"]["units"]["position"][player_0.team_id][unit_id],
                            last_obs["player_0"]["units"]["energy"][player_0.team_id][unit_id],
                            last_obs["player_0"]["relic_nodes"],
                            step,
                            last_obs["player_0"]["relic_nodes_mask"]
                        )
                        
                        next_state = player_0._state_representation(
                            obs["player_0"]["units"]["position"][player_0.team_id][unit_id],
                            obs["player_0"]["units"]["energy"][player_0.team_id][unit_id],
                            obs["player_0"]["relic_nodes"],
                            step + 1,
                            obs["player_0"]["relic_nodes_mask"]
                        )
                        
                        player_0.memory.push(
                            current_state,
                            last_actions["player_0"][unit_id][0],
                            rewards["player_0"],
                            next_state,
                            dones["player_0"]
                        )

                # Learn from experiences
                player_0.learn(
                    step, last_obs["player_0"], actions["player_0"], 
                    obs["player_0"], rewards["player_0"], dones["player_0"]
                )

            # Check game termination
            if dones["player_0"] or dones["player_1"]:
                game_done = True
                if training:
                    player_0.save_model()

            step += 1

    env.close()
    if training:
        player_0.save_model()

# Training the learning agent against the example agent
# Replace `Agent` and `ExampleAgent` with the appropriate class implementations
evaluate_agents(Agent, ExampleAgent, training=True, games_to_play=100)

# Evaluate the trained agent
evaluate_agents(Agent, ExampleAgent, training=False, games_to_play=10)
