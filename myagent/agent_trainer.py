
from luxai_s3.wrappers import LuxAIS3GymEnv
from myagent.agent import Agent

def evaluate_agents(agent_1_cls, agent_2_cls, seed=None, training=True, games_to_play=3):
    env = LuxAIS3GymEnv(numpy_output=True)
    obs, info = env.reset(seed=seed)
    
    env_cfg = info["params"]  

    player_0 = Agent("player_0", info["params"], training=training)
    player_1 = Agent("player_1", info["params"], training=training)

    for i in range(games_to_play):
        obs, info = env.reset(seed=seed)
        game_done = False
        step = 0
        last_obs = None
        last_actions = None
        print(f"{i}")
        while not game_done:
            
            actions = {}
            
            # Store current observation for learning
            if training:
                last_obs = {
                    "player_0": obs["player_0"].copy(),
                    "player_1": obs["player_1"].copy()
                }

            # Get actions
            for agent in [player_0, player_1]:
                actions[agent.player] = agent.act(step=step, obs=obs[agent.player])

            if training:
                last_actions = actions.copy()

            # Environment step
            obs, rewards ,terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] | truncated[k] for k in terminated}
            rewards = {
                "player_0": obs["player_0"]["team_points"][player_0.team_id],
                "player_1": obs["player_1"]["team_points"][player_1.team_id]
            }  
            # Store experiences and learn
            if training and last_obs is not None:
                # Store experience for each unit
                for agent in [player_0, player_1]:
                    for unit_id in range(env_cfg["max_units"]):
                        if obs[agent.player]["units_mask"][agent.team_id][unit_id]:
                            current_state = agent._state_representation(
                                last_obs[agent.player]["units"]["position"][agent.team_id][unit_id],
                                last_obs[agent.player]["units"]["energy"][agent.team_id][unit_id],
                                last_obs[agent.player]["relic_nodes"],
                                step,
                                last_obs[agent.player]["relic_nodes_mask"]
                            )
                            
                            next_state = agent._state_representation(
                                obs[agent.player]["units"]["position"][agent.team_id][unit_id],
                                obs[agent.player]["units"]["energy"][agent.team_id][unit_id],
                                obs[agent.player]["relic_nodes"],
                                step + 1,
                                obs[agent.player]["relic_nodes_mask"]
                            )
                            
                            agent.memory.push(
                                current_state,
                                last_actions[agent.player][unit_id][0],
                                rewards[agent.player],
                                next_state,
                                dones[agent.player]
                            )
                
                # Learn from experiences
                player_0.learn(step, last_obs["player_0"], actions["player_0"], 
                             obs["player_0"], rewards["player_0"], dones["player_0"])
                player_1.learn(step, last_obs["player_1"], actions["player_1"], 
                             obs["player_1"], rewards["player_1"], dones["player_1"])

            if dones["player_0"] or dones["player_1"]:
                game_done = True
                if training:
                    player_0.save_model()
                    player_1.save_model()

            step += 1

    env.close()
    if training:
      player_0.save_model()
      player_1.save_model()

# Training
# evaluate_agents(Agent, Agent, training=True, games_to_play=5000)

# Evaluation
#evaluate_agents(Agent, Agent, training=False, games_to_play=1)

