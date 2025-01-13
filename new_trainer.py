import random
from luxai_s3.wrappers import LuxAIS3GymEnv
from relicbound.agent import Agent as ExampleAgent
from myagent.agent import Agent


def evaluate_agents(learning_agent_cls, example_agent_cls, seed=None, training=True, games_to_play=3):
    env = LuxAIS3GymEnv(numpy_output=True)
    obs, info = env.reset(seed=seed)
    
    env_cfg = info["params"]  

    # Play multiple games
    for i in range(games_to_play):
        # Randomly decide which agent is player_0 and player_1
        if random.choice([True, False]):
            agents = {
                "player_0": learning_agent_cls("player_0", env_cfg, training=training),
                "player_1": example_agent_cls("player_1", env_cfg),
            }
        else:
            agents = {
                "player_0": example_agent_cls("player_0", env_cfg),
                "player_1": learning_agent_cls("player_1", env_cfg, training=training),
            }

        obs, info = env.reset()
        game_done = False
        step = 0
        last_obs = None
        last_actions = None
        print(f"Game {i+1}/{games_to_play}")

        while not game_done:
            actions = {}

            # Store current observation for learning
            if training and isinstance(agents["player_0"], learning_agent_cls):
                last_obs = {
                    "player_0": obs["player_0"].copy(),
                    "player_1": obs["player_1"].copy()
                }

            # Get actions for both players
            for player, agent in agents.items():
                actions[player] = agent.act(step=step, obs=obs[player])

            if training and isinstance(agents["player_0"], learning_agent_cls):
                last_actions = actions.copy()

            # Environment step
            obs, rewards, terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] | truncated[k] for k in terminated}
            rewards = {
                "player_0": obs["player_0"]["team_points"][agents["player_0"].team_id],
                "player_1": obs["player_1"]["team_points"][agents["player_1"].team_id],
            }
            
            # Store experiences and learn
            for player, agent in agents.items():
                if training and isinstance(agent, learning_agent_cls) and last_obs is not None:
                    for unit_id in range(env_cfg["max_units"]):
                        if obs[player]["units_mask"][agent.team_id][unit_id]:
                            current_state = agent._state_representation(
                                last_obs[player]["units"]["position"][agent.team_id][unit_id],
                                last_obs[player]["units"]["energy"][agent.team_id][unit_id],
                                last_obs[player]["relic_nodes"],
                                step,
                                last_obs[player]["relic_nodes_mask"]
                            )
                            
                            next_state = agent._state_representation(
                                obs[player]["units"]["position"][agent.team_id][unit_id],
                                obs[player]["units"]["energy"][agent.team_id][unit_id],
                                obs[player]["relic_nodes"],
                                step + 1,
                                obs[player]["relic_nodes_mask"]
                            )
                            
                            agent.memory.push(
                                current_state,
                                last_actions[player][unit_id][0],
                                rewards[player],
                                next_state,
                                dones[player]
                            )

                    agent.learn(
                        step, last_obs[player], actions[player], 
                        obs[player], rewards[player], dones[player]
                    )

            # Check game termination
            if dones["player_0"] or dones["player_1"]:
                game_done = True
                if training:
                    for agent in agents.values():
                        if isinstance(agent, learning_agent_cls):
                            agent.save_model()

            step += 1

    env.close()
    if training:
        for agent in agents.values():
            if isinstance(agent, learning_agent_cls):
                agent.save_model()


# Training the learning agent against the example agent
evaluate_agents(Agent, ExampleAgent, training=True, games_to_play=10)

# Evaluate the trained agent
evaluate_agents(Agent, ExampleAgent, training=False, games_to_play=1)
