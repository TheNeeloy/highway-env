import gym
import highway_env

# create anomaly exit env
env = gym.make("anomaly-exit-v0")

for m in range(5):                                  # generate 5 scenarios
    done = False                                    # variable to check when to start next scenario
    obs = env.reset()                               # reset env at start
    while not done:                                 # when done, start next scenario
        env.render()                                # render env
        action = 4                                  # keep ego vehicle going straight at slow speed (DiscreteMetaAction)
        obs, reward, done, info = env.step(action)  # step env forward
