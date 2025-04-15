from custom_taxi_env import CustomTaxiEnv

env = CustomTaxiEnv()

state = env.reset()
done = False

while not done:
    env.render()
    action = env.action_space.sample()  # rastgele aksiyon
    next_state, reward, done, _ = env.step(action)
    print("Reward:", reward)
