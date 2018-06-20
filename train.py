import numpy as np

from dqn import DQNAgent
from game import Game


if __name__ == "__main__":
    # parameters
    n_epochs = 1000
    warmup = 500
    steps = 0

    # environment, agent
    env = Game()
    agent = DQNAgent(env.enable_actions, env.name, epsilon=3e-1)
    
    try:
        for e in range(n_epochs):
            # reset
            frame = 0
            loss = 0.0
            Q_max = 0.0
            total_reward = 0
            env.reset()
            state_t_1, reward_t, terminal = env.step(np.random.choice(env.enable_actions))

            while not terminal:
                state_t = state_t_1

                # execute action in environment
                if steps > warmup:
                    action_t = agent.select_action(state_t)
                else:
                    action_t = np.random.choice(env.enable_actions)

                # observe environment
                state_t_1, reward_t, terminal = env.step(action_t)
                total_reward += reward_t

                # store experience
                agent.store_experience(state_t, action_t, reward_t, state_t_1, terminal)
                print(agent.tmp_q_values, np.argmax(agent.tmp_q_values))

                # for log
                frame += 1
                steps += 1

                if steps > warmup:
                    loss += agent.current_loss
                    Q_max += np.max(agent.Q_values([state_t]))

            # experience replay
            # warmup中は学習しない
            if steps > warmup:
                agent.backword()

            print("epoch: {:03d}/{:03d} |  loss: {:.4f} | Q_max: {:.4f} | total reward: {} | steps: {}".format(
                e, n_epochs - 1, loss / frame, Q_max / frame, total_reward, steps))

    except KeyboardInterrupt:
        pass
    finally:
        # ブラウザーを終了する。
        env.driver.quit()  
        # save model
        agent.save_model()
