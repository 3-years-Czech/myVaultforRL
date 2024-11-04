import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GuessNumberEnv(gym.Env):
    """
    A simple environment for a guess-the-number game using Gymnasium.
    
    The goal is to guess a randomly generated number within a specified range.
    The action space is discrete, representing the possible guesses.
    The observation space is converted to a Box space for compatibility.
    The reward is +1 for a correct guess, -1 for an incorrect guess, and 0 for other actions.
    The episode always runs for a fixed number of steps, even if the correct number is guessed earlier.
    """
    
    def __init__(self, max_number=10, max_steps=10):
        super(GuessNumberEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Discrete(max_number)  # Actions are numbers from 0 to max_number-1
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([max_number-1, 2]), dtype=np.float32)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        
        # Environment parameters
        self.max_number = max_number
        self.max_steps = max_steps
        self.target_number = None
        self.current_step = 0
        self.correct_guess_step = None  # Step at which the correct guess was made
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        """
        super().reset(seed=seed)
        self.target_number = self.np_random.integers(0, self.max_number)
        self.current_step = 0
        self.correct_guess_step = None
        return np.array([0.0, 0.0])  # Initial observation and info
    
    def step(self, action):
        """
        Execute one time step within the environment.
        """
        self.current_step += 1
        reward = 0
        done = False
        
        if action < self.target_number:
            feedback = 0  # Too low
            reward = -1
        elif action > self.target_number:
            feedback = 2  # Too high
            reward = -1
        else:
            feedback = 1  # Correct
            reward = 1
            if self.correct_guess_step is None:
                self.correct_guess_step = self.current_step
        
        # Convert (current_guess, feedback) to a Box value
        observation = np.array([action, feedback], dtype=np.float32)
        
        # Episode always runs for a fixed number of steps
        if self.current_step >= self.max_steps:
            done = True
        
        return observation, reward, done, False, {}  # Observation, reward, done, truncated, info
    
    def render(self, mode='human'):
        """
        Render the environment to the screen.
        """
        pass
    
    def close(self):
        """
        Close the environment.
        """
        pass

def opt_method(obs):
    import math
    action, feedback = obs
    if feedback == 1:
        return action
    if feedback == 0:
        return round((action+10.0)/2.0)
    if feedback == 2:
        return math.floor(action/2.0)

# Example usage
if __name__ == "__main__":
    env = GuessNumberEnv(max_number=10, max_steps=10)
    
    # Reset the environment
    rew_list = []
    for i in range(1000):
        observation = env.reset()
        rew = 0
        # Run a few steps
        for _ in range(env.max_steps):
            action = opt_method(observation)
            observation, reward, done, truncated, info = env.step(action)
            rew += reward
        rew_list.append(rew)
        
    print(sum(rew_list)/len(rew_list))
    # Close the environment
    env.close()