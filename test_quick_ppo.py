# Example usage
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env(max_number=10, max_steps=10):
    def _init():
        return GuessNumberEnv(max_number, max_steps)
    return _init

if __name__ == "__main__":
    from env.number import GuessNumberEnv
    from rl_quick_algorithm.ppo import run
    vec_env = DummyVecEnv([make_env() for _ in range(100)])
    
    # print(vec_env.reset())
    time = run(vec_env)
    
    print(f'total time = {time}')

    vec_env.close()