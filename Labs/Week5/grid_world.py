import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle

class GridWorld():
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    
    # interface for grid-world environments
    def __init__(self, rows, cols, agent_start=(0, 0), max_steps=200):
        # transitions are saturate cast
        self.rows = rows
        self.cols = cols
        self.agent_start = self.saturate_cast(agent_start)
        self.max_steps = max_steps
        self.action_space = [self.LEFT, self.RIGHT, self.UP, self.DOWN]
        
        self.is_reset = False
        self.agent = agent_start
        self.rewards = np.zeros(shape=(self.rows, self.cols))
        self.terminal = np.zeros(shape=(self.rows, self.cols), dtype=bool)
        
        # rendered reward precision in decimals
        self.r_prec = 0
    
    @classmethod
    def action_str(cls, action):
        if action == cls.LEFT:
            return "LEFT "
        elif action == cls.RIGHT:
            return "RIGHT"
        elif action == cls.UP:
            return "UP   "
        elif action == cls.DOWN:
            return "DOWN "
        else:
            return "UNKNOWN"
    
    def saturate_cast(self, pos):
        # saturate x = (x, y) to grid boundaries
        x, y = pos
        x = min(max(0, x), self.cols-1)
        y = min(max(0, y), self.rows-1)
        return (x, y)
    
    def update_terminal(self, x, y, make_terminal=True):
        # cell is set to terminal
        self.terminal[y][x] = make_terminal
        
    def update_reward(self, x, y, reward):
        self.rewards[y][x] = reward
        
    def print_trajectory(cls, states, actions, rewards):
        for i in range(len(actions)):
            print(f'{i}: {states[i]}, {cls.action_str(actions[i])} --> {states[i+1]}, {rewards[i]}')
    
    def reset(self):
        self.agent = self.agent_start
        self.is_reset = True
        self.n_steps = 0
        return self.agent
    
    def step(self, action):
        if not self.is_reset:
            print("Reset the environment before calling step.")
            return
        
        x, y = self.agent
        if action == self.LEFT:
            x -= 1
        elif action == self.RIGHT:
            x += 1
        elif action == self.UP:
            y += 1
        elif action == self.DOWN:
            y -= 1
        self.agent = self.saturate_cast((x, y))
        x, y = self.agent
        reward = self.rewards[y][x]
        self.n_steps += 1
        done = self.terminal[y][x] or self.n_steps >= self.max_steps
        if done:
            self.is_reset = False
        return self.agent, reward, done, {}
    
    def _create_fig(self, x=10, y=8):
        """generates figure with square grid cells within (x, y) figure size.
        
        Args:
            x (float): size of figure along x axis
            y (float): size of figure along y axis
        
        Returns:
            fig, ax
        """
        w = min(x/self.cols, y/self.rows)
        fig, ax = plt.subplots(1, 1, figsize=(w*self.cols, w*self.rows))
        
        return fig, ax
    
    def render_grid(self, ax):
        for i in range(self.rows-1):
            ax.plot([-0.5, self.cols+0.5], [i+0.5, i+0.5], color='gray', zorder=0)
        for i in range(self.cols-1):
            ax.plot([i+0.5, i+0.5], [-0.5, self.rows+0.5], color='gray', zorder=0)
            
        # layout
        ax.set_ylim([-0.5, self.rows-0.5])
        ax.set_xlim([-0.5, self.cols-0.5])
        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows)) 
    
    def render_terminal(self, ax):
         # plot terminal and rewards
        for y in range(self.rows):
            for x in range(self.cols):
                if self.terminal[y, x]:
                    ax.add_artist(Rectangle(xy=(x-0.5, y-0.5), width=1, height=1, zorder=-1, color='lightgrey'))

    def render_agent(self, ax, agent=None):
        if agent is None:
            agent = self.agent
        
        ax.add_artist(Circle(xy=agent, radius=0.45, color='green', zorder=3))
    
    def render_action(self, ax, action, agent=None, length=0.4, hw=0.15, hl=0.2, ox=0.2, oy=0.2):
        x, y = self.agent if agent is None else agent
        
        dx, dy = (0, 0)
        if action == self.LEFT:
            dx -= 1
        elif action == self.RIGHT:
            dx += 1
        elif action == self.UP:
            dy += 1
        elif action == self.DOWN:
            dy -= 1
        ax.arrow(x+ox*dx, y+oy*dy, dx*length, dy*length, head_width=hw, head_length=hl, linewidth=2, fc='k', ec='k', zorder=4)

        
    def render_rollout(self, s, a=None, r=None, show_reward=False, jitter=None, ax=None):
        
        if ax is None:
            fig, ax = env._create_fig()
        self.render_grid(ax)
        pos = np.array([np.array(x) for x in s]).astype(np.float64)
        if jitter is None:
            pos += 0.2*(np.random.uniform(size=(len(s),2))-0.5)
        else:
            pos += jitter[:len(s),:]
        ax.plot(pos[:,0], pos[:,1], zorder=4, color='k')
        ax.add_artist(Circle(xy=s[0], radius=0.45, color='gray', zorder=3))
        ax.add_artist(Circle(xy=s[-1], radius=0.45, color='green', zorder=3))
        if a is not None:
            self.render_action(ax, action=a, agent=s[-1])
        if show_reward:
            x,y = s[-1]
            reward_string = f'{r:.0f}' if r < 0 else f'+{r:.0f}'
            ax.text(x=x, y=y, s=reward_string, va='center', ha='center', size=150/max(self.rows, self.cols), zorder=4, color='white')

    def animate_rollout(self, s, a, r):
        # add jitter first
        jitter = 0.2*(np.random.uniform(size=(len(s),2))-0.5)
        env.render_rollout([s[0]], a[0], r=None, show_reward=False, jitter=jitter)
        plt.savefig(f'partial_rollout_0.png', dpi=DPI)
        for i in range(len(s)-2):
            env.render_rollout(s[:i+2], a[i+1], r[i], show_reward=True, jitter=jitter)
            plt.savefig(f'partial_rollout_{i+1}.png', dpi=DPI)
        env.render_rollout(s, a=None, r=r[-1], show_reward=True, jitter=jitter)
        plt.savefig(f'partial_rollout_{len(s)}.png', dpi=DPI)
            
    def render_reward(self, ax, x, y, color='k', zorder=2):
        ax.text(x=x, y=y, s=f'{self.rewards[y][x]:.{self.r_prec}f}', va='center', ha='center', size=150/max(self.rows, self.cols), zorder=zorder, color=color)
            
    def render_rewards(self, ax):
        # plot terminal and rewards
        for y in range(self.rows):
            for x in range(self.cols):
                self.render_reward(ax, x, y)
                               
    def render(self, agent=None, action=None, 
               save=False, fig=None, ax=None):
        
        fig, ax = self._create_fig()
        self.render_grid(ax)
        self.render_terminal(ax)
        self.render_rewards(ax)
        self.render_agent(ax, agent=agent)
        if action is not None:
            self.render_action(ax, action=action, agent=agent)
        if save:
            plt.savefig(f'{int(time.time())}.png', dpi=1200)