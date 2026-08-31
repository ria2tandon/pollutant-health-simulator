import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch


# Resident class class
class Agent:
    def __init__(self, id, x, y, ses, mobility):
        self.id = id
        self.x = x  # Coordinate x and y values
        self.y = y
        self.ses = ses  # 'low' or 'high'
        self.mobility = mobility  # 0–1 value, basically percent chance they have to be able to move
        self.state = 'healthy'  # 'healthy','sick','chronic','deceased'

    # Update agent state based on pollution
    def update_state(self, pollution_grid, grid_size):
        if self.state == 'deceased':
            return

        # Calculate pollution in Moore neighborhood (3x3)
        pollution_val = 0
        max_pollution = 0
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = int(self.x + dx), int(self.y + dy)
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    cell_pollution = pollution_grid[nx, ny]
                    pollution_val += cell_pollution
                    max_pollution = max(max_pollution, cell_pollution)
                    count += 1

        avg_pollution = pollution_val / max(count, 1)

        # Compound effect so it affects agents non-linearly
        pollution_effect = min(1.0, max_pollution)  # worst nearby cell
        compound_effect = pollution_effect ** 3

        # Health impacts can only happen above pollution threshold
        p = pollution_grid[self.y, self.x]
        if p > 0.05:
            if self.state == 'healthy':
                sickness_prob = p * 3
                if random.random() < sickness_prob:
                    self.state = 'sick'

            elif self.state == 'sick':
                death_prob = 0.4 * compound_effect / 3

                # Recovery is harder in polluted areas
                recovery_chance = 0.7 * (1 - compound_effect)

                # Either recover, die, or become chronic
                if random.random() < recovery_chance:
                    self.state = 'healthy'
                elif random.random() < death_prob:
                    self.state = 'deceased'
                else:
                    self.state = 'chronic'

            elif self.state == 'chronic':
                # Can't become healthy again
                death_prob = 0.05 * compound_effect / 3
                if random.random() < death_prob:
                    self.state = 'deceased'

    def move(self, grid_size, pollution_grid, occupied_positions):
      if self.state == 'deceased':
          return

      original_pos = (self.x, self.y)
      moved = False

      if random.random() < self.mobility:
          current_pollution = pollution_grid[int(self.y), int(self.x)]
          possible_moves = []

          for dx in [-1, 0, 1]:
              for dy in [-1, 0, 1]:
                  if dx == 0 and dy == 0:  # Ignore current location
                      continue

                  new_x = int(self.x + dx)
                  new_y = int(self.y + dy)

                  # Ensure new_x and new_y are within bounds
                  if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
                      # Check if position is occupied (only considering the set of occupied positions)
                      if (new_x, new_y) not in occupied_positions:
                          # Check pollution levels before moving
                          move_pollution = pollution_grid[new_y, new_x]
                          if self.ses == 'high' and move_pollution < current_pollution:
                              possible_moves.append((new_x, new_y, move_pollution))
                          elif self.ses == 'low' and move_pollution < current_pollution:
                              possible_moves.append((new_x, new_y, move_pollution))

          if possible_moves:
              pollution_values = [x[2] for x in possible_moves]
              min_pollution_index = pollution_values.index(min(pollution_values))
              best_x, best_y, _ = possible_moves[min_pollution_index]

              # Only remove original position from occupied positions if it's present
              if original_pos in occupied_positions:
                  occupied_positions.remove(original_pos)

              # Add the new position to the occupied positions
              if (best_x, best_y) not in occupied_positions:
                  self.x, self.y = best_x, best_y
                  moved = True

          if moved:
              occupied_positions.add((self.x, self.y))
          else:
              occupied_positions.add(original_pos)
      else:
          occupied_positions.add(original_pos)




# create city class:
class CityModel:
    def __init__(self, grid_size=50, num_agents=600, pollution_bias=0.8):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.pollution_bias = pollution_bias
        self.agents_low = []
        self.agents_high = []
        self.pollution_grid_low = np.zeros((grid_size, grid_size))
        self.pollution_grid_high = np.zeros((grid_size, grid_size))
        self.time_step = 0
        self.cmap = 'binary'
        self.agent_positions = set()  # Rried spatial hash for agent positions
        self.factories_low = []
        self.factories_high = []
        LinearSegmentedColormap.from_list('pollution', ['white', 'black'])
        self.initialize_model()

    def initialize_model(self):
        self.zones = np.random.choice(
            ['industrial', 'residential', 'green'],
            size=(self.grid_size, self.grid_size),
            p=[0.1, 0.5, 0.4]
        )

        # Splits number of factories into low and high SES grids
        industrial_cells = [tuple(c) for c in np.argwhere(self.zones == 'industrial')]
        num_factories = int(0.01 * self.grid_size ** 2)  # 25 factories total
        n_low = int(self.pollution_bias * num_factories)
        n_high = num_factories - n_low

        # Assign factories for low SES
        low_choices = random.sample(industrial_cells, k=n_low)
        for fx, fy in low_choices:
            self.factories_low.append((fx, fy))
            self.pollution_grid_low[fx, fy] = 1.0  # Pollution at factory location is full

        # Assign factories for high SES
        high_choices = random.sample(industrial_cells, k=n_high)
        for fx, fy in high_choices:
            self.factories_high.append((fx, fy))
            self.pollution_grid_high[fx, fy] = 1.0  # ^

        # Create agents at initial positions
        for i in range(self.num_agents):
            if i % 2 == 0:  # low SES -> every other one
                x = random.randint(0, self.grid_size - 1)
                y = random.randint(0, self.grid_size - 1)
                mob = random.uniform(0.0, 0.05)  # lower mobility
                self.agents_low.append(Agent(i, x, y, 'low', mob))
                # Add agent to spatial hash (by their position)
                self.agent_positions.add((x, y))
            else:  # high SES
                x = random.randint(0, self.grid_size - 1)
                y = random.randint(0, self.grid_size - 1)
                mob = random.uniform(0.1, 0.5)  # higher mobility
                self.agents_high.append(Agent(i, x, y, 'high', mob))
                # Add agent to spatial hash (by their position)
                self.agent_positions.add((x, y))

    def diffuse_pollution(self):
      # Sets of polluted cells (factories and affected cells)
      polluted_cells_low = set(self.factories_low)
      polluted_cells_high = set(self.factories_high)

      # Add neighboring cells to the polluted set (for diffusion)
      for fx, fy in self.factories_low:
          for dx in range(-2, 3):  # Expanding diffusion to 5x5 neighborhood (larger than Moore 3x3)
              for dy in range(-2, 3):
                  nx, ny = fx + dx, fy + dy
                  if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                      polluted_cells_low.add((nx, ny))

      for fx, fy in self.factories_high:
          for dx in range(-2, 3):  # Expanding diffusion to 5x5 neighborhood (larger than Moore 3x3)
              for dy in range(-2, 3):
                  nx, ny = fx + dx, fy + dy
                  if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                      polluted_cells_high.add((nx, ny))

      # Diffuse pollution only in these affected cells
      new_low = np.zeros_like(self.pollution_grid_low)
      new_high = np.zeros_like(self.pollution_grid_high)

      # Process only the affected cells (now including a larger neighborhood)
      for x, y in polluted_cells_low:
          if (x, y) in self.factories_low:
              new_low[x, y] = 1.0  # Highest pollution at factory location
          elif (x, y) in polluted_cells_low:
              total = 0
              cnt = 0
              # Apply a medium pollution level for 3x3 neighborhood
              for dx in range(-1, 2):  # 3x3 neighborhood around the factory
                  for dy in range(-1, 2):
                      nx, ny = x + dx, y + dy
                      if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                          total += self.pollution_grid_low[nx, ny]
                          cnt += 1
              # Apply medium pollution value in surrounding area
              new_low[x, y] = total / cnt * 0.7  # Medium pollution

          else:  # For 5x5 area
              total = 0
              cnt = 0
              for dx in range(-2, 3):  # Larger 5x5 neighborhood
                  for dy in range(-2, 3):
                      nx, ny = x + dx, y + dy
                      if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                          total += self.pollution_grid_low[nx, ny]
                          cnt += 1
              # Apply low pollution value in distant area
              new_low[x, y] = total / cnt * 0.4  # Low pollution value

      # Apply the same logic for high SES pollution diffusion
      for x, y in polluted_cells_high:
          if (x, y) in self.factories_high:
              new_high[x, y] = 1.0  # Highest pollution at factory location
          elif (x, y) in polluted_cells_high:
              total = 0
              cnt = 0
              # Apply a medium pollution level for 3x3 neighborhood
              for dx in range(-1, 2):  # 3x3 neighborhood around the factory
                  for dy in range(-1, 2):
                      nx, ny = x + dx, y + dy
                      if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                          total += self.pollution_grid_high[nx, ny]
                          cnt += 1
              # Apply medium pollution value in surrounding area
              new_high[x, y] = total / cnt * 0.7  # Medium pollution

          else:  # For 5x5 area
              total = 0
              cnt = 0
              for dx in range(-2, 3):  # Larger 5x5 neighborhood
                  for dy in range(-2, 3):
                      nx, ny = x + dx, y + dy
                      if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                          total += self.pollution_grid_high[nx, ny]
                          cnt += 1
              # Apply low pollution value in distant area
              new_high[x, y] = total / cnt * 0.4  # Low pollution value

      # Update the pollution grids with the newly computed values
      self.pollution_grid_low = new_low
      self.pollution_grid_high = new_high


    def step(self):
        self.time_step += 1
        # Pollution keeps diffusing
        self.diffuse_pollution()
        # Occupied locations -> less realistic
        occ_low = set()
        occ_high = set()
        for a in self.agents_low:
            if (self.time_step != 1):
                a.update_state(self.pollution_grid_low, self.grid_size)
            a.move(self.grid_size, self.pollution_grid_low, occ_low)
        for a in self.agents_high:
            if (self.time_step != 1):
                a.update_state(self.pollution_grid_high, self.grid_size)
            a.move(self.grid_size, self.pollution_grid_high, occ_high)


    def visualize(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6)) # Makes two grids initially
        fig.suptitle(f"City Model – Time Step #{self.time_step}", fontsize=16)

        # Scatter agents
        ax1.imshow(self.pollution_grid_low, cmap=self.cmap, alpha=1, extent=(0, self.grid_size, 0, self.grid_size))
        ax1.set_title("Low-SES Area")
        for a in self.agents_low:
            c = {'healthy': 'green', 'sick': 'red', 'chronic': 'orange', 'deceased': 'darkred'}[a.state]
            ax1.scatter(a.x + 0.5, self.grid_size - (a.y + 0.5), color=c, marker='o', s=40, edgecolors='black')

        ax2.imshow(self.pollution_grid_high, cmap=self.cmap, alpha=1, extent=(0, self.grid_size, 0, self.grid_size))
        ax2.set_title("High-SES Area")
        for a in self.agents_high:
            c = {'healthy': 'green', 'sick': 'red', 'chronic': 'orange', 'deceased': 'darkred'}[a.state]
            ax2.scatter(a.x + 0.5, self.grid_size - (a.y + 0.5), color=c, marker='o', s=40, edgecolors='black')

        # Legend
        legend_elems = [
            Patch(facecolor='green', label='Healthy'),
            Patch(facecolor='red', label='Sick'),
            Patch(facecolor='orange', label='Chronic'),
            Patch(facecolor='darkred', label='Deceased'),
            plt.Line2D([0], [0], marker='o', color='w', label='Resident',
                       markerfacecolor='white', markersize=10, markeredgecolor='black'),
        ]
        ax2.legend(handles=legend_elems, loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # At the end of 50 steps, output stats for ease
    def get_stats(self):
        stats = {
            'time_step': self.time_step,
            'healthy_low': 0, 'sick_low': 0, 'chronic_low': 0, 'deceased_low': 0,  # low ses values
            'healthy_high': 0, 'sick_high': 0, 'chronic_high': 0, 'deceased_high': 0,  # high ses values
            'total_low': len(self.agents_low),
            'total_high': len(self.agents_high),
        }
        for a in self.agents_low:
            stats[f"{a.state}_low"] += 1
        for a in self.agents_high:
            stats[f"{a.state}_high"] += 1
        return stats

if __name__ == "__main__":
    model = CityModel(grid_size=50, num_agents=600, pollution_bias=0.8)
    n = 50  # number of timesteps, keep it low (50 takes 3 minutes)
    for i in range(n):
        model.step()
        if i == 0 or (i + 1) % 5 == 0:  # Reduce frequency of visualization
            model.visualize()

    stats = model.get_stats()
    print("\nFinal Statistics:")
    for k, v in stats.items():
        print(f"{k}: {v}")
