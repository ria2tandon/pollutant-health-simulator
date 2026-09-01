import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch


# Resident class class



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

        # RESET STATE FOR EACH NEW RUN
        self.pollution_grid_low = np.zeros((self.grid_size, self.grid_size))
        self.pollution_grid_high = np.zeros((self.grid_size, self.grid_size))
        self.factories_low = []
        self.factories_high = []

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



    def steps(self):
        self.time_step += 1
        # Pollution keeps diffusing
        self.diffuse_pollution()
          # Occupied locations -> less realistic
        occ_low = set()
        occ_high = set()
        for a in self.agents_low:
            if self.time_step != 1:
                a.update_state(self.pollution_grid_low, self.grid_size)
            a.move(self.grid_size, self.pollution_grid_low, occ_low)

        for a in self.agents_high:
            if self.time_step != 1:
                a.update_state(self.pollution_grid_high, self.grid_size)
            a.move(self.grid_size, self.pollution_grid_high, occ_high)

    # Heatmap
    def visualize_results(self, avg_stats):
        matrix = np.zeros((5, 3))

        # Each square is low-high
        for i, bias in enumerate([0.5, 0.6, 0.7, 0.8, 0.9]):
            matrix[i, 0] = avg_stats[bias]['low_sick'] - avg_stats[bias]['high_sick']
            matrix[i, 1] = avg_stats[bias]['low_chronic'] - avg_stats[bias]['high_chronic']
            matrix[i, 2] = avg_stats[bias]['low_deceased'] - avg_stats[bias]['high_deceased']

        # Colorblind friendly
        fig, ax = plt.subplots(figsize=(10, 6))
        cax = ax.imshow(matrix, cmap='coolwarm', aspect='auto', interpolation='nearest', vmin=-20, vmax=20)
        fig.colorbar(cax)

        # Formatting, indicate low-high
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Sick', 'Chronic', 'Dead'])
        ax.set_yticks(range(5))
        ax.set_yticklabels([f'{bias}' for bias in [0.5, 0.6, 0.7, 0.8, 0.9]])
        ax.set_xlabel('Agent State')
        ax.set_ylabel('Pollution Bias')
        ax.set_title('Differences in Sick, Chronically Ill, and Dead Patients by SES (Low - High)')
        plt.tight_layout()
        plt.show()

    def get_stats(self):
        stats = {
            'time_step': self.time_step,
            'healthy_low': 0, 'sick_low': 0, 'chronic_low': 0, 'deceased_low': 0,  # low SES values
            'healthy_high': 0, 'sick_high': 0, 'chronic_high': 0, 'deceased_high': 0,  # high SES values
            'total_low': len(self.agents_low),
            'total_high': len(self.agents_high),
        }

        # Count the states of the low SES agents
        for a in self.agents_low:
            stats[f"{a.state}_low"] += 1

        # Count the states of the high SES agents
        for a in self.agents_high:
            stats[f"{a.state}_high"] += 1

        return stats

def run_multiple_runs():
    runs = [0.5, 0.6, 0.7, 0.8, 0.9]
    avg_stats = {
        bias: {'low_sick': 0, 'high_sick': 0,
               'low_chronic': 0, 'high_chronic': 0,
               'low_deceased': 0, 'high_deceased': 0}
        for bias in runs
    }

    num_steps = 50
    num_reps = 5

    for bias in runs:
        for _ in range(num_reps):
            model = CityModel(grid_size=50, num_agents=1000, pollution_bias=bias)

            # cumulative sums over time
            total_stats = {'sick_low': 0, 'sick_high': 0,
                           'chronic_low': 0, 'chronic_high': 0,
                           'deceased_low': 0, 'deceased_high': 0}

            for _ in range(num_steps):
                model.steps()
                stats = model.get_stats()

                total_stats['sick_low']     += stats['sick_low']
                total_stats['sick_high']    += stats['sick_high']
                total_stats['chronic_low']  += stats['chronic_low']
                total_stats['chronic_high'] += stats['chronic_high']
                total_stats['deceased_low'] += stats['deceased_low']
                total_stats['deceased_high']+= stats['deceased_high']

            # Average across time for this single run
            for key in total_stats:
                total_stats[key] /= num_steps

            # Add to overall average across repetitions
            avg_stats[bias]['low_sick']     += total_stats['sick_low']
            avg_stats[bias]['high_sick']    += total_stats['sick_high']
            avg_stats[bias]['low_chronic']  += total_stats['chronic_low']
            avg_stats[bias]['high_chronic'] += total_stats['chronic_high']
            avg_stats[bias]['low_deceased'] += total_stats['deceased_low']
            avg_stats[bias]['high_deceased']+= total_stats['deceased_high']

    # Average across the number of repetitions
    for bias in runs:
        for key in avg_stats[bias]:
            avg_stats[bias][key] /= num_reps

    # Visualize final averaged differences
    model.visualize_results(avg_stats)


# # Get average from 5 runs for each level
# def run_multiple_runs():

#     model = CityModel(grid_size=50, num_agents=300, pollution_bias=0.8)
#     runs = [0.5, 0.6, 0.7, 0.8, 0.9]
#     avg_stats = {run: {'low_sick': 0, 'high_sick': 0, 'low_chronic': 0, 'high_chronic': 0, 'low_deceased': 0, 'high_deceased': 0} for run in runs}

#     for bias in runs:
#         for _ in range(5):
#             model = CityModel(grid_size=50, num_agents=300, pollution_bias=bias)
#             for _ in range(50):
#                 model.steps()
#             stats = model.get_stats()

#             # Add stats to total
#             avg_stats[bias]['low_sick'] += stats['sick_low']
#             avg_stats[bias]['high_sick'] += stats['sick_high']
#             avg_stats[bias]['low_chronic'] += stats['chronic_low']
#             avg_stats[bias]['high_chronic'] += stats['chronic_high']
#             avg_stats[bias]['low_deceased'] += stats['deceased_low']
#             avg_stats[bias]['high_deceased'] += stats['deceased_high']

#     for bias in runs:
#         # Get avg
#         avg_stats[bias]['low_sick'] /= 5
#         avg_stats[bias]['high_sick'] /= 5
#         avg_stats[bias]['low_chronic'] /= 5
#         avg_stats[bias]['high_chronic'] /= 5
#         avg_stats[bias]['low_deceased'] /= 5
#         avg_stats[bias]['high_deceased'] /= 5

#     model.visualize_results(avg_stats)

if __name__ == "__main__":
    run_multiple_runs()
