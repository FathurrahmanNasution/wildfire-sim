# TerrainMap.py
import math
import os
from TerrainTile import TerrainTile

class TerrainMap:
    MAP_PATH = os.path.join(os.getcwd(), 'maps') + os.sep

    def __init__(self, mapfile=None, size=0):
        """
        If mapfile given -> load from maps/<mapfile>.
        Else -> create an empty grass map of given size.
        """
        if mapfile is not None:
            path = os.path.join(self.MAP_PATH, mapfile)
            with open(path, 'r', encoding='utf-8') as f:
                tokens = f.read().split()
            size = int(math.sqrt(len(tokens)))
            self.grid = []
            for i in range(size):
                row = [TerrainTile(tokens[i*size + j]) for j in range(size)]
                self.grid.append(row)
        else:
            self.grid = []
            for i in range(size):
                self.grid.append([TerrainTile('grass') for _ in range(size)])

    def in_bounds(self, r, c):
        s = len(self.grid)
        return 0 <= r < s and 0 <= c < s

    def spread_fire(self, row, col):
        """If tile is burning, attempt to light 4-neighbors."""
        if self.grid[row][col].is_burning:
            adjacent = [(row-1, col), (row, col+1), (row+1, col), (row, col-1)]
            for r, c in adjacent:
                if self.in_bounds(r, c) and not self.grid[r][c].is_burning:
                    self.grid[r][c].light()

    def __str__(self):
        lines = []
        for row in self.grid:
            lines.append(' '.join(str(t) for t in row))
        return '\n'.join(lines)
