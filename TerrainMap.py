# TerrainMap.py
import math
import os
from TerrainTile import TerrainTile

class TerrainMap():
    MAP_PATH = os.path.join(os.getcwd(), 'maps') + os.sep

    def __init__(self, mapfile=None, size=0):
        if mapfile is not None:
            path = os.path.join(self.MAP_PATH, mapfile)
            with open(path) as f:
                tiles = f.read().split()
            size = int(math.sqrt(len(tiles)))
            self.grid = []
            for i in range(size):
                self.grid.append([TerrainTile(tiles[i*size + j]) for j in range(size)])
        else:
            self.grid = []
            for i in range(size):
                self.grid.append([TerrainTile('grass') for j in range(size)])

    def spread_fire(self, row, col):
        if self.grid[row][col].is_burning:
            adjacent = [(row-1, col), (row, col+1), (row+1, col), (row, col-1)]
            for r, c in adjacent:
                if self.in_bounds(r, c) and not self.grid[r][c].is_burning:
                    self.grid[r][c].light()

    def in_bounds(self, row, col):
        size = len(self.grid)
        return 0 <= row < size and 0 <= col < size

    def __str__(self):
        s = ''
        for row in range(len(self.grid)):
            for col in range(len(self.grid)):
                s += str(self.grid[row][col]) + ' '
        s += '\n'
        return s
