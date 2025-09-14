# terrain.py
# Fixed TerrainTile + TerrainMap for 3D wildfire simulator
import math, os, random

TERRAIN_TYPES = {
    'city':      (0,   5000, 35),
    'river':     (0,   0,    100),
    'forest':    (1000,10000,20),
    'dry_forest':(200, 10000,5),
    'brush':     (500, 2500, 20),
    'dry_brush': (0,   2500, 0),
    'grass':     (500, 500,  20),
    'dry_grass': (0,   500,  0),
    'stone':     (0,   0,    100),
    'swamp':     (2000,3000,  60)
}

class TerrainTile:
    def __init__(self, ttype='grass'):
        self.type = ttype
        self.set_params(*TERRAIN_TYPES.get(ttype, TERRAIN_TYPES['grass']))

    def set_params(self, moisture, material, resistance):
        self.is_burning = False
        self.is_burnt = False
        self.moisture = moisture
        self.material = material
        self.resistance = resistance

    def burn(self):
        if self.is_burning:
            if self.material >= 100:
                self.material -= 100
            else:
                self.material = 0
                self.is_burning = False
                self.is_burnt = True
                self.resistance = 100

    def light(self):
        if not self.is_burning and not self.is_burnt:
            roll = random.randint(0, 100)
            if roll > self.resistance:
                if self.moisture >= 100:
                    self.moisture -= 100
                else:
                    self.moisture = 0
                    self.is_burning = True
                    return True
        return False

    def __str__(self):
        return self.type

class TerrainMap:
    MAP_PATH = os.path.join(os.getcwd(), 'maps') + os.sep

    def __init__(self, mapfile=None, size=0):
        if mapfile is not None:
            path = os.path.join(self.MAP_PATH, mapfile)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Handle both space-separated and newline-separated formats
            if '\n' in content:
                lines = content.split('\n')
                self.grid = []
                for line in lines:
                    if line.strip():
                        row_tokens = line.strip().split()
                        row = [TerrainTile(token) for token in row_tokens]
                        self.grid.append(row)
            else:
                # Space-separated format - try to determine grid size
                tokens = content.split()
                total_tokens = len(tokens)
                size = int(math.sqrt(total_tokens))
                
                # Verify it's actually a square
                if size * size != total_tokens:
                    # If not square, try to make it square by padding or truncating
                    print(f"Warning: Map is not square ({total_tokens} tokens). Creating {size}x{size} grid.")
                    tokens = tokens[:size*size]  # Truncate if too many
                    while len(tokens) < size*size:  # Pad if too few
                        tokens.append('grass')
                
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