# WildfireGUI.py
import sys
import os
import math
import random
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPainter, QImage, QColor
from PyQt5.QtWidgets import (QComboBox, QLabel, QPushButton, QRadioButton,
                             QHBoxLayout, QVBoxLayout, QWidget, QApplication,
                             QMessageBox)
from TerrainMap import TerrainMap
from TerrainTile import TerrainTile

# Configuration
TERRAIN_TYPE_LIST = ['City', 'River', 'Forest', 'Dry Forest', 'Brush',
                     'Dry Brush', 'Grass', 'Dry Grass']
TILE_SIZE = 32
SCALE = 3
MENU_WIDTH = 120
PANEL_XPOS = 200
PANEL_YPOS = 100
MIN_GRID = 10
MAX_GRID = 20
SIM_SPEED = 160
DEFAULT_MAP = 'test_map.txt'
USER_FILE = 'user_custom.txt'
WINDOW_TITLE = 'Wildfire Simulator - Enhanced (Raster + Bresenham + AA)'

# Algorithms

def bresenham(x0, y0, x1, y1):
    """Bresenham line algorithm: yields integer (x,y) points from (x0,y0) to (x1,y1)."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 >= x0 else -1
    sy = 1 if y1 >= y0 else -1
    if dy <= dx:
        err = dx // 2
        while x != x1:
            yield x, y
            x += sx
            err -= dy
            if err < 0:
                y += sy
                err += dx
        yield x, y
    else:
        err = dy // 2
        while y != y1:
            yield x, y
            y += sy
            err -= dx
            if err < 0:
                x += sx
                err += dy
        yield x, y


def rasterize_circle_filled(img, cx, cy, radius, color):
    r = int(radius)
    w = img.width()
    h = img.height()
    for dx in range(-r, r + 1):
        max_y = int((r*r - dx*dx) ** 0.5)
        px = cx + dx
        if px < 0 or px >= w:
            continue
        for dy in range(-max_y, max_y + 1):
            py = cy + dy
            if 0 <= py < h:
                img.setPixel(px, py, color.rgba())


def rasterize_rect(img, x0, y0, x1, y1, color):
    w = img.width(); h = img.height()
    # clamp coordinates and draw rect by setting pixels
    x0c = max(0, x0); y0c = max(0, y0)
    x1c = min(w, x1); y1c = min(h, y1)
    for x in range(x0c, x1c):
        for y in range(y0c, y1c):
            img.setPixel(x, y, color.rgba())


class Ember():
    def __init__(self, path_pixels):
        self.path = path_pixels
        self.pos_index = 0
        self.alive = True

    def update(self, step=1):
        self.pos_index += step
        if self.pos_index >= len(self.path):
            self.alive = False

    def current_pos(self):
        if 0 <= self.pos_index < len(self.path):
            return self.path[self.pos_index]
        elif self.path:
            return self.path[-1]
        return (0, 0)

    def landing_pos(self):
        if self.path:
            return self.path[-1]
        return None


class WildfireGUI(QWidget):
    def __init__(self):
        super().__init__()
        # load map
        self.map = TerrainMap(DEFAULT_MAP)
        # UI / state
        self.click_action = 'Light Fire'
        self.tile_paint = 'city'
        self.tile_cache = {}
        self.embers = []
        # prepare images and UI
        self.prepare_tile_images()
        self.set_dimensions()
        self.create_menu()
        # timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(SIM_SPEED)
        self.show()

    def set_dimensions(self):
        size = len(self.map.grid)
        self.width = TILE_SIZE*size + MENU_WIDTH
        self.height = TILE_SIZE*size
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(PANEL_XPOS, PANEL_YPOS, self.width, self.height)

    def create_menu(self):
        main_layout = QHBoxLayout()
        spacer_layout = QHBoxLayout()
        spacer_layout.addStretch(1)
        menu_layout = QVBoxLayout()
        # click action radios
        menu_layout.addWidget(QLabel('Action on click:'))
        self.create_rb(menu_layout, 'Light Fire', True)
        self.create_rb(menu_layout, 'Paint Tile', False)
        # tile choose
        menu_layout.addWidget(QLabel('Tile to paint:'))
        tile_select = QComboBox()
        tile_select.addItems(TERRAIN_TYPE_LIST)
        tile_select.currentIndexChanged.connect(self.set_tile_paint)
        menu_layout.addWidget(tile_select)
        # new map size
        menu_layout.addWidget(QLabel('New map size:'))
        size_select = QComboBox()
        size_select.addItems([str(i) for i in range(MIN_GRID, MAX_GRID+1)])
        size_select.currentIndexChanged.connect(self.set_empty_map)
        menu_layout.addWidget(size_select)
        # save/load
        btn_save = QPushButton('Save', self)
        btn_save.clicked.connect(self.save)
        menu_layout.addWidget(btn_save)
        btn_load = QPushButton('Load', self)
        btn_load.clicked.connect(self.load)
        menu_layout.addWidget(btn_load)
        main_layout.addLayout(spacer_layout)
        main_layout.addLayout(menu_layout)
        self.setLayout(main_layout)

    def create_rb(self, layout, name, is_checked):
        radiobutton = QRadioButton(name)
        radiobutton.name = name
        radiobutton.setChecked(is_checked)
        radiobutton.toggled.connect(self.set_click_action)
        layout.addWidget(radiobutton)

    def set_click_action(self):
        self.click_action = self.sender().name

    def set_tile_paint(self, i):
        self.tile_paint = TERRAIN_TYPE_LIST[i].lower().replace(' ', '_')

    def set_empty_map(self, i):
        size = int(MIN_GRID + i)
        self.map = TerrainMap(None, size)
        self.set_dimensions()
        QMessageBox.information(self, "New Map", f"Created a new {size}x{size} map (filled with grass).")
        self.update()

    def save(self):
        # ensure maps folder exists
        os.makedirs(self.map.MAP_PATH, exist_ok=True)
        path = os.path.join(self.map.MAP_PATH, USER_FILE)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(str(self.map))
        QMessageBox.information(self, "Save Successful", f"Map saved to:\n{path}")

    def load(self):
        path = os.path.join(self.map.MAP_PATH, USER_FILE)
        if not os.path.exists(path):
            QMessageBox.warning(self, "Load Failed", f"No saved map found at:\n{path}")
            return
        self.map = TerrainMap(USER_FILE)
        self.set_dimensions()
        self.update()
        QMessageBox.information(self, "Load Successful", f"Map loaded from:\n{path}")

    def prepare_tile_images(self):
        """
        Create cached images for non-river tiles and a simple base for river.
        Rivers are drawn dynamically in paintEvent so they auto-rotate.
        """
        tile_px = TILE_SIZE
        big_px = tile_px * SCALE

        def make_base_image_for(terrain_key):
            big = QImage(big_px, big_px, QImage.Format_ARGB32)
            big.fill(Qt.transparent)

            if terrain_key == 'river':
                # simple base (not used for auto-rotated rivers)
                water_color = QColor(60, 130, 200, 255)
                rasterize_rect(big, 0, big_px//3, big_px, big_px*2//3, water_color)
            elif terrain_key in ('forest', 'dry_forest'):
                canopy = QColor(20, 120, 20, 255) if terrain_key == 'forest' else QColor(170, 140, 40, 255)
                trunk = QColor(90, 50, 30, 255)
                bg = QColor(30, 80, 30, 255) if terrain_key == 'forest' else QColor(120, 100, 60, 255)
                rasterize_rect(big, 0, 0, big_px, big_px, bg)
                radii = [big_px//3, big_px//4, big_px//5]
                centers = [(big_px//2, big_px//2 - big_px//8), (big_px//3, big_px//2), (big_px*2//3, big_px//2)]
                for (cx, cy), r in zip(centers, radii):
                    rasterize_circle_filled(big, cx, cy, r, canopy)
                rasterize_rect(big, big_px//2 - big_px//20, big_px*2//3, big_px//2 + big_px//20, big_px, trunk)
            elif terrain_key in ('brush', 'dry_brush'):
                bg = QColor(100, 140, 80, 255) if terrain_key == 'brush' else QColor(150, 130, 80, 255)
                rasterize_rect(big, 0, 0, big_px, big_px, bg)
                bush = QColor(20, 100, 20, 255) if terrain_key == 'brush' else QColor(140, 110, 60, 255)
                for _ in range(5):
                    cx = random.randint(big_px//10, big_px*9//10)
                    cy = random.randint(big_px//4, big_px*3//4)
                    r = random.randint(big_px//12, big_px//8)
                    rasterize_circle_filled(big, cx, cy, r, bush)
            elif terrain_key in ('grass', 'dry_grass'):
                bg = QColor(100, 180, 80, 255) if terrain_key == 'grass' else QColor(170, 150, 90, 255)
                rasterize_rect(big, 0, 0, big_px, big_px, bg)
            else:
                bg = QColor(150, 150, 150, 255)
                rasterize_rect(big, 0, 0, big_px, big_px, bg)
                block = QColor(110, 110, 110, 255)
                for rr in range(2):
                    for cc in range(2):
                        x0 = big_px//8 + cc*(big_px//2)
                        y0 = big_px//8 + rr*(big_px//2)
                        rasterize_rect(big, x0, y0, x0+big_px//3, y0+big_px//3, block)

            small = big.scaled(tile_px, tile_px, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            return small

        for key in ['city','river','forest','dry_forest','brush','dry_brush','grass','dry_grass']:
            base = make_base_image_for(key)
            # create burning and burnt overlays
            burning = QImage(base)
            qp = QPainter(burning)
            qp.setCompositionMode(QPainter.CompositionMode_SourceOver)
            qp.setBrush(QColor(255, 80, 20, 100))
            qp.setPen(Qt.NoPen)
            qp.drawEllipse(tile_px//8, tile_px//8, tile_px*3//4, tile_px*3//4)
            qp.end()
            burnt = QImage(base)
            qp2 = QPainter(burnt)
            qp2.fillRect(0, 0, tile_px, tile_px, QColor(0, 0, 0, 120))
            qp2.end()
            self.tile_cache[(key, 'normal')] = base
            self.tile_cache[(key, 'burning')] = burning
            self.tile_cache[(key, 'burnt')] = burnt

    def paintEvent(self, _):
        """
        Draw grid. Rivers are generated on the fly (auto-rotating) depending on neighbors.
        """
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing, True)

        size = len(self.map.grid)

        for row in range(size):
            for col in range(size):
                x = TILE_SIZE * col
                y = TILE_SIZE * row
                tile = self.map.grid[row][col]
                key = tile.type
                variant = 'normal'
                if tile.is_burning:
                    variant = 'burning'
                elif tile.is_burnt:
                    variant = 'burnt'

                # --- Auto-rotate rivers (draw dynamically) ---
                if key == 'river':
                    w, h = TILE_SIZE, TILE_SIZE
                    img = QImage(w, h, QImage.Format_ARGB32)
                    img.fill(Qt.transparent)
                    river_color = QColor(60, 130, 200, 255)

                    # check neighbors
                    up    = self.map.in_bounds(row-1, col) and self.map.grid[row-1][col].type == 'river'
                    down  = self.map.in_bounds(row+1, col) and self.map.grid[row+1][col].type == 'river'
                    left  = self.map.in_bounds(row, col-1) and self.map.grid[row][col-1].type == 'river'
                    right = self.map.in_bounds(row, col+1) and self.map.grid[row][col+1].type == 'river'

                    # draw river segments using rasterization (rects)
                    if up or down:
                        rasterize_rect(img, w//3, 0, 2*w//3, h, river_color)
                    if left or right:
                        rasterize_rect(img, 0, h//3, w, 2*h//3, river_color)

                    # T / cross handling: if multiple neighbors, both rects will draw and form junctions
                    # isolated -> default horizontal
                    if not (up or down or left or right):
                        rasterize_rect(img, 0, h//3, w, 2*h//3, river_color)

                    qp.drawImage(x, y, img)

                # --- Other terrains use cached images for speed ---
                else:
                    img = self.tile_cache.get((key, variant), self.tile_cache.get(('grass','normal')))
                    qp.drawImage(x, y, img)

                # Burning and burnt overlays
                if tile.is_burning:
                    qp.setPen(Qt.NoPen)
                    qp.setBrush(QColor(255, 200, 60, 120))
                    qp.drawEllipse(x + TILE_SIZE//4, y + TILE_SIZE//6, TILE_SIZE//2, TILE_SIZE//2)
                if tile.is_burnt:
                    qp.setPen(Qt.NoPen)
                    qp.setBrush(QColor(0, 0, 0, 80))
                    qp.drawRect(x, y, TILE_SIZE, TILE_SIZE)

        # draw embers on top
        qp.setPen(Qt.NoPen)
        for ember in list(self.embers):
            px, py = ember.current_pos()
            qp.setBrush(QColor(255, 160, 40, 220))
            qp.drawEllipse(px-2, py-2, 4, 4)

        qp.end()

    def mousePressEvent(self, QMouseEvent):
        col = int(QMouseEvent.pos().x()/TILE_SIZE)
        row = int(QMouseEvent.pos().y()/TILE_SIZE)
        if (self.map.in_bounds(row, col)):
            if self.click_action == 'Light Fire':
                tile = self.map.grid[row][col]
                if tile.resistance < 100:
                    tile.is_burning = True
            else:
                self.map.grid[row][col] = TerrainTile(self.tile_paint)
            self.update()

    def tick(self):
        size = len(self.map.grid)
        burning_positions = []
        for r in range(size):
            for c in range(size):
                if self.map.grid[r][c].is_burning:
                    burning_positions.append((r, c))
        for r, c in burning_positions:
            self.map.spread_fire(r, c)
            if random.random() < 0.08:
                self.spawn_ember_from(r, c)
        for r in range(size):
            for c in range(size):
                self.map.grid[r][c].burn()
        for ember in list(self.embers):
            ember.update(step=2)
            if not ember.alive:
                lx, ly = ember.landing_pos()
                tr = int(ly / TILE_SIZE)
                tc = int(lx / TILE_SIZE)
                if self.map.in_bounds(tr, tc):
                    self.map.grid[tr][tc].light()
                try:
                    self.embers.remove(ember)
                except ValueError:
                    pass
        self.update()

    def spawn_ember_from(self, row, col):
        size = len(self.map.grid)
        max_rdist = 3
        for _ in range(6):
            dr = random.randint(-max_rdist, max_rdist)
            dc = random.randint(-max_rdist, max_rdist)
            tr = row + dr
            tc = col + dc
            if self.map.in_bounds(tr, tc) and not (tr == row and tc == col):
                break
        else:
            return
        x0 = col * TILE_SIZE + TILE_SIZE//2
        y0 = row * TILE_SIZE + TILE_SIZE//2
        x1 = tc * TILE_SIZE + TILE_SIZE//2
        y1 = tr * TILE_SIZE + TILE_SIZE//2
        path = list(bresenham(x0, y0, x1, y1))
        if len(path) > 3:
            jittered = []
            for (px, py) in path:
                jittered.append((px + random.randint(-1,1), py + random.randint(-1,1)))
            path = jittered
        ember = Ember(path)
        self.embers.append(ember)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = WildfireGUI()
    sys.exit(app.exec_())