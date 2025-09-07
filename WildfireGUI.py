import sys
import os
import random
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtCore import QTimer, Qt, QPoint
from PyQt5.QtGui import QPainter, QImage, QColor
from PyQt5.QtWidgets import (QComboBox, QLabel, QPushButton, QRadioButton,
                             QHBoxLayout, QVBoxLayout, QWidget, QApplication,
                             QMessageBox)
from TerrainMap import TerrainMap
from TerrainTile import TerrainTile

# Config
TERRAIN_TYPE_LIST = ['City','River','Forest','Dry Forest','Brush',
                     'Dry Brush','Grass','Dry Grass','Stone','Swamp']
TILE_SIZE = 32
SCALE = 3
MENU_WIDTH = 140
PANEL_XPOS = 200
PANEL_YPOS = 100
MIN_GRID = 10
MAX_GRID = 20
SIM_SPEED = 160
DEFAULT_MAP = 'test_map.txt'
USER_FILE = 'user_custom.txt'
WINDOW_TITLE = 'Wildfire Simulator'

# ----------------- Basic Algorithms -----------------

def bresenham(x0, y0, x1, y1):
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

def rasterize_rect_to_image(img, x0, y0, x1, y1, color):
    w = img.width(); h = img.height()
    sx = max(0, x0); ex = min(w, x1)
    sy = max(0, y0); ey = min(h, y1)
    c = color.rgba()
    for x in range(sx, ex):
        for y in range(sy, ey):
            img.setPixel(x, y, c)

def rasterize_circle_filled_to_image(img, cx, cy, radius, color):
    r = int(radius)
    w = img.width(); h = img.height()
    c = color.rgba()
    for dx in range(-r, r+1):
        max_y = int((r*r - dx*dx) ** 0.5)
        px = cx + dx
        if px < 0 or px >= w: continue
        for dy in range(-max_y, max_y+1):
            py = cy + dy
            if 0 <= py < h:
                img.setPixel(px, py, c)

def rasterize_polygon_scanline(img, vertices, color):
    if not vertices: return
    xs = [p[0] for p in vertices]; ys = [p[1] for p in vertices]
    ymin = max(0, min(ys)); ymax = min(img.height()-1, max(ys))
    c = color.rgba()
    n = len(vertices)
    for y in range(ymin, ymax+1):
        nodes = []
        j = n - 1
        for i in range(n):
            xi, yi = vertices[i]; xj, yj = vertices[j]
            if (yi < y and yj >= y) or (yj < y and yi >= y):
                if yj == yi:
                    xint = xi
                else:
                    xint = int(xi + (y - yi) * (xj - xi) / (yj - yi))
                nodes.append(xint)
            j = i
        nodes.sort()
        for k in range(0, len(nodes), 2):
            if k+1 < len(nodes):
                a = max(0, nodes[k]); b = min(img.width()-1, nodes[k+1])
                for x in range(a, b+1):
                    img.setPixel(x, y, c)

# ----------------- Renderer -----------------

class PrimitiveRenderer:
    def __init__(self, tile_px=TILE_SIZE, scale=SCALE):
        self.tile_px = tile_px
        self.big_px = tile_px * scale
        self.scale = scale
        self.cache = {}
        self._make_all()

    def _make_all(self):
        keys = ['city','river','forest','dry_forest','brush','dry_brush','grass','dry_grass','stone','swamp']
        for k in keys:
            if k == 'river':
                for mask in range(16):
                    img = self._make_river_big(mask)
                    small = img.scaled(self.tile_px, self.tile_px, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                    self.cache[('river', mask)] = small
                continue
            big = self._make_big_for(k)
            small = big.scaled(self.tile_px, self.tile_px, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.cache[(k, 'normal')] = small
            burning = QImage(small)
            qp = QPainter(burning)
            qp.setCompositionMode(QPainter.CompositionMode_SourceOver)
            qp.setBrush(QColor(255, 80, 20, 110))
            qp.setPen(Qt.NoPen)
            qp.drawEllipse(self.tile_px//8, self.tile_px//8, self.tile_px*3//4, self.tile_px*3//4)
            qp.end()
            burnt = QImage(small)
            qp2 = QPainter(burnt)
            qp2.fillRect(0, 0, self.tile_px, self.tile_px, QColor(0,0,0,140))
            qp2.end()
            self.cache[(k, 'burning')] = burning
            self.cache[(k, 'burnt')] = burnt

    def _make_big_for(self, k):
        img = QImage(self.big_px, self.big_px, QImage.Format_ARGB32)
        img.fill(Qt.transparent)

        # colors per terrain
        if k == 'city':
            bg = QColor(180, 180, 180)
            rasterize_rect_to_image(img, 0, 0, self.big_px, self.big_px, bg)

            block = QColor(130, 130, 130)
            shadow = QColor(90, 90, 90)

            # 2x2 blok gedung
            for r in range(2):
                for c in range(2):
                    x0 = self.big_px//8 + c*(self.big_px//2)
                    y0 = self.big_px//8 + r*(self.big_px//2)
                    x1 = x0 + self.big_px//3
                    y1 = y0 + self.big_px//3
                    # isi gedung
                    rasterize_rect_to_image(img, x0, y0, x1, y1, block)
                    # shading sisi bawah
                    rasterize_rect_to_image(img, x0, y1-3, x1, y1, shadow)
                    # shading sisi kanan
                    rasterize_rect_to_image(img, x1-3, y0, x1, y1, shadow)

            # outline halus pakai QPainter
            qp = QPainter(img)
            qp.setPen(QColor(50, 50, 50))
            for r in range(2):
                for c in range(2):
                    x0 = self.big_px//8 + c*(self.big_px//2)
                    y0 = self.big_px//8 + r*(self.big_px//2)
                    w = self.big_px//3
                    h = self.big_px//3
                    qp.drawRect(x0, y0, w, h)
            qp.end()

        elif k in ('forest', 'dry_forest'):
            canopy = QColor(20,120,40) if k=='forest' else QColor(150,120,40)
            bg = QColor(30,80,30) if k=='forest' else QColor(120,100,60)
            rasterize_rect_to_image(img, 0, 0, self.big_px, self.big_px, bg)
            # tree circles
            radii = [self.big_px//3, self.big_px//4, self.big_px//5]
            centers = [(self.big_px//2, self.big_px//2 - self.big_px//8),
                    (self.big_px//3, self.big_px//2),
                    (self.big_px*2//3, self.big_px//2)]
            for (cx, cy), r in zip(centers, radii):
                rasterize_circle_filled_to_image(img, cx, cy, r, canopy)
            trunk = QColor(90,50,30)
            rasterize_rect_to_image(img, self.big_px//2 - self.big_px//20, self.big_px*2//3,
                                    self.big_px//2 + self.big_px//20, self.big_px, trunk)

        elif k in ('brush', 'dry_brush'):
            bg = QColor(100,140,80) if k=='brush' else QColor(150,130,80)
            rasterize_rect_to_image(img, 0, 0, self.big_px, self.big_px, bg)
            bush = QColor(20,100,20) if k=='brush' else QColor(140,110,60)
            for i in range(6):
                cx = random.randint(self.big_px//10, self.big_px*9//10)
                cy = random.randint(self.big_px//4, self.big_px*3//4)
                r = random.randint(self.big_px//20, self.big_px//12)
                rasterize_circle_filled_to_image(img, cx, cy, r, bush)

        elif k in ('grass','dry_grass'):
            bg = QColor(100,180,80) if k=='grass' else QColor(170,150,90)
            rasterize_rect_to_image(img, 0, 0, self.big_px, self.big_px, bg)
            # subtle small dots
            dark = QColor(80,120,60) if k=='grass' else QColor(140,120,80)
            for i in range(self.big_px//6):
                rasterize_circle_filled_to_image(img,
                                                random.randint(0, self.big_px-1),
                                                random.randint(0, self.big_px-1),
                                                1, dark)

        elif k == 'stone':
            bg = QColor(120, 120, 120)
            rasterize_rect_to_image(img, 0, 0, self.big_px, self.big_px, bg)

            tri = [(self.big_px//2, self.big_px//6),
                (self.big_px//6, self.big_px*5//6),
                (self.big_px*5//6, self.big_px*5//6)]
            main_color = QColor(90, 90, 90)
            shadow = QColor(60, 60, 60)

            # isi segitiga
            rasterize_polygon_scanline(img, tri, main_color)
            # shading tipis di bawah (geser 3px)
            rasterize_polygon_scanline(img, [(x, y+3) for (x,y) in tri], shadow)

            # outline halus pakai QPainter
            qp = QPainter(img)
            qp.setPen(QColor(40,40,40))
            qp.drawPolygon(*[QPoint(x,y) for (x,y) in tri])
            qp.end()

        elif k == 'swamp':
            bg = QColor(50,100,60)
            rasterize_rect_to_image(img, 0, 0, self.big_px, self.big_px, bg)
            water = QColor(40,70,120,200)
            rasterize_circle_filled_to_image(img, self.big_px//2, self.big_px//2, self.big_px//3, water)

        else:
            # default filler (neutral)
            bg = QColor(150,150,150)
            rasterize_rect_to_image(img, 0, 0, self.big_px, self.big_px, bg)

        return img


    def _make_river_big(self, mask):
        big = QImage(self.big_px, self.big_px, QImage.Format_ARGB32)
        big.fill(Qt.transparent)
        qp = QPainter(big)
        qp.setRenderHint(QPainter.Antialiasing, True)
        qp.setPen(Qt.NoPen)
        qp.setBrush(QColor(60,130,200))
        w = self.big_px; h = self.big_px
        t = w//3
        cx0, cx1 = (w-t)//2, (w+t)//2
        cy0, cy1 = (h-t)//2, (h+t)//2
        qp.drawRect(cx0, cy0, t, t)
        if mask & 1: qp.drawRect(cx0, 0, t, cy1)
        if mask & 2: qp.drawRect(cx0, cy0, w-cx0, t)
        if mask & 4: qp.drawRect(cx0, cy0, t, h-cy0)
        if mask & 8: qp.drawRect(0, cy0, cx1, t)
        qp.end()
        return big

    def get_tile_image(self, key, neighbors_mask=None, variant='normal'):
        if key == 'river':
            mask = neighbors_mask if neighbors_mask is not None else 0
            return self.cache.get(('river', mask))
        return self.cache.get((key, variant), self.cache.get(('grass','normal')))

# ----------------- GUI -----------------

class WildfireGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.map = TerrainMap(DEFAULT_MAP)
        self.renderer = PrimitiveRenderer(tile_px=TILE_SIZE, scale=SCALE)
        self.tile_cache = self.renderer.cache
        self.embers = []
        self.click_action = 'Light Fire'
        self.tile_paint = 'city'
        self.set_dimensions()
        self.create_menu()
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(SIM_SPEED)
        self.show()

    def set_dimensions(self):
        size = len(self.map.grid)
        self.width = TILE_SIZE * size + MENU_WIDTH
        self.height = TILE_SIZE * size
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(PANEL_XPOS, PANEL_YPOS, self.width, self.height)

    def create_menu(self):
        main_layout = QHBoxLayout()
        spacer_layout = QHBoxLayout(); spacer_layout.addStretch(1)
        menu_layout = QVBoxLayout()
        menu_layout.addWidget(QLabel('Action on click:'))
        self.create_rb(menu_layout, 'Light Fire', True)
        self.create_rb(menu_layout, 'Paint Tile', False)
        menu_layout.addWidget(QLabel('Tile to paint:'))
        tile_select = QComboBox(); tile_select.addItems(TERRAIN_TYPE_LIST)
        tile_select.currentIndexChanged.connect(self.set_tile_paint)
        menu_layout.addWidget(tile_select)
        menu_layout.addWidget(QLabel('New map size:'))
        size_select = QComboBox(); size_select.addItems([str(i) for i in range(MIN_GRID, MAX_GRID+1)])
        size_select.currentIndexChanged.connect(self.set_empty_map)
        menu_layout.addWidget(size_select)
        btn_save = QPushButton('Save', self); btn_save.clicked.connect(self.save); menu_layout.addWidget(btn_save)
        btn_load = QPushButton('Load', self); btn_load.clicked.connect(self.load); menu_layout.addWidget(btn_load)
        main_layout.addLayout(spacer_layout); main_layout.addLayout(menu_layout)
        self.setLayout(main_layout)

    def create_rb(self, layout, name, checked):
        rb = QRadioButton(name); rb.name = name; rb.setChecked(checked); rb.toggled.connect(self.set_click_action)
        layout.addWidget(rb)

    def set_click_action(self): self.click_action = self.sender().name
    def set_tile_paint(self, i): self.tile_paint = TERRAIN_TYPE_LIST[i].lower().replace(' ', '_')
    def set_empty_map(self, i):
        size = int(MIN_GRID + i)
        self.map = TerrainMap(None, size)
        self.set_dimensions()
        self.update()

    def save(self):
        os.makedirs(self.map.MAP_PATH, exist_ok=True)
        with open(self.map.MAP_PATH + USER_FILE, 'w', encoding='utf-8') as f:
            f.write(str(self.map))
        QMessageBox.information(self, "Saved", "Map saved to maps/" + USER_FILE)

    def load(self):
        try:
            self.map = TerrainMap(USER_FILE)
            self.set_dimensions()
            self.update()
            QMessageBox.information(self, "Loaded", "Map loaded from maps/" + USER_FILE)
        except Exception as e:
            QMessageBox.warning(self, "Load failed", str(e))

    def paintEvent(self, _):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing, True)
        size = len(self.map.grid)
        for r in range(size):
            for c in range(size):
                x = TILE_SIZE * c; y = TILE_SIZE * r
                tile = self.map.grid[r][c]; key = tile.type
                if key == 'river':
                    up = (r-1)>=0 and self.map.grid[r-1][c].type == 'river'
                    right = (c+1)<size and self.map.grid[r][c+1].type == 'river'
                    down = (r+1)<size and self.map.grid[r+1][c].type == 'river'
                    left = (c-1)>=0 and self.map.grid[r][c-1].type == 'river'
                    mask = (1 if up else 0) | (2 if right else 0) | (4 if down else 0) | (8 if left else 0)
                    img = self.renderer.get_tile_image('river', neighbors_mask=mask)
                    if img: qp.drawImage(x, y, img)
                else:
                    variant = 'normal'
                    if tile.is_burning: variant = 'burning'
                    elif tile.is_burnt: variant = 'burnt'
                    img = self.renderer.get_tile_image(key, variant=variant)
                    if img: qp.drawImage(x, y, img)
                if tile.is_burning:
                    qp.setPen(Qt.NoPen)
                    qp.setBrush(QColor(255,200,60,120))
                    qp.drawEllipse(x+TILE_SIZE//4, y+TILE_SIZE//6, TILE_SIZE//2, TILE_SIZE//2)
                if tile.is_burnt:
                    qp.setPen(Qt.NoPen)
                    qp.setBrush(QColor(0,0,0,80))
                    qp.drawRect(x, y, TILE_SIZE, TILE_SIZE)
        qp.setPen(Qt.NoPen)
        for ember in list(self.embers):
            px, py = ember.current_pos()
            qp.setBrush(QColor(255,160,40,220))
            qp.drawEllipse(px-2, py-2, 4, 4)
        qp.end()

    def mousePressEvent(self, ev):
        col = int(ev.pos().x() / TILE_SIZE)
        row = int(ev.pos().y() / TILE_SIZE)
        if self.map.in_bounds(row, col):
            if self.click_action == 'Light Fire':
                tile = self.map.grid[row][col]
                if tile.resistance < 100:
                    tile.is_burning = True
            else:
                self.map.grid[row][col] = TerrainTile(self.tile_paint)
            self.update()

    def tick(self):
        size = len(self.map.grid)
        burning_positions = [(r,c) for r in range(size) for c in range(size) if self.map.grid[r][c].is_burning]
        for r,c in burning_positions:
            self.map.spread_fire(r,c)
            if random.random() < 0.08: self.spawn_ember_from(r,c)
        for r in range(size):
            for c in range(size):
                self.map.grid[r][c].burn()
        for ember in list(self.embers):
            ember.update(step=2)
            if not ember.alive:
                lx, ly = ember.landing_pos()
                tr = int(ly/TILE_SIZE); tc = int(lx/TILE_SIZE)
                if self.map.in_bounds(tr, tc):
                    self.map.grid[tr][tc].light()
                try: self.embers.remove(ember)
                except ValueError: pass
        self.update()

    def spawn_ember_from(self, row, col):
        size = len(self.map.grid); max_rdist = 3
        for _ in range(8):
            dr = random.randint(-max_rdist, max_rdist)
            dc = random.randint(-max_rdist, max_rdist)
            tr = row+dr; tc = col+dc
            if self.map.in_bounds(tr, tc) and not (tr==row and tc==col): break
        else: return
        x0 = col*TILE_SIZE+TILE_SIZE//2; y0 = row*TILE_SIZE+TILE_SIZE//2
        x1 = tc*TILE_SIZE+TILE_SIZE//2; y1 = tr*TILE_SIZE+TILE_SIZE//2
        path = list(bresenham(x0,y0,x1,y1))
        if len(path) > 3:
            path = [(px+random.randint(-1,1), py+random.randint(-1,1)) for (px,py) in path]
        self.embers.append(Ember(path))

class Ember:
    def __init__(self, path): self.path = path; self.pos_index = 0; self.alive = True
    def update(self, step=1):
        self.pos_index += step
        if self.pos_index >= len(self.path): self.alive = False
    def current_pos(self):
        if 0 <= self.pos_index < len(self.path): return self.path[self.pos_index]
        elif self.path: return self.path[-1]
        return (0,0)
    def landing_pos(self): return self.path[-1] if self.path else None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = WildfireGUI()
    sys.exit(app.exec_())