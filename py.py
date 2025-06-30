import pygame
from collections import deque
import heapq
import time
import random

maze = [
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 0],
]

start = (0, 0)
goal = (7, 8)

# مواقع الوحوش (مثلاً 3 وحوش في متاهة)
monsters = [(3, 6), (5, 7), (7, 3)]

cell_size = 60
rows, cols = len(maze), len(maze[0])
width = cols * cell_size
height = rows * cell_size + 150

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 255)
RED = (255, 0, 0)
DARKRED = (150, 0, 0)
GREEN = (0, 255, 0)
GRAY = (200, 200, 200)
DARKGRAY = (50, 50, 50)
YELLOW = (255, 255, 0)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# نعدل جميع دوال البحث لإضافة تحقق من الوحوش (عدم المرور بمواقعهم)
def is_safe(cell):
    return cell not in monsters

def astar(maze, start, goal):
    frontier = []
    heapq.heappush(frontier, (heuristic(start, goal), 0, start))
    visited = set()
    parent = {}
    explored = []
    steps = 0

    while frontier:
        est_total, cost, current = heapq.heappop(frontier)
        steps += 1
        explored.append(current)

        if current == goal:
            path = reconstruct_path(parent, start, goal)
            return path, explored, steps, parent

        if current in visited:
            continue
        visited.add(current)

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0]+dx, current[1]+dy
            neighbor = (nx, ny)
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny]==0 and is_safe(neighbor):
                if neighbor not in visited:
                    new_cost = cost + 1
                    est = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(frontier, (est, new_cost, neighbor))
                    parent[neighbor] = current
    return [], explored, steps, parent

def gbfs(maze, start, goal):
    frontier = []
    heapq.heappush(frontier, (heuristic(start, goal), start))
    visited = set()
    parent = {}
    explored = []
    steps = 0

    while frontier:
        h, current = heapq.heappop(frontier)
        steps += 1
        explored.append(current)

        if current == goal:
            path = reconstruct_path(parent, start, goal)
            return path, explored, steps, parent

        if current in visited:
            continue
        visited.add(current)

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0]+dx, current[1]+dy
            neighbor = (nx, ny)
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny]==0 and is_safe(neighbor):
                if neighbor not in visited:
                    heapq.heappush(frontier, (heuristic(neighbor, goal), neighbor))
                    parent[neighbor] = current
    return [], explored, steps, parent

def bfs(maze, start, goal):
    queue = deque([start])
    visited = set([start])
    parent = {}
    explored = []
    steps = 0

    while queue:
        current = queue.popleft()
        steps += 1
        explored.append(current)

        if current == goal:
            path = reconstruct_path(parent, start, goal)
            return path, explored, steps, parent

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0]+dx, current[1]+dy
            neighbor = (nx, ny)
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny]==0 and is_safe(neighbor):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
    return [], explored, steps, parent

def dfs(maze, start, goal):
    stack = [start]
    visited = set()
    parent = {}
    explored = []
    steps = 0

    while stack:
        current = stack.pop()
        steps += 1
        explored.append(current)

        if current == goal:
            path = reconstruct_path(parent, start, goal)
            return path, explored, steps, parent

        if current in visited:
            continue
        visited.add(current)

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0]+dx, current[1]+dy
            neighbor = (nx, ny)
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny]==0 and is_safe(neighbor):
                if neighbor not in visited:
                    parent[neighbor] = current
                    stack.append(neighbor)
    return [], explored, steps, parent

def ucs(maze, start, goal):
    frontier = []
    heapq.heappush(frontier, (0, start))
    visited = set()
    parent = {}
    explored = []
    steps = 0

    while frontier:
        cost, current = heapq.heappop(frontier)
        steps += 1
        explored.append(current)

        if current == goal:
            path = reconstruct_path(parent, start, goal)
            return path, explored, steps, parent

        if current in visited:
            continue
        visited.add(current)

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0]+dx, current[1]+dy
            neighbor = (nx, ny)
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny]==0 and is_safe(neighbor):
                if neighbor not in visited:
                    heapq.heappush(frontier, (cost+1, neighbor))
                    parent[neighbor] = current
    return [], explored, steps, parent

def reconstruct_path(parent, start, goal):
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = parent.get(node, start)  # safety fallback
    path.append(start)
    path.reverse()
    return path

class Button:
    def __init__(self, text, x, y, w, h, callback):
        self.rect = pygame.Rect(x,y,w,h)
        self.text = text
        self.callback = callback
        self.color = GRAY

    def draw(self, screen, font):
        pygame.draw.rect(screen, self.color, self.rect)
        pygame.draw.rect(screen, DARKGRAY, self.rect, 2)
        text_surface = font.render(self.text, True, BLACK)
        screen.blit(text_surface, (self.rect.x+10, self.rect.y+10))

    def check_click(self, pos):
        if self.rect.collidepoint(pos):
            self.callback()

def draw_maze(screen, maze, path, explored, start, goal, monsters):
    for i in range(rows):
        for j in range(cols):
            rect = pygame.Rect(j*cell_size, i*cell_size+100, cell_size, cell_size)
            color = WHITE if maze[i][j] == 0 else BLACK
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, DARKGRAY, rect, 1)

    # نرسم الوحوش
    for mx, my in monsters:
        rect = pygame.Rect(my*cell_size, mx*cell_size+100, cell_size, cell_size)
        pygame.draw.rect(screen, DARKRED, rect)
        pygame.draw.circle(screen, RED, rect.center, cell_size//3)

    # نرسم الأماكن التي تم استكشافها
    for cell in explored:
        i, j = cell
        rect = pygame.Rect(j*cell_size, i*cell_size+100, cell_size, cell_size)
        pygame.draw.rect(screen, YELLOW, rect)

    # نرسم المسار
    for cell in path:
        i, j = cell
        rect = pygame.Rect(j*cell_size, i*cell_size+100, cell_size, cell_size)
        pygame.draw.rect(screen, GREEN, rect)

    # نقطة البداية
    rect = pygame.Rect(start[1]*cell_size, start[0]*cell_size+100, cell_size, cell_size)
    pygame.draw.rect(screen, BLUE, rect)

    # نقطة الهدف
    rect = pygame.Rect(goal[1]*cell_size, goal[0]*cell_size+100, cell_size, cell_size)
    pygame.draw.rect(screen, RED, rect)

def move_monsters():
    global monsters
    new_positions = []
    for mx, my in monsters:
        # احصل على خلايا فارغة مجاورة
        neighbors = []
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = mx+dx, my+dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0 and (nx, ny) not in monsters and (nx, ny) != start:
                neighbors.append((nx, ny))
        if neighbors:
            new_pos = random.choice(neighbors)
        else:
            new_pos = (mx, my)
        new_positions.append(new_pos)
    monsters = new_positions

pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Maze Search with Monsters")
font = pygame.font.SysFont('arial', 24)
small_font = pygame.font.SysFont('arial', 18)

search_methods = {
    "A* Search": astar,
    "Greedy Best First": gbfs,
    "Breadth First Search": bfs,
    "Depth First Search": dfs,
    "Uniform Cost Search": ucs,
}

selected_method = None
path = []
explored = []
steps = 0
parent = {}
status_text = "اختر طريقة البحث واضغط زر البحث"

def run_search():
    global path, explored, steps, parent, status_text, selected_method, monsters
    if not selected_method:
        status_text = "يرجى اختيار طريقة البحث أولاً"
        return

    # إذا الوحوش على نقطة البداية فخسارة مباشرة
    if start in monsters:
        status_text = "لقد التقط الوحوش اللاعب في البداية! خسرت!"
        path.clear()
        explored.clear()
        return

    path, explored, steps, parent = search_methods[selected_method](maze, start, goal)
    if not path:
        status_text = "لم يتم العثور على حل!"
    else:
        # تأكد أن المسار لا يحتوي على وحوش (للأمان فقط)
        if any(pos in monsters for pos in path):
            status_text = "الوصول إلى وحش! خسرت!"
            path = []
        else:
            status_text = f"تم إيجاد المسار بـ {steps} خطوات باستخدام {selected_method}."

def select_method(method):
    global selected_method, status_text
    selected_method = method
    status_text = f"تم اختيار {method}"

# أزرار اختيار طريقة البحث
buttons = []
for i, method in enumerate(search_methods):
    buttons.append(Button(method, 10 + i*160, 10, 150, 40, lambda m=method: select_method(m)))
search_button = Button("ابحث", width-160, 10, 150, 40, run_search)

clock = pygame.time.Clock()
monster_move_counter = 0

running = True
while running:
    screen.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            for btn in buttons + [search_button]:
                btn.check_click(pos)

    # نرسم واجهة المستخدم
    for btn in buttons + [search_button]:
        btn.draw(screen, font)

    # نرسم المتاهة مع الوحوش والمسار
    draw_maze(screen, maze, path, explored, start, goal, monsters)

    # نعرض حالة البحث
    status_surface = small_font.render(status_text, True, BLACK)
    screen.blit(status_surface, (10, height - 40))

    # تحريك الوحوش كل 60 إطار تقريباً
    monster_move_counter += 1
    if monster_move_counter > 60:
        move_monsters()
        monster_move_counter = 0

        # إذا الوحوش تحركت إلى نقطة البداية -> خسارة
        if start in monsters:
            status_text = "الوحوش التقطت اللاعب! خسرت!"
            path.clear()
            explored.clear()

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
