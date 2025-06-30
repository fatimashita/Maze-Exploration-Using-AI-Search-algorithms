from collections import deque
import heapq
import pygame
import time
import math
import csv
import os
from datetime import datetime
import csv
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# اسم ملف التخزين
filename = "results.csv"

def store_result_csv(maze_name, algorithm_name, path_length, nodes_expanded, exec_time):
    if not os.path.exists(filename):
        print("ملف النتائج غير موجود. سيتم إنشاؤه تلقائيًا الآن.")
        # إنشاء ملف فارغ مع عناوين الأعمدة الأساسية (مثال)
    # نفتح الملف في وضع الإضافة (append) عشان ما نحذف البيانات القديمة
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # نكتب صف جديد للنتيجة
        writer.writerow([maze_name, algorithm_name, path_length, nodes_expanded, exec_time])

def read_results_csv():
    results = []
    try:
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                # نحول البيانات لأنواع مناسبة
                maze = row[0]
                algo = row[1]
                path_len = int(row[2])
                nodes = int(row[3])
                time = float(row[4])
                results.append({
                    "maze": maze,
                    "algorithm": algo,
                    "path_length": path_len,
                    "nodes_expanded": nodes,
                    "execution_time": time
                })
    except FileNotFoundError:
        print("ملف النتائج غير موجود. سيتم إنشاءه عند أول تخزين.")
    return results

# قراءة وطباعة النتائج من الملف
all_results = read_results_csv()
for r in all_results:
    print(f"Maze: {r['maze']}, Algorithm: {r['algorithm']}, Path Length: {r['path_length']}, Nodes Expanded: {r['nodes_expanded']}, Time: {r['execution_time']:.4f}s")


maze = [
    [0, 1, 0, 0, 0, 1, 0, 0, 0],
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

cell_size = 60 # حجم الخلية الآن  بكسل
rows, cols = len(maze), len(maze[0])
width = cols * cell_size
height = rows * cell_size + 100 # مساحة إضافية للواجهة
# تهيئة شاشة pygame بحجم ديناميكي
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Dynamic Maze Display")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GRAY = (200, 200, 200)
DARKGRAY = (50, 50, 50)
YELLOW = (255, 255, 0)
BTN_BLUE= (70,130,180)
BTN_TEXT= (255,255,255)

# زر تبديل الهيريستيك
HEURISTIC_BTN_RECT = pygame.Rect(10, height - 40, 150, 30)

heuristic_type = "manhattan"  # القيم: "manhattan" أو "euclidean"
pygame.init()
FONT = pygame.font.SysFont(None, 30)
def draw_heuristic_button(screen):
    pygame.draw.rect(screen, BTN_BLUE, HEURISTIC_BTN_RECT, border_radius=8)
    text = FONT.render("Heuristic", True, WHITE)
    screen.blit(text, (HEURISTIC_BTN_RECT.x + 10, HEURISTIC_BTN_RECT.y + 5))

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def heuristic(a, b):
    return manhattan(a, b) if heuristic_type == "manhattan" else euclidean(a, b)

def astar(maze, start, goal):
    rows, cols = len(maze), len(maze[0])  # Ensure bounds are dynamically set within function
    frontier = []
    heapq.heappush(frontier, (heuristic(start, goal), 0, start))
    visited = set()
    parent = {}
    cost_so_far = {start: 0}
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
            if 0 <= nx < rows and 0 <= ny < cols:
                if maze[nx][ny] == 0:
                    neighbor = (nx, ny)
                    new_cost = cost_so_far[current] + 1
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        est = new_cost + heuristic(neighbor, goal)
                        heapq.heappush(frontier, (est, new_cost, neighbor))
                        parent[neighbor] = current

    return [], explored, steps, parent


def gbfs(maze, start, goal):
    rows, cols = len(maze), len(maze[0])
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
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny]==0:
                if neighbor not in visited:
                    heapq.heappush(frontier, (heuristic(neighbor, goal), neighbor))
                    parent[neighbor] = current
    return [], explored, steps, parent

def bfs(maze, start, goal):
    rows, cols = len(maze), len(maze[0])
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
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny]==0:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
    return [], explored, steps, parent

def dfs(maze, start, goal):
    rows, cols = len(maze), len(maze[0])
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
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny]==0:
                if neighbor not in visited:
                    parent[neighbor] = current
                    stack.append(neighbor)
    return [], explored, steps, parent

def dls(maze, current, goal, depth, visited, parent, explored, steps):
    rows, cols = len(maze), len(maze[0])  # <-- هذا السطر ضروري
    if depth == 0 and current == goal:
        return True
    if depth > 0:
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    explored.append(neighbor)
                    steps[0] += 1
                    if dls(maze, neighbor, goal, depth - 1, visited, parent, explored, steps):
                        return True
    return False


def iddfs(maze, start, goal, max_depth=50):
    steps = [0]
    for depth in range(max_depth):
        visited = set([start])
        parent = {}
        explored = [start]
        if dls(maze, start, goal, depth, visited, parent, explored, steps):
            path = reconstruct_path(parent, start, goal)
            return path, explored, steps[0], parent
    return [], [], steps[0], {}

def ucs(maze, start, goal):
    rows, cols = len(maze), len(maze[0])
    frontier = []
    heapq.heappush(frontier, (0, start))
    cost_so_far = {start: 0}
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
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny]==0:
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heapq.heappush(frontier, (new_cost, neighbor))
                    parent[neighbor] = current
    return [], explored, steps, parent

def reconstruct_path(parent, start, goal):
    path = []
    node = goal
    while node != start:
        path.append(node)
        if node not in parent:
            return []  # مسار غير موجود
        node = parent[node]
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

def draw_maze(screen, maze, path, explored, start, goal,rows,cols):
    for i in range(rows):
        for j in range(cols):
            rect = pygame.Rect(j*cell_size, i*cell_size+90, cell_size, cell_size)
            color = WHITE
            if (i,j) == start:
                color = BLUE
            elif (i,j) == goal:
                color = RED
            elif (i,j) in path:
                color = GREEN
            elif (i,j) in explored:
                color = GRAY
            elif maze[i][j] == 1:
                color = BLACK
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, DARKGRAY, rect, 1)
    
def draw_tree(screen, parent_map, start):
    for child, parent in parent_map.items():
        if parent is None:
            continue
        x1 = parent[1]*cell_size + cell_size//2
        y1 = parent[0]*cell_size + cell_size//2 + 90
        x2 = child[1]*cell_size + cell_size//2
        y2 = child[0]*cell_size + cell_size//2 + 90
        pygame.draw.line(screen, YELLOW, (x1,y1), (x2,y2), 2)

def main():
    global start, goal, heuristic_type
    pygame.init()
    
    # حجم النافذة الابتدائي
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption("Maze Solver Interactive")
    font = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()

    easy_maze = [[0, 1, 0, 0, 0],
                 [0, 1, 0, 1, 0],
                 [0, 0, 0, 1, 0],
                 [1, 1, 0, 1, 0],
                 [0, 0, 0, 0, 0]]

    medium_maze = [[0, 1, 0, 1, 0, 1, 0],
                   [0, 1, 0, 1, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1, 0],
                   [1, 1, 1, 1, 0, 1, 0],
                   [0, 0, 0, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0, 1, 1],
                   [0, 0, 0, 1, 0, 0, 0]]

    hard_maze = [[0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
                 [0, 1, 0, 1, 0, 0, 0, 1, 1, 0],
                 [0, 1, 0, 1, 1, 1, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
                 [1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                 [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]

    maze = easy_maze
    rows, cols = len(maze), len(maze[0])
    start = (0, 0)
    goal = (rows - 1, cols - 1)
    heuristic_type = "manhattan"

    path, explored, steps_count, parent_map = [], [], 0, {}
    step = 0
    elapsed_time = 0
    algorithm = ""
    selecting_start = False
    selecting_goal = False

    algorithms = {
        "BFS": bfs,
        "DFS": dfs,
        "UCS": ucs,
        "A*": astar,
        "GBFS": gbfs,
        "IDDFS": iddfs
    }

    def run_algorithm(algo_name):
        nonlocal path, explored, step, steps_count, elapsed_time, algorithm, parent_map
        algorithm = algo_name
        algo_func = algorithms[algo_name]
        start_time = time.time()
        path, explored, steps_count, parent_map = algo_func(maze, start, goal)
        elapsed_time = time.time() - start_time
        step = 0
        maze_name = "Maze-1"
        path_length = len(path)
        store_result_csv(maze_name, algorithm, path_length, steps_count, elapsed_time)

    def set_start():
        nonlocal selecting_start, selecting_goal
        selecting_start = True
        selecting_goal = False

    def set_goal():
        nonlocal selecting_start, selecting_goal
        selecting_start = False
        selecting_goal = True

    # ✅ هذا هو التعديل الوحيد!
    def change_maze(new_maze):
        nonlocal maze, rows, cols, path, explored, steps_count, parent_map
        global start, goal

        maze = new_maze
        rows, cols = len(maze), len(maze[0])
        start = (0, 0)
        goal = (rows - 1, cols - 1)
        path, explored, steps_count, parent_map = [], [], 0, {}
        if algorithm != "":
            run_algorithm(algorithm)

    buttons = []
    x_btn = 10
    for algo_name in algorithms.keys():
        buttons.append(Button(algo_name, x_btn, 10, 80, 40, lambda a=algo_name: run_algorithm(a)))
        x_btn += 90

    buttons.append(Button("Set Start", x_btn, 10, 100, 40, set_start))
    x_btn += 110
    buttons.append(Button("Set Goal", x_btn, 10, 100, 40, set_goal))
    x_btn += 110

    buttons.append(Button("Easy Maze", x_btn, 10, 100, 40, lambda: change_maze(easy_maze)))
    x_btn += 110
    buttons.append(Button("Medium Maze", x_btn, 10, 120, 40, lambda: change_maze(medium_maze)))
    x_btn += 130
    buttons.append(Button("Hard Maze", x_btn, 10, 100, 40, lambda: change_maze(hard_maze)))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if HEURISTIC_BTN_RECT.collidepoint(pos):
                    heuristic_type = "euclidean" if heuristic_type == "manhattan" else "manhattan"
                    if algorithm != "":
                        run_algorithm(algorithm)

                for button in buttons:
                    button.check_click(pos)

                if selecting_start or selecting_goal:
                    x_cell = pos[0] // cell_size
                    y_cell = (pos[1] - 70) // cell_size
                    if 0 <= y_cell < rows and 0 <= x_cell < cols and maze[y_cell][x_cell] == 0:
                        if selecting_start:
                            start = (y_cell, x_cell)
                            selecting_start = False
                        else:
                            goal = (y_cell, x_cell)
                            selecting_goal = False
                        path, explored, steps_count, parent_map = [], [], 0, {}
                        if algorithm != "":
                            run_algorithm(algorithm)

        if not running:
            break

        screen.fill(WHITE)
        draw_maze(screen, maze, path[:step + 1], explored[:step + 1], start, goal, rows, cols)
        draw_tree(screen, parent_map, start)
        draw_heuristic_button(screen)

        for btn in buttons:
            btn.draw(screen, font)

        info_text = f"Algorithm: {algorithm} | Steps: {steps_count} | Explored: {len(explored)} | Time: {elapsed_time:.4f}s | Heuristic: {heuristic_type.capitalize()}"
        info_surface = font.render(info_text, True, BLACK)
        screen.blit(info_surface, (10, 60))

        pygame.display.flip()
        clock.tick(5)

        if path and step < len(path) - 1:
            step += 1
            pygame.time.delay(200)

    pygame.quit()

if __name__ == '__main__':
    main()
