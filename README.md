# Comp_graph_hw
## 1 — Отрезок по введённым с консоли координатам (реализовать по алгоритму Брезенхема)
```python
import matplotlib.pyplot as plt
from PIL import Image # Импорт библиотек
import numpy as np

class GridDrawer:
    def __init__(self, width, height, background_color='white'):
        self.img = Image.new('RGB', (width, height), background_color) # Создаем класс GridDrawer для рисования сетки и линий
        self.width = width
        self.height = height

    def draw_line(self, start, end, color): # Метод draw_line рисует линию по алгоритму Брезенхэма
        x0, y0 = start
        x1, y1 = end
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            self.img.putpixel((x0, y0), color)
            if (x0, y0) == (x1, y1):
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def draw_grid(self, step, color): # Метод draw_grid рисует сетку
        for x in range(0, self.width, step):
            self.draw_line((x, 0), (x, self.height - 1), color)
        for y in range(0, self.height, step):
            self.draw_line((0, y), (self.width - 1, y), color)

    def show_image(self):
        plt.imshow(np.asarray(self.img))
        plt.axis('off')
        plt.show()
                                            # Методы show_image и save_image для работы с изображением
    def save_image(self, filename):
        self.img.save(filename)

def get_coordinate(prompt, max_val): # Запрос координат
    while True:
        try:
            value = int(input(f"{prompt} (0-{max_val}): "))
            if 0 <= value <= max_val:
                return value
            else:
                print(f"Please enter a value between 0 and {max_val}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

width, height = 200, 200 # Устанавливаем значение сетки и полей
grid_drawer = GridDrawer(width, height)
grid_color = (200, 200, 200)
grid_drawer.draw_grid(10, grid_color)

x0 = get_coordinate("Input x0-coordinate", width - 1) # Запрос координат линии 
y0 = get_coordinate("Input y0-coordinate", height - 1)
x1 = get_coordinate("Input x1-coordinate", width - 1)
y1 = get_coordinate("Input y1-coordinate", height - 1)

line_color = (0, 0, 0) # Рисование линии, отображение и сохранение изображения
grid_drawer.draw_line((x0, y0), (x1, y1), line_color)
grid_drawer.show_image()
grid_drawer.save_image('LineDrawing.png')
```
![Рисунок 1](https://github.com/GOSICK070404/Comp_graph_hw/blob/main/1.png)
## 2 — Круг (реализовать с помощью алгоритма Брезенхема)
```python
import matplotlib.pyplot as plt # Импорт библиотек
import numpy as np

class CircleDrawer: # Создание класса CircleDrawer, который будет содержать методы, необходимые для рисования круга

    def __init__(self, radius): 
        self.radius = radius
        self.points = []

    def calculate_points(self): # Метод calculate_points рассчитывает координаты точек
        x, y = 0, self.radius
        d = 3 - 2 * self.radius
        while x <= y:
            self.symmetric_points(x, y)
            if d <= 0:
                d += 4 * x + 6
            else:
                d += 4 * (x - y) + 10
                y -= 1
            x += 1
        self.points = self.sort_points_by_angle(set(self.points))

    def symmetric_points(self, x, y): # Метод symmetric_points добавляет симметричные точки
        self.points += [
            (x, y), (-x, y), (x, -y), (-x, -y),
            (y, x), (-y, x), (y, -x), (-y, -x)
        ]

    def sort_points_by_angle(self, points): # Метод sort_points_by_angle сортирует точки
        points = sorted(points, key=lambda p: np.arctan2(p[1], p[0]))
        return list(points) + [list(points)[0]]

    def plot_circle(self): # Метод отвечающий за постройку круга
        x_coords, y_coords = zip(*self.points)
        plt.plot(x_coords, y_coords, color='black')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f'Круг с радиусом {self.radius}')
        plt.grid(True)
        plt.show()

def main(): # Инициализация и вызов методов
    try:
        radius = int(input("Введите радиус: "))
        if radius <= 0:
            print("Радиус должен быть положительным целым числом.")
            return

        drawer = CircleDrawer(radius)
        drawer.calculate_points()
        drawer.plot_circle()

    except ValueError:
        print("Введите допустимое целое число для радиуса.")

if __name__ == "__main__":
    main()
```
![Рисунок 2](https://github.com/GOSICK070404/Comp_graph_hw/blob/main/2.png)
## 3 — Циферблат (реализовать с помощью алгоритма Брезенхема)
```python
import matplotlib.pyplot as plt
import numpy as np

class CircleDrawer:
    
    def __init__(self, radius): 
        self.radius = radius
        self.points = []

    def calculate_points(self):  # Рассчитывает координаты точек для построения круга
        x, y = 0, self.radius
        d = 3 - 2 * self.radius
        while x <= y:
            self.symmetric_points(x, y)
            if d <= 0:
                d += 4 * x + 6
            else:
                d += 4 * (x - y) + 10
                y -= 1
            x += 1
        self.points = self.sort_points_by_angle(set(self.points))

    def symmetric_points(self, x, y):  # Добавляет симметричные точки
        self.points += [
            (x, y), (-x, y), (x, -y), (-x, -y),
            (y, x), (-y, x), (y, -x), (-y, -x)
        ]

    def sort_points_by_angle(self, points):  # Сортирует точки
        points = sorted(points, key=lambda p: np.arctan2(p[1], p[0]))
        return list(points) + [list(points)[0]]

    def plot_circle_with_ticks(self):  # Строит круг с 12 линиями, как на циферблате
        x_coords, y_coords = zip(*self.points)
        plt.plot(x_coords, y_coords, color='black')
        
        # Добавляем 12 линий как на циферблате
        for i in range(12):
            angle = 2 * np.pi * i / 12  # Угол для каждой линии
            start_x = (self.radius * 0.8) * np.cos(angle)  # Начало линии немного ближе к центру
            start_y = (self.radius * 0.8) * np.sin(angle)
            end_x = self.radius * np.cos(angle)  # Конец линии на окружности
            end_y = self.radius * np.sin(angle)
            
            plt.plot([start_x, end_x], [start_y, end_y], color='red', lw=1.5)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f'Круг с радиусом {self.radius} и 12 линиями как на циферблате')
        plt.grid(True)
        plt.show()

def main():
    try:
        radius = int(input("Введите радиус: "))
        if radius <= 0:
            print("Радиус должен быть положительным целым числом.")
            return

        drawer = CircleDrawer(radius)
        drawer.calculate_points()
        drawer.plot_circle_with_ticks()

    except ValueError:
        print("Введите допустимое целое число для радиуса.")

if __name__ == "__main__":
    main()
```
![Рисунок 3](https://github.com/GOSICK070404/Comp_graph_hw/blob/main/3.png)
## 4 — Алгоритм Сезерленда-Коэна (реализовать в виде кода)
```python
import matplotlib.pyplot as plt

# Определяем коды областей для отсечения
IN_WINDOW = 0  # 0000
LEFT_EDGE = 1   # 0001
RIGHT_EDGE = 2  # 0010
BOTTOM_EDGE = 4 # 0100
TOP_EDGE = 8    # 1000

# Функция для вычисления кода точки
def get_region_code(x, y, x_low, y_low, x_high, y_high):
    
    region_code = IN_WINDOW
    if x < x_low:    # Слева от окна
        region_code |= LEFT_EDGE
    elif x > x_high: # Справа от окна
        region_code |= RIGHT_EDGE
    if y < y_low:    # Ниже окна
        region_code |= BOTTOM_EDGE
    elif y > y_high: # Выше окна
        region_code |= TOP_EDGE
    return region_code

# Алгоритм Сазерленда-Коэна для отсечения отрезков
def sutherland_hodgman_clip(x_start, y_start, x_end, y_end, x_low, y_low, x_high, y_high):
    code_start = get_region_code(x_start, y_start, x_low, y_low, x_high, y_high)
    code_end = get_region_code(x_end, y_end, x_low, y_low, x_high, y_high)
    accepted = False

    while True:
        if code_start == 0 and code_end == 0:  # Обе точки внутри окна
            accepted = True
            break
        elif code_start & code_end != 0:  # Обе точки снаружи, отрезок вне окна
            break
        else:
            intersection_x, intersection_y = 0.0, 0.0
            # Выбираем точку, находящуюся снаружи
            if code_start != 0:
                out_code = code_start
            else:
                out_code = code_end

            # Находим пересечение с границами окна
            if out_code & TOP_EDGE:  # Пересечение с верхней границей
                intersection_x = x_start + (x_end - x_start) * (y_high - y_start) / (y_end - y_start)
                intersection_y = y_high
            elif out_code & BOTTOM_EDGE:  # Пересечение с нижней границей
                intersection_x = x_start + (x_end - x_start) * (y_low - y_start) / (y_end - y_start)
                intersection_y = y_low
            elif out_code & RIGHT_EDGE:  # Пересечение с правой границей
                intersection_y = y_start + (y_end - y_start) * (x_high - x_start) / (x_end - x_start)
                intersection_x = x_high
            elif out_code & LEFT_EDGE:  # Пересечение с левой границей
                intersection_y = y_start + (y_end - y_start) * (x_low - x_start) / (x_end - x_start)
                intersection_x = x_low

            # Заменяем точку снаружи на точку пересечения и пересчитываем код
            if out_code == code_start:
                x_start, y_start = intersection_x, intersection_y
                code_start = get_region_code(x_start, y_start, x_low, y_low, x_high, y_high)
            else:
                x_end, y_end = intersection_x, intersection_y
                code_end = get_region_code(x_end, y_end, x_low, y_low, x_high, y_high)

    if accepted:
        return x_start, y_start, x_end, y_end
    else:
        return None

# Функция для визуализации отсечения линий
def display_plot(lines_to_draw, x_low, y_low, x_high, y_high):
    fig, ax = plt.subplots()

    # Рисуем окно отсечения
    ax.plot([x_low, x_high, x_high, x_low, x_low],
            [y_low, y_low, y_high, y_high, y_low], 'k-', lw=2)

    # Рисуем отрезки до отсечения
    for line in lines_to_draw:
        x1, y1, x2, y2 = line
        ax.plot([x1, x2], [y1, y2], 'r--', label='До отсечения')

    # Отсечение линий
    for line in lines_to_draw:
        clipped_result = sutherland_hodgman_clip(*line, x_low, y_low, x_high, y_high)
        if clipped_result:
            x1, y1, x2, y2 = clipped_result
            ax.plot([x1, x2], [y1, y2], 'g-', lw=2, label='После отсечения')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Отсечение отрезков алгоритмом Сазерленда-Коэна')
    plt.grid(True)
    plt.show()

# Пример использования
if __name__ == "__main__":

    # Задаем окно отсечения
    x_low, y_low = 25, 20
    x_high, y_high = 175, 80

    # Отрезки для отсечения
    lines_to_clip = [
        (0, 60, 50, 100),
        (0, 40, 75, 100),
        (0, 20, 100, 100),
        (25, 0, 150, 100),
        (50, 0, 175, 100),
        (75, 0, 175, 80),
        (100, 0, 175, 60),
        (125, 0, 175, 40),
        (150, 0, 175, 20),
        (0, 0, 125, 100)
    ]

    # Визуализация
    display_plot(lines_to_clip, x_low, y_low, x_high, y_high)
```
![Рисунок 4](https://github.com/GOSICK070404/Comp_graph_hw/blob/main/4.png)
## 5 — Алгоритм Цирруса-Бека (реализовать в виде кода)
```python
import numpy as np
import matplotlib.pyplot as plt

# Функция для вычисления скалярного произведения двух векторов
def scalar_projection(vec1, vec2):
    return vec1[0] * vec2[0] + vec1[1] * vec2[1]

# Алгоритм Цируса-Бека для отсечения отрезков
def clip_line_cyrus_beck(point_start, point_end, poly):
    direction_vec = np.array(point_end) - np.array(point_start)  # Вектор направления отрезка
    t_min = 0  # Параметр t на вхождении
    t_max = 1  # Параметр t на выходе

    for j in range(len(poly)):
        # Получаем текущую грань многоугольника
        corner1 = poly[j]
        corner2 = poly[(j + 1) % len(poly)]
        edge_vec = np.array(corner2) - np.array(corner1)
        normal_vec = np.array([-edge_vec[1], edge_vec[0]])  # Нормаль к текущей грани

        # Вектор от начальной точки отрезка к вершине грани
        offset_vec = np.array(point_start) - np.array(corner1)

        # Вычисляем значения числителя и знаменателя
        numerator = -scalar_projection(offset_vec, normal_vec)
        denominator = scalar_projection(direction_vec, normal_vec)

        if denominator != 0:
            t_value = numerator / denominator
            if denominator > 0:  # Вход в многоугольник
                t_min = max(t_min, t_value)
            else:  # Выход из многоугольника
                t_max = min(t_max, t_value)

            if t_min > t_max:
                return None  # Отрезок невидим

    if t_min <= t_max:
        # Вычисляем точки пересечения с границами многоугольника
        clipped_start = point_start + t_min * direction_vec
        clipped_end = point_start + t_max * direction_vec
        return clipped_start, clipped_end
    return None

# Функция для визуализации отсечения отрезков
def display_clipping_results(segment_list, poly_vertices):
    fig, ax = plt.subplots()

    # Рисуем многоугольник
    poly_vertices.append(poly_vertices[0])  # Замыкаем контур многоугольника
    poly_vertices = np.array(poly_vertices)
    ax.plot(poly_vertices[:, 0], poly_vertices[:, 1], 'k-', lw=2)

    # Рисуем отрезки до отсечения
    for segment in segment_list:
        start, end = segment
        ax.plot([start[0], end[0]], [start[1], end[1]], 'r--', label='До отсечения')

    # Отсекаем и рисуем пересечения
    for segment in segment_list:
        clipped_segment = clip_line_cyrus_beck(np.array(segment[0]), np.array(segment[1]), poly_vertices[:-1].tolist())
        if clipped_segment:
            new_start, new_end = clipped_segment
            ax.plot([new_start[0], new_end[0]], [new_start[1], new_end[1]], 'g-', lw=2, label='После отсечения')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Отсечение отрезков алгоритмом Цируса-Бека')
    plt.grid(True)
    plt.show()

# Пример использования
if __name__ == "__main__":

    # Задаем выпуклый многоугольник
    poly_vertices = [
        [0, 0],
        [100, 0],
        [80, 80],
        [20, 80]
    ]

    # Отрезки для отсечения
    segment_list = [
        ([0, 0], [120, 120]),
        ([0, 20], [100, 120]),
        ([0, 40], [80, 120]),
         ([0, 60], [60, 120]),
        ([20, 0], [120, 100]),
        ([40, 0], [120, 80]),
        ([60, 0], [120, 60]),
        ([80, 0], [120, 40]),
        ([100, 0], [120, 20])
    ]

    # Визуализация
    display_clipping_results(segment_list, poly_vertices)
```
![Рисунок 5](https://github.com/GOSICK070404/Comp_graph_hw/blob/main/5.png)
## 6 — Алгоритм заполнения замкнутых областей посредством "затравки"
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas

def generate_shape_image(points, dimensions=(100, 100)):
    figure, axis = plt.subplots()
    figure.set_size_inches(dimensions[0] / figure.dpi, dimensions[1] / figure.dpi)
    axis.set_xlim(0, dimensions[1])
    axis.set_ylim(0, dimensions[0])
    axis.invert_yaxis()
    axis.axis('off')

    # Создание формы
    shape = Polygon(points, closed=True, edgecolor='black', facecolor='white')
    axis.add_patch(shape)

    # Преобразование в массив
    canvas = Canvas(figure)
    canvas.draw()
    image_data = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').reshape(dimensions[0], dimensions[1], 4)
    plt.close(figure)

    return image_data[:, :, :3].copy()

def is_within_threshold(pixel, brightness_limit=68):
    # Оцениваем пиксели, светлее которых считаем фоном
    return np.mean(pixel) > brightness_limit

def region_fill(img, x_coord, y_coord, color):
    if not is_within_threshold(img[x_coord, y_coord]):
        return

    stack_coords = [(x_coord, y_coord)]

    while stack_coords:
        current_x, current_y = stack_coords.pop()
        if is_within_threshold(img[current_x, current_y]):
            img[current_x, current_y] = color

            for delta_x, delta_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_x, next_y = current_x + delta_x, current_y + delta_y
                if 0 <= next_x < img.shape[0] and 0 <= next_y < img.shape[1] and is_within_threshold(img[next_x, next_y]):
                    stack_coords.append((next_x, next_y))

# Устанавливаем координаты для произвольного многоугольника
points = points = [(50, 5), (61, 35), (98, 35), (68, 57), (79, 91), (50, 70), (21, 91), (32, 57), (2, 35), (39, 35)
]
img_array = generate_shape_image(points)

fill_color = np.array([100, 100, 0], dtype=np.uint8)  # Заливочный цвет

# Убираем промежуточные оттенки серого вдоль границ
dark_threshold = 100
img_array[np.all((img_array[:, :, 0] < dark_threshold) & 
                 (img_array[:, :, 1] < dark_threshold) & 
                 (img_array[:, :, 2] < dark_threshold), axis=-1)] = [255, 255, 255]

# Отображение исходной картинки
plt.subplot(1, 2, 1)
plt.title("До")
plt.imshow(img_array)

# Применение заполнения области
region_fill(img_array, 50, 50, fill_color)

# Результат
plt.subplot(1, 2, 2)
plt.title("После")
plt.imshow(img_array)
plt.show()
```
![Рисунок 6](https://github.com/GOSICK070404/Comp_graph_hw/blob/main/6.png) 
## 7 — Алгоритм заполнения замкнутых областей посредством горизонтального сканирования
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas

def draw_shape(vertices, dimensions=(100, 100)):
    fig, ax = plt.subplots()
    fig.set_size_inches(dimensions[0] / fig.dpi, dimensions[1] / fig.dpi)
    ax.set_xlim(0, dimensions[1])
    ax.set_ylim(0, dimensions[0])
    ax.invert_yaxis()
    ax.axis('off')

    # Создание и добавление фигуры
    shape = Polygon(vertices, closed=True, edgecolor='black', facecolor='white')
    ax.add_patch(shape)

    # Преобразование в массив
    canvas = Canvas(fig)
    canvas.draw()
    image_array = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').reshape(dimensions[0], dimensions[1], 4)
    plt.close(fig)

    return image_array[:, :, :3].copy()

def is_light_color(pixel, brightness_threshold=68):
    # Проверка светлых пикселей на основе порога яркости
    return np.mean(pixel) > brightness_threshold

def scanline_fill(image, x, y, color):
    if not is_light_color(image[x, y]):
        return

    stack = [(x, y)]

    while stack:
        current_x, current_y = stack.pop()
        if is_light_color(image[current_x, current_y]):
            image[current_x, current_y] = color

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_x, next_y = current_x + dx, current_y + dy
                if 0 <= next_x < image.shape[0] and 0 <= next_y < image.shape[1] and is_light_color(image[next_x, next_y]):
                    stack.append((next_x, next_y))

# Массив точек
def fill_shape(points):
    x_bounds = (min(points[:, 0]), max(points[:, 0]))
    y_bounds = (min(points[:, 1]), max(points[:, 1]))

    fill_coords = []

    # Сканирование горизонтальных линий
    for line_y in range(int(y_bounds[0]), int(y_bounds[1]) + 1):
        intersections = []

        # Поиск пересечений с линиями границы
        for i in range(len(points)):
            p1, p2 = points[i], points[(i + 1) % len(points)]
            if (p1[1] > line_y) != (p2[1] > line_y):
                intersect_x = (p2[0] - p1[0]) * (line_y - p1[1]) / (p2[1] - p1[1]) + p1[0]
                intersections.append(intersect_x)

        intersections.sort()

        # Заполнение областей между пересечениями
        for i in range(0, len(intersections), 2):
            fill_coords.append((intersections[i], line_y))
            fill_coords.append((intersections[i + 1], line_y))

    return fill_coords

# Координаты точек фигуры
shape_points = np.array([(1, 2.5), (2.5, 4.5), (4, 2.5), (2.5, 0.5)])
filled_coords = fill_shape(shape_points)

# Визуализация
plt.fill(shape_points[:, 0], shape_points[:, 1], 'yellow')
plt.xlim(0, 6)
plt.ylim(0, 5)
plt.show()
```
![Рисунок 7](https://github.com/GOSICK070404/Comp_graph_hw/blob/main/7.png)
## 1.1 — Вращение фигуры
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import imageio

# Параметры
import os
output_file = os.path.join(os.path.expanduser("~"), "Desktop", "animated_rotate.gif")
frames = []
num_frames = 60

# Функция для вращения точки вокруг осей
def apply_rotation(vec, angles):
    x_angle, y_angle, z_angle = np.radians(angles)

    rot_x = np.array([
        [1, 0, 0],
        [0, np.cos(x_angle), -np.sin(x_angle)],
        [0, np.sin(x_angle), np.cos(x_angle)]
    ])

    rot_y = np.array([
        [np.cos(y_angle), 0, np.sin(y_angle)],
        [0, 1, 0],
        [-np.sin(y_angle), 0, np.cos(y_angle)]
    ])

    rot_z = np.array([
        [np.cos(z_angle), -np.sin(z_angle), 0],
        [np.sin(z_angle), np.cos(z_angle), 0],
        [0, 0, 1]
    ])

    return rot_z @ rot_y @ rot_x @ vec

# Создание случайных точек
n_points = 50
phi = np.linspace(0, 2 * np.pi, int(np.sqrt(n_points)), endpoint=False)
theta = np.linspace(0, np.pi, int(np.sqrt(n_points)), endpoint=True)
phi, theta = np.meshgrid(phi, theta)
phi, theta = phi.flatten(), theta.flatten()
radius = 1
x = radius * np.sin(theta) * np.cos(phi)
y = radius * np.sin(theta) * np.sin(phi)
z = radius * np.cos(theta)
points = np.column_stack((x, y, z))
points = np.column_stack((x, y, z))

# Выпуклая оболочка
hull = ConvexHull(points)

# Генерация цветов для каждой грани
face_colors = np.random.rand(len(hull.simplices), 3)

# Настройка визуализации
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for frame in range(num_frames):
    ax.clear()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_box_aspect([1, 1, 1])

    # Расчёт углов поворота для текущего кадра
    rotation_angles = (frame * 6, frame * 3, frame * 2)

    # Применение вращения к точкам
    rotated = np.array([apply_rotation(pt, rotation_angles) for pt in points])

    # Отображение оболочки
    for idx, simplex in enumerate(hull.simplices):
        verts = rotated[simplex]
        poly = Poly3DCollection([verts], color=face_colors[idx], edgecolor="k", alpha=0.8)
        ax.add_collection3d(poly)

    # Сохранение текущего кадра
    fig.canvas.draw()
    frame_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame_data = frame_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(frame_data)

# Экспорт анимации в GIF
imageio.mimsave(output_file, frames, fps=15)
print(f"Анимация сохранена в: {output_file}")
```
## 1.2 — Вращение фигуры без показа невидимых частей
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import imageio

# Параметры
import os
output_file = os.path.join(os.path.expanduser("~"), "Desktop", "animated_rotate_front.gif")
frames = []
num_frames = 60

# Функция для вращения точки вокруг осей
def apply_rotation(vec, angles):
    x_angle, y_angle, z_angle = np.radians(angles)

    rot_x = np.array([
        [1, 0, 0],
        [0, np.cos(x_angle), -np.sin(x_angle)],
        [0, np.sin(x_angle), np.cos(x_angle)]
    ])

    rot_y = np.array([
        [np.cos(y_angle), 0, np.sin(y_angle)],
        [0, 1, 0],
        [-np.sin(y_angle), 0, np.cos(y_angle)]
    ])

    rot_z = np.array([
        [np.cos(z_angle), -np.sin(z_angle), 0],
        [np.sin(z_angle), np.cos(z_angle), 0],
        [0, 0, 1]
    ])

    return rot_z @ rot_y @ rot_x @ vec

# Создание случайных точек
n_points = 50
phi = np.linspace(0, 2 * np.pi, int(np.sqrt(n_points)), endpoint=False)
theta = np.linspace(0, np.pi, int(np.sqrt(n_points)), endpoint=True)
phi, theta = np.meshgrid(phi, theta)
phi, theta = phi.flatten(), theta.flatten()
radius = 1
x = radius * np.sin(theta) * np.cos(phi)
y = radius * np.sin(theta) * np.sin(phi)
z = radius * np.cos(theta)
points = np.column_stack((x, y, z))
points = np.column_stack((x, y, z))

# Выпуклая оболочка
hull = ConvexHull(points)

# Генерация цветов для каждой грани
face_colors = np.random.rand(len(hull.simplices), 3)

# Настройка визуализации
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for frame in range(num_frames):
    ax.clear()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_box_aspect([1, 1, 1])

    # Расчёт углов поворота для текущего кадра
    rotation_angles = (frame * 6, frame * 3, frame * 2)

    # Применение вращения к точкам
    rotated = np.array([apply_rotation(pt, rotation_angles) for pt in points])

    # Отображение оболочки
    for idx, simplex in enumerate(hull.simplices):
        verts = rotated[simplex]
        poly = Poly3DCollection([verts], color=face_colors[idx], edgecolor="k", alpha=1)
        ax.add_collection3d(poly)

    # Сохранение текущего кадра
    fig.canvas.draw()
    frame_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame_data = frame_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(frame_data)

# Экспорт анимации в GIF
imageio.mimsave(output_file, frames, fps=15)
print(f"Анимация сохранена в: {output_file}")
```
## 1.3 — Вращение фигуры с отражением лучей из какой-то точки
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import imageio

# Параметры GIF
import os
output_file = os.path.join(os.path.expanduser("~"), "Desktop", "animated_rotate_light.gif")
num_frames = 240  # Количество кадров
fps = 60          # Частота кадров

# Универсальная функция вращения точки вокруг осей
def rotate_3d(point, angles):
    ax, ay, az = np.radians(angles)
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax), np.cos(ax)]])

    Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                   [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]])

    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az), np.cos(az), 0],
                   [0, 0, 1]])

    return Rz @ Ry @ Rx @ point

# Генерация точек на сфере
num_points = 50
phi = np.linspace(0, 2 * np.pi, int(np.sqrt(num_points)), endpoint=False)
theta = np.linspace(0, np.pi, int(np.sqrt(num_points)), endpoint=True)
phi, theta = np.meshgrid(phi, theta)
phi, theta = phi.flatten(), theta.flatten()

x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)
points = np.column_stack((x, y, z))

# Создание выпуклой оболочки
hull = ConvexHull(points)

# Генерация случайных цветов для граней
colors = np.random.rand(len(hull.simplices), 3)

# Настройка источника света
light_position = np.array([2, 2, 2])
base_intensity = 0.2  # Минимальный уровень освещения

# Создание фигуры для визуализации
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
frames = []

for frame in range(num_frames):
    ax.clear()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_box_aspect([1, 1, 1])

    # Углы поворота для текущего кадра
    angles = (frame * 1.5, frame * 0.75, frame * 0.5)

    # Применение вращения к точкам
    rotated_points = np.array([rotate_3d(p, angles) for p in points])

    # Визуализация граней оболочки с учётом освещения
    for idx, simplex in enumerate(hull.simplices):
        triangle = rotated_points[simplex]

        # Нормаль к грани
        v0, v1 = triangle[1] - triangle[0], triangle[2] - triangle[0]
        normal = np.cross(v0, v1)
        normal /= np.linalg.norm(normal)

        # Направление света
        face_center = np.mean(triangle, axis=0)
        light_dir = light_position - face_center
        light_dir /= np.linalg.norm(light_dir)

        # Интенсивность освещения
        intensity = np.dot(normal, light_dir)
        intensity = np.clip(intensity, 0, 1)

        # Окончательная интенсивность
        final_intensity = np.clip(intensity + base_intensity, 0, 1)
        face_color = colors[idx] * final_intensity

        # Добавление грани
        poly = Poly3DCollection([triangle], color=face_color, edgecolor='k', alpha=0.9)
        ax.add_collection3d(poly)

    # Сохранение кадра
    fig.canvas.draw()
    frame_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    frame_data = frame_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(frame_data)

# Экспорт анимации в GIF
imageio.mimsave(output_file, frames, fps=fps)
print(f"GIF сохранён в файл: {output_file}")
```
# РК1 - ДЗ1
## 1 <Поляков> сравнение производительности алгоритма Брезенхема построения отрезков и метода из библиотеки pyopengl.
```python
import time
import random
import math
from skimage.draw import line as skimage_line

# Генерация случайных отрезков по случайным координатам
def generate_segments(num_lines):
    return [(random.randint(0, 999), random.randint(0, 999), random.randint(0, 999), random.randint(0, 999)) for _ in range(num_lines)]

# Алгоритм Брезенхема
def bresenham_line_create(x1, y1, x2, y2):
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return points

# Реализация с использованием библиотеки skimage
def skimage_line_create(x1, y1, x2, y2):
    rr, cc = skimage_line(x1, y1, x2, y2)
    return list(zip(rr, cc))

# Замер времени для алгоритма Брезенхема
def bresenham_time_measure(segments):
    total_bresenham_pixels = 0
    start_time = time.time()
    for x1, y1, x2, y2 in segments:
        points = bresenham_line_create(x1, y1, x2, y2)
        total_bresenham_pixels += len(points)
    end_time = time.time()
    bresenham_time = end_time - start_time
    bresenham_density = total_bresenham_pixels / bresenham_time
    print("Время выполнения алгоритма Брезенхема:", bresenham_time)
    print("Скорость отрисовки для алгоритма Брезенхема (пиксели/сек):", bresenham_density)

# Замер времени для метода skimage
def skimage_time_measure(segments):
    total_skimage_pixels = 0
    start_time = time.time()
    for x1, y1, x2, y2 in segments:
        points = skimage_line_create(x1, y1, x2, y2)
        total_skimage_pixels += len(points)
    end_time = time.time()
    skimage_time = end_time - start_time
    skimage_density = total_skimage_pixels / skimage_time
    print("Время выполнения метода skimage:", skimage_time)
    print("Скорость отрисовки для метода skimage (пиксели/сек):", skimage_density)

# Основная функция
def main():
    num_lines = 1000000
    segments = generate_segments(num_lines)

    # Запуск замеров
    bresenham_time_measure(segments)
    skimage_time_measure(segments)

if __name__ == "__main__":
    main()
```
![Рисунок 8](https://github.com/GOSICK070404/Comp_graph_hw/blob/main/8.png)
