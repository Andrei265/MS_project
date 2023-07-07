import sys
import random
import math
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QGridLayout, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QGraphicsView, QGraphicsScene
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor, QPainterPath
from heapq import heappop, heappush
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas


class HistogramCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plotHistogram(self, damage_percentages):
        self.ax.hist(damage_percentages, bins=10, range=(0, 100))
        self.ax.set_xlabel('Процент повреждений')
        self.ax.set_ylabel('Частота')
        self.ax.set_title('Распределение повреждений от высокоточного оружия')
        self.draw()

class MonteCarloWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monte Carlo Simulation")
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.num_targets_input = QLineEdit()
        self.num_vto_input = QLineEdit()
        self.num_trials_input = QLineEdit()
        self.det_radius_input = QLineEdit()
        self.det_flight_range_input = QLineEdit()
        self.det_stddev_input = QLineEdit()
        self.plot_button = QPushButton("Построить график")

        layout.addWidget(QLabel("Количество целей:"))
        layout.addWidget(self.num_targets_input)
        layout.addWidget(QLabel("Количество ВТО:"))
        layout.addWidget(self.num_vto_input)
        layout.addWidget(QLabel("Количество испытаний:"))
        layout.addWidget(self.num_trials_input)
        layout.addWidget(QLabel("Радиус поражения ВТО:"))
        layout.addWidget(self.det_radius_input)
        layout.addWidget(QLabel("Дальность полета ВТО:"))
        layout.addWidget(self.det_flight_range_input)
        layout.addWidget(QLabel("СКО ВТО:"))
        layout.addWidget(self.det_stddev_input)
        layout.addWidget(self.plot_button)

        self.plot_button.clicked.connect(self.plotGraph)

        # Создание экземпляра HistogramCanvas
        self.histogram_canvas = HistogramCanvas(self)

        # Добавление HistogramCanvas в компоновку
        layout.addWidget(self.histogram_canvas)
        self.avg_damage_label = QLabel("Средний процент повреждений:")
        #self.prob_destroyed_label = QLabel("Вероятность поражения всей инфраструктуры:")
        layout.addWidget(self.avg_damage_label)
        #layout.addWidget(self.prob_destroyed_label)

    def updateLabels(self, avg_damage, prob_destroyed):
            self.avg_damage_label.setText(f"Средний процент повреждений: {avg_damage:.2f}%")
            #self.prob_destroyed_label.setText(f"Вероятность поражения всей инфраструктуры: {prob_destroyed:.2f}")

    def plotGraph(self):
        num_targets = int(self.num_targets_input.text()) # Количество целей
        num_weapons = int(self.num_vto_input.text()) # Количество ВТО
        num_trials = int(self.num_trials_input.text()) # Количество испытаний

        # Параметры высокоточного оружия (ВТО)
        det_radius = int(self.det_radius_input.text())  # Радиус поражения ВТО
        det_flight_range = int(self.det_flight_range_input.text())  # Дальность полета ВТО
        det_stddev = int(self.det_stddev_input.text())  # СКО ВТО

        damage_percentages = []  # Список для хранения процентов повреждений

        destroyed_infrastructure_count = 0  # Счетчик разрушенной инфраструктуры

        # Для каждого испытания запускаем цикл для поражения транспортных узлов
        for _ in range(num_trials):
            # Генерируем случайные координаты оружия и целей
            weapon_coordinates = []
            target_coordinates = []

            # Генерируем случайные координаты для высокоточного оружия
            for _ in range(num_weapons):
                weapon_x = random.uniform(0, 1000)
                weapon_y = random.uniform(0, 1000)
                weapon_coordinates.append((weapon_x, weapon_y))

            # Генерируем случайные координаты для целей
            for _ in range(num_targets):
                target_x = random.uniform(0, 1000)
                target_y = random.uniform(0, 1000)
                target_coordinates.append((target_x, target_y))

            # Проверяем каждое оружие и цель на возможное поражение
            damage_count = 0  # Счетчик поврежденных транспортных узлов
            for weapon in weapon_coordinates:
                for target in target_coordinates:
                    weapon_x, weapon_y = weapon
                    target_x, target_y = target

                    # Проверяем, если цель находится внутри радиуса поражения ВТО
                    distance = ((weapon_x - target_x) ** 2 + (weapon_y - target_y) ** 2) ** 0.5
                    if distance <= det_radius:
                        # Генерируем случайную ошибку с использованием СКО ВТО
                        rand_error = float(random.normalvariate(0, det_stddev))/10.0

                        # Вычисляем урон в зависимости от расстояния и ошибки
                        damage_ratio = 25 -  rand_error

                        # Увеличиваем счетчик поврежденных узлов
                        if damage_ratio > 0:
                            damage_count += 1

            # Вычисляем процент повреждений
            damage_percentage = (damage_count / num_targets) * 100

            # Добавляем процент повреждений в список
            damage_percentages.append(damage_percentage)

            # Проверяем, если все узлы повреждены, увеличиваем счетчик разрушенной инфраструктуры
            if damage_count == num_targets:
                destroyed_infrastructure_count += 1

        # Выводим средний процент повреждений
        avg_damage = sum(damage_percentages) / len(damage_percentages)
        print(f"Средний процент повреждений: {avg_damage:.2f}%")

        # Вычисляем вероятность поражения всей инфраструктуры
        prob_destroyed_infrastructure = destroyed_infrastructure_count / num_trials
        #print(f"Вероятность поражения всей инфраструктуры: {prob_destroyed_infrastructure:.2f}")

        # Построение гистограммы распределения повреждений
        self.histogram_canvas.plotHistogram(damage_percentages)
        avg_damage = sum(damage_percentages) / len(damage_percentages) % 100
        prob_destroyed_infrastructure = destroyed_infrastructure_count / num_trials

        self.histogram_canvas.plotHistogram(damage_percentages)
        self.updateLabels(avg_damage, prob_destroyed_infrastructure)

        
class GraphWidget(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 0.8

        if event.angleDelta().y() > 0:
            # Увеличение масштаба при прокрутке вверх
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            # Уменьшение масштаба при прокрутке вниз
            self.scale(zoom_out_factor, zoom_out_factor)


    def drawGraph(self, adjacency_matrix, optimal_route=None):
        self.scene.clear()

        num_vertices = len(adjacency_matrix)
        node_positions = self.calculateNodePositions(num_vertices)

        arrow_pen = QPen(Qt.black)
        arrow_pen.setWidth(2)
        arrow_pen.setCapStyle(Qt.RoundCap)
        arrow_pen.setJoinStyle(Qt.RoundJoin)

        # создание вершин
        for i in range(num_vertices):
            for j in range(num_vertices):
                if i == j:
                    adjacency_matrix[i][j] = 0
                if adjacency_matrix[i][j] > 0:
                    x1, y1 = node_positions[i]
                    x2, y2 = node_positions[j]

                    # создание линии стрелки
                    path = QPainterPath(QPointF(x1, y1))
                    path.lineTo(x2, y2)

                    # Расчт координат стрелок
                    angle = math.atan2(y2 - y1, x2 - x1)
                    arrowhead_length = 15
                    arrowhead_angle = math.radians(30)
                    arrowhead_x = x2 - arrowhead_length * math.cos(angle + arrowhead_angle)
                    arrowhead_y = y2 - arrowhead_length * math.sin(angle + arrowhead_angle)

                    arrowhead_path = QPainterPath()
                    arrowhead_path.moveTo(x2, y2)
                    arrowhead_path.lineTo(arrowhead_x, arrowhead_y)
                    arrowhead_path.lineTo(x2 - arrowhead_length * math.cos(angle - arrowhead_angle),
                                        y2 - arrowhead_length * math.sin(angle - arrowhead_angle))
                    path.addPath(arrowhead_path)

                    if optimal_route and (i, j) in optimal_route:
                        arrow_pen.setColor(Qt.red)
                    else:
                        arrow_pen.setColor(Qt.black)

                    self.scene.addPath(path, arrow_pen)

                    weight = str(adjacency_matrix[i][j])
                    self.scene.addText(weight).setPos((x1 + x2) / 2, (y1 + y2) / 2)
                    arrow_pen.setColor(Qt.black)

        for i in range(num_vertices):
            node = self.scene.addEllipse(node_positions[i][0] - 10, node_positions[i][1] - 10, 20, 20)
            self.scene.addText(str(i + 1)).setPos(node_positions[i][0] - 10, node_positions[i][1] - 10)
            node.setBrush(Qt.white)

        

    def calculateNodePositions(self, num_vertices):
        positions = []
        angle = 360 / num_vertices
        radius = 100
        center_x = self.width() / 2
        center_y = self.height() / 2

        for i in range(num_vertices):
            x = center_x + radius * math.cos(math.radians(angle * i))
            y = center_y + radius * math.sin(math.radians(angle * i))
            positions.append((x, y))

        return positions


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graph Application")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.adjacency_matrix = [[0] * 10 for _ in range(10)]

        self.matrix_inputs = []
        self.graph_widget = GraphWidget()
        self.start_point_input = QLineEdit()
        self.end_point_input = QLineEdit()
        self.generate_button = QPushButton("Автозаполнение")
        self.build_graph_button = QPushButton("Построить граф")
        self.find_route_button = QPushButton("Найти оптимальный маршрут")
        self.monte_carlo_widget = MonteCarloWidget() 

        #self.build_graph_button.setEnabled(False)

        self.layout.addLayout(self.createMatrixInputsLayout())
        self.layout.addLayout(self.createButtonLayout())

        graph_layout = QHBoxLayout()
        graph_layout.addWidget(self.graph_widget)
        graph_layout.addWidget(self.monte_carlo_widget) 

        route_layout = QHBoxLayout()
        route_layout.addWidget(QLabel("Начальная точка:"))
        route_layout.addWidget(self.start_point_input)
        route_layout.addWidget(QLabel("Конечная точка:"))
        route_layout.addWidget(self.end_point_input)

        self.layout.addLayout(graph_layout)
        self.layout.addLayout(route_layout)

        self.generate_button.clicked.connect(self.generateMatrix)
        self.build_graph_button.clicked.connect(self.buildGraph)
        self.find_route_button.clicked.connect(self.findOptimalRoute)
        self.monte_carlo_widget.plot_button.clicked.connect(self.monte_carlo_widget.plotGraph) 

    def createMatrixInputsLayout(self):
        layout = QGridLayout()

        for i in range(10):
            row_inputs = []
            for j in range(10):
                input_field = QLineEdit()
                input_field.setMaximumWidth(40)
                layout.addWidget(input_field, i, j)
                row_inputs.append(input_field)
            self.matrix_inputs.append(row_inputs)

        return layout

    def createButtonLayout(self):
        layout = QHBoxLayout()

        layout.addWidget(self.generate_button)
        layout.addWidget(self.build_graph_button)
        layout.addWidget(self.find_route_button)

        return layout

    def generateMatrix(self):
        self.resetMatrixInputs()

        num_non_zero = random.randint(1, 15)
        available_cells = [(i, j) for i in range(10) for j in range(10)]
        random.shuffle(available_cells)

        for _ in range(num_non_zero):
            i, j = available_cells.pop()
            value = random.randint(1, 10)
            if i != j:
                self.matrix_inputs[i][j].setText(str(value))
                self.adjacency_matrix[i][j] = value
            else:
                self.matrix_inputs[i][j].setText(str(0))
                self.adjacency_matrix[i][j] = 0

        # Заполнение пустых полей нулями
        for i in range(10):
            for j in range(10):
                if not self.matrix_inputs[i][j].text():
                    self.matrix_inputs[i][j].setText('0')
                    self.adjacency_matrix[i][j] = 0

    def resetMatrixInputs(self):
        for row_inputs in self.matrix_inputs:
            for input_field in row_inputs:
                input_field.clear()

        self.adjacency_matrix = [[0] * 10 for _ in range(10)]

    def buildGraph(self):
        adjacency_matrix = []
        for row_inputs in self.matrix_inputs:
            row_values = []
            for input_field in row_inputs:
                value = int(input_field.text()) if input_field.text() else 0
                row_values.append(value)
            adjacency_matrix.append(row_values)

        self.graph_widget.drawGraph(adjacency_matrix)

    def findOptimalRoute(self):
        start_point = int(self.start_point_input.text()) - 1
        end_point = int(self.end_point_input.text()) - 1

        num_vertices = len(self.adjacency_matrix)
        distances = [float('inf')] * num_vertices  # Расстояния до вершин
        distances[start_point] = 0
        optimal_route = []  # Оптимальный маршрут
        previous_vertices = [-1] * num_vertices  # Предыдущие вершины в оптимальном маршруте

        # Очередь с приоритетом для обработки вершин
        pq = [(0, start_point)]

        while pq:
            current_distance, current_vertex = heappop(pq)

            # Если достигнута конечная вершина, завершаем поиск
            if current_vertex == end_point:
                break

            # Пропускаем уже обработанные вершины
            if current_distance > distances[current_vertex]:
                continue

            # Перебираем соседние вершины
            for neighbor in range(num_vertices):
                weight = self.adjacency_matrix[current_vertex][neighbor]
                if weight > 0:
                    distance = current_distance + weight

                    # Если найдено более короткое расстояние, обновляем
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous_vertices[neighbor] = current_vertex
                        heappush(pq, (distance, neighbor))

        if distances[end_point] == float('inf'):
            message = f"Пути из вершины {start_point + 1} в вершину {end_point + 1} не существует"
            QMessageBox.warning(self, "Ошибка", message)
        else:
            # Восстанавливаем оптимальный маршрут
            current_vertex = end_point
            while current_vertex != start_point:
                previous_vertex = previous_vertices[current_vertex]
                optimal_route.append((previous_vertex, current_vertex))
                current_vertex = previous_vertex

            # Разворачиваем оптимальный маршрут
            optimal_route.reverse()

            self.graph_widget.drawGraph(self.adjacency_matrix, optimal_route)
        self.build_graph_button.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())