"""Este módulo contém a classe GameEngine, que é responsável por gerenciar a lógica do jogo,
incluindo a leitura de poses-alvo, detecção de poses do jogador, cálculo de distância entre poses
e renderização das poses na tela. 
Ele também inclui a classe Metric para mostrar na tela a distância entre a pose alvo e a pose 
detectada.
"""

import json

import cv2
import numpy as np

# Define a conectividade entre os pontos para desenhar um esqueleto
POSE_STRUCTURE = {
    "head": {
        "keypoints": [0, 1, 2, 3, 4],
        # Conexões entre os keypoints 
        "edges": [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)], 
        "color": (0, 255, 255)
    },
    "torso": {
        "keypoints": [5, 6, 11, 12],
        "edges": [(5, 6), (11, 12), (5, 11), (6, 12)],
        "color": (0, 255, 0)
    },
    "arms": {
        "keypoints": [7, 8, 9, 10],
        "edges": [(5, 7), (7, 9), (6, 8), (8, 10)],
        "color": (0, 0, 255, 50)
    },
    "legs": {
        "keypoints": [13, 14, 15, 16],
        "edges": [(11, 13), (13, 15), (12, 14), (14, 16)],
        "color": (255, 0, 0) 
    }
}

class GameEngine:
    """Classe responsável por gerenciar a lógica do jogo."""

    def __init__(
            self, 
            distance_threshold = 10.0, 
            conf_threshold = 0.5,
            position = (0, 0), 
            height = 400
            ):
        """Inicializa o GameEngine.
        
        Parametros:
        - distance_threshold: O limiar de distância para considerar a pose como alcançada.
        - conf_threshold: O limiar de confiança para considerar um keypoint como detectado.
        - position: A posição (x, y) onde a pose-alvo deve ser centralizada na imagem.
        - height: A altura (em pixels) da pose-alvo, que determina o tamanho da pose desenhada na tela.
        """
        
        self.position = np.array(position)
        self.height = height
        self.distance_threshold = distance_threshold
        self.conf_threshold = conf_threshold
        # Carrega as poses-alvo a partir do arquivo JSON. O arquivo deve conter um dicionário 
        # onde as chaves são os nomes das poses e os valores são listas de coordenadas dos keypoints
        self.poses = json.load(open("poses.json"))
        # Seleciona uma pose-alvo inicial
        self.target_pose = self.new_pose()
        # Cria um objeto SmoothMetric para suavizar a distância entre a pose detectada e a pose-alvo,
        # e exibir essa distância na tela. O valor inicial é definido como o dobro do limiar de 
        # distância para que a pose não seja considerada alcançada no início do jogo, dando ao 
        # jogador tempo para se posicionar.
        self.distance_smoother = SmoothMetric(
            initial_value=2*distance_threshold,name="Distance", alpha=0.1, font_scale=0.8)

    def new_pose(self):
        """Seleciona uma nova pose-alvo aleatória a partir do conjunto de poses carregado."""

        # Seleciona uma pose
        pose = self.poses[np.random.choice(list(self.poses.keys()))]
        # Normaliza a pose para que a altura seja igual a self.height 
        pose = np.array(pose) * self.height
        # Centraliza a pose, de forma que o centro do esqueleto seja o ponto (0, 0)
        pose = pose - np.mean(pose, axis=0)
        # Translada a pose para a posição desejada na tela, adicionando o deslocamento definido 
        # por self.position
        pose = self.position + pose

        return pose

    def process_frame(self, frame):

        # Detecta a pose do jogador
        detected_pose, kpt_confidence = self.detect_pose(frame)
        if detected_pose is None:
            # Se não for possível detectar uma pose, desenha apenas a pose-alvo no frame para 
            # servir como referência visual
            tgt_conf = np.ones(len(self.target_pose)) 
            frame_with_target = self.draw_pose(frame, self.target_pose, tgt_conf, brightness=0.4)
            return frame_with_target
        
        # Calcula a distância entre a pose detectada e a pose-alvo
        distance = self.pose_distance(detected_pose, kpt_confidence, self.target_pose)
        self.distance_smoother.update(distance)

        distance = self.distance_smoother.get_value()
        if distance < self.distance_threshold:
            # Comandos para quando a pose-alvo for alcançada (por exemplo, atualizar a pontuação, 
            # mostrar uma mensagem, etc.)
            ...
            # Seleciona uma nova pose-alvo para o próximo desafio
            self.new_pose()

        # Desenha a pose-alvo e a pose detectada no frame para fornecer feedback visual ao jogador. 
        # A pose-alvo é desenhada com menor brilho para servir como referência, enquanto a 
        # pose detectada é desenhada com brilho total para destacar a posição atual do jogador.
        tgt_conf = np.ones(len(self.target_pose)) 
        frame_with_target = self.draw_pose(frame, self.target_pose, tgt_conf, brightness=0.4)
        display_img = self.draw_pose(frame_with_target, detected_pose, kpt_confidence, brightness=1.)

        # Adiciona a distância suavizada entre a pose detectada e a pose-alvo no frame
        display_img = self.distance_smoother.draw(display_img)

        return display_img

    def detect_pose(self, frame):
        """Detecta a pose de um jogador em uma imagem."""

        # Para simular a detecção, geramos coordenadas aleatórias próximas à pose-alvo. 
        # Isso é apenas um exemplo e deve ser substituído pelo modelo YOLO26
        np.random.seed(0)
        pose = self.target_pose + 60 + np.random.randint(-20, 20, size=(17, 2))
        # Simula uma confiança de detecção perfeita para todos os keypoints. Na prática, o modelo 
        # de detecção retorna valores de confiança, para cada keypoint
        kpt_confidence = np.ones(len(pose))  

        return pose, kpt_confidence
    
    def pose_distance(self, pose1, kpt_confidence, pose2):   
        """Calcula a distância entre duas poses usando a média das distâncias entre os keypoints
        correspondentes.
        """

        pose1 = np.array(pose1)
        pose2 = np.array(pose2)
        dist = 0.0
        count = 0
        for p1, c, p2 in zip(pose1, kpt_confidence, pose2):
            if c > self.conf_threshold:
                dist += np.linalg.norm(p1 - p2)
                count += 1
        if count > 0:
            mean_distance = dist / count
        else:
            # Se nenhum keypoint tiver confiança suficiente, definimos a distância como um valor 
            # alto
            mean_distance = 2 * self.distance_threshold

        return mean_distance

    def draw_pose(self, img, pose, kpt_confidence, radius=5, thickness=2, brightness=0.5):
        """Plota os keypoints e as conexões (esqueleto) em uma imagem."""

        img_copy = img.copy()
        
        for part, data in POSE_STRUCTURE.items():
            color = np.array(data["color"])
            color = (color * brightness).astype(np.uint8).tolist()
            
            for edge in data["edges"]:
                idx1, idx2 = edge

                # Checa se ambos os pontos têm confiança suficiente para serem desenhados
                if kpt_confidence[idx1] > self.conf_threshold and kpt_confidence[idx2] > self.conf_threshold:    
                    pt1 = (int(pose[idx1][0]), int(pose[idx1][1]))
                    pt2 = (int(pose[idx2][0]), int(pose[idx2][1]))

                    # Em alguns casos, os pontos podem ser (0, 0) se não forem detectados 
                    # corretamente, então é bom checar isso antes de desenhar a linha
                    if pt1 != (0, 0) and pt2 != (0, 0):
                        cv2.line(img_copy, pt1, pt2, color, thickness)
            
            # Desenha os keypoints como círculos
            for kp_idx in data["keypoints"]:

                pt = (int(pose[kp_idx][0]), int(pose[kp_idx][1]))
                if kpt_confidence[kp_idx] > self.conf_threshold and pt != (0, 0):
                    cv2.circle(img_copy, pt, radius, color, -1)

        return img_copy
    
class SmoothMetric:
    """Suaviza valores de uma métrica para evitar flutuações bruscas de valores e exibe o valor
    suavizado na tela.
    """
    def __init__(self, initial_value=0.0, name="Metric", alpha=0.05, font_scale=2.0):
        """Inicializa o objeto Metric.
        
        Parametros:
        - name: O nome da métrica a ser exibida na tela.
        - alpha: O fator de suavização para o cálculo da média móvel exponencial (EMA). 
                 Valores mais próximos de 0 dão mais peso aos valores antigos, enquanto valores 
                 mais próximos de 1 dão mais peso aos valores recentes.
        - font_scale: O tamanho da fonte para exibir a métrica na tela.
        """

        self.name = name
        self.font_scale = font_scale
        
        # Configura os pesos para a média móvel exponencial (EMA)
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self.step = 0
        # Valor inicial da métrica suavizada (EMA)
        self.value_ema = initial_value


    def update(self, value):
        """Adiciona um novo valor à métrica suavizada."""

        self.step += 1
        # Calcula a média móvel exponencial (EMA) usando o novo valor e o valor anterior da EMA
        self.value_ema = (self.alpha * value) + (self.beta * self.value_ema)

    def get_value(self):
        """Retorna o valor suavizado da métrica. Aplica correção de viés para os primeiros passos,
        quando a medida ainda não está estabilizada.
        """
        if self.step == 0:
            return self.value_ema

        bias_correction = 1.0 - (self.beta ** self.step)
        value_corrected = self.value_ema / bias_correction

        return value_corrected

    def draw(self, frame):
        """Desenha a métrica suavizada no frame."""

        value = self.get_value()

        if frame.ndim == 3 and frame.shape[2] == 3:
            color = (0, 255, 0)
        else:
            color = 255

        text = f"{self.name}: {int(value)}"
        cv2.putText(frame, text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, 2)

        return frame