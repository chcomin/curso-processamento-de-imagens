"""Funções auxiliares para o projeto."""

import numpy as np
import timm
import torch
from PIL import Image
from timm.data.transforms_factory import create_transform


def detect(img, model, class_name, conf=0.2):
    """Detecta objetos da categoria especificada na imagem usando o modelo YOLO, e retorna uma
    máscara binária combinada de todas as instâncias detectadas da categoria.

    Parâmetros:
    - img: imagem de entrada (PIL Image)
    - model: modelo YOLO pré-treinado para detecção de objetos e segmentação de instâncias
    - class_name: nome da classe a ser detectada (string, deve ser uma das classes reconhecidas 
        pelo modelo)
    - conf: limiar de confiança para considerar uma detecção válida (float entre 0 e 1). 

    Retorna:
    - combined_mask: máscara binária combinada de todas as instâncias detectadas da categoria, 
    onde os pixels pertencentes a qualquer uma das instâncias são marcados como True. Retorna
    None se nenhuma instância da categoria for detectada.
    """

    results = model(
    img, 
    conf=conf, 
    # Retorna máscaras com a mesma resolução da imagem de entrada
    retina_masks=True  
    )[0]

    # Encontra o índice da classe especificada na lista de classes do modelo 
    class_index = list(results.names.values()).index(class_name)

    # Percorre os resultados da detecção, e para cada detecção da classe especificada, extrai 
    # a máscara
    mask_list = []
    for result in results:
        obj_box = result.boxes
        obj_mask = result.masks.data[0]
        if int(obj_box.cls.item()) == class_index:
            mask_list.append(obj_mask.cpu().numpy())

    if len(mask_list) == 0:
        return None
    # Cria um array com dimensão n x H x W, onde n é o número de máscaras detectadas, e H e W 
    # são as dimensões da imagem de entrada
    combined_mask = np.stack(mask_list, axis=0)
    # Combina as máscaras em um único array binário, onde os pixels pertencentes a qualquer uma 
    # das máscaras são marcados como True
    combined_mask = np.any(combined_mask, axis=0)

    combined_mask = Image.fromarray(combined_mask.astype(np.uint8) * 255)

    return combined_mask

def crop(img, mask):
    """Recorta a caixa delimitadora da região da imagem correspondente à máscara, retornando a
    imagem e a máscara recortada.

    Parâmetros:
    - img: imagem de entrada (PIL Image)
    - mask: máscara binária, onde os pixels do objeto são brancos e o fundo é preto (PIL Image)

    Retorna:
    - obj_img: imagem recortada da região correspondente à máscara (PIL Image)
    - obj_mask: máscara recortada da região correspondente à máscara (PIL Image)
    """

    # Encontra os índices dos pixels onde o valor da máscara é True, e calcula a caixa delimitadora
    inds = np.nonzero(np.array(mask))
    box = [inds[1].min().item(), inds[0].min().item(), inds[1].max().item(), inds[0].max().item()]

    # Recorta a imagem e a máscara usando a caixa delimitadora
    obj_img = img.crop(box)
    obj_mask = mask.crop(box)

    return obj_img, obj_mask

def paste(background, obj_img, obj_mask, obj_size, obj_pos):
    """Redimensiona a imagem do objeto e a máscara para o tamanho especificado, e cola o objeto na
    posição especificada sobre a imagem de fundo, usando a máscara para preservar a transparência.

    Parâmetros:
    - background: imagem de fundo (PIL Image)
    - obj_img: imagem do objeto a ser colado (PIL Image)
    - obj_mask: máscara binária do objeto, onde os pixels do objeto são brancos e o fundo é preto (PIL Image)
    - obj_size: tupla (largura, altura) com o tamanho para redimensionar a imagem do objeto e a máscara
    - obj_pos: tupla (x, y) com a posição (coordenadas do canto superior esquerdo) onde o objeto deve ser colado sobre a imagem de fundo

    Retorna:
    - mixed_img: nova imagem resultante da colagem do objeto sobre a imagem de fundo (PIL Image)
    """

    obj_img = obj_img.resize(obj_size)
    obj_mask = obj_mask.resize(obj_size)

    mixed_img = background.copy()
    mixed_img.paste(obj_img, obj_pos, mask=obj_mask)

    return mixed_img

def load_model(model_name):
    """Carrega a CNN pré-treinada e as classes do ImageNet a partir do nome de um modelo."""

    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    
    data_config = timm.data.resolve_data_config(model=model)
    transforms = create_transform(**data_config)
    
    return model, transforms

def apply_model(img, model, transforms):
    """Aplica a CNN pré-treinada na imagem e retorna a predição do modelo."""

    img_t = transforms(img)
    batch = img_t.unsqueeze(0)

    with torch.inference_mode():
        output = model(batch)

    output = output[0]
    probs = torch.softmax(output, dim=0)
    
    return probs

def build_results(probs, classes, true_class):
    """Constrói um dicionário com os resultados da predição do modelo. Retorna a classe
    verdadeira, a classe prevista e a segunda classe mais provável, junto com suas respectivas
    probabilidades.
    """

    values, indices = torch.topk(probs, k=2)
    
    prob_true = probs[classes.index(true_class)].item()

    results = {
        "true_class": (true_class, prob_true),
        "predicted_class": (classes[indices[0]], values[0].item()),
        "second_predicted": (classes[indices[1]], values[1].item()),
    }

    return results